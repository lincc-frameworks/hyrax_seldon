"""Turn raw per-point light-curve columns into SELDON's model inputs.

This is the implementation of :meth:`hyrax_seldon.models.seldon.HyraxSeldon.prepare_inputs`.
Datasets emit raw values (MJD, flux on the ELAsTiCC ``FLUXCAL`` scale, flux error, band
name, detection flag); the preprocessing itself is SELDON's own ``LightCurvePreprocessor``
and ``enhanced_collate_fn``, run per batch with the settings and statistics of the run config
(see :func:`hyrax_seldon._seldon.preprocess`), as SELDON's
``SELDON_Data_Generator_forecaster.py`` does for new data.

Before handing light curves to SELDON this module does what SELDON's parquet loader does:
it keeps finite points (and only detections, if the run excluded non-detections), divides
flux and error by ``zero_point_flux``, and stores values as float32. SELDON's preprocessor
is then run without augmentations or resampling noise, so the encoder sees the earliest
``max_points`` points. SELDON would pick a random ``max_length`` window from a longer light
curve; here it gets the earliest ``max_length`` points instead.

SELDON cannot encode a light curve with fewer than ``min_points`` points, and neither can
this module: such a batch raises.
"""

import numpy as np

from hyrax_seldon import _seldon
from hyrax_seldon.settings import SeldonSettings

#: Name of the batch group the light-curve fields are read from.
DATA_GROUP = "data"


def prepare_inputs(data_dict: dict, config: dict) -> dict[str, np.ndarray]:
    """Build SELDON's model inputs from a collated Hyrax batch.

    Parameters
    ----------
    data_dict : dict
        The collated batch, with a ``"data"`` group holding the per-point fields named in
        ``[model.HyraxSeldon]``, padded per row, each optionally with a ``<field>_mask``.
    config : dict
        The Hyrax runtime configuration.

    Returns
    -------
    dict[str, numpy.ndarray]
        See :func:`encode_light_curves`.
    """
    if DATA_GROUP not in data_dict:
        raise RuntimeError(
            f"HyraxSeldon reads light curves from the '{DATA_GROUP}' group of the data request, but "
            f"the batch only has {sorted(data_dict)}. Name the light-curve dataset '{DATA_GROUP}' "
            "in [data_request]."
        )
    return encode_batch(data_dict[DATA_GROUP], SeldonSettings.from_config(config))


def _select_columns(data: dict, settings: SeldonSettings):
    """Return ``(time, flux, flux_err, band, detected, valid)``, each ``(batch, length)``."""
    fields = (
        settings.time_field,
        settings.flux_field,
        settings.flux_err_field,
        settings.band_field,
        settings.detected_field,
    )
    missing = [name for name in fields if name not in data]
    if missing:
        raise RuntimeError(
            f"HyraxSeldon could not find the field(s) {missing} in the batch. Available fields: "
            f"{sorted(data)}. Set time_field / flux_field / flux_err_field / band_field / "
            "detected_field in [model.HyraxSeldon] and list them in the data request's 'fields'."
        )

    time = np.asarray(data[settings.time_field], dtype=np.float64)
    flux = np.asarray(data[settings.flux_field], dtype=np.float64)
    flux_err = np.asarray(data[settings.flux_err_field], dtype=np.float64)
    band = np.asarray(data[settings.band_field])
    detected = np.asarray(data[settings.detected_field]).astype(bool)

    shapes = {array.shape for array in (time, flux, flux_err, band, detected)}
    if len(shapes) > 1 or time.ndim != 2:
        raise RuntimeError(
            "HyraxSeldon expects every light-curve field to collate to the same (batch, length) "
            f"shape, but got shapes {sorted(shapes)}."
        )

    valid = np.ones(time.shape, dtype=bool)
    for name in fields:
        mask = data.get(f"{name}_mask")
        if mask is not None:
            valid &= np.asarray(mask).astype(bool)
    valid &= np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    if not settings.include_non_detections:
        valid &= detected

    return time, flux, flux_err, band, detected, valid


def _band_indices(band: np.ndarray, settings: SeldonSettings) -> np.ndarray:
    """Map an array of band names to SELDON band indices."""
    names, inverse = np.unique(band.astype(str), return_inverse=True)
    lookup = np.array([settings.band_to_index(name) for name in names], dtype=np.int64)
    return lookup[inverse].reshape(band.shape)


def encode_light_curves(light_curves, settings: SeldonSettings) -> dict[str, np.ndarray]:
    """Encode light curves of valid points into SELDON's model inputs.

    Parameters
    ----------
    light_curves : list[tuple]
        One ``(time, flux, flux_err, band_idx, detected)`` tuple of per-point arrays per light
        curve, in any order: MJD, flux and flux error on the scale SELDON divides by
        ``zero_point_flux``, SELDON band index, and detection flag.
    settings : SeldonSettings
        Settings from the run config.

    Returns
    -------
    dict[str, numpy.ndarray]
        With ``P = max_points`` and ``L = max_length`` from the run config:

        - ``x_input`` ``(B, P, 4)`` float32: time, flux, flux error, detection flag of the
          earliest ``P`` points (the encoder input)
        - ``band_idx`` ``(B, P, 1)`` int64, ``pad_mask_part`` ``(B, P)`` bool
        - ``time_full`` ``(B, L, 1)`` float32: times of the earliest ``L`` points (where the
          decoder is evaluated)
        - ``band_idx_full`` ``(B, L, 1)`` int64, ``pad_mask_full`` ``(B, L)`` bool
        - ``n_points`` ``(B,)`` int64: valid points in each light curve, before trimming

    Raises
    ------
    ValueError
        If a light curve has fewer than ``min_points`` points; SELDON cannot encode it.
    """
    n_points = np.array([len(light_curve[0]) for light_curve in light_curves], dtype=np.int64)
    too_short = np.flatnonzero(n_points < settings.min_points)
    if len(too_short):
        raise ValueError(
            f"Light curves at batch rows {too_short.tolist()} have {n_points[too_short].tolist()} "
            f"points, fewer than the run config's min_points={settings.min_points}; SELDON cannot "
            "encode them."
        )

    arrays, band_indices, detection_flags = [], [], []
    for time, flux, flux_err, band_idx, detected in light_curves:
        # SELDON picks a random window from longer light curves; keep the earliest instead.
        keep = slice(None)
        if len(time) > settings.max_length:
            keep = np.argsort(time, kind="stable")[: settings.max_length]
        arrays.append(
            np.column_stack(
                [
                    np.asarray(time)[keep],
                    np.asarray(flux)[keep] / settings.zero_point_flux,
                    np.asarray(flux_err)[keep] / settings.zero_point_flux,
                ]
            ).astype(np.float32)
        )
        band_indices.append(np.asarray(band_idx, dtype=np.int64)[keep])
        detection_flags.append(np.asarray(detected, dtype=bool)[keep])

    dataset_config = settings.run_config["dataset"]["config"]
    batch = _seldon.preprocess(
        arrays,
        band_indices,
        detection_flags,
        flux_stats=dataset_config["flux_stats"],
        band_weights=settings.band_weights,
        flux_normalization=dataset_config.get("flux_normalization", "minmax"),
        time_normalization=dataset_config.get("time_normalization", "std"),
        min_points=settings.min_points,
        max_points=settings.max_points,
        max_length=settings.max_length,
        t_offset=settings.t_offset,
    )
    return {
        "x_input": batch["x_input"],
        "band_idx": batch["band_idx"],
        "pad_mask_part": batch["pad_mask_part"],
        "time_full": batch["time_full"],
        "band_idx_full": batch["band_idx_full"],
        "pad_mask_full": batch["pad_mask_full"],
        "n_points": n_points,
    }


def encode_batch(data: dict, settings: SeldonSettings) -> dict[str, np.ndarray]:
    """Encode a batch of padded light-curve columns into SELDON's model inputs.

    Parameters
    ----------
    data : dict
        The batch's ``"data"`` group: per-point fields of shape ``(batch, length)``.
    settings : SeldonSettings
        Settings from the run config.

    Returns
    -------
    dict[str, numpy.ndarray]
        See :func:`encode_light_curves`.
    """
    time, flux, flux_err, band, detected, valid = _select_columns(data, settings)

    band_idx = np.zeros(band.shape, dtype=np.int64)
    band_idx[valid] = _band_indices(band[valid], settings)

    columns = (time, flux, flux_err, band_idx, detected)
    return encode_light_curves(
        [tuple(column[i][valid[i]] for column in columns) for i in range(len(time))], settings
    )
