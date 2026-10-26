"""Turn raw per-point light-curve columns into SELDON's model inputs.

This is the implementation of :meth:`hyrax_seldon.models.seldon.HyraxSeldon.prepare_inputs`.
Datasets emit raw values (MJD, flux on the ELAsTiCC ``FLUXCAL`` scale, flux error, band
name, detection flag); everything the model does to them is configured by the SELDON run
config, through :class:`~hyrax_seldon.settings.SeldonSettings`.

It reproduces SELDON's inference-relevant preprocessing chain
(``TAE/datasets/lightcurve_dataloader2.py``, ``lightcurve_preprocessor.py``) with
deterministic point selection:

1. Keep finite points (and only detections, if the run excluded non-detections); divide
   flux and error by ``zero_point_flux``; sort by time.
2. Keep the most recent ``max_length`` points for the full arrays, and the most recent
   ``max_points`` of those for the encoder.
3. Flux: ``sign(f) * log10(1 + |f|)``, then divide by the same transform of
   ``flux_stats.max`` and subtract 0.5, with errors propagated (SELDON's own transforms).
4. Time: ``(t - t0) / t_max + t_offset``, with ``t0`` the earliest encoder point.
5. Pad: values are edge-padded and band indices zero-padded, with boolean padding masks,
   exactly as SELDON pads.

There is no random window, resampling noise or augmentation.
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
        See :func:`encode_batch`.
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


def _edge_pad(values: np.ndarray, length: int) -> np.ndarray:
    """Pad the first axis to ``length`` by repeating the last row, as SELDON does."""
    if len(values) == 0:
        return np.zeros((length, *values.shape[1:]), dtype=values.dtype)
    widths = [(0, length - len(values))] + [(0, 0)] * (values.ndim - 1)
    return np.pad(values, widths, mode="edge")


def encode_light_curve(time, flux, flux_err, band_idx, detected, settings: SeldonSettings, transform=None):
    """Encode one light curve of valid points into SELDON's model inputs.

    Parameters
    ----------
    time, flux, flux_err : numpy.ndarray
        Per-point MJD, flux and flux error, in any order. Flux is on the scale SELDON
        divides by ``zero_point_flux``.
    band_idx : numpy.ndarray
        Per-point SELDON band index.
    detected : numpy.ndarray
        Per-point detection flag.
    settings : SeldonSettings
        Settings from the run config.
    transform : optional
        SELDON's flux transform for ``settings.flux_max``; built if not given.

    Returns
    -------
    dict[str, numpy.ndarray]
        One row of the arrays :func:`encode_batch` returns.
    """
    if transform is None:
        transform = _seldon.flux_transform(settings.flux_max)

    max_points, max_length = settings.max_points, settings.max_length

    n_points = len(time)
    order = np.argsort(time, kind="stable")[-max_length:]
    time = np.asarray(time, dtype=np.float64)[order]
    flux = np.asarray(flux, dtype=np.float64)[order] / settings.zero_point_flux
    flux_err = np.asarray(flux_err, dtype=np.float64)[order] / settings.zero_point_flux
    band_idx = np.asarray(band_idx, dtype=np.int64)[order]
    detected = np.asarray(detected, dtype=bool)[order]

    n_full = len(time)
    n_part = min(n_full, max_points)

    if n_full:
        flux, flux_err = transform.transform(flux, flux_err)
        t0 = time[n_full - n_part]
        time = (time - t0) / settings.t_max + settings.t_offset

    light_curve = np.stack([time, flux, flux_err, detected.astype(np.float64)], axis=1)
    part = slice(n_full - n_part, n_full)

    pad_mask_full = _seldon.padding_mask(max_length).transform(time)
    pad_mask_part = _seldon.padding_mask(max_points).transform(time[part])

    return {
        "x_input": _edge_pad(light_curve[part], max_points).astype(np.float32),
        "band_idx": np.pad(band_idx[part], (0, max_points - n_part))[:, None],
        "pad_mask_part": pad_mask_part,
        "time_full": _edge_pad(time, max_length)[:, None].astype(np.float32),
        "band_idx_full": np.pad(band_idx, (0, max_length - n_full))[:, None],
        "pad_mask_full": pad_mask_full,
        "n_points": np.int64(n_points),
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
        With ``P = max_points`` and ``L = max_length`` from the run config:

        - ``x_input`` ``(B, P, 4)`` float32: time, flux, flux error, detection flag of the
          most recent ``P`` points (the encoder input)
        - ``band_idx`` ``(B, P, 1)`` int64, ``pad_mask_part`` ``(B, P)`` bool
        - ``time_full`` ``(B, L, 1)`` float32: times of the most recent ``L`` points (where
          the decoder is evaluated)
        - ``band_idx_full`` ``(B, L, 1)`` int64, ``pad_mask_full`` ``(B, L)`` bool
        - ``n_points`` ``(B,)`` int64: valid points in each light curve, before trimming
    """
    time, flux, flux_err, band, detected, valid = _select_columns(data, settings)

    band_idx = np.zeros(band.shape, dtype=np.int64)
    band_idx[valid] = _band_indices(band[valid], settings)

    transform = _seldon.flux_transform(settings.flux_max)
    rows = [
        encode_light_curve(
            time[i][valid[i]],
            flux[i][valid[i]],
            flux_err[i][valid[i]],
            band_idx[i][valid[i]],
            detected[i][valid[i]],
            settings,
            transform,
        )
        for i in range(len(time))
    ]
    return {key: np.stack([row[key] for row in rows]) for key in rows[0]}
