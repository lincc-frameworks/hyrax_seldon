"""``HyraxSeldon``: SELDON's light-curve model running under Hyrax.

The model is SELDON's ``SELDONEnergyClassifierDictModel``, built with SELDON's own factory
from the run config named by ``[model.HyraxSeldon].seldon_config`` and stored as
``self.model``.

Loading accepts a SELDON training run's Lightning checkpoint
(``<run_dir>/checkpoints/*.ckpt``) directly. SELDON's Lightning experiment also holds the
model as ``model``, so the checkpoint's weights load into ``HyraxSeldon`` key for key. Point
``seldon_config`` at the same run's ``hparams.yaml``.

Saving writes the run config next to the weights as ``seldon_config.yaml``, so weights saved
by Hyrax keep the settings they were trained with.

Datasets emit raw per-point values in the terms of SELDON's parquet files: MJD, flux on the
ELAsTiCC ``FLUXCAL`` scale, flux error, band name, and ELAsTiCC ``PHOTFLAG``.
:meth:`HyraxSeldon.prepare_inputs` removes the padding Hyrax's collation added and hands the
light curves to SELDON, which prepares them with the run's own data module and preprocessor,
as it prepares new data (see :func:`hyrax_seldon._seldon.preprocess`).
"""

import logging
import shutil
from pathlib import Path

import ignite.distributed as idist
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from hyrax.models.model_registry import hyrax_model

from hyrax_seldon import _seldon
from hyrax_seldon.settings import LSST_BANDS, SeldonSettings

logger = logging.getLogger(__name__)

#: File name of the SELDON run config written next to saved weights.
SELDON_CONFIG_FILENAME = "seldon_config.yaml"

#: Name of the batch group the light-curve fields are read from.
DATA_GROUP = "data"

#: The entries of SELDON's prepared batch that are passed on to the model.
MODEL_INPUTS = (
    "x_input",
    "band_idx",
    "pad_mask_part",
    "time_full",
    "band_idx_full",
    "pad_mask_full",
    "n_detections",
)


@hyrax_model
class HyraxSeldon(nn.Module):
    """SELDON's energy-based classifier autoencoder, for inference with Hyrax.

    ``infer_batch`` returns every output of SELDON's ``forward()`` (see
    :meth:`infer_batch`). Training is not supported yet.

    Parameters
    ----------
    config : dict
        The Hyrax runtime configuration. ``[model.HyraxSeldon]`` names the SELDON run
        config the model is built from.
    data_sample : optional
        Unused; the architecture comes from the run config.
    """

    def __init__(self, config, data_sample=None):
        super().__init__()
        # Must be set before __init__ returns: the @hyrax_model decorator reads self.config
        # immediately afterwards to build the criterion, optimizer, and scheduler.
        self.config = config
        self.settings = SeldonSettings.from_config(config)
        self.model = _seldon.build_model(self.settings.model_params, compile=self.settings.compile)

    def forward(self, x, t, band_idx_part, band_idx_full, mask=None):
        """Run SELDON's ``forward()``; see :meth:`prepare_inputs` for inputs."""
        return self.model(x, t, band_idx_part, band_idx_full, mask=mask)

    def train_batch(self, batch):
        """Training is not supported yet.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "HyraxSeldon does not support training yet; it runs inference from SELDON checkpoints. "
            "Point [infer].model_weights_file or [infer_stream].model_weights_file at a SELDON run's "
            ".ckpt, and [model.HyraxSeldon].seldon_config at the run's hparams.yaml."
        )

    def decode(self, decoder_inputs, t, band_idx):
        """Evaluate SELDON's decoder and flux-error network, exactly as ``forward()`` does.

        Parameters
        ----------
        decoder_inputs : torch.Tensor
            ``(B, latent)`` decoder inputs from ``forward()``.
        t : torch.Tensor
            ``(B, T, 1)`` normalized query times.
        band_idx : torch.Tensor
            ``(B, T, 1)`` SELDON band indices.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            ``(reconstructed, error)``, each ``(B, T, 1)``; ``error`` is ``None`` if the model
            has no flux-error network.
        """
        model = self.model
        reconstructed, _ = model.decoder(decoder_inputs, t, band_idx, return_parameters=True)
        if model.uq_net is None:
            return reconstructed, None
        error = model.uq_net(
            torch.cat(
                [
                    reconstructed,
                    model.decoder.band_emb(band_idx[:, :, 0]),
                    decoder_inputs[:, None, :].expand(-1, reconstructed.shape[1], -1),
                    t,
                ],
                dim=-1,
            )
        )
        return reconstructed, F.softplus(error)

    def _reconstruction_grid(self, decoder_inputs, time_full):
        """Evaluate the decoder on a fixed grid of times for each LSST band."""
        settings = self.settings
        n_batch = decoder_inputs.shape[0]
        n_bands, n_grid = len(LSST_BANDS), settings.grid_points
        device = decoder_inputs.device

        offsets = torch.linspace(*settings.grid_horizon, n_grid, device=device, dtype=time_full.dtype)
        # time_full is edge-padded, so its last entry is its latest observation.
        t_last = time_full[:, -1, 0]
        grid_t = t_last[:, None, None] + offsets[None, None, :] / settings.t_max
        grid_t = grid_t.expand(n_batch, n_bands, n_grid).reshape(n_batch, n_bands * n_grid, 1)

        bands = torch.tensor([settings.band_index_map[band] for band in LSST_BANDS], device=device)
        grid_bands = bands[:, None].expand(n_bands, n_grid).reshape(1, n_bands * n_grid, 1)
        grid_bands = grid_bands.expand(n_batch, -1, -1)

        reconstructed, error = self.decode(decoder_inputs, grid_t, grid_bands)
        results = {
            "grid_time": offsets[None, :].expand(n_batch, n_grid),
            "grid_reconstructed": reconstructed.reshape(n_batch, n_bands, n_grid),
        }
        if error is not None:
            results["grid_error"] = error.reshape(n_batch, n_bands, n_grid)
        return results

    def infer_batch(self, batch):
        """Run SELDON's ``forward()`` and return all of its outputs.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            The output of :meth:`prepare_inputs`, on the model's device.

        Returns
        -------
        dict[str, torch.Tensor]
            Every ``forward()`` output, as SELDON produces it, with ``parameters.<name>``
            flattened to ``parameters_<name>`` and the always-``None`` ``encoding_sequence``
            dropped. ``z_sample``, ``class_idx`` and ``class_emb`` are stochastic: SELDON
            predicts the class from a posterior sample even in eval mode. Also:

            - ``pad_mask_full`` ``(B, max_length)``: which entries of the per-point outputs
              (``reconstructed``, ``error``) are real points, in time order. A light curve
              longer than ``max_length`` gets SELDON's window of ``max_length`` points.
            - ``short_lightcurve`` ``(B,)``: the light curve has fewer than ``min_points``
              detections. SELDON's loader excludes such light curves from training.
            - with ``reconstruction_grid`` on: ``grid_time`` ``(B, grid_points)`` in days
              relative to the last observation in ``pad_mask_full``, and ``grid_reconstructed`` /
              ``grid_error`` ``(B, 6, grid_points)`` for the bands u, g, r, i, z, y.
        """
        out = self.forward(
            batch["x_input"],
            batch["time_full"],
            batch["band_idx"],
            batch["band_idx_full"],
            mask=batch["pad_mask_part"],
        )

        results = {}
        for key, value in out.items():
            if key == "encoding_sequence":
                continue
            if key == "parameters":
                results.update({f"parameters_{name}": param for name, param in value.items()})
            else:
                results[key] = value

        results["pad_mask_full"] = batch["pad_mask_full"]
        results["short_lightcurve"] = batch["n_detections"] < self.settings.min_points

        if self.settings.reconstruction_grid:
            results.update(self._reconstruction_grid(out["decoder_inputs"], batch["time_full"]))

        return results

    @staticmethod
    def light_curve_frame(data: dict, settings: SeldonSettings) -> pd.DataFrame:
        """Remove a batch's padding, giving one row per light curve in SELDON's parquet schema.

        Parameters
        ----------
        data : dict
            The batch's ``"data"`` group: per-point fields of shape ``(batch, length)``. Entries
            where any field's ``<field>_mask`` is false are padding.
        settings : SeldonSettings
            Names the fields.

        Returns
        -------
        pandas.DataFrame
            Per-point list columns ``MJD``, ``FLUXCAL``, ``FLUXCALERR``, ``BAND`` and ``PHOTFLAG``.
        """
        fields = {
            "MJD": settings.time_field,
            "FLUXCAL": settings.flux_field,
            "FLUXCALERR": settings.flux_err_field,
            "BAND": settings.band_field,
            "PHOTFLAG": settings.photflag_field,
        }
        missing = [name for name in fields.values() if name not in data]
        if missing:
            raise RuntimeError(
                f"HyraxSeldon could not find the field(s) {missing} in the batch. Available fields: "
                f"{sorted(data)}. Set time_field / flux_field / flux_err_field / band_field / "
                "photflag_field in [model.HyraxSeldon] and list them in the data request's 'fields'."
            )

        columns = {column: np.asarray(data[name]) for column, name in fields.items()}
        shapes = {values.shape for values in columns.values()}
        if len(shapes) > 1 or columns["MJD"].ndim != 2:
            raise RuntimeError(
                "HyraxSeldon expects every light-curve field to collate to the same (batch, length) "
                f"shape, but got shapes {sorted(shapes)}."
            )

        valid = np.ones(columns["MJD"].shape, dtype=bool)
        for name in fields.values():
            mask = data.get(f"{name}_mask")
            if mask is not None:
                valid &= np.asarray(mask).astype(bool)

        return pd.DataFrame(
            {
                column: [row[keep] for row, keep in zip(values, valid, strict=True)]
                for column, values in columns.items()
            }
        )

    @staticmethod
    def prepare_inputs(data_dict):
        """Build SELDON's model inputs from a collated batch.

        Parameters
        ----------
        data_dict : dict
            The collated batch, with a ``"data"`` group holding the per-point light-curve
            fields named in ``[model.HyraxSeldon]``, padded per row, each optionally with a
            ``<field>_mask``.

        Returns
        -------
        dict[str, numpy.ndarray]
            With ``P = max_points`` and ``L = max_length`` from the run config:

            - ``x_input`` ``(B, P, 4)`` float32: time, flux, flux error, detection flag of the
              points the encoder sees
            - ``band_idx`` ``(B, P, 1)`` int64, ``pad_mask_part`` ``(B, P)`` bool
            - ``time_full`` ``(B, L, 1)`` float32: times where the decoder is evaluated
            - ``band_idx_full`` ``(B, L, 1)`` int64, ``pad_mask_full`` ``(B, L)`` bool
            - ``n_detections`` ``(B,)`` int64: detections in each light curve, as SELDON's
              loader counts them

        Raises
        ------
        ValueError
            If a light curve has fewer than ``min_points`` points; SELDON cannot encode it.
        """
        # This function's source is written out to prepare_inputs.py next to the saved
        # weights and re-executed at load time, so it must import what it needs itself.
        from hyrax import get_context

        from hyrax_seldon import _seldon
        from hyrax_seldon.models.seldon import DATA_GROUP, MODEL_INPUTS, HyraxSeldon
        from hyrax_seldon.settings import SeldonSettings

        if DATA_GROUP not in data_dict:
            raise RuntimeError(
                f"HyraxSeldon reads light curves from the '{DATA_GROUP}' group of the data request, "
                f"but the batch only has {sorted(data_dict)}. Name the light-curve dataset "
                f"'{DATA_GROUP}' in [data_request]."
            )
        settings = SeldonSettings.from_config(get_context()["config"])
        light_curves = HyraxSeldon.light_curve_frame(data_dict[DATA_GROUP], settings)
        batch = _seldon.preprocess(settings.data_module, light_curves)
        return {key: batch[key] for key in MODEL_INPUTS}


# @hyrax_model installs Hyrax's save/load on the class, replacing any defined in its body,
# so the run-config handling wraps them here.
_hyrax_save = HyraxSeldon.save
_hyrax_load = HyraxSeldon.load


def _save(self, save_path: Path):
    """Save the weights, ``prepare_inputs.py``, and the SELDON run config as ``seldon_config.yaml``."""
    save_path = Path(save_path)
    _hyrax_save(self, save_path)
    destination = save_path.parent / SELDON_CONFIG_FILENAME
    if destination.resolve() != self.settings.seldon_config_path:
        shutil.copyfile(self.settings.seldon_config_path, destination)


def _load_lightning_checkpoint(self, checkpoint_path: Path):
    """Load the weights from a SELDON training run's Lightning checkpoint."""
    device = idist.device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    self.to(device)
    self.load_state_dict(checkpoint["state_dict"])


def _load(self, load_path: Path):
    """Load weights, warning if they were saved with a different SELDON run config.

    A ``.ckpt`` is a SELDON training run's Lightning checkpoint, whose run config is the
    run's ``hparams.yaml`` (``<run_dir>/checkpoints/<name>.ckpt``). Anything else is weights
    saved by :meth:`HyraxSeldon.save`, with the run config beside them as ``seldon_config.yaml``.
    """
    load_path = Path(load_path)
    if load_path.suffix == ".ckpt":
        _load_lightning_checkpoint(self, load_path)
        saved_config = load_path.parent.parent / "hparams.yaml"
    else:
        _hyrax_load(self, load_path)
        saved_config = load_path.parent / SELDON_CONFIG_FILENAME

    if not saved_config.exists():
        logger.warning(
            f"No SELDON run config at {saved_config}, so the weights in {load_path} cannot be checked "
            f"against the SELDON run config in use ({self.settings.seldon_config_path})."
        )
    elif saved_config.resolve() != self.settings.seldon_config_path:
        saved = _seldon.resolve_transmission_paths(_seldon.load_yaml(saved_config))
        if saved != self.settings.run_config:
            logger.warning(
                f"The weights in {load_path} were saved with the SELDON run config {saved_config}, "
                f"which differs from the one in use ({self.settings.seldon_config_path}). Set "
                f"[model.HyraxSeldon].seldon_config = '{saved_config}' to use the settings they "
                "were trained with."
            )


HyraxSeldon.save = _save
HyraxSeldon.load = _load
