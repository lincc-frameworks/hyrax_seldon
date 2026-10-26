"""``HyraxSeldon``: SELDON's light-curve model running under Hyrax.

The model is SELDON's ``SELDONEnergyClassifierDictModel``, built with SELDON's own factory
from the run config named by ``[model.HyraxSeldon].seldon_config`` and stored as
``self.model``. That attribute name matches the ``model.`` prefix of SELDON's Lightning
checkpoints, so a converted checkpoint loads into ``HyraxSeldon`` key for key (see
:mod:`hyrax_seldon.checkpoints`).

Saving writes the run config next to the weights as ``seldon_config.yaml``, so a checkpoint
keeps the settings it was trained with.
"""

import logging
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from hyrax.models.model_registry import hyrax_model

from hyrax_seldon import _seldon
from hyrax_seldon.settings import LSST_BANDS, SeldonSettings

logger = logging.getLogger(__name__)

#: File name of the SELDON run config written next to saved weights.
SELDON_CONFIG_FILENAME = "seldon_config.yaml"

#: The value for ``[model].name`` that selects this model.
MODEL_NAME = "hyrax_seldon.models.seldon.HyraxSeldon"


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
        """Run SELDON's ``forward()``; see :func:`hyrax_seldon.encoding.encode_batch` for inputs."""
        return self.model(x, t, band_idx_part, band_idx_full, mask=mask)

    def train_batch(self, batch):
        """Training is not supported yet.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "HyraxSeldon does not support training yet; it runs inference from converted SELDON "
            "checkpoints. Use seldon-convert-checkpoint and the infer or infer_stream verbs."
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
        # time_full is edge-padded, so its last entry is the most recent observation.
        t_last = time_full[:, -1, 0]
        grid_t = t_last[:, None, None] + offsets[None, None, :] / settings.t_max
        grid_t = grid_t.expand(n_batch, n_bands, n_grid).reshape(n_batch, n_bands * n_grid, 1)

        bands = torch.tensor([settings.band_to_index(band) for band in LSST_BANDS], device=device)
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
              (``reconstructed``, ``error``) are real points. Those are the most recent
              ``max_length`` valid points, in time order.
            - ``short_lightcurve`` ``(B,)``: the encoder saw fewer than ``min_points``
              detections, which SELDON never trains on.
            - with ``reconstruction_grid`` on: ``grid_time`` ``(B, grid_points)`` in days
              relative to the last observation, and ``grid_reconstructed`` /
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
        detections = (batch["x_input"][..., 3] > 0) & batch["pad_mask_part"]
        results["short_lightcurve"] = detections.sum(dim=1) < self.settings.min_points

        if self.settings.reconstruction_grid:
            results.update(self._reconstruction_grid(out["decoder_inputs"], batch["time_full"]))

        return results

    @staticmethod
    def prepare_inputs(data_dict):
        """Build SELDON's model inputs from a collated batch.

        The real work is in :func:`hyrax_seldon.encoding.prepare_inputs`, configured by the
        SELDON run config; this is a shim so that the function written out to
        ``prepare_inputs.py`` next to saved weights stays short and self-contained.

        Parameters
        ----------
        data_dict : dict
            The collated batch, with a ``"data"`` group holding the per-point light-curve
            fields named in ``[model.HyraxSeldon]``.

        Returns
        -------
        dict[str, numpy.ndarray]
            See :func:`hyrax_seldon.encoding.encode_batch`.
        """
        # This function's source is written out to prepare_inputs.py next to the saved
        # weights and re-executed at load time, so it must import what it needs itself.
        from hyrax import get_context

        from hyrax_seldon.encoding import prepare_inputs

        return prepare_inputs(data_dict, get_context()["config"])


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


def _load(self, load_path: Path):
    """Load weights, warning if they were saved with a different SELDON run config."""
    load_path = Path(load_path)
    _hyrax_load(self, load_path)

    saved_config = load_path.parent / SELDON_CONFIG_FILENAME
    if not saved_config.exists():
        logger.warning(
            f"No {SELDON_CONFIG_FILENAME} next to {load_path}, so these weights cannot be checked "
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
