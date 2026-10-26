"""Convert a SELDON run directory into weights Hyrax can load.

A SELDON training run writes ``hparams.yaml`` (the full resolved config, including the
dataset statistics) and Lightning checkpoints under ``checkpoints/``. The checkpoints hold
weights only, with every key prefixed ``model.``. The conversion builds a ``HyraxSeldon``
from the run's ``hparams.yaml``, loads the weights strictly, and saves them with
``HyraxSeldon.save``, which writes:

- ``<name>.pth``: the weights, in the form Hyrax's loader expects
- ``prepare_inputs.py``: the model's input preparation, as Hyrax saves for every model
- ``seldon_config.yaml``: a copy of the run's ``hparams.yaml``

Point ``[infer].model_weights_file`` (or ``[infer_stream]``, or ``[train]`` to fine-tune) at
the ``.pth``, and ``[model.HyraxSeldon].seldon_config`` at the ``seldon_config.yaml``.
"""

import argparse
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

#: The prefix SELDON's Lightning experiment puts on every model weight.
CHECKPOINT_PREFIX = "model."


def find_checkpoint(run_dir: Path, checkpoint: str | None = None) -> Path:
    """Return the checkpoint to convert from a SELDON run directory.

    Parameters
    ----------
    run_dir : Path
        A SELDON run directory, containing ``checkpoints/*.ckpt``.
    checkpoint : str, optional
        File name (or stem) of the checkpoint to use. Required when there is more than one.

    Returns
    -------
    Path
    """
    checkpoint_dir = Path(run_dir) / "checkpoints"
    available = sorted(checkpoint_dir.glob("*.ckpt"))
    if checkpoint is not None:
        matches = [path for path in available if checkpoint in (path.name, path.stem)]
        if not matches:
            raise FileNotFoundError(
                f"No checkpoint {checkpoint!r} in {checkpoint_dir}. Available: {[p.name for p in available]}"
            )
        return matches[0]
    if not available:
        raise FileNotFoundError(f"No .ckpt files in {checkpoint_dir}.")
    if len(available) > 1:
        raise ValueError(
            f"{checkpoint_dir} has {len(available)} checkpoints; choose one with --checkpoint: "
            f"{[p.name for p in available]}"
        )
    return available[0]


def load_seldon_state_dict(checkpoint_path: Path, allow_pickle: bool = False) -> dict:
    """Read a SELDON Lightning checkpoint and return the bare SELDON model's weights.

    Parameters
    ----------
    checkpoint_path : Path
        A ``.ckpt`` written by SELDON's training.
    allow_pickle : bool
        Load with ``weights_only=False``. Only for checkpoints from a trusted source that
        hold objects ``torch.load``'s safe loader refuses.

    Returns
    -------
    dict[str, torch.Tensor]
        The weights with the ``model.`` prefix removed.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=not allow_pickle)
    state_dict = checkpoint["state_dict"]
    unprefixed = [key for key in state_dict if not key.startswith(CHECKPOINT_PREFIX)]
    if unprefixed:
        raise ValueError(
            f"{checkpoint_path} has weights outside the SELDON model (no '{CHECKPOINT_PREFIX}' prefix): "
            f"{unprefixed[:5]}"
        )
    return {key[len(CHECKPOINT_PREFIX) :]: value for key, value in state_dict.items()}


def hyrax_config(seldon_config: Path) -> dict:
    """Return a Hyrax runtime config selecting ``HyraxSeldon`` built from ``seldon_config``."""
    from hyrax import Hyrax

    from hyrax_seldon.models.seldon import MODEL_NAME

    h = Hyrax(setup_logging=False)
    h.set_config("model.name", MODEL_NAME)
    h.set_config("model.HyraxSeldon.seldon_config", str(seldon_config))
    return h.config


def convert_run_dir(
    run_dir: Path, out_dir: Path, checkpoint: str | None = None, allow_pickle: bool = False
) -> Path:
    """Convert a SELDON run directory's checkpoint into Hyrax weights.

    Parameters
    ----------
    run_dir : Path
        A SELDON run directory: ``hparams.yaml`` and ``checkpoints/*.ckpt``.
    out_dir : Path
        Where to write ``<name>.pth``, ``prepare_inputs.py`` and ``seldon_config.yaml``.
    checkpoint : str, optional
        Which checkpoint to convert, when the run has more than one.
    allow_pickle : bool
        See :func:`load_seldon_state_dict`.

    Returns
    -------
    Path
        The written ``.pth`` file.

    Raises
    ------
    RuntimeError
        If the checkpoint's weights do not match the model the run config builds exactly.
    """
    from hyrax_seldon.models.seldon import HyraxSeldon

    run_dir = Path(run_dir)
    hparams = run_dir / "hparams.yaml"
    if not hparams.is_file():
        raise FileNotFoundError(f"{run_dir} has no hparams.yaml; is it a SELDON run directory?")
    checkpoint_path = find_checkpoint(run_dir, checkpoint)

    model = HyraxSeldon(config=hyrax_config(hparams))
    state_dict = load_seldon_state_dict(checkpoint_path, allow_pickle=allow_pickle)
    try:
        model.model.load_state_dict(state_dict, strict=True)
    except RuntimeError as err:
        raise RuntimeError(
            f"The weights in {checkpoint_path} do not match the SELDON model built from {hparams}."
        ) from err

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / f"{checkpoint_path.stem}.pth"
    model.save(weights_path)
    logger.info(f"Converted {checkpoint_path} to {weights_path}")
    return weights_path


def main(argv=None):
    """Console script: ``seldon-convert-checkpoint <run_dir> --out-dir <dir>``."""
    parser = argparse.ArgumentParser(
        prog="seldon-convert-checkpoint",
        description="Convert a SELDON run directory (hparams.yaml + checkpoints/*.ckpt) into weights "
        "for hyrax_seldon's HyraxSeldon model.",
    )
    parser.add_argument("run_dir", type=Path, help="SELDON run directory")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write the weights to")
    parser.add_argument("--checkpoint", help="Checkpoint file name, when the run has more than one")
    parser.add_argument(
        "--allow-pickle",
        action="store_true",
        help="Load the checkpoint with torch.load(weights_only=False). Only for trusted checkpoints.",
    )
    args = parser.parse_args(argv)

    weights_path = convert_run_dir(args.run_dir, args.out_dir, args.checkpoint, args.allow_pickle).resolve()
    print(f"Wrote {weights_path}")
    print("Set in your Hyrax config:")
    print('  [model]\n  name = "hyrax_seldon.models.seldon.HyraxSeldon"')
    print(f'  [model.HyraxSeldon]\n  seldon_config = "{weights_path.parent / "seldon_config.yaml"}"')
    print(f'  [infer]\n  model_weights_file = "{weights_path}"')
