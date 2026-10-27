"""The only module in ``hyrax_seldon`` that imports SELDON (the ``TAE`` package).

Importing SELDON has side effects that are contained here:

- ``TAE.util`` registers the OmegaConf resolvers ``sum`` and ``len`` at import, and
  registering a resolver name twice raises. The import is done once, lazily, and a clash
  with a resolver some other library registered first is reported clearly.
- ``TAE.models.factory`` eagerly imports every SELDON model module (and matplotlib,
  scipy, pandas). That cost is only paid when a model is actually built.
- ``RecurrentODEEncoderV2``, ``ODESolver`` and ``SpectrumBasisDecoder`` call
  ``nn.Module.compile()`` in their constructors. :func:`build_model` undoes that unless
  compilation is requested.

- ``LightCurvePreprocessor``'s constructor reseeds NumPy's global random number generator
  (``np.random.seed(42)``) and prints. :func:`preprocess` restores the generator's state
  afterwards and captures the print.

SELDON's data modules (``FlexibleLightCurveDataLoader``, which starts a
``multiprocessing.Manager``) are never used at runtime.
"""

import contextlib
import copy
import importlib
import io
import sys
from pathlib import Path

import numpy as np

#: The SELDON model class this port supports.
SUPPORTED_MODEL = "SELDONEnergyClassifierDictModel"


def _import(module_name: str):
    """Import a SELDON module, making sure ``TAE.util``'s resolvers are registered first."""
    if "TAE.util" not in sys.modules:
        from omegaconf import OmegaConf

        clashes = [name for name in ("sum", "len") if OmegaConf.has_resolver(name)]
        if clashes:
            raise RuntimeError(
                f"Cannot import SELDON: OmegaConf resolver(s) {clashes} are already registered by "
                "another library, and SELDON's TAE.util registers resolvers with the same names "
                "at import time."
            )
        importlib.import_module("TAE.util")
    return importlib.import_module(module_name)


def tae_data_dir() -> Path:
    """Return the directory of transmission-curve files installed with SELDON."""
    import TAE

    return Path(TAE.__file__).parent / "data"


def load_yaml(path) -> dict:
    """Load a SELDON yaml config and resolve its interpolations.

    Interpolations are resolved against the full config, because ``model_params``
    references ``dataset.config`` and ``global_params``.

    Parameters
    ----------
    path : str or Path
        A SELDON config: a repo yaml or a run's ``hparams.yaml``.

    Returns
    -------
    dict
        The resolved config as plain Python containers.
    """
    _import("TAE.util")
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(path))
    OmegaConf.resolve(config)
    return OmegaConf.to_container(config)


def resolve_transmission_paths(config: dict) -> dict:
    """Point every transmission-curve file in a SELDON config at the installed ``TAE/data``.

    Run configs carry absolute paths from the machine they were trained on. The yaml anchor
    for the filter dictionary is expanded on load, so every ``filedict`` in the config is
    rewritten, not just the top-level ``transmission_filters``. Files are matched by name.

    Parameters
    ----------
    config : dict
        A resolved SELDON config. It is not modified.

    Returns
    -------
    dict
        A copy of ``config`` with every ``filedict`` value replaced by an installed path.

    Raises
    ------
    FileNotFoundError
        If a transmission file named in the config is not installed with SELDON.
    """
    data_dir = tae_data_dir()

    def resolve(path):
        installed = data_dir / Path(path).name
        if not installed.exists():
            raise FileNotFoundError(
                f"SELDON transmission file {Path(path).name!r} (from {path!r}) is not in the "
                f"installed SELDON data directory {data_dir}."
            )
        return str(installed)

    def walk(node):
        if isinstance(node, dict):
            return {
                key: (
                    {band: resolve(path) for band, path in value.items()}
                    if key == "filedict" and isinstance(value, dict)
                    else walk(value)
                )
                for key, value in node.items()
            }
        if isinstance(node, list):
            return [walk(value) for value in node]
        return node

    return walk(config)


def build_model(model_params: dict, compile: bool = False):
    """Build a SELDON model with SELDON's own factory.

    Parameters
    ----------
    model_params : dict
        The ``model_params`` section of a resolved SELDON config (``name`` and ``config``),
        with transmission paths already resolved. It is not modified.
    compile : bool
        Keep the ``nn.Module.compile()`` calls SELDON makes in its constructors. When false
        (the default), they are undone and the model runs eagerly.

    Returns
    -------
    torch.nn.Module
        The randomly initialized SELDON model.
    """
    factory = _import("TAE.models.factory")
    model_params = copy.deepcopy(model_params)
    model = factory.get_module(model_params["name"], model_params["config"])

    if not compile:
        for module in model.modules():
            module._compiled_call_impl = None

    return model


#: Class name given to every light curve handed to SELDON's preprocessor. It requires a class
#: name and weight per light curve, but they only feed its training bookkeeping.
PLACEHOLDER_CLASS = "Unknown"


def preprocess(
    light_curves, band_indices, detection_flags, flux_stats, band_weights, **options
) -> dict[str, np.ndarray]:
    """Run SELDON's ``LightCurvePreprocessor`` and ``enhanced_collate_fn`` over a batch.

    The preprocessor is built without augmentations and without resampling noise, so it
    selects the earliest ``max_points`` points for the encoder (``FullSample``). Light curves
    longer than ``max_length`` get a random window from SELDON, so trim them first.

    Parameters
    ----------
    light_curves : list[numpy.ndarray]
        One ``(n, 3)`` float32 array per light curve: time (MJD), flux, flux error, with flux
        already divided by the run's ``zero_point_flux``.
    band_indices : list[numpy.ndarray]
        One ``(n,)`` int64 array of SELDON band indices per light curve.
    detection_flags : list[numpy.ndarray]
        One ``(n,)`` bool array per light curve.
    flux_stats : dict
        The run config's ``dataset.config.flux_stats``.
    band_weights : dict
        The run config's ``dataset.config.band_weights``.
    **options
        ``LightCurvePreprocessor`` settings from the run config: ``flux_normalization``,
        ``time_normalization``, ``min_points``, ``max_points``, ``max_length``, ``t_offset``.

    Returns
    -------
    dict[str, numpy.ndarray]
        The tensors of ``enhanced_collate_fn``'s batch, as numpy arrays.
    """
    preprocessor_module = _import("TAE.datasets.lightcurve_preprocessor")
    collate = _import("TAE.datasets.lightcurve_dataset").enhanced_collate_fn
    import torch

    meta_data = {"detection_flag": [flags.astype(np.float32)[:, None] for flags in detection_flags]}

    rng_state = np.random.get_state()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            preprocessor = preprocessor_module.LightCurvePreprocessor(
                light_curves=light_curves,
                class_names=[PLACEHOLDER_CLASS] * len(light_curves),
                class_weights={PLACEHOLDER_CLASS: 1.0},
                band_indices=band_indices,
                band_weights=band_weights,
                flux_stats=flux_stats,
                meta_data=meta_data,
                augmentations=None,
                resample=False,
                **options,
            )
        batch = collate([preprocessor[i] for i in range(len(light_curves))])
    finally:
        np.random.set_state(rng_state)

    return {key: value.numpy() for key, value in batch.items() if isinstance(value, torch.Tensor)}
