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

- ``FlexibleLightCurveDataLoader``'s constructor starts a ``multiprocessing.Manager`` it
  never uses. :func:`build_data_module` builds it without one.
- SELDON's data module and ``LightCurvePreprocessor`` print progress to stdout and stderr,
  and the preprocessor's constructor reseeds NumPy's global random number generator
  (``np.random.seed(42)``). :func:`preprocess` captures the output and restores the
  generator's state afterwards.
"""

import contextlib
import copy
import importlib
import io
import sys
from pathlib import Path
from unittest import mock

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


#: ``dataset.config`` settings overridden when the run's data module prepares new light curves:
#: they come from Hyrax rather than from files, carry no class, and are encoded without the
#: training augmentations or resampling noise.
INFERENCE_OVERRIDES = {
    "lightcurve_path": None,
    "fine_tune_data": None,
    "class_filter": None,
    "augmentations": None,
    "resample": False,
    "num_workers": 0,
    "pin_memory": False,
}


def build_data_module(dataset: dict):
    """Build a SELDON run's data module, to prepare new light curves with.

    Parameters
    ----------
    dataset : dict
        The ``dataset`` section of a resolved SELDON run config (``name`` and ``config``), with
        transmission paths already resolved. It is not modified.

    Returns
    -------
    FlexibleLightCurveDataLoader
        The data module, configured as in the run except for :data:`INFERENCE_OVERRIDES`. Its
        ``flux_stats`` and ``band_weights`` are the run's.
    """
    loader_module = _import("TAE.datasets.lightcurve_dataloader2")
    factory = _import("TAE.datasets.dataset_factory")

    config = copy.deepcopy(dataset["config"])
    config.update(INFERENCE_OVERRIDES)
    with (
        mock.patch.object(loader_module, "Manager", lambda: None),
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        return factory.get_module(dataset["name"], config)


def preprocess(data_module, light_curves) -> dict[str, np.ndarray]:
    """Prepare a batch of light curves for the model the way SELDON prepares new data.

    The light curves go through the run's data module (``load_data``, ``init_class_weights``,
    ``init_light_curves``, ``init_band_index``, ``init_meta_data``), then through a
    ``LightCurvePreprocessor`` built as the data module's ``setup()`` builds it, and SELDON's
    ``enhanced_collate_fn``. The data module keeps the run's statistics: its
    ``init_data_stats`` is not called.

    Every light curve is kept, including those with fewer than ``min_points`` detections, which
    SELDON's ``load_data`` would drop.

    Parameters
    ----------
    data_module : FlexibleLightCurveDataLoader
        From :func:`build_data_module`. Each call replaces the light curves it holds.
    light_curves : pandas.DataFrame
        One row per light curve in the schema of SELDON's parquet files: per-point list columns
        ``MJD``, ``FLUXCAL``, ``FLUXCALERR``, ``BAND`` and ``PHOTFLAG``.

    Returns
    -------
    dict[str, numpy.ndarray]
        The tensors of ``enhanced_collate_fn``'s batch, as numpy arrays, and ``n_detections``:
        each light curve's detection count as SELDON's ``load_data`` computes it (``nobs``).

    Raises
    ------
    ValueError
        If a light curve has fewer than ``min_points`` points, which SELDON cannot encode, or a
        band the data module's band index map does not know.
    """
    preprocessor_module = _import("TAE.datasets.lightcurve_preprocessor")
    collate = _import("TAE.datasets.lightcurve_dataset").enhanced_collate_fn
    import torch

    min_points = data_module.min_points
    rng_state = np.random.get_state()
    samples, too_short = [], []
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            data_module.lightcurve_path = light_curves
            # load_data drops light curves with fewer than min_points detections; keep them all.
            data_module.min_points = 0
            try:
                data_module.load_data()
            finally:
                data_module.min_points = min_points
            data_module.init_class_weights()
            data_module.init_light_curves()
            try:
                data_module.init_band_index()
            except KeyError as err:
                raise ValueError(
                    f"Band {err.args[0]!r} is not one of the bands this SELDON model knows: "
                    f"{sorted(data_module.band_index_map)}."
                ) from None
            data_module.init_meta_data()

            preprocessor = preprocessor_module.LightCurvePreprocessor(
                light_curves=data_module.full_light_curve_array,
                class_names=data_module.full_class_names,
                class_weights=data_module.class_weights,
                band_indices=data_module.band_array,
                band_weights=data_module.band_weights,
                flux_stats=data_module.flux_stats,
                min_points=data_module.min_points,
                max_points=data_module.max_points,
                meta_data=data_module.meta_data,
                **data_module.kwargs,
            )
            for index in range(len(preprocessor)):
                try:
                    samples.append(preprocessor[index])
                except AssertionError:
                    # SELDON asserts that the encoder gets at least min_points points.
                    too_short.append(index)
    finally:
        np.random.set_state(rng_state)

    if too_short:
        lengths = [len(data_module.full_light_curve_array[index]) for index in too_short]
        raise ValueError(
            f"Light curves at batch rows {too_short} have {lengths} points, fewer than the run "
            f"config's min_points={min_points}; SELDON cannot encode them."
        )

    batch = {key: value.numpy() for key, value in collate(samples).items() if isinstance(value, torch.Tensor)}
    batch["n_detections"] = data_module.full_light_curves["nobs"].to_numpy(dtype=np.int64)
    return batch
