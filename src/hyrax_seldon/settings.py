"""Read and validate the SELDON run config a ``HyraxSeldon`` model is built from.

``[model.HyraxSeldon].seldon_config`` points at a SELDON run config (a run's
``hparams.yaml``). Everything about the model - architecture, preprocessing settings,
dataset statistics, class map - comes from that file rather than being copied into TOML.
SELDON settings this port does not reproduce are rejected with :class:`NotSupportedError`
rather than silently ignored.
"""

import functools
from dataclasses import dataclass, field
from pathlib import Path

from hyrax_seldon import _seldon

#: The six LSST bands, in wavelength order.
LSST_BANDS = ("u", "g", "r", "i", "z", "y")

#: The SELDON data module whose preprocessing this port runs.
SUPPORTED_DATA_MODULE = "FlexibleLightCurveDataLoader"

#: ``dataset.config`` settings that must have one specific value, with SELDON's default
#: for when the key is absent.
_REQUIRED_DATASET_VALUES = {
    # (key, required value, SELDON default)
    "mode": ("parquet", "pickle"),
    "include_meta_data": (True, False),
    "flux_normalization": ("minmax", "minmax"),
    "time_normalization": ("std", "std"),
    "log_flux": (True, True),
    "gumbel_transform": (False, False),
    "zero_non_detections": (False, False),
    "convert_bands_to_sncosmo_from_filename": (False, False),
}


class NotSupportedError(ValueError):
    """A SELDON run-config setting that ``hyrax_seldon`` does not support."""

    def __init__(self, key: str, value, detail: str = ""):
        msg = f"SELDON setting {key} = {value!r} is not supported by hyrax_seldon."
        if detail:
            msg += f" {detail}"
        super().__init__(msg)
        self.key = key
        self.value = value


@functools.lru_cache(maxsize=8)
def _read_run_config(path: str, mtime_ns: int) -> dict:
    """Load and resolve a run config. Cached on the file's path and modification time."""
    return _seldon.resolve_transmission_paths(_seldon.load_yaml(path))


@functools.lru_cache(maxsize=8)
def _data_module(path: str, mtime_ns: int):
    """Build a run config's SELDON data module. Cached on the file's path and modification time."""
    return _seldon.build_data_module(_read_run_config(path, mtime_ns)["dataset"])


def _number(key: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NotSupportedError(key, value, "A number is required.")
    return float(value)


def _positive_int(key: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise NotSupportedError(key, value, "A positive integer is required.")
    return value


def _check_run_config(path: Path, run_config: dict):
    """Raise if ``run_config`` is not a SELDON run config this port supports."""
    model_name = run_config.get("model_params", {}).get("name")
    if model_name != _seldon.SUPPORTED_MODEL:
        raise NotSupportedError(
            "model_params.name", model_name, f"Only {_seldon.SUPPORTED_MODEL} is supported."
        )

    dataset = run_config.get("dataset", {})
    if dataset.get("name") != SUPPORTED_DATA_MODULE:
        raise NotSupportedError(
            "dataset.name", dataset.get("name"), f"Only {SUPPORTED_DATA_MODULE} is supported."
        )

    dataset_config = dataset.get("config", {})
    flux_stats = dataset_config.get("flux_stats")
    if not isinstance(flux_stats, dict) or "max" not in flux_stats or "t_max" not in flux_stats:
        raise ValueError(
            f"{path} has no dataset.config.flux_stats, so it is not the run config of a SELDON "
            "training run. Point seldon_config at the run's hparams.yaml, which SELDON writes "
            "with the dataset statistics the model was trained with."
        )
    if not isinstance(dataset_config.get("class_map"), dict):
        raise ValueError(f"{path} has no dataset.config.class_map, so it is not a SELDON run config.")
    if not isinstance(dataset_config.get("band_weights"), dict):
        raise ValueError(f"{path} has no dataset.config.band_weights, so it is not a SELDON run config.")

    for key, (required, default) in _REQUIRED_DATASET_VALUES.items():
        value = dataset_config.get(key, default)
        if value != required:
            raise NotSupportedError(f"dataset.config.{key}", value, f"It must be {required!r}.")

    if dataset_config.get("t_max") == "infer":
        raise NotSupportedError("dataset.config.t_max", "infer", "A fixed time scale is required.")


@dataclass(frozen=True)
class SeldonSettings:
    """Everything ``hyrax_seldon`` needs from the Hyrax config and the SELDON run config.

    Build one with :meth:`from_config`.
    """

    seldon_config_path: Path
    run_config: dict = field(repr=False)
    #: The run's SELDON data module (``FlexibleLightCurveDataLoader``), which prepares light
    #: curves for the model; see :func:`hyrax_seldon._seldon.preprocess`.
    data_module: object = field(repr=False, compare=False)
    compile: bool
    time_field: str
    flux_field: str
    flux_err_field: str
    band_field: str
    photflag_field: str
    reconstruction_grid: bool
    grid_points: int
    grid_horizon: tuple[float, float]

    # From the run config's dataset.config
    t_max: float
    min_points: int
    class_map: dict[str, int] = field(repr=False)

    @property
    def model_params(self) -> dict:
        """The run config's ``model_params``: the SELDON model name and its config."""
        return self.run_config["model_params"]

    @property
    def class_names(self) -> list[str]:
        """Class names, ordered by the class index the model predicts."""
        return [name for name, _ in sorted(self.class_map.items(), key=lambda item: item[1])]

    @property
    def band_index_map(self):
        """SELDON's band name -> band index map, as the run's data module built it."""
        return self.data_module.band_index_map

    @classmethod
    def from_config(cls, config: dict) -> "SeldonSettings":
        """Read ``[model.HyraxSeldon]`` and the SELDON run config it points at.

        Parameters
        ----------
        config : dict
            The Hyrax runtime configuration.

        Returns
        -------
        SeldonSettings

        Raises
        ------
        ValueError
            If ``seldon_config`` is not set, or the file is not a SELDON run config.
        NotSupportedError
            If the run config uses a SELDON setting this port does not support.
        """
        model_config = config["model"]["HyraxSeldon"]

        seldon_config = model_config["seldon_config"]
        if not seldon_config:
            raise ValueError(
                "config['model']['HyraxSeldon']['seldon_config'] must be set to a SELDON run "
                "config: the hparams.yaml of a SELDON training run, or the seldon_config.yaml "
                "HyraxSeldon saves next to its weights."
            )
        path = Path(seldon_config).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SELDON run config {path} does not exist.")

        mtime_ns = path.stat().st_mtime_ns
        run_config = _read_run_config(str(path), mtime_ns)
        _check_run_config(path, run_config)

        grid_horizon = model_config["grid_horizon"]
        if (
            not isinstance(grid_horizon, (list, tuple))
            or len(grid_horizon) != 2
            or not grid_horizon[0] < grid_horizon[1]
        ):
            raise ValueError(
                "config['model']['HyraxSeldon']['grid_horizon'] must be [start, end] in days "
                f"relative to the last observation, with start < end. Got {grid_horizon!r}."
            )

        data_module = _data_module(str(path), mtime_ns)
        dataset_config = run_config["dataset"]["config"]
        return cls(
            seldon_config_path=path,
            run_config=run_config,
            data_module=data_module,
            compile=bool(model_config["compile"]),
            time_field=str(model_config["time_field"]),
            flux_field=str(model_config["flux_field"]),
            flux_err_field=str(model_config["flux_err_field"]),
            band_field=str(model_config["band_field"]),
            photflag_field=str(model_config["photflag_field"]),
            reconstruction_grid=bool(model_config["reconstruction_grid"]),
            grid_points=_positive_int("model.HyraxSeldon.grid_points", model_config["grid_points"]),
            grid_horizon=(float(grid_horizon[0]), float(grid_horizon[1])),
            t_max=_number("dataset.config.flux_stats.t_max", dataset_config["flux_stats"]["t_max"]),
            min_points=_positive_int("dataset.config.min_points", data_module.min_points),
            class_map={str(name): int(index) for name, index in dataset_config["class_map"].items()},
        )
