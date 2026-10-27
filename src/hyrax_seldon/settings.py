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


def _number(key: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NotSupportedError(key, value, "A number is required.")
    return float(value)


def _positive_int(key: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise NotSupportedError(key, value, "A positive integer is required.")
    return value


def band_index_map(dataset_config: dict) -> dict[str, int]:
    """Return SELDON's band name -> band index map for a run config's ``dataset.config``.

    For ``BandIndexMap`` in ``files`` mode, SELDON numbers bands by the sorted keys of the
    filter dictionary (``TAE/datasets/bandpasses.py``).
    """
    spec = dataset_config.get("band_index_map")
    if not isinstance(spec, dict) or spec.get("name") != "BandIndexMap":
        raise NotSupportedError(
            "dataset.config.band_index_map", spec, "Only BandIndexMap in 'files' mode is supported."
        )
    mode = spec.get("config", {}).get("mode")
    if mode != "files":
        raise NotSupportedError(
            "dataset.config.band_index_map.config.mode", mode, "Only 'files' mode is supported."
        )
    return {band: index for index, band in enumerate(sorted(spec["config"]["filedict"]))}


@dataclass(frozen=True)
class SeldonSettings:
    """Everything ``hyrax_seldon`` needs from the Hyrax config and the SELDON run config.

    Build one with :meth:`from_config`.
    """

    seldon_config_path: Path
    run_config: dict = field(repr=False)
    compile: bool
    time_field: str
    flux_field: str
    flux_err_field: str
    band_field: str
    detected_field: str
    reconstruction_grid: bool
    grid_points: int
    grid_horizon: tuple[float, float]

    # From the run config's dataset.config
    flux_max: float
    t_max: float
    t_offset: float
    zero_point_flux: float
    min_points: int
    max_points: int
    max_length: int
    include_non_detections: bool
    band_index: dict[str, int] = field(repr=False)
    band_weights: dict[int, float] = field(repr=False)
    class_map: dict[str, int] = field(repr=False)

    @property
    def model_params(self) -> dict:
        """The run config's ``model_params``: the SELDON model name and its config."""
        return self.run_config["model_params"]

    @property
    def class_names(self) -> list[str]:
        """Class names, ordered by the class index the model predicts."""
        return [name for name, _ in sorted(self.class_map.items(), key=lambda item: item[1])]

    def band_to_index(self, band: str) -> int:
        """Return SELDON's band index for a band name.

        Single-character names are lower-cased first, as SELDON does (ELAsTiCC's ``Y``
        is SELDON's ``y``).

        Raises
        ------
        ValueError
            If the band is not one the model knows.
        """
        key = band.lower() if len(band) == 1 else band
        try:
            return self.band_index[key]
        except KeyError:
            raise ValueError(
                f"Band {band!r} is not one of the bands this SELDON model knows: {sorted(self.band_index)}."
            ) from None

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

        run_config = _read_run_config(str(path), path.stat().st_mtime_ns)
        return cls._from_run_config(path, run_config, model_config)

    @classmethod
    def _from_run_config(cls, path: Path, run_config: dict, model_config: dict) -> "SeldonSettings":
        model_name = run_config.get("model_params", {}).get("name")
        if model_name != _seldon.SUPPORTED_MODEL:
            raise NotSupportedError(
                "model_params.name", model_name, f"Only {_seldon.SUPPORTED_MODEL} is supported."
            )

        dataset_config = run_config.get("dataset", {}).get("config", {})
        flux_stats = dataset_config.get("flux_stats")
        if not isinstance(flux_stats, dict) or "max" not in flux_stats or "t_max" not in flux_stats:
            raise ValueError(
                f"{path} has no dataset.config.flux_stats, so it is not the run config of a SELDON "
                "training run. Point seldon_config at the run's hparams.yaml, which SELDON writes "
                "with the dataset statistics the model was trained with."
            )
        class_map = dataset_config.get("class_map")
        if not isinstance(class_map, dict):
            raise ValueError(f"{path} has no dataset.config.class_map, so it is not a SELDON run config.")
        band_weights = dataset_config.get("band_weights")
        if not isinstance(band_weights, dict):
            raise ValueError(f"{path} has no dataset.config.band_weights, so it is not a SELDON run config.")

        for key, (required, default) in _REQUIRED_DATASET_VALUES.items():
            value = dataset_config.get(key, default)
            if value != required:
                raise NotSupportedError(f"dataset.config.{key}", value, f"It must be {required!r}.")

        if dataset_config.get("t_max") == "infer":
            raise NotSupportedError("dataset.config.t_max", "infer", "A fixed time scale is required.")

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

        return cls(
            seldon_config_path=path,
            run_config=run_config,
            compile=bool(model_config["compile"]),
            time_field=str(model_config["time_field"]),
            flux_field=str(model_config["flux_field"]),
            flux_err_field=str(model_config["flux_err_field"]),
            band_field=str(model_config["band_field"]),
            detected_field=str(model_config["detected_field"]),
            reconstruction_grid=bool(model_config["reconstruction_grid"]),
            grid_points=_positive_int("model.HyraxSeldon.grid_points", model_config["grid_points"]),
            grid_horizon=(float(grid_horizon[0]), float(grid_horizon[1])),
            flux_max=_number("dataset.config.flux_stats.max", flux_stats["max"]),
            t_max=_number("dataset.config.flux_stats.t_max", flux_stats["t_max"]),
            t_offset=_number("dataset.config.t_offset", dataset_config.get("t_offset", 0.0)),
            zero_point_flux=_number(
                "dataset.config.zero_point_flux", dataset_config.get("zero_point_flux", 1.0)
            ),
            min_points=_positive_int("dataset.config.min_points", dataset_config.get("min_points", 5)),
            max_points=_positive_int("dataset.config.max_points", dataset_config.get("max_points", 100)),
            max_length=_positive_int("dataset.config.max_length", dataset_config.get("max_length", 400)),
            include_non_detections=bool(dataset_config.get("include_non_detections", False)),
            band_index=band_index_map(dataset_config),
            band_weights={int(index): float(weight) for index, weight in band_weights.items()},
            class_map={str(name): int(index) for name, index in class_map.items()},
        )
