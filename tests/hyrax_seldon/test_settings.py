import pytest
import yaml
from seldon_fixtures import TARGET_CONFIG

from hyrax_seldon import _seldon
from hyrax_seldon.settings import LSST_BANDS, NotSupportedError, SeldonSettings


def _write_run_config(tmp_path, run_config_path, edit):
    """Copy the synthetic run config with ``edit(config)`` applied, and return its path."""
    with open(run_config_path) as f:
        config = yaml.safe_load(f)
    edit(config)
    path = tmp_path / "hparams.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(config, f)
    return path


def test_reads_run_config(seldon_config):
    """Settings come from the run config's dataset.config and flux_stats."""
    settings = SeldonSettings.from_config(seldon_config)

    assert settings.max_points == 64
    assert settings.max_length == 512
    assert settings.min_points == 32
    assert settings.t_max == 20.0
    assert settings.t_offset == 0.0
    assert settings.zero_point_flux == 1.0
    assert settings.include_non_detections
    assert settings.flux_max > 0
    assert len(settings.class_names) == 10
    assert settings.model_params["name"] == "SELDONEnergyClassifierDictModel"


def test_lsst_band_indices(seldon_config):
    """The band index map is SELDON's sorted filter-dictionary order."""
    settings = SeldonSettings.from_config(seldon_config)

    assert {band: settings.band_to_index(band) for band in LSST_BANDS} == {
        "u": 40,
        "g": 37,
        "r": 39,
        "i": 38,
        "z": 42,
        "y": 41,
    }
    # Single-character bands are lower-cased, as SELDON does for ELAsTiCC's "Y".
    assert settings.band_to_index("Y") == 41
    with pytest.raises(ValueError, match="'J'"):
        settings.band_to_index("J")


def test_transmission_paths_resolved_to_installed_files(seldon_config):
    """Cluster paths in a run config point at the installed TAE/data files."""
    settings = SeldonSettings.from_config(seldon_config)
    data_dir = _seldon.tae_data_dir()

    band_map = settings.run_config["dataset"]["config"]["band_index_map"]["config"]["filedict"]
    assert all(path.startswith(str(data_dir)) for path in band_map.values())


def test_seldon_config_required(model_config):
    """seldon_config = false is an error that says what to set."""
    with pytest.raises(ValueError, match="seldon_config"):
        SeldonSettings.from_config({"model": {"HyraxSeldon": model_config}})


def test_plain_repo_yaml_rejected(model_config):
    """A repo yaml has no dataset statistics, so it is not a run config."""
    from pathlib import Path

    import TAE

    model_config["seldon_config"] = str(Path(TAE.__file__).parent / "configs" / TARGET_CONFIG)
    with pytest.raises(ValueError, match="flux_stats"):
        SeldonSettings.from_config({"model": {"HyraxSeldon": model_config}})


@pytest.mark.parametrize(
    "key, value",
    [
        ("include_meta_data", False),
        ("t_max", "infer"),
        ("flux_normalization", "std"),
        ("time_normalization", "minmax"),
        ("mode", "pickle"),
        ("zero_non_detections", True),
        ("log_flux", False),
        ("gumbel_transform", True),
        ("convert_bands_to_sncosmo_from_filename", True),
    ],
)
def test_unsupported_settings_rejected(tmp_path, run_config_path, model_config, key, value):
    """Unsupported SELDON settings raise an error naming the key and value."""

    def edit(config):
        config["dataset"]["config"][key] = value

    model_config["seldon_config"] = str(_write_run_config(tmp_path, run_config_path, edit))
    with pytest.raises(NotSupportedError, match=f"dataset.config.{key} = {value!r}"):
        SeldonSettings.from_config({"model": {"HyraxSeldon": model_config}})


def test_unsupported_model_rejected(tmp_path, run_config_path, model_config):
    """Only SELDONEnergyClassifierDictModel is supported."""

    def edit(config):
        config["model_params"]["name"] = "SELDONDictModel"

    model_config["seldon_config"] = str(_write_run_config(tmp_path, run_config_path, edit))
    with pytest.raises(NotSupportedError, match="model_params.name = 'SELDONDictModel'"):
        SeldonSettings.from_config({"model": {"HyraxSeldon": model_config}})


def test_unsupported_band_map_rejected(tmp_path, run_config_path, model_config):
    """Only a file-based BandIndexMap is supported."""

    def edit(config):
        config["dataset"]["config"]["band_index_map"]["config"]["mode"] = "sncosmo"

    model_config["seldon_config"] = str(_write_run_config(tmp_path, run_config_path, edit))
    with pytest.raises(NotSupportedError, match="band_index_map.config.mode = 'sncosmo'"):
        SeldonSettings.from_config({"model": {"HyraxSeldon": model_config}})


def test_bad_grid_horizon_rejected(seldon_config):
    """grid_horizon must be [start, end] with start < end."""
    seldon_config["model"]["HyraxSeldon"]["grid_horizon"] = [10.0, -10.0]
    with pytest.raises(ValueError, match="grid_horizon"):
        SeldonSettings.from_config(seldon_config)


def test_run_config_without_band_weights_rejected(tmp_path, run_config_path, model_config):
    """SELDON's preprocessor needs the run's band weights."""

    def edit(config):
        del config["dataset"]["config"]["band_weights"]

    model_config["seldon_config"] = str(_write_run_config(tmp_path, run_config_path, edit))
    with pytest.raises(ValueError, match="band_weights"):
        SeldonSettings.from_config({"model": {"HyraxSeldon": model_config}})
