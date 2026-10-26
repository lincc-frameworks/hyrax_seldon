import copy
import tomllib
from pathlib import Path

import pytest
from seldon_fixtures import make_elasticc_parquet, make_seldon_run_dir

import hyrax_seldon

DEFAULT_CONFIG = Path(hyrax_seldon.__file__).parent / "default_config.toml"


@pytest.fixture(scope="session")
def elasticc_parquet(tmp_path_factory):
    """A small ELAsTiCC-schema parquet file of synthetic light curves."""
    path = tmp_path_factory.mktemp("elasticc") / "light_curves.parquet"
    make_elasticc_parquet(path, n_objects=20)
    return path


@pytest.fixture(scope="session")
def seldon_run_dir(tmp_path_factory, elasticc_parquet):
    """A SELDON run directory made by SELDON's own training path."""
    return make_seldon_run_dir(tmp_path_factory.mktemp("seldon_logs"), elasticc_parquet)


@pytest.fixture(scope="session")
def run_config_path(seldon_run_dir):
    """The synthetic run's hparams.yaml."""
    return seldon_run_dir / "hparams.yaml"


@pytest.fixture
def model_config():
    """The [model.HyraxSeldon] defaults from hyrax_seldon's default_config.toml."""
    with open(DEFAULT_CONFIG, "rb") as f:
        return tomllib.load(f)["model"]["HyraxSeldon"]


@pytest.fixture
def seldon_config(model_config, run_config_path):
    """A minimal config dict with [model.HyraxSeldon] pointing at the synthetic run config."""
    model_config["seldon_config"] = str(run_config_path)
    return {"model": {"HyraxSeldon": model_config}}


@pytest.fixture(scope="session")
def converted_weights(tmp_path_factory, seldon_run_dir):
    """The synthetic run's checkpoint, converted with seldon-convert-checkpoint."""
    from hyrax_seldon.checkpoints import convert_run_dir

    return convert_run_dir(seldon_run_dir, tmp_path_factory.mktemp("converted"))


@pytest.fixture(scope="session")
def _hyrax_config(converted_weights):
    from hyrax_seldon.checkpoints import hyrax_config

    return hyrax_config(converted_weights.parent / "seldon_config.yaml")


@pytest.fixture
def hyrax_config(_hyrax_config):
    """A full Hyrax runtime config selecting HyraxSeldon, built from the converted run config."""
    return copy.deepcopy(_hyrax_config)
