import numpy as np
import pandas as pd
import pytest
from seldon_fixtures import prepare_inputs

from hyrax_seldon.collate import collate_light_curve_field
from hyrax_seldon.datasets.elasticc_parquet import ElasticcParquetDataset
from hyrax_seldon.settings import SeldonSettings

FIELDS = ["time", "flux", "flux_err", "band", "photflag"]
COLUMNS = {"time": "MJD", "flux": "FLUXCAL", "flux_err": "FLUXCALERR", "band": "BAND", "photflag": "PHOTFLAG"}


@pytest.fixture
def dataset(elasticc_parquet):
    """The dataset over the synthetic ELAsTiCC parquet file."""
    config = {"data_set": {"ElasticcParquetDataset": {"read_kwargs": {}}}}
    return ElasticcParquetDataset(config, data_location=elasticc_parquet)


def _collate(dataset, indices):
    """Collate the dataset's per-point fields for ``indices`` as Hyrax does."""
    samples = [{field: getattr(dataset, f"get_{field}")(i) for field in FIELDS} for i in indices]
    batch = {}
    for field in FIELDS:
        batch.update(getattr(dataset, f"collate_{field}")(samples))
    return batch


@pytest.fixture
def seldon_batch(elasticc_parquet, seldon_config):
    """SELDON's own training data path over the same parquet file, without augmentations.

    The run's data module is set up from the file as SELDON's train.py does, and every light
    curve is prepared by its LightCurvePreprocessor and enhanced_collate_fn.
    """
    import copy

    from TAE.datasets.dataset_factory import get_module as get_dataset
    from TAE.datasets.lightcurve_dataset import enhanced_collate_fn

    run_config = SeldonSettings.from_config(seldon_config).run_config
    dataset_config = copy.deepcopy(run_config["dataset"]["config"])
    dataset_config.update(
        lightcurve_path=[str(elasticc_parquet)], num_workers=0, augmentations=None, resample=False
    )
    loader = get_dataset(run_config["dataset"]["name"], dataset_config)
    try:
        loader.setup()
        # The run was trained on this file, so setup() recomputes the run's statistics.
        assert loader.flux_stats == pytest.approx(run_config["dataset"]["config"]["flux_stats"])
        return enhanced_collate_fn([loader.dataset[i] for i in range(len(loader.dataset))])
    finally:
        loader.manager.shutdown()


def test_length_and_ids(dataset, elasticc_parquet):
    """One item per parquet row, identified by SNID."""
    table = pd.read_parquet(elasticc_parquet)

    assert len(dataset) == len(table)
    assert [dataset.get_object_id(i) for i in range(len(dataset))] == [str(s) for s in table["SNID"]]


def test_getters_emit_raw_columns(dataset, elasticc_parquet):
    """Every point is emitted as stored, saturated ones included."""
    row = pd.read_parquet(elasticc_parquet).iloc[3]
    assert (np.asarray(row["PHOTFLAG"]) & 1024).any()

    for field, column in COLUMNS.items():
        np.testing.assert_array_equal(getattr(dataset, f"get_{field}")(3), np.asarray(row[column]))
    assert dataset.get_time(3).dtype == np.float64
    assert dataset.get_photflag(3).dtype == np.int64


def test_prepared_inputs_match_seldon(dataset, seldon_config, seldon_batch):
    """The dataset and prepare_inputs give exactly what SELDON's own data path gives."""
    ours = prepare_inputs(_collate(dataset, range(len(dataset))), seldon_config)

    assert len(seldon_batch["x_input"]) == len(dataset)
    for key in ["time_full", "band_idx", "band_idx_full", "pad_mask_part", "pad_mask_full"]:
        np.testing.assert_array_equal(ours[key], seldon_batch[key].numpy(), err_msg=key)
    # setup() recomputes the flux statistics as float32, where the run config stores them as
    # floats; the scaled fluxes differ in their last float32 bit.
    np.testing.assert_allclose(ours["x_input"], seldon_batch["x_input"].numpy(), rtol=0, atol=1e-7)


def test_collate_pads_with_masks(dataset):
    """Per-point fields collate padded to the longest light curve, with a mask."""
    samples = [{field: getattr(dataset, f"get_{field}")(i) for field in FIELDS} for i in (0, 1)]
    lengths = [len(sample["time"]) for sample in samples]

    for field in FIELDS:
        collated = getattr(dataset, f"collate_{field}")(samples)
        assert collated[field].shape == (2, max(lengths))
        assert collated[f"{field}_mask"].sum(axis=1).tolist() == lengths
        for row, sample in enumerate(samples):
            np.testing.assert_array_equal(collated[field][row, : lengths[row]], sample[field])


def test_collate_padding_values():
    """Padding is zero or an empty string, never a value that looks like data."""
    out = collate_light_curve_field([{"band": np.array(["u"])}, {"band": np.array(["g", "r"])}], "band")
    assert out["band"].tolist() == [["u", ""], ["g", "r"]]
    assert out["band_mask"].tolist() == [[True, False], [True, True]]

    out = collate_light_curve_field(
        [{"photflag": np.array([4096])}, {"photflag": np.array([], dtype=np.int64)}], "photflag"
    )
    assert out["photflag"].tolist() == [[4096], [0]]
    assert out["photflag_mask"].tolist() == [[True], [False]]


def _write_infer_config(tmp_path, parquet, weights, reconstruction_grid=False):
    config_file = tmp_path / "infer_config.toml"
    config_file.write_text(
        f"""
[general]
results_dir = "{tmp_path / "results"}"

[data_loader]
batch_size = 6

[model]
name = "hyrax_seldon.models.seldon.HyraxSeldon"

[model.HyraxSeldon]
seldon_config = "{weights.parent.parent / "hparams.yaml"}"
reconstruction_grid = {str(reconstruction_grid).lower()}

[infer]
model_weights_file = "{weights}"

[data_request.infer.data]
dataset_class = "hyrax_seldon.datasets.elasticc_parquet.ElasticcParquetDataset"
data_location = "{parquet}"
primary_id_field = "object_id"
fields = ["time", "flux", "flux_err", "band", "photflag"]
"""
    )
    return config_file


EXPECTED_KEYS = {
    "reconstructed",
    "error",
    "z_mean",
    "z_sigma",
    "z_var",
    "z_sample",
    "z",
    "decoder_inputs",
    "memory",
    "parameters_global_offset",
    "parameters_global_amplitude",
    "parameters_redshift",
    "class_idx",
    "class_emb",
    "elbo",
    "pad_mask_full",
    "short_lightcurve",
}
GRID_KEYS = {"grid_time", "grid_reconstructed", "grid_error"}


@pytest.mark.parametrize("reconstruction_grid", [False, True])
def test_infer_end_to_end(tmp_path, elasticc_parquet, seldon_checkpoint, reconstruction_grid):
    """h.infer() over ELAsTiCC parquet writes every output key, one row per object."""
    from hyrax import Hyrax

    h = Hyrax(
        config_file=_write_infer_config(tmp_path, elasticc_parquet, seldon_checkpoint, reconstruction_grid)
    )
    results = h.infer()

    table = pd.read_parquet(elasticc_parquet)
    assert len(results) == len(table)
    assert sorted(results.ids()) == sorted(str(s) for s in table["SNID"])
    expected = EXPECTED_KEYS | (GRID_KEYS if reconstruction_grid else set())
    assert set(results.keys()) - {"object_id"} == expected

    first = results[0]
    assert np.asarray(first["reconstructed"]).shape == (512, 1)
    assert np.asarray(first["z_mean"]).shape == (12,)
    assert np.isfinite(np.asarray(first["reconstructed"])).all()
    if reconstruction_grid:
        assert np.asarray(first["grid_reconstructed"]).shape == (6, 64)
