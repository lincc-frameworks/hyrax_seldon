import numpy as np
import pandas as pd
import pytest
from seldon_fixtures import SATURATION_FLAG, target_yaml_text

from hyrax_seldon.collate import collate_light_curve_field
from hyrax_seldon.datasets.elasticc_parquet import ElasticcParquetDataset

FIELDS = ["time", "flux", "flux_err", "band", "detected"]


@pytest.fixture
def dataset(elasticc_parquet):
    """The dataset over the synthetic ELAsTiCC parquet file."""
    config = {"data_set": {"ElasticcParquetDataset": {"read_kwargs": {}}}}
    return ElasticcParquetDataset(config, data_location=elasticc_parquet)


@pytest.fixture(scope="module")
def seldon_loader(elasticc_parquet):
    """SELDON's own data module, set up from the same parquet file with the target config."""
    from omegaconf import OmegaConf
    from TAE.datasets.dataset_factory import get_module as get_dataset

    from hyrax_seldon.settings import band_index_map

    config = OmegaConf.create(target_yaml_text())
    OmegaConf.resolve(config)
    dataset_config = OmegaConf.to_container(config)["dataset"]["config"]
    dataset_config.update(lightcurve_path=[str(elasticc_parquet)], num_workers=0, augmentations=None)

    band_index = band_index_map(dataset_config)
    loader = get_dataset("FlexibleLightCurveDataLoader", dataset_config)
    loader.setup()
    loader.test_band_index = band_index
    yield loader
    loader.manager.shutdown()


def test_length_and_ids(dataset, elasticc_parquet):
    """One item per parquet row, identified by SNID."""
    table = pd.read_parquet(elasticc_parquet)

    assert len(dataset) == len(table)
    assert [dataset.get_object_id(i) for i in range(len(dataset))] == [str(s) for s in table["SNID"]]


def test_getters_drop_saturated_points(dataset, elasticc_parquet):
    """Saturated points are dropped; every other point is emitted raw, in file order."""
    row = pd.read_parquet(elasticc_parquet).iloc[3]
    kept = (np.asarray(row["PHOTFLAG"]) & SATURATION_FLAG) == 0
    assert not kept.all()

    np.testing.assert_array_equal(dataset.get_time(3), np.asarray(row["MJD"])[kept])
    np.testing.assert_array_equal(dataset.get_flux(3), np.asarray(row["FLUXCAL"])[kept])
    np.testing.assert_array_equal(dataset.get_flux_err(3), np.asarray(row["FLUXCALERR"])[kept])
    np.testing.assert_array_equal(dataset.get_band(3), np.asarray(row["BAND"])[kept])
    np.testing.assert_array_equal(dataset.get_detected(3), (np.asarray(row["PHOTFLAG"]) > 0)[kept])
    assert dataset.get_time(3).dtype == np.float64
    assert dataset.get_detected(3).dtype == bool
    assert (~dataset.get_detected(3)).any()


def test_getters_match_seldon_extraction(dataset, seldon_loader):
    """The points, bands and detection flags match what SELDON's own parquet loader extracts."""
    assert len(seldon_loader.full_light_curve_array) == len(dataset)
    band_index = seldon_loader.test_band_index

    for i in range(len(dataset)):
        light_curve = seldon_loader.full_light_curve_array[i]
        np.testing.assert_array_equal(light_curve[:, 0], dataset.get_time(i).astype(np.float32))
        np.testing.assert_array_equal(light_curve[:, 1], dataset.get_flux(i).astype(np.float32))
        np.testing.assert_array_equal(light_curve[:, 2], dataset.get_flux_err(i).astype(np.float32))
        np.testing.assert_array_equal(
            seldon_loader.meta_data["detection_flag"][i][:, 0], dataset.get_detected(i).astype(np.float32)
        )
        bands = [band_index[b.lower()] for b in dataset.get_band(i)]
        np.testing.assert_array_equal(seldon_loader.band_array[i], bands)


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
    """Padding is zero, empty-string or false, never a value that looks like data."""
    out = collate_light_curve_field([{"band": np.array(["u"])}, {"band": np.array(["g", "r"])}], "band")
    assert out["band"].tolist() == [["u", ""], ["g", "r"]]
    assert out["band_mask"].tolist() == [[True, False], [True, True]]

    out = collate_light_curve_field(
        [{"detected": np.array([True])}, {"detected": np.array([], dtype=bool)}], "detected"
    )
    assert out["detected"].dtype == bool
    assert out["detected_mask"].tolist() == [[True], [False]]


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
fields = ["time", "flux", "flux_err", "band", "detected"]
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
