import numpy as np
import pytest
import torch
from hyrax_seldon.models.seldon import HyraxSeldon
from hyrax_seldon.settings import SeldonSettings
from seldon_fixtures import (
    DETECTION_FLAG,
    FIRST_DETECTION_FLAG,
    SATURATION_FLAG,
    collate_light_curves,
    prepare_inputs,
    random_light_curve,
)


@pytest.fixture
def settings(seldon_config):
    """Settings read from the synthetic SELDON run config."""
    return SeldonSettings.from_config(seldon_config)


def test_output_shapes_and_dtypes(seldon_config):
    """Every array has a fixed per-row shape, with the dtypes the model expects."""
    rng = np.random.default_rng(2)
    light_curves = [random_light_curve(rng, 40), random_light_curve(rng, 90)]
    out = prepare_inputs(collate_light_curves(light_curves), seldon_config)

    expected = {
        "x_input": ((2, 64, 4), np.float32),
        "band_idx": ((2, 64, 1), np.int64),
        "pad_mask_part": ((2, 64), np.bool_),
        "time_full": ((2, 512, 1), np.float32),
        "band_idx_full": ((2, 512, 1), np.int64),
        "pad_mask_full": ((2, 512), np.bool_),
        "n_detections": ((2,), np.int64),
    }
    assert {key: (value.shape, value.dtype) for key, value in out.items()} == expected


def test_light_curve_frame_removes_padding(settings):
    """Each row holds its own points, unpadded, in SELDON's parquet columns."""
    rng = np.random.default_rng(3)
    light_curves = [random_light_curve(rng, 40), random_light_curve(rng, 90)]

    frame = HyraxSeldon.light_curve_frame(collate_light_curves(light_curves), settings)

    assert list(frame.columns) == ["MJD", "FLUXCAL", "FLUXCALERR", "BAND", "PHOTFLAG"]
    for row, light_curve in enumerate(light_curves):
        for column, values in zip(frame.columns, light_curve, strict=True):
            np.testing.assert_array_equal(frame[column][row], values)


def test_masked_points_dropped(seldon_config):
    """Padding and masked points never reach SELDON."""
    rng = np.random.default_rng(5)
    light_curve = random_light_curve(rng, 50)
    batch = collate_light_curves([light_curve])
    batch["flux_err_mask"][0, 5] = False

    keep = np.ones(50, bool)
    keep[5] = False
    out = prepare_inputs(batch, seldon_config)
    reference = prepare_inputs(
        collate_light_curves([tuple(column[keep] for column in light_curve)]), seldon_config
    )

    for key in out:
        np.testing.assert_array_equal(out[key], reference[key])


def test_seldon_selects_points_by_photflag(seldon_config):
    """SELDON drops saturated points and counts detections from PHOTFLAG."""
    rng = np.random.default_rng(6)
    mjd, flux, flux_err, band, _ = random_light_curve(rng, 60)
    photflag = np.zeros(60, np.int64)
    photflag[:40] = DETECTION_FLAG
    photflag[0] |= FIRST_DETECTION_FLAG
    photflag[[1, 2, 45]] |= SATURATION_FLAG

    out = prepare_inputs(collate_light_curves([(mjd, flux, flux_err, band, photflag)]), seldon_config)

    # Saturated points are dropped; non-detections are kept, as the run config includes them.
    assert out["pad_mask_full"].sum() == 57
    assert out["n_detections"].tolist() == [38]
    assert out["x_input"][0, :, 3].sum() == 38


def test_light_curves_with_few_detections_kept(seldon_config, settings):
    """A light curve with fewer than min_points detections is encoded, not dropped."""
    rng = np.random.default_rng(7)
    light_curves = [random_light_curve(rng, 80), random_light_curve(rng, 60, detected_fraction=0.2)]

    out = prepare_inputs(collate_light_curves(light_curves), seldon_config)

    assert out["x_input"].shape[0] == 2
    assert out["n_detections"][1] < settings.min_points <= out["n_detections"][0]


@pytest.mark.parametrize("n_points", [0, 31])
def test_too_few_points_rejected(seldon_config, n_points):
    """SELDON cannot encode a light curve with fewer than min_points points, so the batch fails."""
    rng = np.random.default_rng(4)
    light_curves = [random_light_curve(rng, 40), random_light_curve(rng, n_points)]

    with pytest.raises(ValueError, match=rf"batch rows \[1\] have \[{n_points}\] points"):
        prepare_inputs(collate_light_curves(light_curves), seldon_config)


def test_unknown_band_rejected(seldon_config):
    """A band the model does not know is an error, not a silent guess."""
    rng = np.random.default_rng(8)
    mjd, flux, flux_err, band, photflag = random_light_curve(rng, 40)
    band[2] = "J"

    # SELDON lower-cases single-character band names before looking them up.
    with pytest.raises(ValueError, match="'j'"):
        prepare_inputs(collate_light_curves([(mjd, flux, flux_err, band, photflag)]), seldon_config)


def test_missing_field_reported(seldon_config):
    """A batch missing a configured field names it."""
    rng = np.random.default_rng(9)
    batch = collate_light_curves([random_light_curve(rng, 40)])
    del batch["photflag"]

    with pytest.raises(RuntimeError, match="photflag"):
        prepare_inputs(batch, seldon_config)


def test_seldon_global_state_restored(seldon_config, capsys):
    """SELDON reseeds NumPy's global generator and prints; neither leaks out."""
    rng = np.random.default_rng(10)
    batch = collate_light_curves([random_light_curve(rng, 40)])

    np.random.seed(123)
    expected = np.random.rand()
    np.random.seed(123)
    prepare_inputs(batch, seldon_config)

    assert np.random.rand() == expected
    assert capsys.readouterr() == ("", "")


def test_data_group_required(seldon_config):
    """prepare_inputs reads the 'data' group into arrays torch takes as-is, and names it when missing."""
    from hyrax.context import run_context

    rng = np.random.default_rng(11)
    batch = collate_light_curves([random_light_curve(rng, 40)])

    with run_context("infer", config=seldon_config):
        out = HyraxSeldon.prepare_inputs({"data": batch, "object_id": np.array(["a"])})
        with pytest.raises(RuntimeError, match="'data'"):
            HyraxSeldon.prepare_inputs({"light_curves": batch})

    for value in out.values():
        torch.from_numpy(value)
