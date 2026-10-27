import numpy as np
import pytest
import torch
from seldon_fixtures import collate_light_curves, random_light_curve

from hyrax_seldon.encoding import encode_batch, encode_light_curves, prepare_inputs
from hyrax_seldon.settings import SeldonSettings


@pytest.fixture
def settings(seldon_config):
    """Settings read from the synthetic SELDON run config."""
    return SeldonSettings.from_config(seldon_config)


def _encode_one(mjd, flux, flux_err, band_idx, detected, settings):
    """Encode a single light curve and return its row."""
    out = encode_light_curves([(mjd, flux, flux_err, band_idx, detected)], settings)
    return {key: value[0] for key, value in out.items()}


def _seldon_reference(light_curves, settings):
    """Run SELDON's own LightCurvePreprocessor and collate, configured from the run config.

    SELDON picks a random window from light curves longer than max_length, so those are
    handed over as their earliest max_length points.
    """
    from TAE.datasets.lightcurve_dataset import enhanced_collate_fn
    from TAE.datasets.lightcurve_preprocessor import LightCurvePreprocessor

    arrays, band_indices, detection_flags = [], [], []
    for mjd, flux, flux_err, band, detected in light_curves:
        keep = slice(None)
        if len(mjd) > settings.max_length:
            keep = np.argsort(mjd, kind="stable")[: settings.max_length]
        arrays.append(np.column_stack([mjd[keep], flux[keep], flux_err[keep]]).astype(np.float32))
        band_indices.append(np.array([settings.band_to_index(b) for b in band[keep]], dtype=np.int64))
        detection_flags.append(detected[keep].astype(np.float32)[:, None])

    dataset_config = settings.run_config["dataset"]["config"]
    preprocessor = LightCurvePreprocessor(
        light_curves=arrays,
        class_names=["SNIa"] * len(arrays),
        class_weights={"SNIa": 1.0},
        band_indices=band_indices,
        band_weights=dataset_config["band_weights"],
        flux_stats=dataset_config["flux_stats"],
        meta_data={"detection_flag": detection_flags},
        flux_normalization=dataset_config["flux_normalization"],
        time_normalization=dataset_config["time_normalization"],
        augmentations=None,
        min_points=settings.min_points,
        max_length=settings.max_length,
        resample=False,
        max_points=settings.max_points,
        t_offset=settings.t_offset,
    )
    return enhanced_collate_fn([preprocessor[i] for i in range(len(arrays))])


def test_matches_seldon_preprocessing(settings):
    """Model inputs equal SELDON's preprocessor + collate configured from the run config."""
    rng = np.random.default_rng(1)
    light_curves = [random_light_curve(rng, n) for n in (100, 300, 700, 40)]
    # Points arrive in no particular order; SELDON sorts them itself.
    light_curves = [
        tuple(np.asarray(column)[order] for column in lc)
        for lc, order in ((lc, rng.permutation(len(lc[0]))) for lc in light_curves)
    ]

    ours = encode_batch(collate_light_curves(light_curves), settings)
    reference = _seldon_reference(light_curves, settings)

    for key in ["x_input", "time_full", "band_idx", "band_idx_full", "pad_mask_part", "pad_mask_full"]:
        np.testing.assert_array_equal(ours[key], reference[key].numpy(), err_msg=key)


def test_output_shapes_and_dtypes(settings):
    """Every array has a fixed per-row shape, with the dtypes the model expects."""
    rng = np.random.default_rng(2)
    light_curves = [random_light_curve(rng, 40), random_light_curve(rng, 90)]
    out = encode_batch(collate_light_curves(light_curves), settings)

    expected = {
        "x_input": ((2, 64, 4), np.float32),
        "band_idx": ((2, 64, 1), np.int64),
        "pad_mask_part": ((2, 64), np.bool_),
        "time_full": ((2, 512, 1), np.float32),
        "band_idx_full": ((2, 512, 1), np.int64),
        "pad_mask_full": ((2, 512), np.bool_),
        "n_points": ((2,), np.int64),
    }
    assert {key: (value.shape, value.dtype) for key, value in out.items()} == expected


def test_encoder_sees_earliest_points(settings):
    """The encoder gets the earliest max_points points, and t0 is the first of them."""
    rng = np.random.default_rng(3)
    _, flux, flux_err, _, detected = random_light_curve(rng, 100)
    mjd = np.arange(100, dtype=np.float64) + 60000.0

    out = _encode_one(mjd, flux, flux_err, np.full(100, 40), detected, settings)

    np.testing.assert_allclose(out["x_input"][:, 0], (mjd[:64] - mjd[0]) / settings.t_max, rtol=1e-6)
    np.testing.assert_array_equal(out["x_input"][:, 3], detected[:64].astype(np.float32))
    assert out["pad_mask_part"].all()
    np.testing.assert_allclose(out["time_full"][:100, 0], (mjd - mjd[0]) / settings.t_max, rtol=1e-6)
    assert out["pad_mask_full"].sum() == 100
    assert out["n_points"] == 100


def test_min_points_top_up_leaves_earliest_selection(settings):
    """SELDON's min_points top-up adds the earliest detections, which the earliest points already hold."""
    mjd = np.arange(100, dtype=np.float64) + 60000.0
    detected = np.zeros(100, bool)
    detected[80:] = True  # no detections among the earliest 64 points
    ones = np.ones(100)

    out = _encode_one(mjd, ones, ones, np.full(100, 40), detected, settings)

    np.testing.assert_allclose(out["x_input"][:, 0], (mjd[:64] - mjd[0]) / settings.t_max, rtol=1e-6)
    assert not out["x_input"][:, 3].any()


def test_full_arrays_keep_earliest_max_length(settings):
    """Light curves longer than max_length keep their earliest max_length points."""
    mjd = np.arange(600, dtype=np.float64) + 60000.0
    ones = np.ones(600)

    out = _encode_one(mjd[::-1], ones, ones, np.full(600, 40), ones.astype(bool), settings)

    assert out["pad_mask_full"].all()
    assert out["n_points"] == 600
    np.testing.assert_allclose(out["time_full"][:, 0], np.arange(512) / settings.t_max, rtol=1e-6)


def test_padding(settings):
    """Values are edge-padded, band indices zero-padded, and masks mark real points."""
    mjd = np.arange(40, dtype=np.float64) + 60000.0
    flux = np.linspace(100.0, 500.0, 40)
    bands = np.tile([37, 38, 39, 40], 10)

    out = _encode_one(mjd, flux, flux / 10, bands, np.ones(40, bool), settings)

    assert out["pad_mask_part"].tolist() == [True] * 40 + [False] * 24
    assert out["pad_mask_full"].tolist() == [True] * 40 + [False] * 472
    np.testing.assert_array_equal(out["x_input"][40:], np.repeat(out["x_input"][39:40], 24, axis=0))
    np.testing.assert_array_equal(out["time_full"][40:, 0], np.full(472, out["time_full"][39, 0]))
    assert out["band_idx"][:, 0].tolist() == bands.tolist() + [0] * 24
    assert out["band_idx_full"][:, 0].tolist() == bands.tolist() + [0] * 472


@pytest.mark.parametrize("n_points", [0, 31])
def test_too_few_points_rejected(settings, n_points):
    """SELDON cannot encode a light curve with fewer than min_points points, so the batch fails."""
    rng = np.random.default_rng(4)
    light_curves = [random_light_curve(rng, 40), random_light_curve(rng, n_points)]

    with pytest.raises(ValueError, match=rf"batch rows \[1\] have \[{n_points}\] points"):
        encode_batch(collate_light_curves(light_curves), settings)


def test_flux_transform(settings):
    """Flux is semilog-scaled by flux_stats.max, with errors propagated."""
    flux = np.concatenate([[-50.0, 0.0, 10.0, settings.flux_max], np.full(36, 100.0)])
    flux_err = np.full(40, 5.0)
    mjd = np.arange(40) + 60000.0

    out = _encode_one(mjd, flux, flux_err, np.full(40, 40), np.ones(40, bool), settings)

    log_max = np.log10(1 + settings.flux_max)
    expected_flux = np.sign(flux[:4]) * np.log10(1 + np.abs(flux[:4])) / log_max - 0.5
    expected_err = flux_err[:4] / (np.abs(flux[:4]) + 1) / np.log(10) / log_max
    np.testing.assert_allclose(out["x_input"][:4, 1], expected_flux, rtol=1e-6)
    np.testing.assert_allclose(out["x_input"][:4, 2], expected_err, rtol=1e-6)
    np.testing.assert_allclose(out["x_input"][3, 1], 0.5, rtol=1e-6)


def test_zero_point_flux(settings):
    """Flux and error are divided by the run config's zero_point_flux first."""
    from dataclasses import replace

    mjd = np.arange(40) + 60000.0
    flux = np.linspace(10.0, 400.0, 40)
    scaled = replace(settings, zero_point_flux=10.0)

    out = _encode_one(mjd, flux, flux, np.full(40, 40), np.ones(40, bool), scaled)
    reference = _encode_one(mjd, flux / 10, flux / 10, np.full(40, 40), np.ones(40, bool), settings)

    np.testing.assert_allclose(out["x_input"], reference["x_input"], rtol=1e-6)


def test_masked_and_non_finite_points_dropped(settings):
    """Padding, NaN values and masked points never reach the model."""
    rng = np.random.default_rng(5)
    mjd, flux, flux_err, band, detected = random_light_curve(rng, 50)
    batch = collate_light_curves([(mjd, flux, flux_err, band, detected)])
    batch["flux"][0, 3] = np.nan
    batch["flux_err_mask"][0, 5] = False

    keep = np.ones(50, bool)
    keep[[3, 5]] = False
    out = encode_batch(batch, settings)
    reference = encode_batch(
        collate_light_curves([(mjd[keep], flux[keep], flux_err[keep], band[keep], detected[keep])]), settings
    )

    for key in out:
        np.testing.assert_array_equal(out[key], reference[key])


def test_non_detections_dropped_when_run_excluded_them(settings):
    """A run trained without non-detections only sees detections."""
    from dataclasses import replace

    rng = np.random.default_rng(6)
    mjd, flux, flux_err, band, detected = random_light_curve(rng, 120, detected_fraction=0.5)
    detections_only = replace(settings, include_non_detections=False)

    out = encode_batch(collate_light_curves([(mjd, flux, flux_err, band, detected)]), detections_only)
    reference = encode_batch(
        collate_light_curves(
            [(mjd[detected], flux[detected], flux_err[detected], band[detected], detected[detected])]
        ),
        settings,
    )

    for key in out:
        np.testing.assert_array_equal(out[key], reference[key])
    assert out["x_input"][0, out["pad_mask_part"][0], 3].all()


def test_unknown_band_rejected(settings):
    """A band the model does not know is an error, not a silent guess."""
    rng = np.random.default_rng(7)
    mjd, flux, flux_err, band, detected = random_light_curve(rng, 40)
    band[2] = "J"

    with pytest.raises(ValueError, match="'J'"):
        encode_batch(collate_light_curves([(mjd, flux, flux_err, band, detected)]), settings)


def test_missing_field_reported(settings):
    """A batch missing a configured field names it."""
    rng = np.random.default_rng(8)
    batch = collate_light_curves([random_light_curve(rng, 40)])
    del batch["detected"]

    with pytest.raises(RuntimeError, match="detected"):
        encode_batch(batch, settings)


def test_seldon_global_state_restored(settings, capsys):
    """SELDON's preprocessor reseeds NumPy's global generator and prints; neither leaks out."""
    rng = np.random.default_rng(9)
    batch = collate_light_curves([random_light_curve(rng, 40)])

    np.random.seed(123)
    expected = np.random.rand()
    np.random.seed(123)
    encode_batch(batch, settings)

    assert np.random.rand() == expected
    assert capsys.readouterr().out == ""


def test_prepare_inputs_reads_data_group(seldon_config, settings):
    """prepare_inputs reads the 'data' group and returns arrays torch can take as-is."""
    rng = np.random.default_rng(10)
    batch = collate_light_curves([random_light_curve(rng, 40)])

    out = prepare_inputs({"data": batch, "object_id": np.array(["a"])}, seldon_config)

    for key, value in encode_batch(batch, settings).items():
        np.testing.assert_array_equal(out[key], value)
        torch.from_numpy(out[key])

    with pytest.raises(RuntimeError, match="'data'"):
        prepare_inputs({"light_curves": batch}, seldon_config)
