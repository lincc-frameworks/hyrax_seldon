import numpy as np
import pytest
import torch
from seldon_fixtures import collate_light_curves, random_light_curve

from hyrax_seldon.encoding import encode_batch, encode_light_curve, prepare_inputs
from hyrax_seldon.settings import SeldonSettings


@pytest.fixture
def settings(seldon_config):
    """Settings read from the synthetic SELDON run config."""
    return SeldonSettings.from_config(seldon_config)


class _MostRecent:
    """Stands in for SELDON's augmenter so SELDON selects the most recent max_points."""

    def __init__(self, max_points):
        self.max_points = max_points

    def transform(self, data, *args):
        return np.arange(max(0, len(data) - self.max_points), len(data))


def _seldon_reference(light_curves, settings):
    """Run SELDON's own LightCurvePreprocessor and collate over the same light curves.

    SELDON chooses a random window over curves longer than max_length (and can never
    choose the last one), so it is handed the sorted, most recent max_length points.
    """
    from TAE.datasets.lightcurve_dataset import enhanced_collate_fn
    from TAE.datasets.lightcurve_preprocessor import LightCurvePreprocessor

    arrays, band_indices, detection_flags = [], [], []
    for mjd, flux, flux_err, band, detected in light_curves:
        order = np.argsort(mjd)[-settings.max_length :]
        arrays.append(np.column_stack([mjd[order], flux[order], flux_err[order]]).astype(np.float32))
        band_indices.append(np.array([settings.band_to_index(b) for b in band[order]]))
        detection_flags.append(detected[order].astype(np.float32)[:, None])

    dataset_config = settings.run_config["dataset"]["config"]
    preprocessor = LightCurvePreprocessor(
        light_curves=arrays,
        class_names=["SNIa"] * len(arrays),
        class_weights={"SNIa": 1.0},
        band_indices=band_indices,
        band_weights={index: 1.0 for index in range(len(settings.band_index))},
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
    preprocessor.augmenter = _MostRecent(settings.max_points)
    return enhanced_collate_fn([preprocessor[i] for i in range(len(arrays))])


def test_parity_with_seldon_preprocessing(settings):
    """Model inputs match SELDON's preprocessor + collate on the same light curves."""
    rng = np.random.default_rng(1)
    light_curves = [random_light_curve(rng, n) for n in (100, 300, 700)]
    # A curve shorter than max_points; every point detected so SELDON's min_points rule holds.
    light_curves.append(random_light_curve(rng, 40, detected_fraction=1.0))

    # hyrax_seldon sorts for itself; hand it the points out of order.
    shuffled = []
    for lc in light_curves:
        order = rng.permutation(len(lc[0]))
        shuffled.append(tuple(np.asarray(column)[order] for column in lc))

    ours = encode_batch(collate_light_curves(shuffled), settings)
    reference = _seldon_reference(light_curves, settings)

    np.testing.assert_allclose(ours["x_input"], reference["x_input"].numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ours["time_full"], reference["time_full"].numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(ours["band_idx"], reference["band_idx"].numpy())
    np.testing.assert_array_equal(ours["band_idx_full"], reference["band_idx_full"].numpy())
    np.testing.assert_array_equal(ours["pad_mask_part"], reference["pad_mask_part"].numpy())
    np.testing.assert_array_equal(ours["pad_mask_full"], reference["pad_mask_full"].numpy())


def test_output_shapes_and_dtypes(settings):
    """Every array has a fixed per-row shape, with the dtypes the model expects."""
    rng = np.random.default_rng(2)
    out = encode_batch(
        collate_light_curves([random_light_curve(rng, 10), random_light_curve(rng, 90)]), settings
    )

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


def test_encoder_sees_most_recent_points(settings):
    """The encoder gets the most recent max_points points, and t0 is the earliest of them."""
    rng = np.random.default_rng(3)
    mjd, flux, flux_err, band, detected = random_light_curve(rng, 100)
    mjd = np.arange(100, dtype=np.float64) + 60000.0

    out = encode_light_curve(mjd, flux, flux_err, np.full(100, 40), detected, settings)

    expected_time = (mjd[36:] - mjd[36]) / settings.t_max
    np.testing.assert_allclose(out["x_input"][:, 0], expected_time, rtol=1e-6)
    np.testing.assert_array_equal(out["x_input"][:, 3], detected[36:].astype(np.float32))
    assert out["pad_mask_part"].all()
    # Full arrays hold all 100 points; earlier points have negative times.
    np.testing.assert_allclose(out["time_full"][:100, 0], (mjd - mjd[36]) / settings.t_max, rtol=1e-6)
    assert out["pad_mask_full"].sum() == 100
    assert out["n_points"] == 100


def test_full_arrays_keep_most_recent_max_length(settings):
    """Light curves longer than max_length keep their most recent max_length points."""
    mjd = np.arange(600, dtype=np.float64) + 60000.0
    ones = np.ones(600)

    out = encode_light_curve(mjd, ones, ones, np.full(600, 40), ones.astype(bool), settings)

    assert out["pad_mask_full"].all()
    assert out["n_points"] == 600
    # The last full point is the most recent observation, 63 days after t0.
    np.testing.assert_allclose(out["time_full"][-1, 0], 63 / settings.t_max, rtol=1e-6)
    np.testing.assert_allclose(out["time_full"][0, 0], -(511 - 63) / settings.t_max, rtol=1e-6)


def test_padding(settings):
    """Values are edge-padded, band indices zero-padded, and masks mark real points."""
    mjd = np.array([60000.0, 60001.0, 60003.0])
    flux = np.array([100.0, 200.0, 300.0])

    out = encode_light_curve(mjd, flux, flux / 10, np.array([37, 38, 39]), np.ones(3, bool), settings)

    assert out["pad_mask_part"].tolist() == [True] * 3 + [False] * 61
    assert out["pad_mask_full"].tolist() == [True] * 3 + [False] * 509
    np.testing.assert_array_equal(out["x_input"][3:], np.repeat(out["x_input"][2:3], 61, axis=0))
    np.testing.assert_array_equal(out["time_full"][3:, 0], np.full(509, out["time_full"][2, 0]))
    assert out["band_idx"][:, 0].tolist() == [37, 38, 39] + [0] * 61
    assert out["band_idx_full"][:, 0].tolist() == [37, 38, 39] + [0] * 509


def test_empty_light_curve(settings):
    """A light curve with no valid points encodes to zeros with all-false masks."""
    empty = np.array([])
    out = encode_light_curve(empty, empty, empty, empty.astype(int), empty.astype(bool), settings)

    assert not out["pad_mask_part"].any()
    assert not out["pad_mask_full"].any()
    assert not out["x_input"].any()
    assert out["n_points"] == 0


def test_flux_transform(settings):
    """Flux is semilog-scaled by flux_stats.max, with errors propagated."""
    flux = np.array([-50.0, 0.0, 10.0, settings.flux_max])
    flux_err = np.array([5.0, 5.0, 5.0, 5.0])
    mjd = np.arange(4) + 60000.0

    out = encode_light_curve(mjd, flux, flux_err, np.full(4, 40), np.ones(4, bool), settings)

    log_max = np.log10(1 + settings.flux_max)
    expected_flux = np.sign(flux) * np.log10(1 + np.abs(flux)) / log_max - 0.5
    expected_err = flux_err / (np.abs(flux) + 1) / np.log(10) / log_max
    np.testing.assert_allclose(out["x_input"][:4, 1], expected_flux, rtol=1e-6)
    np.testing.assert_allclose(out["x_input"][:4, 2], expected_err, rtol=1e-6)
    np.testing.assert_allclose(out["x_input"][3, 1], 0.5, rtol=1e-6)


def test_zero_point_flux(settings):
    """Flux and error are divided by the run config's zero_point_flux first."""
    from dataclasses import replace

    mjd = np.arange(4) + 60000.0
    flux = np.array([10.0, 20.0, 30.0, 40.0])
    scaled = replace(settings, zero_point_flux=10.0)

    out = encode_light_curve(mjd, flux, flux, np.full(4, 40), np.ones(4, bool), scaled)
    reference = encode_light_curve(mjd, flux / 10, flux / 10, np.full(4, 40), np.ones(4, bool), settings)

    np.testing.assert_array_equal(out["x_input"], reference["x_input"])


def test_masked_and_non_finite_points_dropped(settings):
    """Padding, NaN values and masked points never reach the model."""
    rng = np.random.default_rng(4)
    mjd, flux, flux_err, band, detected = random_light_curve(rng, 20)
    batch = collate_light_curves([(mjd, flux, flux_err, band, detected)])
    batch["flux"][0, 3] = np.nan
    batch["flux_err_mask"][0, 5] = False

    keep = np.ones(20, bool)
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

    rng = np.random.default_rng(5)
    mjd, flux, flux_err, band, detected = random_light_curve(rng, 50, detected_fraction=0.5)
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
    rng = np.random.default_rng(6)
    mjd, flux, flux_err, band, detected = random_light_curve(rng, 5)
    band[2] = "J"

    with pytest.raises(ValueError, match="'J'"):
        encode_batch(collate_light_curves([(mjd, flux, flux_err, band, detected)]), settings)


def test_missing_field_reported(settings):
    """A batch missing a configured field names it."""
    rng = np.random.default_rng(7)
    batch = collate_light_curves([random_light_curve(rng, 5)])
    del batch["detected"]

    with pytest.raises(RuntimeError, match="detected"):
        encode_batch(batch, settings)


def test_prepare_inputs_reads_data_group(seldon_config, settings):
    """prepare_inputs reads the 'data' group and returns arrays torch can take as-is."""
    rng = np.random.default_rng(8)
    batch = collate_light_curves([random_light_curve(rng, 30)])

    out = prepare_inputs({"data": batch, "object_id": np.array(["a"])}, seldon_config)

    for key, value in encode_batch(batch, settings).items():
        np.testing.assert_array_equal(out[key], value)
        torch.from_numpy(out[key])

    with pytest.raises(RuntimeError, match="'data'"):
        prepare_inputs({"light_curves": batch}, seldon_config)
