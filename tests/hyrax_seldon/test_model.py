import copy
import logging
import shutil

import numpy as np
import pytest
import torch
import yaml
from hyrax_seldon.models.seldon import SELDON_CONFIG_FILENAME, HyraxSeldon
from seldon_fixtures import collate_light_curves, prepare_inputs, random_light_curve

DETERMINISTIC_KEYS = [
    "reconstructed",
    "error",
    "z_mean",
    "z_sigma",
    "z_var",
    "z",
    "decoder_inputs",
    "memory",
    "elbo",
]
STOCHASTIC_KEYS = ["z_sample", "class_idx", "class_emb"]
PARAMETER_KEYS = ["global_offset", "global_amplitude", "redshift"]

#: How far outputs on the runtime's device may be from SELDON's on the CPU, as a fraction of
#: the reference's largest magnitude. CPU and GPU kernels round differently, and the encoder's
#: ODE solve compounds that; the test asks for the same model, not bit-identical arithmetic.
DEVICE_MARGIN = 0.02


@pytest.fixture
def model(hyrax_config, seldon_checkpoint):
    """HyraxSeldon with the synthetic run's checkpoint loaded, in eval mode.

    Hyrax's loader puts the model on the runtime's device (CUDA or MPS when available), so
    these tests cover the device code paths.
    """
    model = HyraxSeldon(config=hyrax_config)
    model.load(seldon_checkpoint)
    return model.eval()


def _device(model):
    return next(model.parameters()).device


def _checkpoint_state_dict(checkpoint_path):
    """The Lightning checkpoint's weights, as SELDON saved them."""
    return torch.load(checkpoint_path, map_location="cpu", weights_only=True)["state_dict"]


def _light_curves(lengths, seed=0, detected_fractions=None):
    rng = np.random.default_rng(seed)
    detected_fractions = detected_fractions or [0.9] * len(lengths)
    pairs = zip(lengths, detected_fractions, strict=True)
    return [random_light_curve(rng, n, fraction) for n, fraction in pairs]


@pytest.fixture
def batch(model, hyrax_config):
    """Prepared model inputs for light curves of several lengths, the last with few detections.

    The inputs are on the model's device, as Hyrax places them.
    """
    light_curves = _light_curves([150, 600, 70, 40], detected_fractions=[0.9, 0.9, 0.9, 0.3])
    prepared = prepare_inputs(collate_light_curves(light_curves), hyrax_config)
    return {key: torch.from_numpy(value).to(_device(model)) for key, value in prepared.items()}


def _cpu(batch):
    return {key: value.cpu() for key, value in batch.items()}


@pytest.fixture
def seldon_model(model, seldon_checkpoint):
    """The reference: the checkpoint loaded into SELDON's model built with SELDON's factory, on the CPU."""
    from TAE.models.factory import get_module

    params = copy.deepcopy(model.settings.model_params)
    seldon_model = get_module(params["name"], params["config"])
    for module in seldon_model.modules():
        module._compiled_call_impl = None
    # SELDON's Lightning experiment holds the model as `model`.
    state_dict = {
        key.removeprefix("model."): value for key, value in _checkpoint_state_dict(seldon_checkpoint).items()
    }
    seldon_model.load_state_dict(state_dict, strict=True)
    return seldon_model.eval()


def test_checkpoint_has_seldon_lightning_layout(seldon_checkpoint):
    """The synthetic run's checkpoint is weights only, every key under SELDON's `model`."""
    checkpoint = torch.load(seldon_checkpoint, map_location="cpu", weights_only=True)

    assert "hyper_parameters" not in checkpoint
    assert len(checkpoint["state_dict"]) == 295
    assert all(key.startswith("model.") for key in checkpoint["state_dict"])


def test_load_lightning_checkpoint(hyrax_config, seldon_checkpoint):
    """Hyrax's load path restores a SELDON Lightning checkpoint's weights exactly."""
    model = HyraxSeldon(config=hyrax_config)
    model.load(seldon_checkpoint)

    original = _checkpoint_state_dict(seldon_checkpoint)
    assert model.state_dict().keys() == original.keys()
    for key, value in model.state_dict().items():
        assert torch.equal(value.cpu(), original[key]), key


def _copy_run_dir(seldon_run_dir, tmp_path):
    """Copy the synthetic run directory, and return the copy's checkpoint."""
    run_dir = tmp_path / "run"
    shutil.copytree(seldon_run_dir, run_dir)
    (checkpoint,) = (run_dir / "checkpoints").glob("*.ckpt")
    return checkpoint


@pytest.mark.parametrize("change", ["missing", "unexpected"])
def test_mismatched_checkpoint_rejected(hyrax_config, seldon_run_dir, tmp_path, change):
    """A checkpoint whose weights do not match the run config's model fails to load."""
    checkpoint_path = _copy_run_dir(seldon_run_dir, tmp_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    if change == "missing":
        checkpoint["state_dict"].pop(next(iter(checkpoint["state_dict"])))
    else:
        checkpoint["state_dict"]["extra.weight"] = torch.zeros(1)
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
        HyraxSeldon(config=hyrax_config).load(checkpoint_path)


def test_load_checkpoint_warns_on_different_run_config(hyrax_config, seldon_run_dir, tmp_path, caplog):
    """A checkpoint whose run's hparams.yaml differs from the run config in use warns."""
    checkpoint_path = _copy_run_dir(seldon_run_dir, tmp_path)
    hparams = checkpoint_path.parent.parent / "hparams.yaml"
    config = yaml.safe_load(hparams.read_text())
    config["dataset"]["config"]["flux_stats"]["max"] *= 2
    hparams.write_text(yaml.safe_dump(config))

    with caplog.at_level(logging.WARNING, logger="hyrax_seldon.models.seldon"):
        HyraxSeldon(config=hyrax_config).load(checkpoint_path)

    assert str(hparams) in caplog.text


def test_load_checkpoint_does_not_warn_on_its_own_run_config(hyrax_config, seldon_checkpoint, caplog):
    """Loading a run's checkpoint with that run's hparams.yaml does not warn."""
    with caplog.at_level(logging.WARNING, logger="hyrax_seldon.models.seldon"):
        HyraxSeldon(config=hyrax_config).load(seldon_checkpoint)

    assert "run config" not in caplog.text


def _call_seldon(seldon_model, batch):
    return seldon_model(
        x=batch["x_input"],
        t=batch["time_full"],
        band_idx_part=batch["band_idx"],
        band_idx_full=batch["band_idx_full"],
        mask=batch["pad_mask_part"],
    )


def _assert_within_margin(ours, reference, name):
    """Assert ``ours`` is within DEVICE_MARGIN of ``reference``, relative to its largest magnitude."""
    difference = (ours.cpu() - reference).abs().max()
    scale = reference.abs().max()
    assert difference <= DEVICE_MARGIN * scale, f"{name}: max difference {difference} vs scale {scale}"


def test_infer_batch_matches_seldon(model, seldon_model, batch):
    """Every deterministic output matches SELDON's model called directly on the CPU."""
    with torch.no_grad():
        ours = model.infer_batch(batch)
        reference = _call_seldon(seldon_model, _cpu(batch))

    for key in DETERMINISTIC_KEYS:
        _assert_within_margin(ours[key], reference[key], key)
    for key in PARAMETER_KEYS:
        _assert_within_margin(ours[f"parameters_{key}"], reference["parameters"][key], key)


def test_stochastic_outputs_match_seldon_under_the_same_seed(model, seldon_model, batch):
    """z_sample, class_idx and class_emb are passed through as SELDON draws them.

    Random draws from one seed differ between devices, so the reference runs on the model's
    device here.
    """
    seldon_model = copy.deepcopy(seldon_model).to(_device(model))
    with torch.no_grad():
        torch.manual_seed(7)
        ours = model.infer_batch(batch)
        torch.manual_seed(7)
        reference = _call_seldon(seldon_model, batch)

    for key in STOCHASTIC_KEYS:
        assert torch.equal(ours[key], reference[key]), key


def test_output_keys_and_fixed_shapes(model, batch):
    """Every output key is present with a fixed per-row shape."""
    with torch.no_grad():
        out = model.infer_batch(batch)

    expected = {
        "reconstructed": (512, 1),
        "error": (512, 1),
        "z_mean": (12,),
        "z_sigma": (12,),
        "z_var": (12,),
        "z_sample": (12,),
        "z": (12,),
        "decoder_inputs": (12,),
        "memory": (128,),
        "parameters_global_offset": (),
        "parameters_global_amplitude": (),
        "parameters_redshift": (),
        "class_idx": (),
        "class_emb": (32,),
        "elbo": (),
        "pad_mask_full": (512,),
        "short_lightcurve": (),
    }
    assert {key: tuple(value.shape[1:]) for key, value in out.items()} == expected
    assert all(value.shape[0] == 4 for value in out.values())
    assert "encoding_sequence" not in out
    assert not any(key.startswith("grid_") for key in out)


def test_short_lightcurve_flag(model, batch):
    """Light curves with fewer than min_points detections, by SELDON's count, are flagged."""
    with torch.no_grad():
        out = model.infer_batch(batch)

    assert out["short_lightcurve"].tolist() == (batch["n_detections"] < 32).tolist()
    assert out["short_lightcurve"].tolist() == [False, False, False, True]


def test_pad_mask_full_passed_through(model, batch):
    """pad_mask_full marks which per-point outputs are real points."""
    with torch.no_grad():
        out = model.infer_batch(batch)

    assert torch.equal(out["pad_mask_full"], batch["pad_mask_full"])
    assert out["pad_mask_full"].sum(dim=1).tolist() == [150, 512, 70, 40]


def test_decode_reproduces_forward(model, batch):
    """decode() evaluates the decoder and error network exactly as forward() does."""
    with torch.no_grad():
        out = model.infer_batch(batch)
        reconstructed, error = model.decode(out["decoder_inputs"], batch["time_full"], batch["band_idx_full"])

    assert torch.equal(reconstructed, out["reconstructed"])
    assert torch.equal(error, out["error"])


def test_save_load_round_trip(model, hyrax_config, batch, tmp_path):
    """Saved weights reload into an identical model, with the run config saved beside them."""
    model.save(tmp_path / "weights.pth")

    assert (tmp_path / "prepare_inputs.py").exists()
    assert (tmp_path / SELDON_CONFIG_FILENAME).read_bytes() == model.settings.seldon_config_path.read_bytes()

    reloaded = HyraxSeldon(config=hyrax_config)
    reloaded.load(tmp_path / "weights.pth")
    reloaded.eval()

    with torch.no_grad():
        first = model.infer_batch(batch)
        second = reloaded.infer_batch(batch)
    for key in DETERMINISTIC_KEYS:
        assert torch.equal(first[key], second[key]), key


def test_load_warns_on_different_run_config(model, hyrax_config, tmp_path, caplog):
    """Loading weights saved with a different run config warns, naming the saved config."""
    model.save(tmp_path / "weights.pth")
    saved = tmp_path / SELDON_CONFIG_FILENAME
    config = yaml.safe_load(saved.read_text())
    config["dataset"]["config"]["flux_stats"]["max"] *= 2
    saved.write_text(yaml.safe_dump(config))

    with caplog.at_level(logging.WARNING, logger="hyrax_seldon.models.seldon"):
        HyraxSeldon(config=hyrax_config).load(tmp_path / "weights.pth")

    assert str(saved) in caplog.text


def test_load_does_not_warn_on_same_run_config(model, hyrax_config, tmp_path, caplog):
    """A byte-for-byte copy of the run config in use is not a mismatch."""
    model.save(tmp_path / "weights.pth")

    with caplog.at_level(logging.WARNING, logger="hyrax_seldon.models.seldon"):
        HyraxSeldon(config=hyrax_config).load(tmp_path / "weights.pth")

    assert "differs" not in caplog.text


def test_train_batch_not_supported(model, batch):
    """Training is not supported yet."""
    with pytest.raises(NotImplementedError, match="training"):
        model.train_batch(batch)


def test_prepare_inputs_uses_run_context(hyrax_config):
    """prepare_inputs reads the config from Hyrax's run context, as verbs provide it."""
    from hyrax.context import run_context

    raw = collate_light_curves(_light_curves([40]))
    with run_context("infer", config=hyrax_config):
        prepared = HyraxSeldon.prepare_inputs({"data": raw, "object_id": np.array(["a"])})

    assert prepared["x_input"].shape == (1, 64, 4)


def test_compile_neutralized_unless_requested(hyrax_config):
    """SELDON's constructor compile() calls are undone unless compile = true."""
    model = HyraxSeldon(config=hyrax_config)
    assert all(module._compiled_call_impl is None for module in model.modules())

    hyrax_config["model"]["HyraxSeldon"]["compile"] = True
    compiled = HyraxSeldon(config=hyrax_config)
    assert any(module._compiled_call_impl is not None for module in compiled.modules())


@pytest.fixture
def grid_model(hyrax_config, seldon_checkpoint):
    """HyraxSeldon with a 3-point reconstruction grid at -10, 0 and +10 days."""
    hyrax_config["model"]["HyraxSeldon"].update(
        reconstruction_grid=True, grid_points=3, grid_horizon=[-10.0, 10.0]
    )
    model = HyraxSeldon(config=hyrax_config)
    model.load(seldon_checkpoint)
    return model.eval()


def test_reconstruction_grid_shapes(grid_model, batch):
    """Grid outputs are present with fixed shapes when the grid is on."""
    with torch.no_grad():
        out = grid_model.infer_batch(batch)

    assert out["grid_time"].shape == (4, 3)
    assert out["grid_reconstructed"].shape == (4, 6, 3)
    assert out["grid_error"].shape == (4, 6, 3)
    assert out["grid_time"][0].tolist() == [-10.0, 0.0, 10.0]
    assert torch.isfinite(out["grid_reconstructed"]).all()


def test_reconstruction_grid_values(grid_model, batch):
    """The grid is the decoder at the last observation's time plus each offset, per LSST band."""
    from hyrax_seldon.settings import LSST_BANDS

    with torch.no_grad():
        out = grid_model.infer_batch(batch)

    settings = grid_model.settings
    n_points = batch["pad_mask_full"].sum(dim=1)
    for row in range(4):
        last = int(n_points[row]) - 1
        # At offset 0, the grid in the band of the last observation is the reconstruction there.
        band = int(batch["band_idx_full"][row, last, 0])
        band_row = [settings.band_index_map[b] for b in LSST_BANDS].index(band)
        torch.testing.assert_close(
            out["grid_reconstructed"][row, band_row, 1], out["reconstructed"][row, last, 0]
        )
        torch.testing.assert_close(out["grid_error"][row, band_row, 1], out["error"][row, last, 0])

    # +10 days, in the r band, through decode() directly.
    t = batch["time_full"][:, -1:, :] + 10.0 / settings.t_max
    r_band = torch.full((4, 1, 1), settings.band_index_map["r"], device=t.device)
    with torch.no_grad():
        reconstructed, _ = grid_model.decode(out["decoder_inputs"], t, r_band)
    torch.testing.assert_close(out["grid_reconstructed"][:, LSST_BANDS.index("r"), 2], reconstructed[:, 0, 0])
