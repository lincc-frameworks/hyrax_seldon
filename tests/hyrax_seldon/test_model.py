import logging

import numpy as np
import pytest
import torch
import yaml
from seldon_fixtures import collate_light_curves, random_light_curve

from hyrax_seldon.checkpoints import find_checkpoint, load_seldon_state_dict
from hyrax_seldon.encoding import encode_batch
from hyrax_seldon.models.seldon import SELDON_CONFIG_FILENAME, HyraxSeldon

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


@pytest.fixture
def model(hyrax_config, converted_weights):
    """HyraxSeldon with the converted synthetic checkpoint loaded, in eval mode, on the CPU.

    Hyrax's loader moves the model to the default accelerator; these tests compare against
    SELDON exactly, so they run on the CPU.
    """
    model = HyraxSeldon(config=hyrax_config)
    model.load(converted_weights)
    return model.to("cpu").eval()


def _light_curves(lengths, seed=0):
    rng = np.random.default_rng(seed)
    return [random_light_curve(rng, n) for n in lengths]


@pytest.fixture
def batch(model):
    """Prepared model inputs for light curves of several lengths, including a short one."""
    prepared = encode_batch(collate_light_curves(_light_curves([150, 600, 70, 12])), model.settings)
    return {key: torch.from_numpy(value) for key, value in prepared.items()}


@pytest.fixture
def seldon_model(model, seldon_run_dir):
    """The same checkpoint loaded into a SELDON model built directly with SELDON's factory."""
    import copy

    from TAE.models.factory import get_module

    params = copy.deepcopy(model.settings.model_params)
    seldon_model = get_module(params["name"], params["config"])
    for module in seldon_model.modules():
        module._compiled_call_impl = None
    seldon_model.load_state_dict(load_seldon_state_dict(find_checkpoint(seldon_run_dir)), strict=True)
    return seldon_model.eval()


def _call_seldon(seldon_model, batch):
    return seldon_model(
        x=batch["x_input"],
        t=batch["time_full"],
        band_idx_part=batch["band_idx"],
        band_idx_full=batch["band_idx_full"],
        mask=batch["pad_mask_part"],
    )


def test_infer_batch_matches_seldon(model, seldon_model, batch):
    """Every deterministic output equals SELDON's model called directly on the same batch."""
    with torch.no_grad():
        ours = model.infer_batch(batch)
        reference = _call_seldon(seldon_model, batch)

    for key in DETERMINISTIC_KEYS:
        assert torch.equal(ours[key], reference[key]), key
    for key in PARAMETER_KEYS:
        assert torch.equal(ours[f"parameters_{key}"], reference["parameters"][key]), key


def test_stochastic_outputs_match_seldon_under_the_same_seed(model, seldon_model, batch):
    """z_sample, class_idx and class_emb are passed through as SELDON draws them."""
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
    """Light curves whose encoder input has fewer than min_points detections are flagged."""
    with torch.no_grad():
        out = model.infer_batch(batch)

    detections = ((batch["x_input"][..., 3] > 0) & batch["pad_mask_part"]).sum(dim=1)
    assert out["short_lightcurve"].tolist() == (detections < 32).tolist()
    assert out["short_lightcurve"].tolist() == [False, False, False, True]


def test_pad_mask_full_passed_through(model, batch):
    """pad_mask_full marks which per-point outputs are real points."""
    with torch.no_grad():
        out = model.infer_batch(batch)

    assert torch.equal(out["pad_mask_full"], batch["pad_mask_full"])
    assert out["pad_mask_full"].sum(dim=1).tolist() == [150, 512, 70, 12]


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
    reloaded.to("cpu").eval()

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

    raw = collate_light_curves(_light_curves([20]))
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
def grid_model(hyrax_config, converted_weights):
    """HyraxSeldon with a 3-point reconstruction grid at -10, 0 and +10 days."""
    hyrax_config["model"]["HyraxSeldon"].update(
        reconstruction_grid=True, grid_points=3, grid_horizon=[-10.0, 10.0]
    )
    model = HyraxSeldon(config=hyrax_config)
    model.load(converted_weights)
    return model.to("cpu").eval()


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
        band_row = [settings.band_to_index(b) for b in LSST_BANDS].index(band)
        torch.testing.assert_close(
            out["grid_reconstructed"][row, band_row, 1], out["reconstructed"][row, last, 0]
        )
        torch.testing.assert_close(out["grid_error"][row, band_row, 1], out["error"][row, last, 0])

    # +10 days, in the r band, through decode() directly.
    t = batch["time_full"][:, -1:, :] + 10.0 / settings.t_max
    r_band = torch.full((4, 1, 1), settings.band_to_index("r"))
    with torch.no_grad():
        reconstructed, _ = grid_model.decode(out["decoder_inputs"], t, r_band)
    torch.testing.assert_close(out["grid_reconstructed"][:, LSST_BANDS.index("r"), 2], reconstructed[:, 0, 0])
