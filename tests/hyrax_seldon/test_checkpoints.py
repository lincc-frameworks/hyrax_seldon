import shutil

import pytest
import torch

from hyrax_seldon import _seldon
from hyrax_seldon.checkpoints import convert_run_dir, find_checkpoint, load_seldon_state_dict, main
from hyrax_seldon.models.seldon import HyraxSeldon
from hyrax_seldon.settings import SeldonSettings


def _copy_run_dir(seldon_run_dir, tmp_path):
    run_dir = tmp_path / "run"
    shutil.copytree(seldon_run_dir, run_dir)
    return run_dir


def test_checkpoint_has_seldon_lightning_layout(seldon_run_dir):
    """The synthetic run has SELDON's layout: hparams.yaml and a weights-only Lightning .ckpt."""
    checkpoint = torch.load(find_checkpoint(seldon_run_dir), map_location="cpu", weights_only=True)

    assert "hyper_parameters" not in checkpoint
    assert len(checkpoint["state_dict"]) == 295
    assert all(key.startswith("model.") for key in checkpoint["state_dict"])


def test_strips_prefix_and_loads_strictly_into_bare_seldon_model(seldon_config, seldon_run_dir):
    """Stripped weights match a bare SELDON model built from the run config, key for key."""
    settings = SeldonSettings.from_config(seldon_config)
    state_dict = load_seldon_state_dict(find_checkpoint(seldon_run_dir))

    model = _seldon.build_model(settings.model_params)
    model.load_state_dict(state_dict, strict=True)


def test_conversion_writes_weights_prepare_inputs_and_run_config(converted_weights, seldon_run_dir):
    """The converter writes the .pth, prepare_inputs.py, and a copy of hparams.yaml."""
    out_dir = converted_weights.parent

    assert converted_weights.name == find_checkpoint(seldon_run_dir).stem + ".pth"
    assert (out_dir / "prepare_inputs.py").exists()
    assert (out_dir / "seldon_config.yaml").read_bytes() == (seldon_run_dir / "hparams.yaml").read_bytes()


def test_converted_weights_equal_checkpoint_weights(converted_weights, seldon_run_dir):
    """The .pth holds the checkpoint's weights under HyraxSeldon's 'model.' attribute."""
    converted = torch.load(converted_weights, weights_only=True)
    original = torch.load(find_checkpoint(seldon_run_dir), map_location="cpu", weights_only=True)

    assert converted.keys() == original["state_dict"].keys()
    for key, value in original["state_dict"].items():
        assert torch.equal(converted[key], value), key


def test_converted_weights_load_with_hyrax(hyrax_config, converted_weights, seldon_run_dir):
    """Hyrax's load path (HyraxSeldon.load) restores the checkpoint's weights."""
    model = HyraxSeldon(config=hyrax_config)
    model.load(converted_weights)

    original = load_seldon_state_dict(find_checkpoint(seldon_run_dir))
    for key, value in model.model.state_dict().items():
        assert torch.equal(value.cpu(), original[key]), key


def test_mismatched_weights_rejected(tmp_path, seldon_run_dir):
    """A checkpoint that does not match the run config's model is an error."""
    run_dir = _copy_run_dir(seldon_run_dir, tmp_path)
    checkpoint_path = find_checkpoint(run_dir)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    checkpoint["state_dict"].pop(next(iter(checkpoint["state_dict"])))
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match="do not match"):
        convert_run_dir(run_dir, tmp_path / "out")


def test_weights_outside_model_rejected(tmp_path, seldon_run_dir):
    """Every checkpoint weight must belong to the SELDON model."""
    run_dir = _copy_run_dir(seldon_run_dir, tmp_path)
    checkpoint_path = find_checkpoint(run_dir)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    checkpoint["state_dict"]["extra.weight"] = torch.zeros(1)
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="extra.weight"):
        load_seldon_state_dict(checkpoint_path)


def test_several_checkpoints_need_a_choice(tmp_path, seldon_run_dir):
    """With several checkpoints, --checkpoint picks one."""
    run_dir = _copy_run_dir(seldon_run_dir, tmp_path)
    original = find_checkpoint(run_dir)
    shutil.copy(original, original.with_name("epoch=1-val_loss=1.00.ckpt"))

    with pytest.raises(ValueError, match="--checkpoint"):
        find_checkpoint(run_dir)
    assert find_checkpoint(run_dir, "epoch=1-val_loss=1.00").name == "epoch=1-val_loss=1.00.ckpt"
    with pytest.raises(FileNotFoundError, match="nope"):
        find_checkpoint(run_dir, "nope")


def test_not_a_run_dir(tmp_path):
    """A directory without hparams.yaml is reported as not a SELDON run."""
    with pytest.raises(FileNotFoundError, match="hparams.yaml"):
        convert_run_dir(tmp_path, tmp_path / "out")


def test_console_script(tmp_path, seldon_run_dir, capsys):
    """seldon-convert-checkpoint converts and prints the config to use."""
    main([str(seldon_run_dir), "--out-dir", str(tmp_path / "out")])

    output = capsys.readouterr().out
    assert "model_weights_file" in output
    assert len(list((tmp_path / "out").glob("*.pth"))) == 1
    assert (tmp_path / "out" / "seldon_config.yaml").exists()
