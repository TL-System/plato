"""Tests for the composable trainer checkpoint facilities."""

from pathlib import Path

import torch

from plato.config import Config
from plato.trainers.composable import ComposableTrainer


def _clone_state_dict(model: torch.nn.Module):
    """Clone a model state dict."""
    return {name: param.clone() for name, param in model.state_dict().items()}


def _build_trainer():
    """Create a composable trainer with a lightweight model."""
    return ComposableTrainer(model=lambda: torch.nn.Linear(4, 2))


def test_save_and_load_roundtrip(temp_config, tmp_path):
    """Saving and loading should round-trip model weights."""
    trainer = _build_trainer()

    # Redirect model path for this test and pick a deterministic file name.
    Config.params["model_path"] = str(tmp_path / "models")
    Path(Config.params["model_path"]).mkdir(parents=True, exist_ok=True)
    Config().trainer = Config().trainer._replace(model_name="checkpoint_roundtrip")

    original_state = _clone_state_dict(trainer.model)

    trainer.save_model()
    checkpoint_path = Path(Config.params["model_path"]) / "checkpoint_roundtrip.pth"
    history_path = checkpoint_path.with_suffix(".pth.pkl")

    assert checkpoint_path.exists()
    assert history_path.exists()

    # Mutate the model to confirm load restores the saved weights.
    for param in trainer.model.parameters():
        param.data.add_(1.0)

    trainer.load_model()

    for name, param in trainer.model.state_dict().items():
        assert torch.allclose(param, original_state[name])


def test_save_model_with_custom_filename(temp_config, tmp_path):
    """Explicit filenames should be honoured when saving checkpoints."""
    trainer = _build_trainer()

    Config.params["model_path"] = str(tmp_path / "models")
    Path(Config.params["model_path"]).mkdir(parents=True, exist_ok=True)

    trainer.save_model(filename="custom_model.pth")

    custom_path = Path(Config.params["model_path"]) / "custom_model.pth"
    history_path = custom_path.with_suffix(".pth.pkl")

    assert custom_path.exists()
    assert history_path.exists()
