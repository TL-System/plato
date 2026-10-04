"""Public SSL worker state and frozen-feature regressions, core dependencies."""

import copy
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from plato.config import Config
from plato.trainers.self_supervised_learning import Trainer
from plato.utils.checkpoint_paths import checkpoint_name
from tests.integration.utils import build_minimal_config, configure_environment
from tests.trainers.test_ssl_training_contract import EncoderModel


def personal_config(*, spawn=False):
    config = build_minimal_config(
        trainer_type="self_supervised_learning", model_name="org/model"
    )
    config["trainer"].update(batch_size=2, loss_criterion="MSELoss")
    if spawn:
        config["trainer"]["max_concurrency"] = 1
    config["algorithm"]["personalization"] = {
        "model_name": "linear_mlp",
        "model_type": "general_multilayer",
        "batch_size": 2,
        "epochs": 1,
        "optimizer": "SGD",
        "lr_scheduler": "ConstantLR",
        "loss_criterion": "CrossEntropyLoss",
    }
    config["parameters"]["personalization"] = {
        "model": {"num_classes": 2},
        "optimizer": {"lr": 0.1},
        "learning_rate": {"factor": 1.0, "total_iters": 1},
    }
    return config


@pytest.mark.slow
def test_ssl_public_spawn_returns_private_head_and_isolates_identity(tmp_path):
    config = personal_config(spawn=True)
    with configure_environment(config, runtime_root=tmp_path):
        torch.manual_seed(15)
        trainer = Trainer(model=EncoderModel())
        template = copy.deepcopy(trainer.local_layers.state_dict())
        trainer.set_client_id(7)
        trainer.device = trainer.context.device = torch.device("cpu")
        trainer.current_round = 2
        encoder = copy.deepcopy(trainer.model.encoder.state_dict())
        reference = copy.deepcopy(trainer.local_layers)
        data = TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.int64))
        with torch.no_grad():
            features = trainer.model.encoder(torch.ones(2, 2))
        for _ in range(2):
            optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
            for _ in range(2):
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(
                    reference(features), torch.zeros(2, dtype=torch.int64)
                )
                loss.backward()
                optimizer.step()
            trainer.train(data, [0, 1, 2, 3])
            for name, value in trainer.local_layers.state_dict().items():
                torch.testing.assert_close(value, reference.state_dict()[name])
            for name, value in trainer.model.encoder.state_dict().items():
                torch.testing.assert_close(value, encoder[name])
        trained = copy.deepcopy(trainer.local_layers.state_dict())
        root = Path(Config.params["model_path"])
        assert (
            root
            / checkpoint_name("org/model", 7, "ssl_local_layers", suffix=".safetensors")
        ).is_file()
        trainer.set_client_id(8)
        for name, value in trainer.local_layers.state_dict().items():
            torch.testing.assert_close(value, template[name])
        trainer.set_client_id(7)
        for name, value in trainer.local_layers.state_dict().items():
            torch.testing.assert_close(value, trained[name])
        reconstructed = Trainer(model=EncoderModel())
        reconstructed.set_client_id(7)
        for name, value in reconstructed.local_layers.state_dict().items():
            torch.testing.assert_close(value, trained[name])


@pytest.mark.parametrize("interrupted", [False, True])
def test_ssl_personalization_freezes_encoder_buffers_gradients_and_preserves_modes(
    tmp_path, interrupted
):
    config = personal_config()
    with configure_environment(config, runtime_root=tmp_path):
        model = EncoderModel()
        model.encoder = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.BatchNorm1d(2),
            torch.nn.Dropout(0.5),
        )
        model.encoder.encoding_dim = 2
        trainer = Trainer(model=model)
        trainer.set_client_id(7)
        trainer.current_round = 2
        trainer.device = trainer.context.device = torch.device("cpu")
        before = copy.deepcopy(model.encoder.state_dict())
        requires_grad = [
            parameter.requires_grad for parameter in model.encoder.parameters()
        ]
        # Old encoder gradients must not survive a head-only run.
        for parameter in model.encoder.parameters():
            parameter.grad = torch.ones_like(parameter)
        data = TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.int64))
        if interrupted:

            def fail(*_):
                raise RuntimeError("interrupted SSL head update")

            trainer.loss_strategy.compute_loss = fail
            with pytest.raises(RuntimeError, match="interrupted SSL"):
                trainer.train_model(
                    {**config["trainer"], "run_id": "head"}, data, [0, 1, 2, 3]
                )
        else:
            trainer.train_model(
                {**config["trainer"], "run_id": "head"}, data, [0, 1, 2, 3]
            )
        for name, value in model.encoder.state_dict().items():
            torch.testing.assert_close(value, before[name])
        assert all(parameter.grad is None for parameter in model.encoder.parameters())
        assert [
            parameter.requires_grad for parameter in model.encoder.parameters()
        ] == requires_grad
        assert all(module.training for module in model.encoder.modules())
