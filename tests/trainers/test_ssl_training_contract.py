"""SSL personalization paths that require only core Torch dependencies."""

import copy

import pytest
import torch
from torch.utils.data import TensorDataset

from plato.trainers.self_supervised_learning import SSLTestingStrategy, Trainer
from plato.trainers.strategies.base import TrainingContext
from tests.integration.utils import build_minimal_config, configure_environment


class EncoderModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2)
        self.encoder.encoding_dim = 2

    def forward(self, samples):
        return self.encoder(samples)


def test_ssl_model_instance_and_personalization_batch_config(tmp_path):
    config = build_minimal_config(trainer_type="self_supervised_learning")
    config["trainer"].update(batch_size=8, loss_criterion="MSELoss")
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
    with configure_environment(config, runtime_root=tmp_path):
        model = EncoderModel()
        trainer = Trainer(model=model)
        assert trainer.model is model
        trainer.device = trainer.context.device = torch.device("cpu")
        trainer.current_round = 2
        data = TensorDataset(torch.eye(2).repeat(2, 1), torch.tensor([0, 1, 0, 1]))
        before = copy.deepcopy(trainer.local_layers.state_dict())
        before_encoder = copy.deepcopy(model.encoder.state_dict())
        trainer.train_model({**config["trainer"], "run_id": "ssl"}, data, [0, 1, 2, 3])
        assert trainer.train_loader.batch_size == 2
        assert any(
            not torch.equal(value, before[key])
            for key, value in trainer.local_layers.state_dict().items()
        )
        for key, value in model.encoder.state_dict().items():
            torch.testing.assert_close(value, before_encoder[key])


def test_ssl_knn_reference_set_is_not_indexed_with_test_sampler(tmp_path):
    config = build_minimal_config()
    with configure_environment(config, runtime_root=tmp_path):
        reference = TensorDataset(
            torch.tensor([[0.0, 0.0], [10.0, 10.0]]), torch.tensor([0, 1])
        )
        testset = TensorDataset(
            torch.tensor([[10.0, 10.0]]).repeat(4, 1), torch.ones(4, dtype=torch.long)
        )
        model = EncoderModel()
        model.encoder = torch.nn.Identity()
        context = TrainingContext()
        context.device = torch.device("cpu")
        strategy = SSLTestingStrategy(personalized_trainset=reference)
        assert (
            strategy.test_model(model, {"batch_size": 2}, testset, [3], context) == 1.0
        )


def test_ssl_empty_evaluation_is_consistent_with_basic(tmp_path):
    config = build_minimal_config()
    with configure_environment(config, runtime_root=tmp_path):
        context = TrainingContext()
        context.device = torch.device("cpu")
        model = EncoderModel()
        empty = TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long))
        data = TensorDataset(torch.eye(2), torch.tensor([0, 1]))
        strategy = SSLTestingStrategy(
            local_layers=torch.nn.Linear(2, 2), personalized_trainset=data
        )
        context.current_round = 2
        assert (
            strategy.test_model(model, {"batch_size": 2}, empty, None, context) == 0.0
        )
        context.current_round = 1
        strategy.personalized_trainset = empty
        with pytest.raises(ValueError, match="nonempty reference"):
            strategy.test_model(model, {"batch_size": 2}, data, None, context)
        assert (
            strategy.test_model(model, {"batch_size": 2}, empty, None, context) == 0.0
        )
