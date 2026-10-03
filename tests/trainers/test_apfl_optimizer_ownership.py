"""Optimizer ownership must preserve a returning client's private momentum."""

import copy

import torch
from torch.utils.data import TensorDataset

from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms.apfl_strategy import (
    APFLStepStrategy,
    APFLUpdateStrategy,
)
from tests.integration.utils import build_minimal_config, configure_environment
from tests.trainers.test_personalized_state_ownership import model


def test_apfl_returning_client_matches_dedicated_optimizer_reference(tmp_path):
    config = build_minimal_config(model_name="org/model")
    config["parameters"]["optimizer"].update(lr=0.2, momentum=0.9)
    with configure_environment(config, runtime_root=tmp_path):

        def create(label):
            update = APFLUpdateStrategy(
                adaptive_alpha=False, model_fn=model, save_path=str(tmp_path / label)
            )
            trainer = ComposableTrainer(
                model=model,
                model_update_strategy=update,
                training_step_strategy=APFLStepStrategy(),
            )
            trainer.set_client_id(1)
            return trainer, update

        reused, update = create("reused")
        dedicated, reference_update = create("dedicated")
        data = TensorDataset(torch.eye(2), torch.tensor([0, 1]))
        run = {**config["trainer"], "run_id": "owner"}
        for round_id in (1, 2):
            for trainer in (reused, dedicated):
                trainer.current_round = round_id
                trainer.model.weight.data.fill_(float(round_id))
                trainer.model.bias.data.zero_()
                trainer.train_model(run.copy(), data, [0, 1])
            if round_id == 1:
                first = copy.deepcopy(update.personalized_model.state_dict())
                reused.set_client_id(2)
                assert "apfl_personalized_optimizer" not in reused.context.state
                reused.train_model(run.copy(), data, [0, 1])
                reused.set_client_id(1)
                update.on_train_start(reused.context)
                for name, value in update.personalized_model.state_dict().items():
                    torch.testing.assert_close(value, first[name])
        for name, value in update.personalized_model.state_dict().items():
            torch.testing.assert_close(
                value, reference_update.personalized_model.state_dict()[name]
            )
        actual_optimizer = reused.context.state["apfl_personalized_optimizer"]
        expected_optimizer = dedicated.context.state["apfl_personalized_optimizer"]
        for actual, expected in zip(
            actual_optimizer.state.values(), expected_optimizer.state.values()
        ):
            torch.testing.assert_close(
                actual["momentum_buffer"], expected["momentum_buffer"]
            )
