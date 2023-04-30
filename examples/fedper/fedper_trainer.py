"""
A personalized federated learning trainer using FedPer.

"""

import os
import logging

from plato.trainers import basic_personalized
from plato.utils.filename_formatter import NameFormatter
from plato.algorithms import fedavg_partial


class Trainer(basic_personalized.Trainer):
    """A personalized federated learning trainer using the FedBABU algorithm."""

    def train_run_end(self, config):
        """Save the trained model to be the personalized model."""
        # copy the trained model to the personalized model
        self.personalized_model.load_state_dict(self.model.state_dict(), strict=True)

        current_round = self.current_round

        personalized_model_name = config["personalized_model_name"]
        save_location = self.get_checkpoint_dir_path()
        filename = NameFormatter.get_format_name(
            client_id=self.client_id,
            model_name=personalized_model_name,
            round_n=current_round,
            run_id=None,
            prefix="personalized",
            ext="pth",
        )
        os.makedirs(save_location, exist_ok=True)
        self.save_personalized_model(filename=filename, location=save_location)

    # pylint: disable=unused-argument
    def test_model(self, config, testset, sampler=None, **kwargs):
        """
        Evaluates the model with the provided test dataset and test sampler.

        Auguments:
        testset: the test dataset.
        sampler: the test sampler. The default is None.
        kwargs (optional): Additional keyword arguments.
        """
        accuracy = super().test_model(config, testset, sampler=None, **kwargs)

        # save the personaliation accuracy to the results dir
        self.checkpoint_personalized_accuracy(
            accuracy=accuracy,
            current_round=self.current_round,
            epoch=config["epochs"],
            run_id=None,
        )

        return accuracy

    def freeze_model(self, model, modules_name=None):
        """Freeze a part of the model."""
        if modules_name is not None:
            frozen_params = []
            for name, param in model.named_parameters():
                if any([param_name in name for param_name in modules_name]):
                    param.requires_grad = False
                    frozen_params.append(name)

            logging.info(
                "[Client #%d] has frozen %s",
                self.client_id,
                fedavg_partial.Algorithm.extract_modules_name(frozen_params),
            )

    def activate_model(self, model, modules_name=None):
        """Unfrozen a part of the model."""
        if modules_name is not None:
            unfrozen_params = []
            for name, param in model.named_parameters():
                if any([param_name in name for param_name in modules_name]):
                    param.requires_grad = True
                    unfrozen_params.append(name)
            logging.info(
                "[Client #%d] has unfrozen %s",
                self.client_id,
                fedavg_partial.Algorithm.extract_modules_name(unfrozen_params),
            )

    def personalized_train_run_start(self, config, **kwargs):
        """According to FedPer, freeze a partial of the model and
        never update it during personalization."""
        eval_outputs = super().personalized_train_run_start(config, **kwargs)
        logging.info(
            "[Client #%d] will freeze %s before performing personalization",
            self.client_id,
            config["frozen_personalized_modules_name"],
        )
        self.freeze_model(
            self.personalized_model, config["frozen_personalized_modules_name"]
        )
        return eval_outputs

    def personalized_train_run_end(self, config):
        """Reactive the personalized model."""
        self.activate_model(
            self.personalized_model, config["frozen_personalized_modules_name"]
        )
