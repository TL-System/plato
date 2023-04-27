"""
A personalized federated learning trainer using FedPer.

"""

import os

from plato.trainers import basic_personalized
from plato.utils.filename_formatter import NameFormatter


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

    def personalized_train_model(self, config, trainset, sampler, **kwargs):
        """Ditto will only evaluate the personalized model."""
        batch_size = config["batch_size"]

        testset = kwargs["testset"]
        testset_sampler = kwargs["testset_sampler"]

        personalized_test_loader = self.get_personalized_data_loader(
            batch_size, testset, testset_sampler.get()
        )

        eval_outputs = self.perform_evaluation(
            personalized_test_loader, self.personalized_model
        )
        accuracy = eval_outputs["accuracy"]

        # save the personaliation accuracy to the results dir
        self.checkpoint_personalized_accuracy(
            accuracy=accuracy, current_round=self.current_round, epoch=0, run_id=None
        )

        if "max_concurrency" in config:

            # save the accuracy directly for latter usage
            # in the eval_test(...)
            model_name = config["personalized_model_name"]
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
            self.save_accuracy(accuracy, filename)
            return None
