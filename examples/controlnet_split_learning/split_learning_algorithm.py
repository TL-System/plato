"""Train the ControlNet model with split learning"""
# pylint:disable=import-error
import time
import logging
import torch

from split_learning import split_learning_algorithm
from plato.config import Config


class Algorithm(split_learning_algorithm.Algorithm):
    """The split learning algorithm to train ControlNet."""

    def extract_features(self, dataset, sampler):
        """Extracting features using layers before the cut_layer."""

        tic = time.perf_counter()

        self.model.to(self.trainer.device)
        self.model.model.to(self.trainer.device)
        self.model.model.eval()

        features_dataset = []

        batch, _ = self.trainer.get_train_samples(
            Config().trainer.batch_size, dataset, sampler
        )
        with torch.no_grad():
            output_dict = self.model.training_step(batch)

        output_dict["control_output"] = output_dict["control_output"].detach().cpu()
        for index, items in enumerate(output_dict["sd_output"]):
            output_dict["sd_output"][index] = items.detach().cpu()
        noise = output_dict["noise"].detach().cpu()
        output_dict["timestep"] = output_dict["timestep"].detach().cpu()
        if not (
            hasattr(Config().parameters.model, "safe")
            and Config().parameters.model.safe
        ):
            output_dict["cond_txt"] = output_dict["cond_txt"].detach().cpu()
        output_dict.pop("noise")
        features_dataset.append((output_dict, noise))

        toc = time.perf_counter()
        logging.warning(
            "[Client #%d] Features extracted from %s examples in %.2f seconds.",
            self.client_id,
            Config().trainer.batch_size,
            toc - tic,
        )

        return features_dataset, toc - tic

    def update_weights_before_cut(self, weights):
        """Update the weights before cut layer, called when testing accuracy."""
        current_weights = self.extract_weights()
        # update the weights of client model
        for key, _ in weights.items():
            if "input_hint_block" in key or "input_blocks.0" in key:
                current_weights[key] = weights[key]

        self.load_weights(current_weights)
