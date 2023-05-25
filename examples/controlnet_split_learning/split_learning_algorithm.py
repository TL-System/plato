"""Train the ControlNet model with split learning"""
import time
import logging
import torch
from plato.config import Config
from split_learning import split_learning_algorithm


class Algorithm(split_learning_algorithm.Algorithm):
    """The split learning algorithm to train ControlNet."""

    def extract_features(self, dataset, sampler):
        """Extracting features using layers before the cut_layer."""
        self.model.to(self.trainer.device)
        self.model.eval()

        tic = time.perf_counter()

        features_dataset = []

        batch = self.trainer.get_train_samples(
            Config().trainer.batch_size, dataset, sampler
        )
        with torch.no_grad():
            batch = batch.to(self.trainer.device)
            output_dict = self.model.training_step(batch)

        output_dict["control_output"] = output_dict["control_output"].cpu()
        output_dict["sd_output"] = output_dict["sd_output"].cpu()
        noise = output_dict["noise"].cpu()
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
