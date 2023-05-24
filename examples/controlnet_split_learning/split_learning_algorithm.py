"""Train the ControlNet model with split learning"""
import time
import logging
import torch
from plato.config import Config
from examples.split_learning import split_learning_algorithm


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
