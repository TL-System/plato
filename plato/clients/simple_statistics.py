"""
A basic federated learning client capable of collecting 
model-related statistics.
"""

import os
import json
import logging
from collections import Counter

import torch

from plato.clients import simple
from plato.utils.filename_formatter import get_format_name

from plato.config import Config


class Client(simple.Client):
    """A basic federated learning client for model statistics collection."""

    def get_data_statistics(self, dataset, dataset_sampler):
        """Get the data statistics."""
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False, sampler=dataset_sampler.get()
        )
        data_labels = []
        for _, label in data_loader:
            data_labels.extend(label.tolist())
        num_samples = len(data_labels)
        labels_sample_count = Counter(data_labels)
        return labels_sample_count, num_samples

    def save_data_statistics(self):
        """Save the data statistics."""
        result_path = Config().params["result_path"]

        save_location = os.path.join(result_path, "client_" + str(self.client_id))
        os.makedirs(save_location, exist_ok=True)
        filename = get_format_name(
            client_id=self.client_id, suffix="data_statistics", ext="json"
        )
        save_file_path = os.path.join(save_location, filename)

        if not os.path.exists(save_file_path):
            train_data_sta, train_count = self.get_data_statistics(
                self.trainset, self.sampler
            )
            test_data_sta, test_count = self.get_data_statistics(
                self.testset, self.testset_sampler
            )

            with open(save_file_path, "w", encoding="utf8") as file_pointer:
                json.dump(
                    {
                        "train": train_data_sta,
                        "test": test_data_sta,
                        "train_size": train_count,
                        "test_size": test_count,
                    },
                    file_pointer,
                )
            logging.info("Saved local data statistics of client[%d] ", self.client_id)

    def save_model_statistics(self, model_attr_name):
        """Get the model statistics."""
        assert model_attr_name in vars(self).keys()

        model_path = Config().params["model_path"]

        to_save_dir = os.path.join(model_path, "client_" + str(self.client_id))
        os.makedirs(to_save_dir, exist_ok=True)

        # the actual models are held the trainer of the client
        model_obj = getattr(self.trainer, model_attr_name)
        # logging the model's info
        file_name_detailed = get_format_name(
            client_id=self.client_id,
            model_name=model_attr_name,
            ext="log",
        )
        file_name_brief = get_format_name(
            client_id=self.client_id,
            model_name=model_attr_name,
            suffix="brief",
            ext="log",
        )
        save_path_detailed = os.path.join(to_save_dir, file_name_detailed)
        save_path_brief = os.path.join(to_save_dir, file_name_brief)

        if not os.path.exists(save_path_detailed):
            with open(save_path_detailed, "w", encoding="utf8") as file:
                file.write(str(model_obj))
        if not os.path.exists(save_path_brief):
            global_parameter_names = list(model_obj.state_dict().keys())
            with open(save_path_brief, "w", encoding="utf8") as file:
                for item in global_parameter_names:
                    file.write(f"{item}\n")

    def configure(self) -> None:
        """Performing the general client's configure and then initialize the
        personalized model for the client."""
        super().configure()
        # save statistics of the defined model
        if (
            hasattr(Config().clients, "logging_model_statistics")
            and Config().clients.logging_model_statistics
        ):
            self.save_model_statistics(model_attr_name="model")

    def _allocate_data(self) -> None:
        """Allocate training or testing dataset of this client."""
        super()._allocate_data()
        if (
            hasattr(Config().clients, "logging_data_statistics")
            and Config().clients.logging_data_statistics
        ):
            self.save_data_statistics()
