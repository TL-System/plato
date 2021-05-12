"""
The HuggingFace datasets.

For more information about the HuggingFace datasets, refer to
https://huggingface.co/docs/datasets/quicktour.html.
"""

import logging

from datasets import load_dataset

from plato.config import Config
from plato.datasources import base


class DataSource(base.DataSource):
    """A data source for HuggingFace datasets."""
    def __init__(self):
        super().__init__()

        dataset_name = Config().data.dataset_name
        logging.info("Dataset: %s", dataset_name)

        if hasattr(Config.data, 'dataset_config'):
            dataset_config = Config().data.dataset_config
        else:
            dataset_config = None

        self.dataset = load_dataset(dataset_name, dataset_config)
        self.trainset = self.dataset['train']
        self.testset = self.dataset['validation']

    def num_train_examples(self):
        return len(self.trainset)

    def num_test_examples(self):
        return len(self.testset)

    def get_train_set(self):
        return self.trainset

    def get_test_set(self):
        return self.testset
