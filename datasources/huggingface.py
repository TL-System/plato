"""
The HuggingFace datasets.

For more information about the HuggingFace datasets, refer to
https://huggingface.co/docs/datasets/quicktour.html.
"""

import logging
from datasets import load_dataset

from config import Config
from datasources import base


class DataSource(base.DataSource):
    """A data source for HuggingFace datasets."""
    def __init__(self):
        super().__init__()

        dataset_name = Config().data.dataset_name
        logging.info("Dataset: %s", dataset_name)

        self.train_set = load_dataset(dataset_name, split='train')
        self.test_set = load_dataset(dataset_name, split='test')

    def num_train_examples(self):
        return len(self.train_set)

    def num_test_examples(self):
        return len(self.test_set)

    def get_train_set(self):
        return self.train_set

    def get_test_set(self):
        return self.test_set
