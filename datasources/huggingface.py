"""
The HuggingFace datasets.

For more information about the HuggingFace datasets, refer to
https://huggingface.co/docs/datasets/quicktour.html.
"""

from datasets import load_dataset

from config import Config
from datasources import base


class DataSource(base.DataSource):
    """A data source for HuggingFace datasets."""
    def __init__(self, path):
        super().__init__(path)

        self.train_set = None
        self.test_set = None

    @staticmethod
    def num_train_examples():
        return Config().data.num_train_examples

    @staticmethod
    def num_test_examples():
        return Config().data.num_test_examples

    @staticmethod
    def num_classes():
        return Config().data.num_classes

    def classes(self):
        """Obtains a list of class names in the dataset."""
        return Config().data.classes

    def get_train_set(self):
        dataset_name = Config().data.dataset_name
        self.train_set = load_dataset(dataset_name, split='train')
        return self.train_set

    def get_test_set(self):
        dataset_name = Config().data.dataset_name
        self.test_set = load_dataset(dataset_name, split='test')
        return self.test_set
