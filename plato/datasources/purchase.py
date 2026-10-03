"""
The Purchase100 dataset.
"""

import logging
import os
from pathlib import Path
from urllib import request

import numpy as np
import torch
from torch.utils import data

from plato.config import Config
from plato.datasources import base
from plato.utils.archive import UnsafeArchiveError, extract_archive


class DataSource(base.DataSource):
    """The Purchase100 dataset."""

    def __init__(self, **kwargs):
        super().__init__()
        root_path = Config().params["data_path"]
        dataset_path = os.path.join(root_path, "dataset_purchase")
        if not os.path.isfile(os.path.join(root_path, "purchase_numpy.npz")):
            self.download_dataset(root_path, dataset_path)

        self.trainset, self.testset = self.extract_data(root_path)

    def download_dataset(self, root_path, dataset_path):
        """Download the Purchase100 dataset."""
        with self._download_guard(root_path):
            cache_path = os.path.join(root_path, "purchase_numpy.npz")
            archive_path = os.path.join(root_path, "tmp_purchase.tgz")
            for artifact in (cache_path, archive_path):
                if Path(artifact).is_symlink():
                    raise UnsafeArchiveError(
                        f"Unsafe dataset artifact symlink: {artifact}"
                    )
            if os.path.isfile(cache_path):
                return
            if not os.path.isfile(dataset_path):
                logging.info("Downloading the Purchase100 dataset...")
                filename = (
                    "https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz"
                )
                request.urlretrieve(filename, archive_path)
                extract_archive(archive_path, root_path)

            logging.info("Processing the dataset...")
            data_set = np.genfromtxt(dataset_path, delimiter=",", ndmin=2)
            X = data_set[:, 1:].astype(np.float64)
            Y = (data_set[:, 0]).astype(np.int32) - 1
            np.savez(cache_path, X=X, Y=Y)

    def extract_data(self, root_path):
        """Extract data."""
        with np.load(os.path.join(root_path, "purchase_numpy.npz")) as dataset:
            X, Y = dataset["X"], dataset["Y"]
        ## randomly shuffle the data without changing the caller's RNG
        indices = np.random.RandomState(0).permutation(len(X))
        X, Y = X[indices], Y[indices]

        ## extract 20000 data samplers for training and testing respectively
        num_train = 20000
        train_data = X[:num_train]
        test_data = X[num_train : num_train * 2]
        train_label = Y[:num_train]
        test_label = Y[num_train : num_train * 2]

        ## create datasets
        train_dataset = VectorDataset(train_data, train_label)
        test_dataset = VectorDataset(test_data, test_label)

        return train_dataset, test_dataset

class VectorDataset(data.Dataset):
    """
    Create a Purchase100 dataset based on features and labels
    """

    def __init__(self, features, labels):
        self.data = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(labels, dtype=torch.long)
        self.classes = [f"Style #{i}" for i in range(100)]

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.size(0)
