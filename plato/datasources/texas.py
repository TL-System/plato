"""
The Texas100 dataset.
"""

import logging
import os
import tarfile
import urllib

import numpy as np
import torch
from torch.utils import data

from plato.config import Config
from plato.datasources import base


class DataSource(base.DataSource):
    """The Texas100 dataset."""

    def __init__(self):
        super().__init__()
        root_path = Config().params["data_path"]
        feat_path = os.path.join(root_path, "texas/100/feats")
        label_path = os.path.join(root_path, "texas/100/labels")
        if not os.path.isdir(root_path):
            os.mkdir(root_path)
        if not os.path.isfile(feat_path):
            self.download_dataset(root_path, feat_path, label_path)

        self.trainset, self.testset = self.extract_data(root_path)

    def download_dataset(self, root_path, feat_path, label_path):
        """Download the Texas100 dataset."""
        logging.info("Downloading the Texas100 dataset...")
        filename = "https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz"
        urllib.request.urlretrieve(filename, os.path.join(root_path, "tmp_texas.tgz"))
        logging.info("Dataset downloaded.")
        tar = tarfile.open(os.path.join(root_path, "tmp_texas.tgz"))
        tar.extractall(path=root_path)

        logging.info("Processing the dataset...")
        data_set_feats = np.genfromtxt(feat_path, delimiter=",")
        data_set_labels = np.genfromtxt(label_path, delimiter=",")
        logging.info("Finish processing the dataset.")

        X = data_set_feats.astype(np.float64)
        Y = data_set_labels.astype(np.int32) - 1
        np.savez(os.path.join(root_path, "texas_numpy.npz"), X=X, Y=Y)

    def extract_data(self, root_path):
        """Extract data."""
        data = np.load(os.path.join(root_path, "texas_numpy.npz"))

        ## randomly shuffle the data
        X, Y = data["X"], data["Y"]
        np.random.seed(0)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
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

    def num_train_examples(self):
        return 20000

    def num_test_examples(self):
        return 20000


class VectorDataset(data.Dataset):
    """
    Create a Texas100 dataset based on features and labels
    """

    def __init__(self, features, labels):
        self.data = torch.stack([torch.FloatTensor(i) for i in features])
        self.targets = torch.stack([torch.LongTensor([i]) for i in labels])[:, 0]
        self.classes = [f"Procedure #{i}" for i in range(100)]

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.size(0)
