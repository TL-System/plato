"""
The Purchase100 dataset.
"""
import os
import logging
import urllib
import tarfile
import torch
import numpy as np
from plato.config import Config
from plato.datasources import base


class DataSource(base.DataSource):
    """The Purchase100 dataset."""

    def __init__(self):
        super().__init__()
        root_path = Config().params['data_path']
        dataset_path = os.path.join(root_path, 'dataset_purchase')
        if not os.path.isdir(root_path):
            os.mkdir(root_path)
        if not os.path.isfile(dataset_path):
            self.download_dataset(root_path, dataset_path)

        self.trainset, self.testset = self.extract_data(root_path)

    def download_dataset(self, root_path, dataset_path):
        """Download the Purchase100 dataset."""
        logging.info('Downloading the Purchase100 dataset...')
        filename = "https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz"
        urllib.request.urlretrieve(filename,
                                   os.path.join(root_path, 'tmp_purchase.tgz'))
        logging.info('Dataset downloaded')
        tar = tarfile.open(os.path.join(root_path, 'tmp_purchase.tgz'))
        tar.extractall(path=root_path)

        logging.info('Reading dataset...')
        data_set = np.genfromtxt(dataset_path, delimiter=',')
        logging.info('Finish reading!')

        X = data_set[:, 1:].astype(np.float64)
        Y = (data_set[:, 0]).astype(np.int32) - 1
        np.savez(os.path.join(root_path, 'purchase_numpy.npz'), X=X, Y=Y)

    def extract_data(self, root_path):
        """Extract data."""
        data = np.load(os.path.join(root_path, 'purchase_numpy.npz'))

        ## randomly shuffle the data
        X, Y = data['X'], data['Y']
        np.random.seed(0)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X, Y = X[indices], Y[indices]

        ## extract 20000 data samplers for training and testing respectively
        num_train = 20000
        train_data = X[:num_train]
        test_data = X[num_train:num_train * 2]
        train_label = Y[:num_train]
        test_label = Y[num_train:num_train * 2]

        ## create datasets
        train_dataset = create_tensor_dataset(train_data, train_label)
        test_dataset = create_tensor_dataset(test_data, test_label)

        return train_dataset, test_dataset

    def num_train_examples(self):
        return 20000

    def num_test_examples(self):
        return 20000


def create_tensor_dataset(features, labels):
    """
        Create a tensor dataset based on features and labels
    """
    tensor_x = torch.stack([torch.FloatTensor(i) for i in features])
    tensor_y = torch.stack([torch.LongTensor([i]) for i in labels])[:, 0]
    dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    return dataset
