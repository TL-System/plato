"""
The FEMNIST Classification dataset.

FEMNIST contains 817851 images, each of which is a 28x28 greyscale image in 1 out of 62 classes.
The dataset is already partitioned by clients' identification.
There are in total 3597 clients, each of which has 227.37 images on average (std is 88.84).
For each client, 90% data samples are used for training, while the rest are used for testing.

Reference for the dataset: Cohen, G., Afshar, S., Tapson, J. and Van Schaik, A.,
EMNIST: Extending MNIST to handwritten letters. In 2017 IEEE IJCNN.
Reference for the related submodule: https://github.com/TalwalkarLab/leaf/tree/master
"""

from __future__ import division
import logging
import os

from torchvision import transforms

from plato.config import Config
from plato.datasources import base

import json
import numpy as np
from torch.utils.data import Dataset


class CustomDictDataset(Dataset):
    """Custom dataset from a dictionary with support of transforms."""
    def __init__(self, files, size, transform=None):
        self.files = files
        self.size = size
        self.transform = transform

    def __getitem__(self, index):
        remain = index
        file_idx = -1
        for i, size in enumerate(self.size):
            if remain < size:
                file_idx = i
                break
            remain -= size

        with open(self.files[file_idx], 'r') as fin:
            data = json.load(fin)

        user = data['users'][0]
        sample = data['user_data'][user]['x'][remain]
        target = data['user_data'][user]['y'][remain]
        if self.transform:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return sum(self.size)


class ReshapeListTransform:
    def __init__(self, new_shape):
        self.new_shape = new_shape

    def __call__(self, img):
        return np.array(img, dtype=np.float32).reshape(self.new_shape)


class DataSource(base.DataSource):
    """The FEMNIST dataset."""
    def __init__(self, client_id=0):
        super().__init__()
        self.trainset = None
        self.testset = None
        self.trainset_size = 0
        self.testset_size = 0

        root_path = os.path.join(Config().data.data_path, 'FEMNIST')
        if client_id == 0:
            data_dir = os.path.join(root_path, 'test')
            data_url = "https://jiangzhifeng.s3.us-east-2.amazonaws.com/FEMNIST/test.zip"
            data_size = 21.1
        else:
            data_dir = os.path.join(root_path, 'train')
            data_url = "https://jiangzhifeng.s3.us-east-2.amazonaws.com/FEMNIST/train.zip"
            data_size = 169.2

        if not os.path.exists(data_dir):
            logging.info(
                f"[Downloading the FEMNIST dataset ({data_size} MB). This may take a while."
            )
            self.download(url=data_url, data_path=root_path)

        files, size = self.read_data(data_dir=data_dir, client_id=client_id)

        _transform = transforms.Compose([
            ReshapeListTransform((28, 28, 1)),
            transforms.ToPILImage(),
            transforms.RandomCrop(28,
                                  padding=2,
                                  padding_mode="constant",
                                  fill=1.0),
            transforms.RandomResizedCrop(28,
                                         scale=(0.8, 1.2),
                                         ratio=(4. / 5., 5. / 4.)),
            transforms.RandomRotation(5, fill=1.0),
            transforms.ToTensor(),
            transforms.Normalize(0.9637, 0.1597),
        ])
        dataset = CustomDictDataset(files=files,
                                    size=size,
                                    transform=_transform)

        if client_id == 0:  # testing set of the server
            self.testset_size = sum(size)
            self.testset = dataset
        else:  # training set of a client
            self.trainset_size = sum(size)
            self.trainset = dataset

    def read_data(self, data_dir, client_id=0):
        files = os.listdir(data_dir)
        files = [f for f in files if f.endswith('.json')]
        files = sorted(files)
        if client_id > 0:
            files = [files[client_id - 1]]
        files = [os.path.join(data_dir, f) for f in files]

        size = []
        for f in files:
            with open(f, 'r') as fin:
                cdata = json.load(fin)
            size.append(cdata['num_samples'][0])

        return files, size

    def num_train_examples(self):
        return self.trainset_size

    def num_test_examples(self):
        return self.testset_size