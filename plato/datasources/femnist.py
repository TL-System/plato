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
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform
        self.samples = []
        self.targets = []

        for file_path in files:
            with open(file_path, 'r') as fin:
                data = json.load(fin)
            user = data['users'][0]
            samples = data['user_data'][user]['x']
            targets = data['user_data'][user]['y']

            self.samples.extend(samples)
            self.targets.extend(targets)

    def __getitem__(self, index):
        sample = self.samples[index]
        target = self.targets[index]
        if self.transform:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.targets)


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

        if client_id == 0:
            files = self.read_data(data_dir=os.path.join(root_path, 'test'), client_id=client_id)
        else:
            files = self.read_data(data_dir=os.path.join(root_path, 'train'), client_id=client_id)

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
        dataset = CustomDictDataset(files=files, transform=_transform)

        if client_id == 0:  # testing set of the server
            self.testset = dataset
        else:  # training set of a client
            self.trainset = dataset

    def read_data(self, data_dir, client_id=0):
        files = os.listdir(data_dir)
        files = [f for f in files if f.endswith('.json')]
        files = sorted(files)
        if client_id > 0:
            files = [files[client_id - 1]]
        files = [os.path.join(data_dir, f) for f in files]

        return files

    def num_train_examples(self):
        return len(self.trainset)

    def num_test_examples(self):
        return len(self.testset)