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

import logging
import os

from torchvision import transforms

from plato.config import Config
from plato.datasources import base

import numpy as np
import json
from collections import defaultdict
import subprocess
from torch.utils.data import Dataset


class CustomDictDataset(Dataset):
    """Custom dataset from a dictionary with support of transforms."""
    def __init__(self, dictionary, transform=None):
        self.xs = dictionary['x']
        self.ys = dictionary['y']
        self.transform = transform

    def __getitem__(self, index):
        x = self.xs[index]
        if self.transform:
            x = self.transform(x)
        y = self.ys[index]
        return x, y

    def __len__(self):
        return len(self.xs)


class ReshapeListTransform:
    def __init__(self, new_shape):
        self.new_shape = new_shape

    def __call__(self, img):
        return np.array(img, dtype=np.float32).reshape(self.new_shape)


class DataSource(base.DataSource):
    """The FEMNIST dataset."""
    def __init__(self):
        super().__init__()
        self.trainset_size = 0
        self.testset_size = 0

        leaf_path = Config().data.data_path  # should be the path to leaf
        train_folder = os.path.join(leaf_path, 'data', 'femnist', 'data', 'train')
        test_folder = os.path.join(leaf_path, 'data', 'femnist', 'data', 'test')

        if not os.path.exists(train_folder) or not os.path.exists(test_folder):
            cmd = './preprocess.sh -s niid --sf 1.0 -k 0 -t sample'
            logging.info(
                "Downloading and partitioning the FEMNIST dataset. This may take a while."
            )
            working_folder = os.path.join(leaf_path, 'data', 'femnist')
            proc = subprocess.Popen(cmd.split(' '), cwd=working_folder)
            proc.wait()

        logging.info(
            "Loading the FEMNIST dataset. This may take a while."
        )
        train_clients, _, train_data, test_data = self.read_data(train_folder, test_folder)
        trainset = self.dict_to_list(train_clients, train_data)
        testset = self.merge_testset(train_clients, test_data)

        _transform = transforms.Compose([
            ReshapeListTransform((28, 28, 1)),
            transforms.ToPILImage(),
            transforms.RandomCrop(28, padding=2, padding_mode="constant", fill=1.0),
            transforms.RandomResizedCrop(28, scale=(0.8, 1.2), ratio=(4. / 5., 5. / 4.)),
            transforms.RandomRotation(5, fill=1.0),
            transforms.ToTensor(),
            transforms.Normalize(0.9637, 0.1597),
        ])

        self.trainset = [CustomDictDataset(dictionary=d, transform=_transform) for d in trainset]
        self.testset = CustomDictDataset(dictionary=testset, transform=_transform)

    def dict_to_list(self, list_of_keys, dictionary):
        result = []
        for key in list_of_keys:
            result.append(dictionary[key])

        return result

    def merge_testset(self, list_of_keys, dictionary):
        first_key = list_of_keys[0]
        result = dictionary[first_key]
        for key in list_of_keys[1:]:
            result['x'].extend(dictionary[key]['x'])
            result['y'].extend(dictionary[key]['y'])

        self.testset_size = len(result['x'])
        return result

    def do_nothing(self):
        pass

    def read_dir_worker(self, files, data_dir):
        clients = []
        groups = []
        data = defaultdict(self.do_nothing)

        for f in files:
            file_path = os.path.join(data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            clients.extend(cdata['users'])
            if 'hierarchies' in cdata:
                groups.extend(cdata['hierarchies'])
            data.update(cdata['user_data'])

        clients = list(sorted(data.keys()))
        return clients, groups, data

    def read_dir(self, data_dir):
        clients = []
        groups = []
        data = defaultdict(lambda: None)

        files = os.listdir(data_dir)
        files = [f for f in files if f.endswith('.json')]

        # no multiprocessing due to memory concerns
        for f in files:
            file_path = os.path.join(data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            clients.extend(cdata['users'])
            if 'hierarchies' in cdata:
                groups.extend(cdata['hierarchies'])
            data.update(cdata['user_data'])

        clients = list(sorted(data.keys()))
        return clients, groups, data

    def read_data(self, train_data_dir, test_data_dir):
        train_clients, train_groups, train_data = self.read_dir(train_data_dir)
        test_clients, test_groups, test_data = self.read_dir(test_data_dir)

        assert train_clients == test_clients
        assert train_groups == test_groups

        return train_clients, train_groups, train_data, test_data

    def num_train_examples(self):
        return self.trainset_size

    def num_test_examples(self):
        return self.testset_size
