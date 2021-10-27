"""
The Federated EMNIST dataset.

The Federated EMNIST dataset originates from the EMNIST dataset, which contains
817851 images, each of which is a 28x28 greyscale image in 1 out of 62 classes.
The difference between the Federated EMNIST dataset and its original counterpart
is that this dataset is already partitioned by the client ID, using the data
provider IDs included in the original EMNIST dataset. As a result of this
partitioning, there are 3597 clients in total, each of which has 227.37 images
on average (std is 88.84). For each client, 90% data samples are used for
training, while the remaining samples are used for testing.

Reference:

G. Cohen, S. Afshar, J. Tapson, and A. Van Schaik, "EMNIST: Extending MNIST to
handwritten letters," in the 2017 International Joint Conference on Neural
Networks (IJCNN).

"""
import json
import logging
import os

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from plato.config import Config
from plato.datasources import base


class CustomDictDataset(Dataset):
    """ Custom dataset from a dictionary with support of transforms. """
    def __init__(self, loaded_data, transform=None):
        """ Initializing the custom dataset. """
        super().__init__()
        self.loaded_data = loaded_data
        self.transform = transform

    def __getitem__(self, index):
        sample = self.loaded_data['x'][index]
        target = self.loaded_data['y'][index]
        if self.transform:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.loaded_data['y'])


class ReshapeListTransform:
    """ The transform that reshapes an image. """
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

        root_path = os.path.join(Config().data.data_path, 'FEMNIST',
                                 'packaged_data')
        if client_id == 0:
            # If we are on the federated learning server
            data_dir = os.path.join(root_path, 'test')
            data_url = "https://jiangzhifeng.s3.us-east-2.amazonaws.com/FEMNIST/test/" \
                       + str(client_id) + ".zip"
        else:
            data_dir = os.path.join(root_path, 'train')
            data_url = "https://jiangzhifeng.s3.us-east-2.amazonaws.com/FEMNIST/train/" \
                       + str(client_id) + ".zip"

        if not os.path.exists(os.path.join(data_dir, str(client_id))):
            logging.info(
                "Downloading the Federated EMNIST dataset "
                "with the client datasets pre-partitioned. This may take a while.",
            )
            self.download(url=data_url, data_path=data_dir)

        loaded_data = DataSource.read_data(
            file_path=os.path.join(data_dir, str(client_id), 'data.json'))

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
        dataset = CustomDictDataset(loaded_data=loaded_data,
                                    transform=_transform)

        if client_id == 0:  # testing dataset on the server
            self.testset = dataset
        else:  # training dataset on one of the clients
            self.trainset = dataset

    @staticmethod
    def read_data(file_path):
        """ Reading the dataset specific to a client_id. """
        with open(file_path, 'r') as fin:
            loaded_data = json.load(fin)
        return loaded_data

    def num_train_examples(self):
        return len(self.trainset)

    def num_test_examples(self):
        return len(self.testset)
