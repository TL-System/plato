"""
The CINIC-10 dataset.
For more information about CINIC-10, refer to
https://github.com/BayesWatch/cinic-10
"""

import os
import urllib
import tarfile
from torchvision import datasets, transforms

from datasets import base


class Dataset(base.Dataset):
    """The CINIC-10 dataset."""
    def __init__(self, path):
        super().__init__(path)

        self.cinic_directory = path + '/CINIC-10'

        # Download and extract CINIC-10 dataset if haven't
        if not os.path.exists(self.cinic_directory):
            if not os.path.exists(self.cinic_directory + '.tar.gz'):
                print('Downloading CINIC-10 dataset...')
                print('It might take a while :(')
                url = 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz'
                urllib.request.urlretrieve(url,
                                           self.cinic_directory + '.tar.gz')
                print('Done!')
            print('Extracting CINIC-10 dataset...')
            print('It might take a while :(')
            tar = tarfile.open(self.cinic_directory + '.tar.gz')
            tar.extractall(path=self.cinic_directory)
            tar.close()
            print('Done!')

        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.47889522, 0.47227842, 0.43047404],
                                 [0.24205776, 0.23828046, 0.25874835])
        ])

    @staticmethod
    def num_train_examples():
        return 90000

    @staticmethod
    def num_test_examples():
        return 90000

    @staticmethod
    def num_classes():
        return 10

    def get_train_set(self):
        return datasets.ImageFolder(root=self.cinic_directory + '/train',
                                    transform=self._transform)

    def get_test_set(self):
        return datasets.ImageFolder(root=self.cinic_directory + '/test',
                                    transform=self._transform)
