"""
Base class for data sources, encapsulating training and testing datasets with
custom augmentations and transforms already accommodated.
"""
import gzip
import logging
import os
import sys
import tarfile
import zipfile
from urllib.parse import urlparse

import requests
from plato.config import Config


class DataSource:
    """
    Training and testing datasets with custom augmentations and transforms
    already accommodated.
    """
    def __init__(self):
        self.trainset = None
        self.testset = None

    @staticmethod
    def download(url, data_path):
        """downloads a dataset from a URL."""
        if not os.path.exists(data_path):
            if Config().clients.total_clients > 1:
                if not hasattr(Config().data, 'concurrent_download'
                               ) or not Config().data.concurrent_download:
                    raise ValueError(
                        "The dataset has not yet been downloaded from the Internet. "
                        "Please re-run with '-d' or '--download' first. ")

            os.makedirs(data_path, exist_ok=True)

        url_parse = urlparse(url)
        file_name = os.path.join(data_path, url_parse.path.split('/')[-1])

        if not os.path.exists(file_name.replace('.gz', '')):
            logging.info("Downloading %s.", url)

            res = requests.get(url, stream=True)
            total_size = int(res.headers["Content-Length"])
            downloaded_size = 0

            with open(file_name, "wb+") as file:
                for chunk in res.iter_content(chunk_size=1024):
                    downloaded_size += len(chunk)
                    file.write(chunk)
                    file.flush()
                    sys.stdout.write("\r{:.1f}%".format(100 * downloaded_size /
                                                        total_size))
                    sys.stdout.flush()
                sys.stdout.write("\n")

            # Unzip the compressed file just downloaded
            logging.info("Decompressing the dataset downloaded.")
            name, suffix = os.path.splitext(file_name)

            if file_name.endswith("tar.gz"):
                tar = tarfile.open(file_name, "r:gz")
                tar.extractall(data_path)
                tar.close()
                os.remove(file_name)
            elif suffix == '.zip':
                logging.info("Extracting %s to %s.", file_name, data_path)
                with zipfile.ZipFile(file_name, 'r') as zip_ref:
                    zip_ref.extractall(data_path)
            elif suffix == '.gz':
                unzipped_file = open(name, 'wb')
                zipped_file = gzip.GzipFile(file_name)
                unzipped_file.write(zipped_file.read())
                zipped_file.close()
                os.remove(file_name)
            else:
                logging.info("Unknown compressed file type.")
                sys.exit()

        if Config().args.download:
            logging.info("The dataset has been successfully downloaded. "
                         "Re-run the experiment without '-d' or '--download'.")
            sys.exit()

    @staticmethod
    def input_shape():
        """ Obtains the input shape of this data source. """
        raise NotImplementedError(
            "Input shape not specified for this data source.")

    def num_train_examples(self) -> int:
        """ Obtains the number of training examples. """
        return len(self.trainset)

    def num_test_examples(self) -> int:
        """ Obtains the number of testing examples. """
        return len(self.testset)

    def classes(self):
        """ Obtains a list of class names in the dataset. """
        return list(self.trainset.classes)

    def targets(self):
        """ Obtains a list of targets (labels) for all the examples
        in the dataset. """
        return self.trainset.targets

    def get_train_set(self):
        """ Obtains the training dataset. """
        return self.trainset

    def get_test_set(self):
        """ Obtains the validation dataset. """
        return self.testset
