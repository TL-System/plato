"""
Base class for datasets.
"""
import gzip
import logging
import os
import sys
import tarfile
import zipfile
from urllib.parse import urlparse

import requests


class DataSource:
    """
    The training or testing dataset that accommodates custom augmentation and transforms.
    """
    def __init__(self):
        self.trainset = None
        self.testset = None

    @staticmethod
    def download(url, data_path):
        """downloading the dataset from a URL."""
        if not os.path.exists(data_path):
            os.makedirs(data_path)

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
                    done = int(100 * downloaded_size / total_size)
                    # show download progress
                    sys.stdout.write("\r[{}{}] {:.2f}%".format(
                        "█" * done, " " * (100 - done),
                        100 * downloaded_size / total_size))
                    sys.stdout.flush()
                sys.stdout.write("\n")

            # Unzip the compressed file just downloaded
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

    def num_train_examples(self) -> int:
        return len(self.trainset)

    def num_test_examples(self) -> int:
        return len(self.testset)

    def classes(self):
        """Obtains a list of class names in the dataset. """
        return list(self.trainset.classes)

    def targets(self):
        """Obtains a list of targets (labels) for all the examples
        in the dataset. """
        return self.trainset.targets

    def get_train_set(self):
        """Obtains the training dataset. """
        return self.trainset

    def get_test_set(self):
        """Obtains the validation dataset. """
        return self.testset
