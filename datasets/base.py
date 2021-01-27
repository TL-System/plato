"""
Base class for datasets.
"""
from abc import ABC, abstractstaticmethod

import os
import sys
import logging
import gzip
from urllib.parse import urlparse
import requests


class Dataset(ABC):
    """
    The training or testing dataset that accommodates custom augmentation and transforms.
    """
    def __init__(self, path):
        self._path = path

    @staticmethod
    def download(url, data_path):
        """downloading the MNIST dataset."""
        url_parse = urlparse(url)
        file_name = os.path.join(data_path, url_parse.path.split('/')[-1])

        if not os.path.exists(file_name.replace('.gz', '')):
            logging.info("Downloading %s.", url)

            res = requests.get(url, stream=True, verify=False)
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
            unzipped_file = open(file_name.replace('.gz', ''), 'wb')
            zipped_file = gzip.GzipFile(file_name)
            unzipped_file.write(zipped_file.read())
            zipped_file.close()

            os.remove(file_name)

    @abstractstaticmethod
    def num_train_examples() -> int:
        pass

    @abstractstaticmethod
    def num_test_examples() -> int:
        pass

    @abstractstaticmethod
    def num_classes() -> int:
        pass

    @abstractstaticmethod
    def get_train_set() -> 'Dataset':
        pass

    @abstractstaticmethod
    def get_test_set() -> 'Dataset':
        pass
