"""
Base class for data sources, encapsulating training and testing datasets with
custom augmentations and transforms already accommodated.
"""

import contextlib
import gzip
import logging
import os
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests


class DataSource:
    """
    Training and testing datasets with custom augmentations and transforms
    already accommodated.
    """

    def __init__(self):
        self.trainset: Any | None = None
        self.testset: Any | None = None

    @staticmethod
    @contextlib.contextmanager
    def _download_guard(data_path: str):
        """Serialise dataset downloads to avoid concurrent corruption."""
        os.makedirs(data_path, exist_ok=True)
        lock_file = os.path.join(data_path, ".download.lock")
        lock_fd = None
        waited = False

        try:
            while True:
                try:
                    lock_fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                    break
                except FileExistsError:
                    if not waited:
                        logging.info(
                            "Another process is preparing the dataset at %s. Waiting.",
                            data_path,
                        )
                        waited = True
                    time.sleep(1)
            yield
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
                try:
                    os.remove(lock_file)
                except FileNotFoundError:
                    pass

    @staticmethod
    def download(url, data_path):
        """Download a dataset from a URL if it is not already available."""
        os.makedirs(data_path, exist_ok=True)
        sentinel = Path(data_path) / ".download_complete"

        if sentinel.exists():
            return

        url_parse = urlparse(url)
        file_name = os.path.join(data_path, url_parse.path.split("/")[-1])

        with DataSource._download_guard(data_path):
            if sentinel.exists():
                return

            logging.info("Downloading %s.", url)

            res = requests.get(url, verify=False, stream=True)
            total_size = int(res.headers.get("Content-Length", 0))
            downloaded_size = 0

            with open(file_name, "wb+") as file:
                for chunk in res.iter_content(chunk_size=1024):
                    if not chunk:
                        continue
                    downloaded_size += len(chunk)
                    file.write(chunk)
                    file.flush()
                    if total_size:
                        sys.stdout.write(f"\r{100 * downloaded_size / total_size:.1f}%")
                        sys.stdout.flush()
                if total_size:
                    sys.stdout.write("\n")

            # Unzip the compressed file just downloaded
            logging.info("Decompressing the dataset downloaded.")
            name, suffix = os.path.splitext(file_name)

            if file_name.endswith("tar.gz"):
                with tarfile.open(file_name, "r:gz") as tar:
                    tar.extractall(data_path)
                os.remove(file_name)
            elif suffix == ".zip":
                logging.info("Extracting %s to %s.", file_name, data_path)
                with zipfile.ZipFile(file_name, "r") as zip_ref:
                    zip_ref.extractall(data_path)
            elif suffix == ".gz":
                with gzip.open(file_name, "rb") as zipped_file:
                    with open(name, "wb") as unzipped_file:
                        unzipped_file.write(zipped_file.read())
                os.remove(file_name)
            else:
                logging.info("Unknown compressed file type for %s.", file_name)
                sys.exit()

            sentinel.touch()

    @staticmethod
    def input_shape():
        """Obtains the input shape of this data source."""
        raise NotImplementedError("Input shape not specified for this data source.")

    def num_train_examples(self) -> int:
        """Obtains the number of training examples."""
        trainset = self.require_trainset()
        return len(trainset)

    def num_test_examples(self) -> int:
        """Obtains the number of testing examples."""
        testset = self.require_testset()
        return len(testset)

    def classes(self):
        """Obtains a list of class names in the dataset."""
        trainset = self.require_trainset()
        classes = getattr(trainset, "classes", None)
        if classes is None:
            raise AttributeError(
                "Training dataset does not expose `classes` attribute."
            )
        return list(classes)

    def targets(self):
        """Obtains a list of targets (labels) for all the examples
        in the dataset."""
        trainset = self.require_trainset()
        targets = getattr(trainset, "targets", None)
        if targets is None:
            raise AttributeError(
                "Training dataset does not expose `targets` attribute."
            )
        return targets

    def get_train_set(self):
        """Obtains the training dataset."""
        return self.require_trainset()

    def get_test_set(self):
        """Obtains the validation dataset."""
        return self.require_testset()

    def require_trainset(self):
        """Return the training dataset, ensuring it is available."""
        if self.trainset is None:
            raise RuntimeError("Training dataset has not been loaded yet.")
        return self.trainset

    def require_testset(self):
        """Return the test dataset, ensuring it is available."""
        if self.testset is None:
            raise RuntimeError("Test dataset has not been loaded yet.")
        return self.testset
