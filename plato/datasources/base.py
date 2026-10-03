"""
Base class for data sources, encapsulating training and testing datasets with
custom augmentations and transforms already accommodated.
"""

import contextlib
import fcntl
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

from plato.utils.archive import UnsafeArchiveError, extract_archive


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
        if Path(data_path).is_symlink():
            raise UnsafeArchiveError(
                f"Unsafe download destination symlink: {data_path}"
            )
        os.makedirs(data_path, exist_ok=True)
        lock_file = os.path.join(data_path, ".download.lock")
        if Path(lock_file).is_symlink():
            raise UnsafeArchiveError(f"Unsafe download lock symlink: {lock_file}")
        # Keep a stable lock inode: unlinking a flock file can let a newcomer
        # bypass a waiter on the previous inode. The kernel releases the lock
        # even if the owning process exits without executing this finally block.
        with open(lock_file, "a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @staticmethod
    def download(url, data_path):
        """Download a dataset from a URL if it is not already available."""
        url_parse = urlparse(url)
        file_name = os.path.join(data_path, url_parse.path.split("/")[-1])
        name, suffix = os.path.splitext(file_name)
        if suffix not in {".gz", ".zip"}:
            raise ValueError(f"Unsupported download archive format: {file_name}")
        if Path(data_path).is_symlink():
            raise UnsafeArchiveError(
                f"Unsafe download destination symlink: {data_path}"
            )
        os.makedirs(data_path, exist_ok=True)
        sentinel = Path(f"{file_name}.complete")

        artifacts = [Path(file_name), sentinel]
        if suffix == ".gz" and not file_name.endswith("tar.gz"):
            artifacts.append(Path(name))
        for artifact in artifacts:
            if artifact.is_symlink():
                raise UnsafeArchiveError(
                    f"Unsafe download artifact symlink: {artifact}"
                )

        if sentinel.exists():
            return

        with DataSource._download_guard(data_path):
            if sentinel.exists():
                return

            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                if attempt == 1:
                    logging.info("Downloading %s.", url)
                else:
                    logging.info(
                        "Retrying download (%s/%s) for %s.",
                        attempt,
                        max_attempts,
                        url,
                    )

                try:
                    with requests.get(url, stream=True, timeout=60) as res:
                        res.raise_for_status()
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
                                    sys.stdout.write(
                                        f"\r{100 * downloaded_size / total_size:.1f}%"
                                    )
                                    sys.stdout.flush()
                            if total_size:
                                sys.stdout.write("\n")
                except requests.RequestException as exc:
                    logging.warning("Download failed for %s: %s", url, exc)
                    Path(file_name).unlink(missing_ok=True)
                    if attempt == max_attempts:
                        raise
                    time.sleep(1)
                    continue

                if total_size and downloaded_size != total_size:
                    logging.warning(
                        "Download size mismatch for %s (expected %s, got %s).",
                        file_name,
                        total_size,
                        downloaded_size,
                    )
                    if os.path.exists(file_name):
                        os.remove(file_name)
                    if attempt == max_attempts:
                        raise RuntimeError(
                            f"Incomplete download for {url}. Please retry."
                        )
                    time.sleep(1)
                    continue

                # Unzip the compressed file just downloaded
                logging.info("Decompressing the dataset downloaded.")

                try:
                    if file_name.endswith("tar.gz"):
                        extract_archive(file_name, data_path)
                        os.remove(file_name)
                    elif suffix == ".zip":
                        logging.info("Extracting %s to %s.", file_name, data_path)
                        extract_archive(file_name, data_path)
                    elif suffix == ".gz":
                        with gzip.open(file_name, "rb") as zipped_file:
                            with open(name, "wb") as unzipped_file:
                                unzipped_file.write(zipped_file.read())
                        os.remove(file_name)
                except (OSError, tarfile.ReadError, zipfile.BadZipFile) as exc:
                    logging.warning("Failed to extract %s: %s", file_name, exc)
                    if os.path.exists(file_name):
                        os.remove(file_name)
                    if attempt == max_attempts:
                        raise
                    time.sleep(1)
                    continue

                sentinel.touch()
                break

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
