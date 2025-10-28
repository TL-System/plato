"""
Base class for benchmarks evaluating trained models.
"""

from typing import Any
from abc import ABC, abstractmethod
import gzip
import logging
import os
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse
import requests
import contextlib, time


class Benchmark(ABC):
    """Base class for model benchmarks."""

    def __init__(self):
        """
        Initialize the benchmark.
        """
        super().__init__()

    @abstractmethod
    def evaluate(self) -> dict[str, Any]:
        """
        Evaluate the model on benchmark tasks.

        evaluate() returns evaluation results.

        Returns:
            Dictionary of evaluation metrics

        Example:
            >>> results = benchmark.evaluate()
            >>> print(results)
            {'task1_accuracy': 0.85, 'overall': 0.875}
        """
        pass

    @abstractmethod
    def get_formatted_result(self) -> str:
        pass

    # Borrowed from plato/datasources/base.py
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
        url_parse = urlparse(url)
        file_name = os.path.join(data_path, url_parse.path.split("/")[-1])
        os.makedirs(data_path, exist_ok=True)
        sentinel = Path(f"{file_name}.complete")

        if sentinel.exists():
            return

        with Benchmark._download_guard(data_path):
            if sentinel.exists():
                return

            logging.info("Downloading %s.", url)

            res = requests.get(url, stream=True, timeout=60)
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
