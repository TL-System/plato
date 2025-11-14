"""
ETT (Electricity Transformer Temperature) datasource for time series forecasting.

Supports all ETT datasets:
- ETTh1, ETTh2: Hourly data (1 point per hour)
- ETTm1, ETTm2: 15-minute data (4 points per hour)

Data from: https://github.com/zhouhaoyi/ETDataset
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from plato.config import Config
from plato.datasources import base


class ETTDataset(Dataset):
    """ETT time series dataset with sliding window."""

    def __init__(self, data, context_length, prediction_length, stride=1):
        """
        Create dataset with sliding windows.

        Args:
            data: pandas DataFrame or numpy array with shape (timesteps, channels)
            context_length: Number of historical timesteps
            prediction_length: Number of future timesteps to predict
            stride: Stride for sliding window
        """
        if isinstance(data, pd.DataFrame):
            # Remove date column if present
            if "date" in data.columns:
                data = data.drop("date", axis=1)
            data = data.values

        self.data = torch.FloatTensor(data)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride

        # Calculate number of valid windows
        total_length = context_length + prediction_length
        self.num_windows = max(0, (len(data) - total_length) // stride + 1)

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        """Return past_values and future_values for PatchTSMixer."""
        start_idx = idx * self.stride
        end_context = start_idx + self.context_length
        end_future = end_context + self.prediction_length

        past_values = self.data[start_idx:end_context]
        future_values = self.data[end_context:end_future]

        return {
            "past_values": past_values,
            "future_values": future_values,
        }


class DataSource(base.DataSource):
    """ETT datasource for time series forecasting (ETTh1, ETTh2, ETTm1, ETTm2)."""

    # Dataset configurations
    DATASET_INFO = {
        "ETTh1": {"freq": "hourly", "points_per_hour": 1},
        "ETTh2": {"freq": "hourly", "points_per_hour": 1},
        "ETTm1": {"freq": "15min", "points_per_hour": 4},
        "ETTm2": {"freq": "15min", "points_per_hour": 4},
    }

    def __init__(self, **kwargs):
        super().__init__()

        # Get dataset name
        dataset_name = kwargs.get(
            "dataset_name", 
            getattr(Config().data, "dataset_name", "ETTh1")
        )
        
        # Validate dataset name
        if dataset_name not in self.DATASET_INFO:
            raise ValueError(
                f"Unknown ETT dataset: {dataset_name}. "
                f"Supported datasets: {list(self.DATASET_INFO.keys())}"
            )

        logging.info("Using %s (Electricity Transformer Temperature) dataset", dataset_name)
        
        dataset_info = self.DATASET_INFO[dataset_name]
        logging.info(
            "Dataset frequency: %s (%d points per hour)",
            dataset_info["freq"],
            dataset_info["points_per_hour"]
        )

        # Get configuration
        context_length = getattr(Config().trainer, "context_length", 512)
        prediction_length = getattr(Config().trainer, "prediction_length", 96)

        # Download and load the data
        data_path = self._download_data(dataset_name)
        df = pd.read_csv(data_path)

        logging.info(
            "Loaded %s dataset with %d timesteps and %d channels",
            dataset_name,
            len(df),
            len(df.columns) - 1,
        )  # -1 for date column

        # Split into train/val/test following the standard ETT split
        # Standard split: 12 months train, 4 months val, 4 months test
        points_per_hour = dataset_info["points_per_hour"]
        train_end = 12 * 30 * 24 * points_per_hour  # 12 months
        val_end = train_end + 4 * 30 * 24 * points_per_hour  # + 4 months

        train_df = df[:train_end]
        val_df = df[train_end:val_end]
        test_df = df[val_end:]

        logging.info(
            "%s split - train: %d, val: %d, test: %d",
            dataset_name,
            len(train_df),
            len(val_df),
            len(test_df),
        )

        # Create datasets with sliding windows
        self.trainset = ETTDataset(
            train_df, context_length, prediction_length, stride=1
        )

        # Use validation set as test set for federated learning
        self.testset = ETTDataset(
            val_df, context_length, prediction_length, stride=prediction_length
        )

        logging.info(
            "Created %d training windows and %d test windows",
            len(self.trainset),
            len(self.testset),
        )

    def _download_data(self, dataset_name):
        """Download ETT dataset from GitHub if not already present."""
        data_dir = Path(Config().params["data_path"]) / "ETT-small"
        data_dir.mkdir(parents=True, exist_ok=True)

        data_file = data_dir / f"{dataset_name}.csv"

        if data_file.exists():
            logging.info("%s.csv already exists", dataset_name)
            return str(data_file)

        # Download from GitHub
        logging.info("Downloading %s.csv from GitHub ...", dataset_name)
        url = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{dataset_name}.csv"

        try:
            import urllib.request

            urllib.request.urlretrieve(url, str(data_file))
            logging.info("Successfully downloaded %s.csv", dataset_name)
        except Exception as e:
            logging.error("Failed to download %s.csv: %s", dataset_name, e)
            raise RuntimeError(
                f"Could not download {dataset_name} dataset from {url}. "
                f"Please download it manually to {data_file}"
            ) from e

        return str(data_file)

    def num_train_examples(self):
        return len(self.trainset)

    def num_test_examples(self):
        return len(self.testset)

    def get_train_set(self):
        return self.trainset

    def get_test_set(self):
        return self.testset
