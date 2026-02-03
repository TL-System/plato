"""
Open-Meteo datasource for weather and solar forecasting.

Supports temperature and solar radiation forecasting for any location using
historical data from Open-Meteo Archive API with interpolation to 5-minute intervals.

Data from: https://open-meteo.com/
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from plato.config import Config
from plato.datasources import base
from plato.utils.openmeteo_api import (
    calculate_date_range,
    fetch_and_interpolate_weather,
)


class OpenMeteoDataset(Dataset):
    """Open-Meteo time series dataset with sliding window."""

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
    """Open-Meteo datasource for weather and solar forecasting."""

    # Task configurations
    TASK_CONFIGS = {
        "temperature": {
            "variables": ["temperature_2m"],
            "num_channels": 1,
            "description": "Temperature forecasting (°C)",
        },
        "solar": {
            "variables": ["shortwave_radiation"],
            "num_channels": 1,
            "description": "Solar power forecasting (W/m²)",
        },
        "weather_multivariate": {
            "variables": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "pressure_msl",
            ],
            "num_channels": 5,
            "description": "Multi-variate weather forecasting",
        },
        "solar_weather_multivariate": {
            "variables": [
                "shortwave_radiation",
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "pressure_msl",
                "cloud_cover",
            ],
            "num_channels": 7,
            "description": "Solar + weather multi-variate forecasting (W/m², °C, %, mm, m/s, hPa, %)",
        },
    }

    def __init__(self, **kwargs):
        super().__init__()

        # Get configuration
        task_type = kwargs.get(
            "task_type", getattr(Config().data, "task_type", "temperature")
        )
        latitude = kwargs.get("latitude", getattr(Config().data, "latitude", 43.65))
        longitude = kwargs.get("longitude", getattr(Config().data, "longitude", -79.38))
        location_name = kwargs.get(
            "location_name", getattr(Config().data, "location_name", "Toronto")
        )
        historical_days = kwargs.get(
            "historical_days", getattr(Config().data, "historical_days", 14)
        )

        # Validate task type
        if task_type not in self.TASK_CONFIGS:
            raise ValueError(
                f"Unknown task type: {task_type}. "
                f"Supported tasks: {list(self.TASK_CONFIGS.keys())}"
            )

        task_config = self.TASK_CONFIGS[task_type]
        variables = task_config["variables"]
        num_channels = task_config["num_channels"]
        self.task_type = task_type
        self.variables = list(variables)

        logging.info(
            "Using Open-Meteo datasource for %s - %s",
            location_name,
            task_config["description"],
        )
        logging.info(
            "Location: lat=%.2f, lon=%.2f, historical_days=%d",
            latitude,
            longitude,
            historical_days,
        )
        logging.info("Variables: %s", ", ".join(variables))

        # Get trainer configuration
        context_length = getattr(Config().trainer, "context_length", 336)
        prediction_length = getattr(Config().trainer, "prediction_length", 288)

        logging.info(
            "Context length: %d timesteps, Prediction length: %d timesteps",
            context_length,
            prediction_length,
        )

        # Fetch and process data
        df = self._fetch_and_process_data(
            latitude, longitude, historical_days, variables, location_name
        )
        self.full_df = df

        logging.info(
            "Loaded %s data with %d timesteps and %d channels",
            task_type,
            len(df),
            num_channels,
        )

        # Split into train/val/test (70/20/10 split)
        total_len = len(df)
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.9)

        # Shift val/test start back by context_length so their first window has history
        val_start = max(0, train_end - context_length)
        test_start = max(0, val_end - context_length)

        train_df = df[:train_end]
        val_df = df[val_start:val_end]
        test_df = df[test_start:]

        logging.info(
            "Data split - train: %d, val: %d, test: %d timesteps",
            len(train_df),
            len(val_df),
            len(test_df),
        )

        # Compute train mean/std per channel for consistent normalization
        eps = 1e-6
        train_mean = train_df.mean()
        train_std = train_df.std().replace(0, eps)
        self.train_mean = train_mean
        self.train_std = train_std

        # Normalize all splits using training statistics
        train_norm = ((train_df - train_mean) / train_std).to_numpy()
        val_norm = ((val_df - train_mean) / train_std).to_numpy()
        test_norm = ((test_df - train_mean) / train_std).to_numpy()

        # Store normalization statistics for inverse transform in the future
        self.normalization_stats = {
            "mean": train_mean.to_dict(),
            "std": train_std.to_dict(),
        }
        logging.info("Normalization - mean: %s", train_mean.to_dict())
        logging.info("Normalization - std: %s", train_std.to_dict())
        logging.info("Data normalized using training set statistics")

        # Create datasets with sliding windows
        self.trainset = OpenMeteoDataset(
            train_norm, context_length, prediction_length, stride=1
        )
        self.valset = OpenMeteoDataset(
            val_norm, context_length, prediction_length, stride=1
        )
        self.testset = OpenMeteoDataset(
            test_norm, context_length, prediction_length, stride=1
        )

        logging.info(
            "Created %d training windows and %d test windows",
            len(self.trainset),
            len(self.testset),
        )

    def _fetch_and_process_data(
        self, latitude, longitude, historical_days, variables, location_name
    ):
        """Fetch data from Open-Meteo API and interpolate to 5-minute intervals."""
        # Calculate date range (end 7 days before today to account for archive delay)
        start_date, end_date = calculate_date_range(
            historical_days=historical_days, end_offset_days=7
        )

        logging.info("Fetching data for date range: %s to %s", start_date, end_date)

        # Set up cache directory
        cache_dir = Path(Config().params["data_path"]) / "openmeteo_cache"

        # Fetch and interpolate data to 5-minute intervals
        df = fetch_and_interpolate_weather(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            variables=variables,
            cache_dir=cache_dir,
            target_freq="5min",
        )

        # Validate data
        if df.empty:
            raise ValueError("Fetched DataFrame is empty")

        if df.isnull().any().any():
            nan_counts = df.isnull().sum()
            logging.warning("Data contains NaN values: %s", nan_counts.to_dict())
            # Fill remaining NaN values
            df = df.ffill().bfill()

        return df

    def num_train_examples(self):
        return len(self.trainset)

    def num_test_examples(self):
        return len(self.testset)

    def get_train_set(self):
        return self.trainset

    def get_test_set(self):
        return self.testset
