"""
Open-Meteo API utility functions for fetching and processing weather data.

This module provides functions to fetch historical weather and solar radiation data
from the Open-Meteo Archive API, with caching and interpolation support.
"""

import hashlib
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests


def _generate_cache_key(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    variables: List[str],
    target_freq: str,
) -> str:
    """Generate a unique cache key based on request parameters."""
    key_string = f"{latitude}_{longitude}_{start_date}_{end_date}_{'_'.join(sorted(variables))}_{target_freq}"
    return hashlib.md5(key_string.encode()).hexdigest()


def _get_cache_path(cache_dir: Path, cache_key: str) -> Path:
    """Get the cache file path for a given cache key."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_key}.csv"


def fetch_and_interpolate_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    variables: List[str],
    cache_dir: Path,
    target_freq: str = "5min",
) -> pd.DataFrame:
    """
    Fetch hourly weather data from Open-Meteo API and interpolate to target frequency.

    This function fetches historical weather data at hourly intervals and interpolates
    it to the specified target frequency (e.g., 5-minute intervals) using linear
    interpolation.

    Args:
        latitude: Latitude of the location (e.g., 43.65 for Toronto)
        longitude: Longitude of the location (e.g., -79.38 for Toronto)
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        variables: List of weather variables to fetch (e.g., ['temperature_2m', 'shortwave_radiation'])
        cache_dir: Directory to store cached data
        target_freq: Target frequency for interpolation (default: '5min')

    Returns:
        pd.DataFrame: DataFrame with DatetimeIndex and interpolated weather data

    Raises:
        requests.RequestException: If API request fails
        ValueError: If data validation fails

    Example:
        >>> df = fetch_and_interpolate_weather(
        ...     latitude=43.65,
        ...     longitude=-79.38,
        ...     start_date='2024-01-01',
        ...     end_date='2024-01-14',
        ...     variables=['temperature_2m'],
        ...     cache_dir=Path('./cache'),
        ...     target_freq='5min'
        ... )
        >>> df.head()
                            temperature_2m
        2024-01-01 00:00:00           5.2
        2024-01-01 00:05:00           5.18
        2024-01-01 00:10:00           5.16
        ...
    """
    # Generate cache key
    cache_key = _generate_cache_key(
        latitude, longitude, start_date, end_date, variables, target_freq
    )
    cache_path = _get_cache_path(cache_dir, cache_key)

    if cache_path.exists():
        logging.info(f"Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    logging.info(
        f"Fetching weather data from Open-Meteo API for {start_date} to {end_date}"
    )
    hourly_df = _fetch_hourly_data(latitude, longitude, start_date, end_date, variables)

    logging.info(f"Interpolating hourly data to {target_freq} intervals")
    interpolated_df = _interpolate_to_frequency(hourly_df, target_freq)

    logging.info(f"Caching interpolated data to {cache_path}")
    interpolated_df.to_csv(cache_path)

    return interpolated_df


def _fetch_hourly_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    variables: List[str],
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch hourly weather data from Open-Meteo Archive API.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        variables: List of hourly weather variables to fetch
        max_retries: Maximum number of retry attempts on failure

    Returns:
        pd.DataFrame: DataFrame with hourly weather data

    Raises:
        requests.RequestException: If all retry attempts fail
        ValueError: If API returns invalid data
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables),
        "timezone": "auto",
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            logging.info(
                f"Requesting data from Open-Meteo (attempt {attempt + 1}/{max_retries})"
            )
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Validate response structure
            if "hourly" not in data:
                raise ValueError("Invalid API response: missing 'hourly' field")

            hourly_data = data["hourly"]
            if "time" not in hourly_data:
                raise ValueError("Invalid API response: missing 'time' field")

            # Create DataFrame
            df = pd.DataFrame(hourly_data)
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)

            # Validate that all requested variables are present
            for var in variables:
                if var not in df.columns:
                    raise ValueError(f"Variable '{var}' not found in API response")

            # Check for missing timestamps (should be continuous hourly data)
            expected_length = (
                pd.to_datetime(end_date) - pd.to_datetime(start_date)
            ).days * 24 + 24
            if len(df) < expected_length * 0.9:  # Allow 10% tolerance
                logging.warning(
                    f"Expected ~{expected_length} hourly records, got {len(df)}"
                )

            logging.info(
                f"Successfully fetched {len(df)} hourly records for {len(variables)} variables"
            )
            return df

        except requests.RequestException as e:
            last_error = e
            logging.warning(
                f"Request failed (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                logging.info(f"Retrying in {wait_time} seconds...")
                import time

                time.sleep(wait_time)

        except (ValueError, KeyError) as e:
            logging.error(f"Data validation failed: {e}")
            raise

    raise requests.RequestException(
        f"Failed to fetch data after {max_retries} attempts. Last error: {last_error}"
    )


def _interpolate_to_frequency(df: pd.DataFrame, target_freq: str) -> pd.DataFrame:
    """
    Interpolate hourly data to target frequency using linear interpolation.

    Args:
        df: DataFrame with hourly data and DatetimeIndex
        target_freq: Target frequency string (e.g., '5min', '15min')

    Returns:
        pd.DataFrame: Interpolated DataFrame with target frequency

    Example:
        If target_freq='5min', this creates 12 data points per hour:
        - Original: 00:00, 01:00, 02:00, ...
        - Interpolated: 00:00, 00:05, 00:10, ..., 00:55, 01:00, ...
    """
    # Resample to target frequency (creates NaN for new timestamps)
    resampled_df = df.resample(target_freq).asfreq()

    # Interpolate missing values using linear interpolation
    interpolated_df = resampled_df.interpolate(method="linear")

    # Forward fill any remaining NaN at the end
    interpolated_df = interpolated_df.ffill()

    # Backward fill any remaining NaN at the beginning
    interpolated_df = interpolated_df.bfill()

    # Validate no NaN values remain
    if interpolated_df.isnull().any().any():
        nan_counts = interpolated_df.isnull().sum()
        logging.warning(f"Interpolation left NaN values: {nan_counts.to_dict()}")

    logging.info(
        f"Interpolated from {len(df)} hourly records to {len(interpolated_df)} {target_freq} records"
    )

    return interpolated_df


def calculate_date_range(historical_days: int, end_offset_days: int = 7) -> tuple:
    """
    Calculate start and end dates for historical data fetch.

    Open-Meteo archive data typically has a delay of several days before becoming
    available. This function calculates appropriate date ranges.

    Args:
        historical_days: Number of days of historical data to fetch
        end_offset_days: Days before today to use as end date (default: 7)
            This accounts for the delay in archive data availability

    Returns:
        tuple: (start_date, end_date) as strings in 'YYYY-MM-DD' format

    Example:
        >>> start, end = calculate_date_range(historical_days=14, end_offset_days=7)
        >>> # If today is 2024-01-21:
        >>> # start = '2024-01-07' (14 days before end)
        >>> # end = '2024-01-14' (7 days before today)
    """
    today = datetime.now()
    end_date = today - timedelta(days=end_offset_days)
    start_date = end_date - timedelta(days=historical_days - 1)

    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
