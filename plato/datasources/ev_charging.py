"""
EV Charging Datasource for Federated Time-Series Forecasting.

Dataset: "EV Charging Reports" – Mendeley Data (dataset1_ev_charging_reports.csv)
  https://data.mendeley.com/datasets/jbks2rcwyj/1

Raw CSV format:
  session_ID ; Garage_ID ; User_ID ; User_type ; Shared_ID ;
  Start_plugin ; Start_plugin_hour ; End_plugout ; End_plugout_hour ;
  El_kWh ; Duration_hours ; month_plugin ; weekdays_plugin ;
  Plugin_category ; Duration_category
  - Datetimes use DD.MM.YYYY HH:MM format.
  - El_kWh uses a comma as the decimal separator.

Preprocessing pipeline
-----------------------
1. Filter to the requested garage (default "AdO1", which has 4 private users),
   or use all garages when ``garage = "all"``.
2. For each user, build a continuous hourly grid from the first to the last
   session hour in the dataset.
3. For every hour, mark is_charging = 1 if the user had an active session,
   else 0; accumulate energy_kwh proportionally over session hours.
4. Scale energy_kwh ∈ [0, 1] using the training-split maximum.
5. Add cyclic time encodings:
     hour_sin = sin(2π · hour / 24)   hour_cos = cos(2π · hour / 24)
     dow_sin  = sin(2π · dow  / 7)    dow_cos  = cos(2π · dow  / 7)
6. Split temporally: 70 % train, 15 % val, 15 % test.
7. Build sliding-window samples:
     past_values   : (context_length,   6)  : all features
     future_values : (prediction_length, 1) : is_charging only

Federated split
---------------
Each client sees only its own user's data.

TOML configuration
------------------
[data]
datasource      = "EVCharging"
datasource_path = "runtime/data/ado1/dataset1_ev_charging_reports.csv"
garage          = "AdO1"   # optional; use "all" for cross-garage user lists
num_users       = 4        # optional

[trainer]
context_length    = 168   # 7 * 24 h
prediction_length = 168   # 7 * 24 h
train_ratio       = 0.70
val_ratio         = 0.15
stride            = 1    # slide 1 hour at a time
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from plato.config import Config


# Exact column names from the Mendeley CSV
_CSV_SEP = ";"
_GARAGE_COL = "Garage_ID"
_USER_COL = "User_ID"
_START_COL = "Start_plugin"
_END_COL = "End_plugout"
_ENERGY_COL = "El_kWh"
_DT_FORMAT = "%d.%m.%Y %H:%M"


# Preprocessing helpers
def _parse_european_float(series: pd.Series) -> pd.Series:
    """Replace comma decimal separator and coerce to float."""
    return (
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )


def _build_hourly_series(
    df: pd.DataFrame,
    garage: str | None,
    num_users: int,
    user_ids: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Build per-user hourly DataFrames from raw session records.

    Parameters:
    user_ids : explicit list of User_ID strings to include.
    num_users : max number of users to take alphabetically when ``user_ids``
                is not given.

    Returns:
    dict mapping user_id (str) -> pd.DataFrame with hourly index and columns:
        is_charging (0/1 float), energy_kwh (float >= 0)
    """
    garage_name = None if garage is None else str(garage).strip()
    use_all_garages = not garage_name or garage_name.lower() in {"all", "*", "any"}

    if use_all_garages:
        df = df.copy()
    else:
        available_garages = sorted(
            df[_GARAGE_COL].astype(str).str.strip().dropna().unique()
        )
        mask = df[_GARAGE_COL].astype(str).str.strip() == garage_name
        df = df[mask].copy()
        if df.empty:
            raise ValueError(
                f"No records found for garage '{garage_name}'. "
                f"Available: {available_garages}"
            )

    # Parse datetimes
    df[_START_COL] = pd.to_datetime(df[_START_COL].str.strip(), format=_DT_FORMAT)
    df[_END_COL] = pd.to_datetime(df[_END_COL].str.strip(), format=_DT_FORMAT)
    df[_ENERGY_COL] = _parse_european_float(df[_ENERGY_COL])

    # Drop invalid rows
    df = df.dropna(subset=[_START_COL, _END_COL])
    df = df[df[_END_COL] > df[_START_COL]]

    # Resolve user list
    available = sorted(df[_USER_COL].dropna().unique())
    if user_ids is not None:
        # Explicit list from config — validate each entry
        missing = [u for u in user_ids if u not in available]
        if missing:
            scope = "all garages" if use_all_garages else f"garage '{garage_name}'"
            raise ValueError(
                f"Users not found in {scope}: {missing}. "
                f"Available: {available}"
            )
        users = list(user_ids)  # preserve config order
    else:
        users = available[:num_users]
    scope = "all garages" if use_all_garages else f"garage '{garage_name}'"
    logging.info("EVCharging: %s -> users %s", scope, users)

    result: dict[str, pd.DataFrame] = {}
    for user in users:
        udf = df[df[_USER_COL] == user]

        # Per-user hourly index: only spans that user's own activity window.
        # Using a global index would pad every user with the same number of
        # zero-charging hours, giving all clients identical dataset sizes.
        user_start = udf[_START_COL].min().floor("h")
        user_end = udf[_END_COL].max().ceil("h")
        hourly_index = pd.date_range(user_start, user_end, freq="h")

        is_charging = pd.Series(0.0, index=hourly_index)
        energy_kwh = pd.Series(0.0, index=hourly_index)

        for _, row in udf.iterrows():
            # All hours touched by this session
            session_hours = pd.date_range(
                row[_START_COL].floor("h"),
                row[_END_COL].floor("h"),
                freq="h",
            )
            valid_hours = session_hours[session_hours.isin(hourly_index)]
            if valid_hours.empty:
                continue
            is_charging[valid_hours] = 1.0
            energy_per_hour = float(row[_ENERGY_COL]) / max(len(valid_hours), 1)
            energy_kwh[valid_hours] += energy_per_hour

        user_df = pd.DataFrame(
            {"is_charging": is_charging, "energy_kwh": energy_kwh},
            index=hourly_index,
        )
        user_df.index.name = "timestamp"
        result[user] = user_df

    return result


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append cyclic hour-of-day and day-of-week columns."""
    hour = df.index.hour.astype(float)
    dow = df.index.dayofweek.astype(float)
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


# Ordered feature columns fed into the model
_FEATURE_COLS = [
    "is_charging",
    "energy_scaled",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


# Torch Dataset
class _EVChargingDataset(Dataset):
    """Sliding-window samples for one user / one split.

    Each sample:
        past_values   : FloatTensor (context_length,   6)
        future_values : FloatTensor (prediction_length, 1)  ← is_charging only
    """

    def __init__(
        self,
        data: np.ndarray,  # shape (T, 6), already normalized
        context_length: int,
        prediction_length: int,
        stride: int = 1,
        starts: list[int] | None = None,  # explicit window start indices
    ):
        super().__init__()
        self.data = torch.FloatTensor(data)
        self.context_length = context_length
        self.prediction_length = prediction_length
        if starts is not None:
            # Caller already computed and partitioned the valid starts.
            self.indices = starts
        else:
            total = context_length + prediction_length
            max_start = len(data) - total
            if max_start < 0:
                logging.warning(
                    "EVCharging: data has only %d steps but needs %d "
                    "(context=%d + prediction=%d) — dataset will be empty.",
                    len(data),
                    total,
                    context_length,
                    prediction_length,
                )
                self.indices = []
            else:
                self.indices = list(range(0, max_start + 1, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        s = self.indices[idx]
        e_ctx = s + self.context_length
        e_pred = e_ctx + self.prediction_length
        return {
            "past_values": self.data[s:e_ctx],  # (ctx, 6)
            "future_values": self.data[e_ctx:e_pred, :1],  # (pred, 1)
        }


# Plato DataSource
class DataSource:
    """EV Charging DataSource for Plato federated learning.

    Each instance represents ONE user (client_id selects the user, 0-indexed
    over the alphabetically sorted user list for the requested garage).

    Typical config (timesfm_ev_charging.toml):

        [data]
        datasource      = "EVCharging"
        datasource_path = "runtime/data/ado1/dataset1_ev_charging_reports.csv"
        garage          = "AdO1"
        num_users       = 4

        [trainer]
        context_length    = 168
        prediction_length = 168
        train_ratio       = 0.70
        val_ratio         = 0.15
        stride            = 24
    """

    def __init__(self, client_id: int = 0, **kwargs):
        cfg = Config()
        data_cfg = cfg.data
        trainer_cfg = cfg.trainer

        # Locate CSV
        csv_path = kwargs.get(
            "datasource_path",
            getattr(data_cfg, "datasource_path", None),
        )
        if csv_path is None:
            raise ValueError(
                "EVCharging requires 'datasource_path' in [data] config, "
                'e.g. datasource_path = "runtime/data/ado1/dataset1_ev_charging_reports.csv"'
            )
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(os.getcwd(), csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"EV charging CSV not found: {csv_path}\n"
                "Download from https://data.mendeley.com/datasets/jbks2rcwyj/1"
            )

        garage_cfg = kwargs.get("garage", getattr(data_cfg, "garage", "AdO1"))
        garage = None if garage_cfg is None else str(garage_cfg)
        garage_name = None if garage is None else garage.strip()

        # Config: users = ["AdO1-1", "AdO1-2", "AdO1-3", "AdO1-4"]
        user_ids_cfg = kwargs.get("users", getattr(data_cfg, "users", None))
        if user_ids_cfg is not None:
            user_ids: list[str] | None = [str(u) for u in user_ids_cfg]
            num_users = len(user_ids)
        else:
            user_ids = None
            num_users = int(kwargs.get("num_users", getattr(data_cfg, "num_users", 4)))

        # Window / split settings
        self.context_length = int(getattr(trainer_cfg, "context_length", 168))
        self.prediction_length = int(getattr(trainer_cfg, "prediction_length", 168))
        train_ratio = float(getattr(trainer_cfg, "train_ratio", 0.70))
        val_ratio = float(getattr(trainer_cfg, "val_ratio", 0.15))
        stride = int(getattr(trainer_cfg, "stride", 1))

        # Load and preprocess
        logging.info("EVCharging: loading %s", csv_path)
        raw_df = pd.read_csv(csv_path, sep=_CSV_SEP, low_memory=False)

        user_series = _build_hourly_series(
            raw_df, garage=garage, num_users=num_users, user_ids=user_ids
        )

        # Preserve config-specified order when user_ids is given
        if user_ids is not None:
            users = [u for u in user_ids if u in user_series]
        else:
            users = sorted(user_series.keys())

        user_index = max(0, client_id - 1)

        if user_index >= len(users):
            scope = (
                "all garages"
                if not garage_name or garage_name.lower() in {"all", "*", "any"}
                else f"garage '{garage_name}'"
            )
            raise ValueError(
                f"client_id={client_id} out of range; "
                f"found {len(users)} users in {scope}: {users}"
            )

        user_key = users[user_index]
        logging.info("EVCharging: client_id=%d -> user '%s'", client_id, user_key)

        user_df = _add_time_features(user_series[user_key])
        raw_array = user_df[
            [
                "is_charging",
                "energy_kwh",
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
            ]
        ].values.astype(np.float32)

        # Split window indices, not raw hours
        window = self.context_length + self.prediction_length
        all_starts = list(range(0, len(raw_array) - window + 1, stride))
        n_windows = len(all_starts)

        n_train_w = max(1, int(n_windows * train_ratio))
        n_val_w = max(0, int(n_windows * val_ratio))
        train_starts = all_starts[:n_train_w]
        val_starts = all_starts[n_train_w : n_train_w + n_val_w]
        test_starts = all_starts[n_train_w + n_val_w :]

        # Energy scaling
        if train_starts:
            train_end = min(len(user_df), train_starts[-1] + window)
        else:
            train_end = max(1, int(len(user_df) * train_ratio))
        energy_max = float(user_df["energy_kwh"].iloc[:train_end].max()) or 1.0

        full_array = raw_array.copy()
        full_array[:, 1] = full_array[:, 1] / energy_max  # -> energy_scaled in [0, 1]

        # Keep the full normalized array for inference scripts
        self.normalized_data = full_array

        self._train_set = _EVChargingDataset(
            full_array,
            self.context_length,
            self.prediction_length,
            stride=stride,
            starts=train_starts,
        )
        self._val_set = _EVChargingDataset(
            full_array,
            self.context_length,
            self.prediction_length,
            stride=stride,
            starts=val_starts,
        )
        self._test_set = _EVChargingDataset(
            full_array,
            self.context_length,
            self.prediction_length,
            stride=stride,
            starts=test_starts,
        )

        logging.info(
            "EVCharging user '%s': %d train / %d val / %d test windows",
            user_key,
            len(self._train_set),
            len(self._val_set),
            len(self._test_set),
        )

    # Plato DataSource interface
    def get_train_set(self) -> _EVChargingDataset:
        return self._train_set

    def get_val_set(self) -> _EVChargingDataset:
        return self._val_set

    def get_test_set(self) -> _EVChargingDataset:
        return self._test_set

    def num_train_examples(self) -> int:
        return len(self._train_set)

    def num_test_examples(self) -> int:
        return len(self._test_set)
