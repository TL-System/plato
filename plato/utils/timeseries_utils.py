"""
Utility functions for time series model detection and handling.
"""

from typing import Optional

# Single source of truth: all known HuggingFace time series model types.
# When adding a new time series model, register it here AND add a loader to
# plato/models/huggingface.py (_TIMESERIES_LOADERS).
TIMESERIES_MODEL_TYPES: frozenset[str] = frozenset({"timesfm", "patchtsmixer"})


def is_timeseries_model(
    model_name: Optional[str] = None,
    model_type: Optional[str] = None,
    dataset_type: Optional[str] = None,
) -> bool:
    """
    Check if a model/dataset is for time series.

    Args:
        model_name: Name of the model
        model_type: Type of model from config
        dataset_type: Type of dataset from config

    Returns:
        True if this is a time series model, False otherwise
    """
    model_name_lower = (model_name or "").lower()
    model_type_lower = (model_type or "").lower()

    # Check explicit model type
    if model_type_lower in TIMESERIES_MODEL_TYPES:
        return True

    # Check if any known time series type appears in the model name
    if any(ts_type in model_name_lower for ts_type in TIMESERIES_MODEL_TYPES):
        return True

    # Generic "timeseries" keyword in name
    if "timeseries" in model_name_lower:
        return True

    # Check dataset type
    if dataset_type and dataset_type.lower() == "timeseries":
        return True

    return False
