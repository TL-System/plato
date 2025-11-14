"""
Utility functions for time series model detection and handling.
"""

from typing import Optional, Tuple


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
    model_name_lower = model_name.lower() if model_name else ""
    model_type_lower = model_type.lower() if model_type else ""

    # Check for PatchTSMixer
    if (
        model_type_lower == "patchtsmixer"
        or "patchtsmixer" in model_name_lower
        or "timeseries" in model_name_lower
    ):
        return True

    # Check dataset type
    if dataset_type and dataset_type.lower() == "timeseries":
        return True

    return False
