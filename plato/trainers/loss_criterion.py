"""
Obtaining the loss criterion for training workloads according to the configuration file.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from torch import nn

from plato.config import Config

_CORE_LOSS_CRITERIA = {
    "L1Loss": nn.L1Loss,
    "MSELoss": nn.MSELoss,
    "BCELoss": nn.BCELoss,
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    "NLLLoss": nn.NLLLoss,
    "PoissonNLLLoss": nn.PoissonNLLLoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "HingeEmbeddingLoss": nn.HingeEmbeddingLoss,
    "MarginRankingLoss": nn.MarginRankingLoss,
    "TripletMarginLoss": nn.TripletMarginLoss,
    "KLDivLoss": nn.KLDivLoss,
}

_SSL_LOSS_CRITERION_NAMES = {
    "NegativeCosineSimilarity": "NegativeCosineSimilarity",
    "NTXentLoss": "NTXentLoss",
    "BarlowTwinsLoss": "BarlowTwinsLoss",
    "DCLLoss": "DCLLoss",
    "DCLWLoss": "DCLWLoss",
    "DINOLoss": "DINOLoss",
    "PMSNCustomLoss": "PMSNCustomLoss",
    "SwaVLoss": "SwaVLoss",
    "PMSNLoss": "PMSNLoss",
    "SymNegCosineSimilarityLoss": "SymNegCosineSimilarityLoss",
    "TiCoLoss": "TiCoLoss",
    "VICRegLoss": "VICRegLoss",
    "VICRegLLoss": "VICRegLLoss",
    "MSNLoss": "MSNLoss",
}


def _require_lightly_loss_module() -> ModuleType:
    """Load `lightly.loss` only for SSL workloads."""
    try:
        return importlib.import_module("lightly.loss")
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "Self-supervised learning loss criteria require the optional "
            "'lightly' package. Install it in environments that run SSL "
            "training workloads."
        ) from exc


def _resolve_loss_criterion(name: str) -> type[Any]:
    """Resolve a configured loss criterion class by name."""
    loss_criterion = _CORE_LOSS_CRITERIA.get(name)
    if loss_criterion is not None:
        return loss_criterion

    ssl_attr = _SSL_LOSS_CRITERION_NAMES.get(name)
    if ssl_attr is not None:
        lightly_loss = _require_lightly_loss_module()
        loss_criterion = getattr(lightly_loss, ssl_attr, None)
        if loss_criterion is None:
            raise AttributeError(
                f"Optional dependency 'lightly' does not provide loss '{name}'."
            )
        return loss_criterion

    raise ValueError(f"Unknown loss criterion: {name}")


def get(**kwargs: Any):
    """Get a loss function with its name from the configuration file."""
    if "loss_criterion" in kwargs:
        loss_criterion_name = kwargs["loss_criterion"]
    elif hasattr(Config(), "trainer") and hasattr(Config().trainer, "loss_criterion"):
        loss_criterion_name = Config().trainer.loss_criterion
    else:
        loss_criterion_name = "CrossEntropyLoss"

    if not isinstance(loss_criterion_name, str):
        raise TypeError("loss_criterion must be provided as a string.")

    if "loss_criterion_params" in kwargs:
        loss_criterion_params = kwargs["loss_criterion_params"]
    elif hasattr(Config(), "parameters") and hasattr(
        Config().parameters, "loss_criterion"
    ):
        loss_criterion_params = Config().parameters.loss_criterion._asdict()
    else:
        loss_criterion_params = {}

    if not isinstance(loss_criterion_params, dict):
        raise TypeError("loss_criterion_params must be a mapping of keyword arguments.")

    loss_criterion = _resolve_loss_criterion(loss_criterion_name)
    return loss_criterion(**loss_criterion_params)
