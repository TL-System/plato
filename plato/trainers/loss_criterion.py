"""
Obtaining the loss criterion for training workloads according to the configuration file.
"""

from typing import Union

from lightly import loss
from torch import nn

from plato.config import Config


def get(**kwargs: str | dict):
    """Get a loss function with its name from the configuration file."""
    registered_loss_criterion = {
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

    ssl_loss_criterion = {
        "NegativeCosineSimilarity": loss.NegativeCosineSimilarity,
        "NTXentLoss": loss.NTXentLoss,
        "BarlowTwinsLoss": loss.BarlowTwinsLoss,
        "DCLLoss": loss.DCLLoss,
        "DCLWLoss": loss.DCLWLoss,
        "DINOLoss": loss.DINOLoss,
        "PMSNCustomLoss": loss.PMSNCustomLoss,
        "SwaVLoss": loss.SwaVLoss,
        "PMSNLoss": loss.PMSNLoss,
        "SymNegCosineSimilarityLoss": loss.SymNegCosineSimilarityLoss,
        "TiCoLoss": loss.TiCoLoss,
        "VICRegLoss": loss.VICRegLoss,
        "VICRegLLoss": loss.VICRegLLoss,
        "MSNLoss": loss.MSNLoss,
    }

    registered_loss_criterion.update(ssl_loss_criterion)

    if "loss_criterion" in kwargs:
        loss_criterion_name = kwargs["loss_criterion"]
    elif hasattr(Config(), "trainer") and hasattr(Config().trainer, "loss_criterion"):
        loss_criterion_name = Config().trainer.loss_criterion
    else:
        loss_criterion_name = "CrossEntropyLoss"

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

    loss_criterion = registered_loss_criterion.get(loss_criterion_name)
    if loss_criterion is None:
        raise ValueError(f"Unknown loss criterion: {loss_criterion_name}")

    return loss_criterion(**loss_criterion_params)
