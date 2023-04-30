"""
Obtaining the loss criterion for training workloads according to the configuration file.
"""
from typing import Union

from torch import nn
from lightly import loss

from plato.config import Config


def get(**kwargs: Union[str, dict]):
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

    loss_criterion_name = (
        kwargs["loss_criterion"]
        if "loss_criterion" in kwargs
        else (
            Config().trainer.loss_criterion
            if hasattr(Config.trainer, "loss_criterion")
            else "CrossEntropyLoss"
        )
    )

    loss_criterion_params = (
        kwargs["loss_criterion_params"]
        if "loss_criterion_params" in kwargs
        else (
            Config().parameters.loss_criterion._asdict()
            if hasattr(Config.parameters, "loss_criterion")
            else {}
        )
    )

    loss_criterion = registered_loss_criterion.get(loss_criterion_name)

    return loss_criterion(**loss_criterion_params)
