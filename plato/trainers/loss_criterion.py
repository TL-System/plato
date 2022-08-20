"""
Obtaining the loss criterion for training workloads according to the configuration file.
"""
from torch import nn

from plato.config import Config


def get():
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

    loss_criterion_name = (
        Config().trainer.loss_criterion
        if hasattr(Config.trainer, "loss_criterion")
        else "CrossEntropyLoss"
    )
    loss_criterion = registered_loss_criterion.get(loss_criterion_name)

    if hasattr(Config().parameters, "loss_criterion"):
        loss_criterion_params = Config().parameters.loss_criterion._asdict()
    else:
        loss_criterion_params = {}

    return loss_criterion(**loss_criterion_params)
