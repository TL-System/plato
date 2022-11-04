"""
Obtaining the loss criterion for training workloads according to the configuration file.
"""
from torch import nn

from plato.config import Config


def get_target_loss(stage_prefix):
    """ Get a loss criterion based on the required prefix that defines the specific stage,
        such as personalized learning stage. """

    loss_criterion_prefix = {None: "loss_criterion", "pers": "pers_loss_criterion"}

    target_loss_criterion = loss_criterion_prefix[stage_prefix]

    return target_loss_criterion


def get(stage_prefix: None):
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

    target_criterion = get_target_loss(stage_prefix)

    loss_criterion_name = (
        getattr(Config().trainer, target_criterion)
        if hasattr(Config.trainer, target_criterion)
        else "CrossEntropyLoss"
    )
    loss_criterion = registered_loss_criterion.get(loss_criterion_name)

    if hasattr(Config().parameters, target_criterion):
        loss_criterion_params = Config().parameters.loss_criterion._asdict()
    else:
        loss_criterion_params = {}

    return loss_criterion(**loss_criterion_params)
