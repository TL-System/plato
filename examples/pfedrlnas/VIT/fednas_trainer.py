"""
Customized Trainer for PerFedRLNAS.
"""
import fednasvit_specific

from plato.trainers import basic
from plato.config import Config


if Config().trainer.type == "basic":
    BasicTrainer = basic.Trainer
else:
    BasicTrainer = basic.TrainerWithTimmScheduler


class Trainer(BasicTrainer):
    """Use special optimizer and loss criterion specific for NASVIT."""

    def get_loss_criterion(self):
        return fednasvit_specific.get_nasvit_loss_criterion()

    def get_optimizer(self, model):
        optimizer = fednasvit_specific.get_optimizer(model)
        return optimizer
