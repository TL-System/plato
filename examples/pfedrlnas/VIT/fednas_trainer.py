from plato.trainers import basic
from plato.config import Config

import fednasvit_specific

if Config().trainer.type == "basic":
    basic_Trainer = basic.Trainer
else:
    basic_Trainer = basic.TrainerWithTimmScheduler


class Trainer(basic_Trainer):
    def get_loss_criterion(self):
        return fednasvit_specific.get_NASVIT_loss_criterion()

    def get_optimizer(self, model):
        optimizer = fednasvit_specific.get_optimizer()
        return optimizer
