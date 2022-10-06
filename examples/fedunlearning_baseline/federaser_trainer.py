""" A federated unlearning trainer that implements the FedEraser Algorithm. """

import numpy as np

from plato.trainers import basic
from plato.config import Config


class Trainer(basic.Trainer):
    """
    When the client enters retraining phase, trainer should conduct calibration training
    which only trains a reduced number of rounds. The ratio of reduction is defined in Config.
    """

    def __init__(self, model=None):
        super().__init__(model)

        # Indication of retraining
        self.updated_epochs = Config().trainer.epochs

    def set_epochs(self, updated_epochs):
        """ Setting the retraining phase for the trainer """
        self.updated_epochs = updated_epochs

    def train_process(self,
                      config,
                      trainset,
                      sampler,
                      cut_layer=None,
                      **kwargs):
        """
        When doing calibration training,reduce the local training epoch by a ratio defined in config
        """

        config['epochs'] = self.updated_epochs

        super().train_process(config, trainset, sampler, cut_layer, **kwargs)
