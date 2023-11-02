"""
A personalized federated learning trainer using FedRep.

Reference:

Collins et al., "Exploiting Shared Representations for Personalized Federated
Learning", in the Proceedings of ICML 2021.

https://arxiv.org/abs/2102.07078

Source code: https://github.com/lgcollins/FedRep
"""

import logging

from plato.trainers import basic


class Trainer(basic.Trainer):
    """A personalized federated learning trainer using the FedRep algorithm."""

    def __init__(self, model=None):
        super().__init__(model)

        self.representation_param_names = []
        self.head_param_names = []

    def set_representation_and_head(self, representation_param_names):
        """Setting the parameter names for global (representation)
        and local (the head) models."""

        # set the parameter names for the representation
        #   As mentioned by Eq. 1 and Fig. 2 of the paper, the representation
        #   behaves as the global model.
        self.representation_param_names = representation_param_names

        # FedRep calls the weights and biases of the final fully-connected layer
        # in each of the models as the "head"
        # This insight is obtained from the source code of FedRep.
        model_parameter_names = self.model.state_dict().keys()

        self.head_param_names = [
            name
            for name in model_parameter_names
            if name not in representation_param_names
        ]

        logging.info(
            "[Client #%s] Representation layers: %s",
            self.client_id,
            self.representation_param_names,
        )
        logging.info(
            "[Client #%s] Head layers: %s", self.client_id, self.head_param_names
        )

    def train_epoch_start(self, config):
        """
        Method called at the beginning of a training epoch.

        The local training stage in FedRep contains two parts:

        - Head optimization:
            Makes τ local gradient-based updates to solve for its optimal head given
            the current global representation communicated by the server.

        - Representation optimization:
            Takes one local gradient-based update with respect to the current representation.
        """
        # As presented in Section 3 of the FedRep paper, the head is optimized
        # for (epochs - 1) while freezing the representation.
        local_epochs = (
            config["local_epochs"] if "local_epochs" in config else config["epochs"] - 1
        )

        if self.current_epoch <= local_epochs:
            for name, param in self.model.named_parameters():
                if name in self.representation_param_names:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        # The representation will then be optimized for only one epoch
        if self.current_epoch > local_epochs:
            for name, param in self.model.named_parameters():
                if name in self.representation_param_names:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
