"""
The federated averaging algorithm for GAN model.
"""
from collections import OrderedDict

from plato.algorithms import fedavg
from plato.trainers.base import Trainer


class Algorithm(fedavg.Algorithm):

    def __init__(self, trainer: Trainer):
        super().__init__(trainer)
        self.netG = self.model.netG
        self.netD = self.model.netD

    def compute_weight_updates(self, weights_received):
        """Extract the weights received from a client and compute the updates."""
        baseline_weights_G, baseline_weights_D = self.extract_weights()

        updates = []
        for weight_G, weight_D in weights_received:
            update_G = OrderedDict()
            for name, current_weight in weight_G.items():
                baseline = baseline_weights_G[name]

                delta = current_weight - baseline
                update_G[name] = delta

            update_D = OrderedDict()
            for name, current_weight in weight_D.items():
                baseline = baseline_weights_D[name]

                delta = current_weight - baseline
                update_D[name] = delta

            updates.append((update_G, update_D))

        return updates

    def update_weights(self, update):
        """ Update the existing model weights. """
        baseline_weights_G, baseline_weights_D = self.extract_weights()
        update_G, update_D = update

        updated_weights_G = OrderedDict()
        for name, weight in baseline_weights_G.items():
            updated_weights_G[name] = weight + update_G[name]

        updated_weights_D = OrderedDict()
        for name, weight in baseline_weights_D.items():
            updated_weights_D[name] = weight + update_D[name]

        return updated_weights_G, updated_weights_D

    def extract_weights(self, model=None):
        """Extract weights from the model."""
        netG = self.netG
        netD = self.netD
        if model is not None:
            netG = self.netG
            netD = self.netD

        weightG = netG.cpu().state_dict()
        weightD = netD.cpu().state_dict()

        return weightG, weightD

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
        weightsG, weightsD = weights
        if weightsG is not None:
            self.netG.load_state_dict(weightsG, strict=True)
        if weightsD is not None:
            self.netD.load_state_dict(weightsD, strict=True)
