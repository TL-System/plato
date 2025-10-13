"""
The federated averaging algorithm for GAN model.
"""

from collections import OrderedDict

from plato.algorithms import fedavg
from plato.trainers.base import Trainer


class Algorithm(fedavg.Algorithm):
    """Federated averaging algorithm for GAN models, used by both the client and the server."""

    def __init__(self, trainer: Trainer):
        super().__init__(trainer=trainer)
        self.generator = self.model.generator
        self.discriminator = self.model.discriminator

    def compute_weight_deltas(self, baseline_weights, weights_received):
        """Extract the weights received from a client and compute the updates."""
        baseline_weights_gen, baseline_weights_disc = baseline_weights

        deltas = []
        for weight_gen, weight_disc in weights_received:
            delta_gen = OrderedDict()
            for name, current_weight in weight_gen.items():
                baseline = baseline_weights_gen[name]

                delta = current_weight - baseline
                delta_gen[name] = delta

            delta_disc = OrderedDict()
            for name, current_weight in weight_disc.items():
                baseline = baseline_weights_disc[name]

                delta = current_weight - baseline
                delta_disc[name] = delta

            deltas.append((delta_gen, delta_disc))

        return deltas

    def update_weights(self, deltas):
        """Update the existing model weights."""
        baseline_weights_gen, baseline_weights_disc = self.extract_weights()
        update_gen, update_disc = deltas

        updated_weights_gen = OrderedDict()
        for name, weight in baseline_weights_gen.items():
            updated_weights_gen[name] = weight + update_gen[name]

        updated_weights_disc = OrderedDict()
        for name, weight in baseline_weights_disc.items():
            updated_weights_disc[name] = weight + update_disc[name]

        return updated_weights_gen, updated_weights_disc

    def extract_weights(self, model=None):
        """Extract weights from the model."""
        generator = self.generator
        discriminator = self.discriminator
        if model is not None:
            generator = model.generator
            discriminator = model.discriminator

        gen_weight = generator.cpu().state_dict()
        disc_weight = discriminator.cpu().state_dict()

        return gen_weight, disc_weight

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
        weights_gen, weights_disc = weights
        # The client might only receive one or none of the Generator
        # and Discriminator model weight.
        if weights_gen is not None:
            self.generator.load_state_dict(weights_gen, strict=True)
        if weights_disc is not None:
            self.discriminator.load_state_dict(weights_disc, strict=True)
