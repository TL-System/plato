"""
The federated averaging algorithm for PyTorch.
"""
import string
from typing import List
from collections import OrderedDict

from plato.algorithms import base


class Algorithm(base.Algorithm):
    """PyTorch-based federated averaging algorithm, used by both the client and the server."""

    def compute_weight_deltas(self, baseline_weights, weights_received):
        """Compute the deltas between baseline weights and weights received."""
        # Calculate updates from the received weights
        deltas = []
        for weight in weights_received:
            delta = OrderedDict()
            for name, current_weight in weight.items():
                baseline = baseline_weights[name]

                # Calculate update
                _delta = current_weight - baseline
                delta[name] = _delta
            deltas.append(delta)

        return deltas

    def update_weights(self, deltas):
        """Updates the existing model weights from the provided deltas."""
        baseline_weights = self.extract_weights()

        updated_weights = OrderedDict()
        for name, weight in baseline_weights.items():
            updated_weights[name] = weight + deltas[name]

        return updated_weights

    def extract_weights(self, model=None):
        """Extracts weights from the model."""
        if model is None:
            return self.model.cpu().state_dict()
        else:
            return model.cpu().state_dict()

    def load_weights(self, weights):
        """Loads the model weights passed in as a parameter."""
        self.model.load_state_dict(weights, strict=True)

    @staticmethod
    def extract_module_names(parameter_names: List[str]):
        """
        Extract module names from the given parameter names. A parameter name is a list of names
        connected by `.`, such as `encoder.conv1.weight`.
        """
        split_char = "."

        # Converting `encoder.conv1.weight`` to [encoder, conv1, weight]
        translator = str.maketrans("", "", string.punctuation)
        splitted_names = [
            [
                subname.translate(translator).lower()
                for subname in name.split(split_char)
            ]
            for name in parameter_names
        ]

        # With [encoder, conv1, weight], [encoder, conv1, bias], diff_idx = 1.
        diff_idx = 0
        for idx, subnames in enumerate(zip(*splitted_names)):
            if len(set(subnames)) > 1:
                diff_idx = idx
                break

        # Extract the first `diff_idx` parameter names as module names
        extracted_names = []
        for para_name in parameter_names:
            splitted_names = para_name.split(split_char)
            core_names = splitted_names[: diff_idx + 1]
            module_name = f"{split_char}".join(core_names)
            if module_name not in extracted_names:
                extracted_names.append(module_name)

        return extracted_names
