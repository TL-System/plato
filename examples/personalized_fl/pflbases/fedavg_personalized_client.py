"""
A base client for personalized federated learning
"""
from collections import OrderedDict

from plato.clients import simple
from plato.config import Config


class Client(simple.Client):
    """
    A base client class for personalized federated learning.
    It will save local layers during outbound ready
        after outbound payloads are processed and ready.
    """

    def outbound_ready(self, report, outbound_processor):
        super().outbound_ready(report, outbound_processor)
        weights = self.algorithm.extract_weights()

        # Save local layers before giving them to the outbound processor
        if hasattr(Config().algorithm, "local_layer_names"):
            # Extract weights of desired local layers
            local_layers = OrderedDict(
                [
                    (name, param)
                    for name, param in weights.items()
                    if any(
                        param_name in name.strip().split(".")
                        for param_name in Config().algorithm.local_layer_names
                    )
                ]
            )
            model_path = Config().params["model_path"]
            model_name = Config().trainer.model_name
            filename = f"{model_path}/{model_name}_{self.client_id}_local_layers.pth"
            self.algorithm.save_local_layers(local_layers, filename)
