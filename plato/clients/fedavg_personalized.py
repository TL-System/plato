"""
A personalized federated learning client that saves its local layers before
sending the shared global model to the server after local training.
"""
from collections import OrderedDict

from plato.clients import simple
from plato.config import Config


class Client(simple.Client):
    """
    A personalized federated learning client that saves its local layers before sending the
    shared global model to the server after local training.
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
