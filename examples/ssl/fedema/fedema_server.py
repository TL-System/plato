"""
Implementation of the server for the FedEMA.
"""

import logging
import os

import utils

from plato.config import Config
from plato.servers import fedavg_personalized as personalized_server


class Server(personalized_server.Server):
    """A server for FedEMA method to compute the model divergence."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

        # Set the lambda used in the paper
        self.clients_divg_scale = {
            client_id: 0.0 for client_id in range(1, self.total_clients + 1)
        }

        # Whether to compute the divergence scale adaptively.
        # Use the default one if not provided.
        self.adaptive_divg_scale = (
            False
            if not hasattr(Config().algorithm, "adaptive_divergence_scale")
            else Config().algorithm.adaptive_divergence_scale
        )
        # If the personalized divergence scale is set to be constant
        #   then all clients share the same scale
        if not self.adaptive_divg_scale:
            # must provide the default value in the config file
            default_scale = Config().algorithm.default_divergence_scale
            self.clients_divg_scale = {
                client_id: default_scale for client_id in self.clients_divg_scale
            }

    def weights_aggregated(self, updates):
        """Get client divergence based on the aggregated weights and
        the client's update.
        """
        # Get the divergence will be computed within how many rounds
        if not hasattr(Config().algorithm, "compute_scale_before_round"):
            divg_before_round = 1
        else:
            divg_before_round = Config().algorithm.compute_scale_before_round

        # To compute the divergence scale adaptively
        # and within the computing rounds
        if self.adaptive_divg_scale and self.current_round <= divg_before_round:
            clients_id = [update.report.client_id for update in updates]

            # Compute the divergence scale based on the distance between
            # the updated local model and the aggregated global model
            encoder_layer_names = Config().algorithm.encoder_layer_names

            logging.info("[Server #%d] Computing divergence scales.", os.getpid())

            for client_update in updates:
                client_parameters = client_update.payload
                client_id = client_update.report.client_id

                if client_id not in clients_id:
                    continue

                aggregated_encoder = utils.extract_encoder(
                    self.algorithm.model.state_dict(), encoder_layer_names
                )

                client_encoder = utils.extract_encoder(
                    model_layers=client_parameters,
                    encoder_layer_names=encoder_layer_names,
                )

                # Compute L2 norm between the aggregated encoder
                # and client encoder
                l2_distance = utils.get_parameters_diff(
                    parameter_a=aggregated_encoder,
                    parameter_b=client_encoder,
                )

                if not hasattr(Config().algorithm, "divergence_scale_tau"):
                    tau = 0.7
                else:
                    tau = Config().algorithm.divergence_scale_tau
                client_divg_scale = tau / l2_distance

                # Assign the divergence scale to the client
                self.clients_divg_scale[client_id] = client_divg_scale

    def customize_server_payload(self, payload):
        """Insert the divergence scale into the server payload."""
        client_scale = self.clients_divg_scale[self.selected_client_id]

        return [payload, client_scale]
