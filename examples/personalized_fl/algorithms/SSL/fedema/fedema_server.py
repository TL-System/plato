"""
Implementation of the server for the FedEMA .

Note:
    Divergence is abbreviated as divg
"""
import os
import logging

import utils
from moving_average import ModelEMA

from plato.config import Config

from pflbases import fedavg_personalized


class Server(fedavg_personalized.Server):
    """A personalized federated learning server using the pFL-CMA's EMA method."""

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

        # The lambda used in the paper
        self.clients_divg_scale = {
            client_id: 0.0 for client_id in range(1, self.total_clients + 1)
        }

        # Whether to compute the divergence scale adaptively
        # Use the default one if not provided
        self.adaptive_divg_scale = (
            False
            if not hasattr(Config().algorithm, "adaptive_divergence_scale")
            else Config().algorithm.adaptive_divergence_scale
        )
        # if the personalized divergence scale is set to be constant
        # then all clients share the same scale
        if not self.adaptive_divg_scale:
            # must provide the default value in the config file
            default_scale = Config().algorithm.default_divergence_scale
            self.clients_divg_scale = {
                client_id: default_scale for client_id in self.clients_divg_scale
            }

        self.tau = (
            0.7
            if not hasattr(Config().algorithm, "divergence_scale_tau")
            else Config().algorithm.divergence_scale_tau
        )
        # Compute the scale before which round
        self.divg_divg_before_round = (
            1
            if not hasattr(Config().algorithm, "compute_scale_before_round")
            else Config().algorithm.compute_scale_before_round
        )

    def weights_aggregated(self, updates):
        """Get client divergence based on the aggregated weights and
        the client's update.
        """
        # Get the clients id required to compute the divergence rate
        # which clients' scales are required to be computed.
        do_clients_id = []

        # if divergence is not required to be computed
        # adaptively
        if not self.adaptive_divg_scale:
            return do_clients_id
        # if the computation round has been passed
        if self.current_round > self.divg_divg_before_round:
            return do_clients_id

        do_clients_id = [update.report.client_id for update in updates]

        # Compute the divergence scale based on the distance between
        # the updated local model and the aggregated global model
        encoder_layer_names = Config().algorithm.encoder_layer_names

        logging.info("[Server #%d] Computing divergence scales.", os.getpid())

        for client_update in updates:
            client_parameters = client_update.payload
            client_id = client_update.report.client_id

            if client_id not in do_clients_id:
                continue

            aggregated_encoder = self.algorithm.extract_encoder()

            client_encoder = utils.extract_encoder(
                model_layers=client_parameters,
                encoder_layer_names=encoder_layer_names,
            )

            # the global L2 norm over a list of tensors.
            l2_distance = ModelEMA.get_parameters_diff(
                parameter_a=aggregated_encoder,
                parameter_b=client_encoder,
            )

            client_divg_scale = self.tau / l2_distance

            self.clients_divg_scale[client_id] = client_divg_scale

    def customize_server_payload(self, payload):
        """Insert the divergence scale into the server payload."""
        client_scale = self.clients_divg_scale[self.selected_client_id]

        return [payload, client_scale]
