"""
Implementation of the server for the FedEMA .

Note:
    Divergence is abbreviated as divg
"""
import os
import logging

from plato.config import Config

from pflbases import fedavg_personalized_server

from moving_average import ModelEMA


class Server(fedavg_personalized_server.Server):
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

        # the lambda used in the paper
        self.clients_divg_scale = {}

        # whether to compute the divergence scale adaptively
        # False: use `default_genrlz_divg_scale`
        self.adaptive_divg_scale = False

        self.default_divg_scale = 0.0
        self.tau = 0.0
        self.divg_divg_before_round = 1
        self.initial_clients_divergence()

    def initial_clients_divergence(self):
        """Initial the clients' lambda by assiging the
        0.0 float values."""

        total_clients = Config().clients.total_clients
        self.clients_divg_scale = {
            client_id: 0.0 for client_id in range(1, total_clients + 1)
        }
        self.tau = (
            0.7
            if not hasattr(Config().algorithm, "divergence_scale_tau")
            else Config().algorithm.divergence_scale_tau
        )
        self.adaptive_divg_scale = (
            False
            if not hasattr(Config().algorithm, "adaptive_divergence_scale")
            else Config().algorithm.adaptive_divergence_scale
        )
        # compute the scale before which round
        self.divg_divg_before_round = (
            1
            if not hasattr(Config().algorithm, "compute_scale_before_round")
            else Config().algorithm.compute_scale_before_round
        )
        # if the personalized divergence scale is set to be constant
        # then all clients share the same scale
        if not self.adaptive_divg_scale:
            # must provide the default value in the config file
            default_scale = Config().algorithm.default_divergence_scale
            self.clients_divg_scale = {
                client_id: default_scale for client_id in range(1, total_clients + 1)
            }

    def compute_divergence_scales(self, updates, clients_id):
        """Compute the divergence scale based on the distance between
        the updated local model and the aggregated global model.
        """
        encoder_modules_name = Config().algorithm.encoder_modules_name

        logging.info("[Server #%d] Computing divergence scales.", os.getpid())

        for client_update in updates:
            client_parameters = client_update.payload
            client_id = client_update.report.client_id

            if client_id not in clients_id:
                continue

            aggregated_encoder = self.algorithm.extract_weights(
                modules_name=encoder_modules_name
            )
            client_encoder = self.algorithm.get_target_weights(
                model_parameters=client_parameters, modules_name=encoder_modules_name
            )

            # the global L2 norm over a list of tensors.
            l2_distance = ModelEMA.get_parameters_diff(
                parameter_a=aggregated_encoder,
                parameter_b=client_encoder,
            )

            client_divg_scale = self.tau / l2_distance

            self.clients_divg_scale[client_id] = client_divg_scale

    def get_computation_clients(self, updates):
        """Get the clients id required to compute the divergence rate."""
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

        return do_clients_id

    def weights_aggregated(self, updates):
        """Get client divergence based on the aggregated weights and
        the client's update.
        """
        # get the clients id required to compute the divergence rate
        clients_id = self.get_computation_clients(updates)
        self.compute_divergence_scales(updates, clients_id)

    def customize_server_payload(self, payload):
        """Insert the divergence scale into the server payload."""
        client_scale = self.clients_divg_scale[self.selected_client_id]

        return [payload, client_scale]
