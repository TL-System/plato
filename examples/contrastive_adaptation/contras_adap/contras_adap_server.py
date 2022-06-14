"""
The implementation for FedEMA's server.

This performs as a backup for potentially latter usage.

Note:
    Generalization is abbreviated as genrlz
    Divergence is abbreviated as divg
"""

import logging

import torch

from plato.config import Config
from plato.servers import fedavg_ssl_base as ssl_server


class Server(ssl_server.Server):
    """A personalized federated learning server using the FedEMA method."""

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)

        # the personalized lambda
        #   - dominated by the generalization
        #   - computed for controlling the moving average update for the
        #     local ang the global online networks
        self.clients_genrlz_divg_scale = {}

        # whether to compute the generalization scale adaptively
        # False means using the default generalization_scale for all
        # clients.
        self.adaptive_genrlz_divg_scale = Config(
        ).server.adaptive_generalization_scale

        self.genrlz_divg_autoscaler_anchor = 0.0
        self.genrlz_divg_autoscaler_round_interval = 0
        self.initial_clients_generalization_scale()

    def initial_clients_generalization_scale(self):
        """ Initial the clients' personalized lambda by assiging the
            0.0 float values. """

        total_clients = Config().clients.total_clients
        # if the personalized generalization scale is set to be constant
        # then all clients share the same scale
        if not self.adaptive_genrlz_divg_scale:
            # must provide the default value in the config file
            default_scale = Config().server.default_generalization_scale
            self.clients_genrlz_divg_scale = {
                client_id: default_scale
                for client_id in range(1, total_clients + 1)
            }
        else:
            # the scale should be computed based on the metric
            # with generalization_autoscaler_anchor
            self.genrlz_divg_autoscaler_anchor = Config(
            ).server.generalization_autoscaler_anchor
            self.genrlz_divg_autoscaler_round_interval = Config(
            ).server.generalization_autoscaler_round_interval

            self.clients_genrlz_divg_scale = {
                client_id: 0.0
                for client_id in range(1, total_clients + 1)
            }

    def compute_generalization_scales(self, updates, to_compute_clients_id):
        """ Compute the generalization scale based on the distance between
            the updated local model and the aggregated global model.

            Note: This deltas computation part of this function can be
                implemeted by using the:

                directly.
        """
        # before performing this function
        # the global aggregation should be completed

        # Then, we computed the deltas between the clients' update
        # and the aggregated model by using the algorithm's  compute_weight_deltas
        # function directly.
        # the type of obtained deltas is a list, in which each item
        # is a OrderDict containing the delta between the update and the
        # the global model. Key of this dict is the name of the parameter.
        deltas = self.algorithm.compute_weight_deltas(updates)

        for i, client_update in enumerate(updates):
            (client_id, __, _, __) = client_update
            if client_id not in to_compute_clients_id:
                continue

            update_global_delta = deltas[i]

            # the global L2 norm over a list of tensors.
            # 1. compute l2 norm for each parameter
            params_l2 = torch.stack([
                torch.linalg.norm(weight_delta, ord=2)
                for _, weight_delta in update_global_delta.items()
            ])
            # 2. compute the global l2 norm
            update_global_distance = torch.linalg.norm(params_l2, ord=2)

            client_divg_scale = self.genrlz_divg_autoscaler_anchor / update_global_distance

            self.clients_genrlz_divg_scale[client_id] = client_divg_scale

    def is_perform_generalization_scale_amend(self, updated_clients_id):
        """ Whether performing the divergece scales update computation. """

        # whether to perfrom the amend
        to_perfome_amend = False
        # which clients' scales are required to be amended.
        to_amend_clients_id = []
        # the updating clients id that have not been computed and assigned
        # the divergecen scale
        to_amend_clients_id = [
            client_id for client_id in updated_clients_id
            if self.clients_genrlz_divg_scale[client_id] == 0
        ]

        # if the generalization scale is desired to be computed and
        # the there existed at least one client in updating clients
        # has not been computed
        # if generalization_autoscaler_round_interval is set to be 0
        #   the convergence scale of all clients will be only amended
        #   once when they have not been amended.
        if self.genrlz_divg_autoscaler_round_interval == 0:
            if not to_amend_clients_id:
                to_perfome_amend = True
        else:
            # else, the clients' dgeneralization scales are desired to be
            # amended continueoulsy
            to_perfome_amend = self.current_round % self.genrlz_divg_autoscaler_round_interval == 0
            to_amend_clients_id = updated_clients_id

        return to_perfome_amend, to_amend_clients_id

    def amend_generalization_scales(self, updates):
        """ Update clients' generalization scales. """
        # determine whether to perform the computation
        to_perfome_amend = False
        if self.adaptive_genrlz_divg_scale:
            updated_clients_id = [
                client_id for (client_id, __, _, __) in updates
            ]

            to_perfome_amend, to_amend_clients_id = self.is_perform_generalization_scale_amend(
                updated_clients_id)

        if to_perfome_amend:
            logging.info(
                f"Performing the generalization scale amend for client: {to_amend_clients_id}"
            )
            # compute the corresponding generalization scales for clients
            self.compute_generalization_scales(updates, to_amend_clients_id)

    async def aggregate_weights(self, updates):
        """Aggregate the reported weight updates from the selected clients."""
        deltas = await self.federated_averaging(updates)
        updated_weights = self.algorithm.update_weights(deltas)
        self.algorithm.load_weights(updated_weights)

        # perform the generalization_scales amend
        self.amend_generalization_scales(updates)

    async def customize_server_response(self, server_response):
        """
            The FedEMA server sends the computed generalization
            scale to the corresponding client.
        """
        # as the customize_server_response is performed for each client
        # within the loop and the client_id is assigned to the
        # self.selected_client_id, the client's corresponding
        # generalization scale can be obtained directly.
        #
        client_gen_divergence_scale = self.clients_genrlz_divg_scale[
            self.selected_client_id]

        # server sends the corresponding scale to the client
        server_response[
            "generalization_divergence_scale"] = client_gen_divergence_scale
        return server_response
