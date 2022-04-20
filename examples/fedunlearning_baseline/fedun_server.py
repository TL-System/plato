"""
A customized server for federated unlearning.

Reference: Liu et al., "The Right to be Forgotten in Federated Learning: An Efficient Realization with Rapid Retraining." in Proc. INFOCOM, 2022 https://arxiv.org/abs/2203.07320

"""
import logging
import pickle
import torch
import torch.nn.functional as F

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """ A federated unlearning server using the Fedun baseline server. """

    def __init__(self):
        super().__init__()

    async def select_clients(self):
        """ Select a subset of the clients and send messages to them to start training. """
        self.updates = []
        self.current_round += 1

        logging.info("\n[%s] Starting round %s/%s.", self, self.current_round,
                     Config().trainer.rounds)

        if Server.client_simulation_mode:
            # In the client simulation mode, the client pool for client selection contains
            # all the virtual clients to be simulated
            self.clients_pool = list(range(1, 1 + self.total_clients))
            if Config().is_central_server():
                self.clients_pool = list(
                    range(Config().clients.per_round + 1,
                          Config().clients.per_round + 1 + self.total_clients))

        else:
            # If no clients are simulated, the client pool for client selection consists of
            # the current set of clients that have contacted the server
            self.clients_pool = list(self.clients)

        # In asychronous FL, avoid selecting new clients to replace those that are still
        # training at this time

        # When simulating the wall clock time, if len(self.reported_clients) is 0, the
        # server has aggregated all reporting clients already
        if self.asynchronous_mode and self.selected_clients is not None and len(
                self.reported_clients) > 0 and len(
                    self.reported_clients) < self.clients_per_round:
            # If self.selected_clients is None, it implies that it is the first iteration;
            # If len(self.reported_clients) == self.clients_per_round, it implies that
            # all selected clients have already reported.

            # Except for these two cases, we need to exclude the clients who are still
            # training.
            training_client_ids = [
                self.training_clients[client_id]['id']
                for client_id in list(self.training_clients.keys())
            ]

            # If the server is simulating the wall clock time, some of the clients who
            # reported may not have been aggregated; they should be excluded from the next
            # round of client selection
            reporting_client_ids = [
                client[1]['client_id'] for client in self.reported_clients
            ]

            selectable_clients = [
                client for client in self.clients_pool
                if client not in training_client_ids
                and client not in reporting_client_ids
            ]

            if self.simulate_wall_time:
                self.selected_clients = self.choose_clients(
                    selectable_clients, len(self.current_processed_clients))
            else:
                self.selected_clients = self.choose_clients(
                    selectable_clients, len(self.reported_clients))
        else:
            self.selected_clients = self.choose_clients(
                self.clients_pool, self.clients_per_round)

        self.current_reported_clients = {}
        self.current_processed_clients = {}

        # There is no need to clear the list of reporting clients if we are
        # simulating the wall clock time on the server. This is because
        # when wall clock time is simulated, the server needs to wait for
        # all the clients to report before selecting a subset of clients for
        # replacement, and all remaining reporting clients will be processed
        # in the next round
        if not self.simulate_wall_time:
            self.reported_clients = []

        if len(self.selected_clients) > 0:
            self.selected_sids = []

            for i, selected_client_id in enumerate(self.selected_clients):
                self.selected_client_id = selected_client_id

                if self.client_simulation_mode:
                    client_id = i + 1
                    if Config().is_central_server():
                        client_id = selected_client_id

                    sid = self.clients[client_id]['sid']

                    if self.asynchronous_mode and self.simulate_wall_time:
                        training_sids = []
                        for client_info in self.reported_clients:
                            training_sids.append(client_info[1]['sid'])

                        # skip if this sid is currently `training' with reporting clients
                        # or it has already been selected in this round
                        while sid in training_sids or sid in self.selected_sids:
                            client_id = client_id % self.clients_per_round + 1
                            sid = self.clients[client_id]['sid']

                        self.selected_sids.append(sid)
                else:
                    sid = self.clients[self.selected_client_id]['sid']

                self.training_clients[self.selected_client_id] = {
                    'id': self.selected_client_id,
                    'starting_round': self.current_round,
                    'start_time': self.wall_time,
                    'update_requested': False
                }

                logging.info("[%s] Selecting client #%d for training.", self,
                             self.selected_client_id)

                server_response = {'id': self.selected_client_id}
                server_response['current_round'] = self.current_round
                payload = self.algorithm.extract_weights()
                payload = self.customize_server_payload(payload)

                if self.comm_simulation:
                    logging.info(
                        "[%s] Sending the current model to client #%d (simulated).",
                        self, self.selected_client_id)

                    # First apply outbound processors, if any
                    payload = self.outbound_processor.process(payload)

                    model_name = Config().trainer.model_name if hasattr(
                        Config().trainer, 'model_name') else 'custom'
                    checkpoint_dir = Config().params['checkpoint_dir']

                    payload_filename = f"{checkpoint_dir}/{model_name}_{self.selected_client_id}.pth"
                    with open(payload_filename, 'wb') as payload_file:
                        pickle.dump(payload, payload_file)
                    server_response['payload_filename'] = payload_filename

                server_response = await self.customize_server_response(
                    server_response)

                # Sending the server response as metadata to the clients (payload to follow)
                await self.sio.emit('payload_to_arrive',
                                    {'response': server_response},
                                    room=sid)

                if not self.comm_simulation:
                    # Sending the server payload to the client
                    logging.info(
                        "[%s] Sending the current model to client #%d.", self,
                        selected_client_id)

                    await self.send(sid, payload, selected_client_id)

    async def aggregate_weights(self, updates):
        """Aggregate the reported weight updates from the selected clients."""
        if 
        update = await self.federated_averaging(updates)
        updated_weights = self.algorithm.update_weights(update)
        self.algorithm.load_weights(updated_weights)