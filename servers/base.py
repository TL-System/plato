"""
The base class for federated learning servers.
"""

from abc import abstractmethod
import json
import sys
import os
import logging
import subprocess
import pickle
import asyncio
import websockets

from config import Config


class Server:
    """The base class for federated learning servers."""
    def __init__(self):
        self.clients = {}
        self.total_clients = 0
        self.selected_clients = None
        self.current_round = 0
        self.model = None
        self.accuracy = 0
        self.reports = []

        # Parameter used by cross-silo FL
        if Config().cross_silo:
            # Number of local aggregation rounds on edge servers
            # of the current global training round
            self.edge_agg_num = Config().cross_silo.rounds
            # This is a flag to prevent an edge server continues choosing clients
            # to do local aggregation when global training is done
            self.is_selected_by_central = False

        # Directory of results (figures etc.)
        self.result_dir = './results/' + Config(
        ).training.dataset + '/' + Config().training.model + '/'

    def register_client(self, client_id, websocket):
        """Adding a newly arrived client to the list of clients."""
        if not client_id in self.clients:
            self.clients[client_id] = websocket

    def unregister_client(self, websocket):
        """Removing an existing client from the list of clients."""
        for key, value in dict(self.clients).items():
            if value == websocket:
                del self.clients[key]

    def start_clients(self, as_server=False):
        """Starting all the clients as separate processes."""
        starting_id = 1

        if as_server:
            total_processes = Config().cross_silo.total_silos
            starting_id += Config().clients.total_clients
        else:
            total_processes = Config().clients.total_clients

        for client_id in range(starting_id, total_processes + starting_id):
            logging.info("Starting client #%s...", client_id)
            command = "python client.py -i {}".format(client_id)
            command += " -c {}".format(Config.args.config)

            if as_server:
                command += " -p {}".format(Config().server.port + client_id)
                logging.info("This client #%s is an edge server.", client_id)

            subprocess.Popen(command, shell=True)

    async def close_connections(self):
        """Closing all WebSocket connections after training completes."""
        for _, client_socket in dict(self.clients).items():
            await client_socket.close()

    async def select_clients(self):
        """Select a subset of the clients and send messages to them to start training."""
        self.reports = []
        self.current_round += 1
        if Config().args.id:
            # To train enough rounds of local aggregations on an edge server,
            # in clients/edges.py line 39-40, we let the edge server wait for a while until
            # reaching target number of local aggregations.
            # The waiting interval is 1 second.

            # However, if a local aggregation can be done in less than 1 second,
            # it would happen that the edge server starts a new round of local aggregation
            # by calling select_clients()
            # even when it reaches target number of local aggregations in that 1 second.
            # Therefore, here we use self.current_round > self.edge_agg_num to check
            # if it already runs enough rounds of local aggregations.

            # But using this condition is not enough. There is another bad situation where
            # during the 1 second waiting due to self.current_round > self.edge_agg_num,
            # a new round of global training starts and this edge server is selected.
            # However, now the self.current_round is 0 due to clients/edges.py line 42,
            # and cannot be added 1 on the above line 86 before choose_clients()
            # for its first local aggregation round.
            # In this case, we need to add 1 to self.current_round (line 111),
            # or this edge server will run one more round of local aggregations.
            while self.current_round > self.edge_agg_num or not self.is_selected_by_central:
                await asyncio.sleep(1)
            if self.is_selected_by_central and self.current_round == 0:
                self.current_round += 1

            logging.info(
                '**** Local aggregation round %s/%s on edge server (client #%s) ****',
                self.current_round, self.edge_agg_num,
                Config().args.id)
        else:
            logging.info('**** Round %s/%s ****', self.current_round,
                         Config().training.rounds)

        self.choose_clients()

        if len(self.selected_clients) > 0:
            for client_id in self.selected_clients:
                socket = self.clients[client_id]
                logging.info("Selecting client #%s for training...", client_id)
                server_response = {'id': client_id, 'payload': True}
                if Config().rl and not Config().args.id:
                    server_response = await self.generate_rl_info(
                        server_response)
                await socket.send(json.dumps(server_response))

                logging.info("Sending the current model...")
                await socket.send(pickle.dumps(self.model.state_dict()))

    async def serve(self, websocket, path):
        """Running a federated learning server."""

        try:
            async for message in websocket:
                data = json.loads(message)
                client_id = data['id']
                logging.info("Server %s: Data received from client #%s",
                             os.getpid(), client_id)

                if 'payload' in data:
                    # an existing client reports new updates from local training
                    client_update = await websocket.recv()
                    report = pickle.loads(client_update)
                    logging.info(
                        "Server {}: Update from client #{} received. Accuracy = {:.2f}%\n"
                        .format(os.getpid(), client_id, 100 * report.accuracy))

                    self.reports.append(report)

                    if len(self.reports) == len(self.selected_clients):
                        self.accuracy = self.process_report()

                        await self.wrap_up_one_round()

                        # Break the loop when the target accuracy is achieved
                        target_accuracy = Config().training.target_accuracy

                        if not Config().args.port:
                            if target_accuracy and self.accuracy >= target_accuracy:
                                logging.info('Target accuracy reached.')
                                self.wrap_up()
                                await self.close_connections()
                                sys.exit()

                            if self.current_round >= Config().training.rounds:
                                logging.info(
                                    'Target number of training rounds reached.'
                                )
                                self.wrap_up()
                                await self.close_connections()
                                sys.exit()

                        await self.select_clients()

                else:
                    # a new client arrives
                    self.register_client(client_id, websocket)

                    if self.current_round == 0 and len(
                            self.clients) >= self.total_clients:
                        logging.info('Server %s: starting FL training...',
                                     os.getpid())
                        await self.select_clients()
        except websockets.ConnectionClosed as exception:
            logging.info("Server %s: WebSockets connection closed abnormally.",
                         os.getpid())
            logging.error(exception)
            sys.exit()

    def wrap_up(self):
        """Wrapping up when the training is done."""

    @abstractmethod
    def configure(self):
        """Configuring the server with initialization work."""

    @abstractmethod
    def choose_clients(self):
        """Choose a subset of the clients to participate in each round."""

    @abstractmethod
    def process_report(self):
        """Process the reports after all clients have sent them to the server."""

    @abstractmethod
    async def wrap_up_one_round(self):
        """Wrapping up when one round of training is done."""
