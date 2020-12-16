"""
The base class for federated learning servers.
"""

from abc import abstractmethod
from dataclasses import dataclass
import json
import sys
import os
import logging
import subprocess
import pickle
import time
import asyncio
import websockets

from config import Config
import utils.plot_figures as plot_figures


@dataclass
class RLReport:
    """RL report sent to the RL agent."""
    rl_state: list
    is_rl_episode_done: bool


class Server:
    """The base class for federated learning servers."""
    def __init__(self):
        self.clients = {}
        self.total_clients = 0
        self.selected_clients = None
        self.current_round = 0
        self.model = None
        self.accuracy = 0
        self.accuracy_list = []
        self.reports = []
        self.round_start_time = 0  # starting time of a gloabl training round
        self.training_time_list = []  # training time of each round
        self.edge_agg_num_list = [
        ]  # number of local aggregation rounds on edge servers of each global training round

        if Config().rl:
            # Parameters used by RL
            self.rl_state = None
            # Parameter of federated learning that is tuned by a RL agent
            self.rl_tuned_para_name = Config().rl.tuned_para
            self.rl_tuned_para_value = None
            self.rl_time_step = 1
            self.get_rl_tuned_para = False
            self.is_rl_episode_done = False
            self.is_rl_step_done = False

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

            if Config().rl:
                command += " -r {}".format(Config.args.rl_config)

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
            logging.info(
                '**** Local aggregation round %s/%s on edge server #%s ****',
                self.current_round,
                Config().cross_silo.rounds,
                Config().args.id)
        else:
            logging.info('**** Round %s/%s ****', self.current_round,
                         Config().training.rounds)

        self.round_start_time = time.time()

        self.choose_clients()

        if len(self.selected_clients) > 0:
            for client_id in self.selected_clients:
                socket = self.clients[client_id]
                logging.info("Selecting client #%s for training...", client_id)
                server_response = {'id': client_id, 'payload': True}
                if Config().rl:
                    server_response[
                        'rl_tuned_para_name'] = self.rl_tuned_para_name
                    server_response[
                        'rl_tuned_para_value'] = self.rl_tuned_para_value
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
                        self.accuracy_list.append(self.accuracy * 100)

                        self.training_time_list.append(time.time() -
                                                       self.round_start_time)

                        # Use accuracy as state for now
                        self.rl_state = self.accuracy

                        # Break the loop when the target accuracy is achieved
                        target_accuracy = Config().training.target_accuracy

                        if not Config().args.port:
                            self.edge_agg_num_list.append(
                                Config().cross_silo.rounds)

                            if target_accuracy and self.accuracy >= target_accuracy:
                                logging.info('Target accuracy reached.')
                                self.plot_figures_of_results()
                                await self.close_connections()
                                if Config().rl:
                                    self.is_rl_episode_done = True
                                else:
                                    sys.exit()

                            if self.current_round >= Config().training.rounds:
                                logging.info(
                                    'Target number of training rounds reached.'
                                )
                                self.plot_figures_of_results()
                                await self.close_connections()
                                if Config().rl:
                                    self.is_rl_episode_done = True
                                else:
                                    sys.exit()

                        if Config().rl and not Config().args.port:
                            self.is_rl_step_done = True

                            while not self.get_rl_tuned_para:
                                await asyncio.sleep(1)
                            self.get_rl_tuned_para = False

                        await self.select_clients()

                else:
                    # a new client arrives
                    self.register_client(client_id, websocket)

                    if self.current_round == 0 and len(
                            self.clients) >= self.total_clients:
                        logging.info('Server %s: starting FL training...',
                                     os.getpid())
                        if Config().rl and not Config().args.port:
                            while not self.get_rl_tuned_para:
                                await asyncio.sleep(1)
                            self.get_rl_tuned_para = False

                        await self.select_clients()

        except websockets.ConnectionClosed as exception:
            logging.info("Server %s: WebSockets connection closed abnormally.",
                         os.getpid())
            logging.error(exception)
            sys.exit()

    async def start_server_by_rl(self, rl_port):
        """Start the FL central server by RL agent."""

        logging.info("FL central server is contacting the RL agent...")
        uri = 'ws://{}:{}'.format(Config().server.address, rl_port)

        try:
            async with websockets.connect(uri,
                                          ping_interval=None,
                                          max_size=2**30) as websocket:
                logging.info("Central Server: Signing in at the RL agent...")
                await websocket.send(json.dumps({'greeting': 'Hi'}))

                while True:
                    logging.info(
                        "Central server is waiting for the tuned parameter of time step %s...",
                        self.rl_time_step)
                    rl_agent_response = await websocket.recv()
                    data = json.loads(rl_agent_response)

                    if data['rl_time_step'] == self.rl_time_step and 'rl_tuned_para_value' in data:
                        logging.info(
                            "Central server has received the tuned parameter of time step %s.",
                            self.rl_time_step)

                        self.rl_tuned_para_value = data['rl_tuned_para_value']

                        # The central server gets the tuned parameter,
                        # and it is ready to start this global training round.
                        self.get_rl_tuned_para = True

                        # Wait until this time step (round of global training) is done
                        while not self.is_rl_step_done:
                            await asyncio.sleep(1)

                        rl_report = RLReport(self.rl_state,
                                             self.is_rl_episode_done)
                        self.is_rl_step_done = False

                        # Send the state to the RL agent when one time step is done
                        logging.info(
                            "Central server is ready to send the state to the RL agent..."
                        )

                        # Sending time step as metadata to the server (state to follow)
                        server_update = {'rl_time_step': self.rl_time_step}
                        await websocket.send(json.dumps(server_update))
                        self.rl_time_step += 1

                        # Sending the current state to the RL agent
                        await websocket.send(pickle.dumps(rl_report))

        except OSError as exception:
            logging.info(
                "FL central server: connection to the RL agent failed.")
            logging.error(exception)

    def plot_figures_of_results(self):
        """Plot figures of results."""
        plot_figures.plot_global_round_vs_accuracy(self.accuracy_list,
                                                   self.result_dir)
        plot_figures.plot_training_time_vs_accuracy(self.accuracy_list,
                                                    self.training_time_list,
                                                    self.result_dir)

        if Config().cross_silo:
            plot_figures.plot_edge_agg_num_vs_accuracy(self.accuracy_list,
                                                       self.edge_agg_num_list,
                                                       self.result_dir)

    @abstractmethod
    def configure(self):
        """Configuring the server with initialization work."""

    @abstractmethod
    def choose_clients(self):
        """Choose a subset of the clients to participate in each round."""

    @abstractmethod
    def process_report(self):
        """Process the reports after all clients have sent them to the server."""
