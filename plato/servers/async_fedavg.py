"""
A simple federated learning server using federated averaging.
"""

import asyncio
import logging
import os
import random
import time
import sys
import pickle
import socketio
from aiohttp import web
from plato.utils import s3
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.utils import csv_processor
from plato.servers import base, fedavg


class Server(fedavg.Server):
    """Federated learning server using federated averaging."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        self.available_clients = {}
        self.selected_clients = set()
        self.current_step = 0
        self.total_steps = Config().server.async_total_steps
        self.test_freq = Config().server.async_test_freq
        self.total_clients = Config().clients.total_clients
        self.max_clients_per_step = Config().server.async_max_clients_per_step
        self.training_start_time = time.perf_counter()

        if hasattr(Config(), 'results'):
            self.recorded_items[0] = 'step'

    async def do_real_stuffs(self):
        # first performs possible aggregation
        if len(self.updates) > 0:
            logging.info(
                "[Server #%d] Received %d clients' reports. Processing.",
                os.getpid(), len(self.updates))
            await self.aggregate_weights(self.updates)

        # then performs possible testing
        if self.current_step > 0 and self.current_step % self.test_freq == 0:
            logging.info("[Server #%d] Testing started.", os.getpid())
            await self.process_aggregated_weights()
            logging.info("[Server #%d] Testing ended.", os.getpid())

        await self.wrap_up()

        # then perform possible client selection
        logging.info("[Server #%d] Starting training.", os.getpid())
        await self.select_clients()

    async def loop_over_steps(self):
        # loop over steps
        tick = -1
        seconds_per_tick = Config().server.async_seconds_per_tick
        ticks_per_step = Config().server.async_ticks_per_step
        start_time = time.time()
        while True:
            # advances the records
            tick += 1
            if tick % ticks_per_step == 0:
                logging.info("[Server #%d] Step %s/%s starts. Do real stuffs.",
                             os.getpid(), self.current_step, self.total_steps)
                await self.do_real_stuffs()
                logging.info("[Server #%d] Real stuffs done.", os.getpid())

            # accounts for the start time drift
            expected_time = tick * seconds_per_tick
            actual_time = time.time() - start_time
            start_time_drift = actual_time - expected_time

            if start_time_drift < seconds_per_tick:
                sleep_time = seconds_per_tick - start_time_drift
                await asyncio.sleep(sleep_time)

            if tick % ticks_per_step == 0:
                logging.info("[Server #%d] Step %s/%s ends.", os.getpid(),
                             self.current_step, self.total_steps)
                self.current_step += 1

    def start(self, port=Config().server.port):
        """ Start running the socket.io server. """
        logging.info("Starting a server at address %s and port %s.",
                     Config().server.address, port)

        ping_interval = Config().server.ping_interval if hasattr(
            Config().server, 'ping_interval') else 3600
        ping_timeout = Config().server.ping_timeout if hasattr(
            Config().server, 'ping_timeout') else 360
        self.sio = socketio.AsyncServer(ping_interval=ping_interval,
                                        max_http_buffer_size=2**31,
                                        ping_timeout=ping_timeout)
        self.sio.register_namespace(
            base.ServerEvents(namespace='/', plato_server=self))

        self.s3_client = None
        try:
            self.s3_client = s3.S3()
        except:
            self.s3_client = None

        current_loop = asyncio.get_event_loop()
        current_loop.create_task(self.loop_over_steps())

        app = web.Application()
        self.sio.attach(app)
        web.run_app(app, host=Config().server.address, port=port)

    async def register_client(self, sid, client_id):
        """Adding a newly arrived client to the list of clients."""
        if not client_id in self.clients:
            # The last contact time is stored for each client
            self.clients[client_id] = {
                'sid': sid,
                'last_contacted': time.perf_counter()
            }
            self.available_clients[client_id] = {
                'sid': sid,
                'available_from': time.perf_counter()
            }
            logging.info("[Server #%d] New client with id #%d arrived.",
                         os.getpid(), client_id)
        else:
            self.clients[client_id]['last_contacted'] = time.perf_counter()
            self.available_clients[client_id] = {
                'sid': sid,
                'available_from': time.perf_counter()
            }
            logging.info("[Server #%d] New contact from Client #%d received.",
                         os.getpid(), client_id)

    async def select_clients(self):
        """Select a subset of the clients and send messages to them to start training."""
        self.updates = []

        logging.info("[Server #%d] Start to select clients at Step %s/%s.",
                     os.getpid(), self.current_step, self.total_steps)

        if hasattr(Config().clients,
                   'simulation') and Config().clients.simulation:
            # In the client simulation mode, the client pool for client selection contains
            # all the virtual clients to be simulated
            raise NotImplementedError
        else:
            # If no clients are simulated, the client pool for client selection consists of
            # the current set of clients that have contacted the server
            self.clients_pool = list(self.available_clients)

        newly_selected_clients = self.choose_clients()
        for client in newly_selected_clients:
            self.selected_clients.add(client)

        if len(newly_selected_clients) > 0:
            for i, selected_client_id in enumerate(newly_selected_clients):
                if hasattr(Config().clients,
                           'simulation') and Config().clients.simulation:
                    raise NotImplementedError
                else:
                    client_id = selected_client_id

                del self.available_clients[client_id]

                sid = self.clients[client_id]['sid']

                logging.info("[Server #%d] Selecting client #%d for training.",
                             os.getpid(), selected_client_id)

                server_response = {'id': selected_client_id}
                server_response = await self.customize_server_response(
                    server_response)

                # Sending the server response as metadata to the clients (payload to follow)
                await self.sio.emit('payload_to_arrive',
                                    {'response': server_response},
                                    room=sid)

                payload = self.algorithm.extract_weights()
                payload = self.customize_server_payload(payload)

                # Sending the server payload to the client
                logging.info(
                    "[Server #%d] Sending the current model to client #%d.",
                    os.getpid(), selected_client_id)
                await self.send(sid, payload, selected_client_id)

    def configure(self):
        """
        Booting the federated learning server by setting up the data, model, and
        creating the clients.
        """

        logging.info("[Server #%d] Configuring the server...", os.getpid())

        target_accuracy = Config().trainer.target_accuracy

        if target_accuracy:
            logging.info("Training: %s steps or %s%% accuracy\n",
                         self.total_steps, 100 * target_accuracy)
        else:
            logging.info("Training: %s steps\n", self.total_steps)

        self.load_trainer()

        if not Config().clients.do_test:
            dataset = datasources_registry.get(client_id=0)
            self.testset = dataset.get_test_set()

        # Initialize the csv file which will record results
        if hasattr(Config(), 'results'):
            result_csv_file = Config().result_dir + 'result.csv'
            csv_processor.initialize_csv(result_csv_file, self.recorded_items,
                                         Config().result_dir)

    async def register_client(self, sid, client_id):
        """Adding a newly arrived client to the list of clients."""
        if not client_id in self.clients:
            # The last contact time is stored for each client
            self.clients[client_id] = {
                'sid': sid,
                'last_contacted': time.perf_counter()
            }
            self.available_clients[client_id] = {
                'sid': sid,
                'available_from': time.perf_counter()
            }
            logging.info("[Server #%d] New client with id #%d arrived.",
                         os.getpid(), client_id)
        else:
            self.clients[client_id]['last_contacted'] = time.perf_counter()
            self.available_clients[client_id] = {
                'sid': sid,
                'available_from': time.perf_counter()
            }
            logging.info("[Server #%d] New contact from Client #%d received.",
                         os.getpid(), client_id)

    async def client_payload_done(self, sid, client_id, object_key):
        """ Upon receiving all the payload from a client, eithe via S3 or socket.io. """
        if object_key is None:
            assert self.client_payload[sid] is not None

            payload_size = 0
            if isinstance(self.client_payload[sid], list):
                for _data in self.client_payload[sid]:
                    payload_size += sys.getsizeof(pickle.dumps(_data))
            else:
                payload_size = sys.getsizeof(
                    pickle.dumps(self.client_payload[sid]))
        else:
            self.client_payload[sid] = self.s3_client.receive_from_s3(
                object_key)
            payload_size = sys.getsizeof(pickle.dumps(
                self.client_payload[sid]))

        logging.info(
            "[Server #%d] Received %s MB of payload data from client #%d.",
            os.getpid(), round(payload_size / 1024**2, 2), client_id)

        self.updates.append((self.reports[sid], self.client_payload[sid]))

        self.selected_clients.remove(client_id)
        self.available_clients[client_id] = {
            'sid': sid,
            'available_from': time.perf_counter()
        }

    async def client_disconnected(self, sid):
        """ When a client disconnected it should be removed from its internal states. """
        for client_id, client in dict(self.clients).items():
            if client['sid'] == sid:
                del self.clients[client_id]
                del self.available_clients[client_id]

                logging.info(
                    "[Server #%d] Client #%d disconnected and removed from this server.",
                    os.getpid(), client_id)

                if client_id in self.selected_clients:
                    self.selected_clients.remove(client_id)

    async def wrap_up(self):
        """Wrapping up when each step of training is done."""
        # Break the loop when the target accuracy is achieved
        target_accuracy = Config().trainer.target_accuracy

        if target_accuracy and self.accuracy >= target_accuracy:
            logging.info("[Server #%d] Target accuracy reached.", os.getpid())
            await self.close()

        if self.current_step >= self.total_steps:
            logging.info("Target number of asynchronous step reached.")
            await self.close()

    def choose_clients(self):
        """Choose a subset of the clients to participate in each step."""
        # Select clients randomly
        nums_to_select = min(self.max_clients_per_step, len(self.clients_pool))
        return random.sample(self.clients_pool, nums_to_select)

    async def process_aggregated_weights(self):
        """Process the aggregated weights by testing the accuracy."""

        # Testing the global model accuracy
        if Config().clients.do_test:
            # Compute the average accuracy from client reports
            self.accuracy = self.accuracy_averaging(self.updates)
            logging.info(
                '[Server #{:d}] Average client accuracy: {:.2f}%.'.format(
                    os.getpid(), 100 * self.accuracy))
        else:
            # Testing the updated model directly at the server
            self.accuracy = await self.trainer.server_test(self.testset)

            logging.info(
                '[Server #{:d}] Global model accuracy: {:.2f}%\n'.format(
                    os.getpid(), 100 * self.accuracy))

        if hasattr(Config().trainer, 'use_wandb'):
            wandb.log({"accuracy": self.accuracy})

        await self.wrap_up_processing_reports()

    async def wrap_up_processing_reports(self):
        """Wrap up processing the reports with any additional work."""

        if hasattr(Config(), 'results'):
            new_row = []
            for item in self.recorded_items:
                item_value = {
                    'step': self.current_step,
                    'accuracy': self.accuracy * 100,
                    'elapsed_time':
                    time.perf_counter() - self.training_start_time
                }[item]
                new_row.append(item_value)

            result_csv_file = Config().result_dir + 'result.csv'

            csv_processor.write_csv(result_csv_file, new_row)