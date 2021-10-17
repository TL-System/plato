"""
A federated learning server with RL Agent.
"""

import logging
import os
import pickle
import sys
import time
from abc import abstractmethod

import socketio
import torch
import torch.nn.functional as F
from aiohttp import web

from plato.algorithms import registry as algorithms_registry
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.models import registry as models_registry
from plato.servers import base, fedavg
from plato.trainers import registry as trainers_registry
from plato.utils import csv_processor, s3


class RLServerEvents(base.ServerEvents):
    """ A custom namespace for socketio.AsyncServer. """

    #pylint: disable=unused-argument
    async def on_agent_alive(self, sid, data):
        """ RL Agent arrived or it sends a heartbeat. """
        await self.plato_server.register_agent(sid, data['agent'],
                                               data['current_rl_episode'])

    async def on_agent_dead(self, sid, data):
        """ RL training or playing finished. """
        await self.plato_server.close()

    async def on_env_reset(self, sid, data):
        """ RL Agent arrived or it sends a heartbeat. """
        await self.plato_server.reset_env(sid, data['current_episode'])

    async def on_update_to_arrive(self, sid, data):
        """ RL Agent sends a new report. """
        await self.plato_server.agent_update_to_arrive(sid,
                                                       data['agent_response'])

    async def on_agent_chunk(self, sid, data):
        """ A chunk of data from the server arrived. """
        await self.plato_server.agent_chunk_arrived(sid, data['data'])

    async def on_agent_update(self, sid, data):
        """ RL Agent sends a new update. """
        await self.plato_server.agent_update_arrived(sid, data['agent'])

    async def on_agent_update_done(self, sid, data):
        """ RL Agent finished sending its updates. """
        await self.plato_server.agent_update_done(sid, data['agent'])


class RLServer(fedavg.Server):
    """ A federated learning server with RL Agent. """
    def __init__(self, trainer=None):
        super().__init__(trainer=trainer)
        self.rl_agent = None
        self.agent_report = None
        self.agent_update = None
        self.agent_chunks = []
        self.current_rl_episode = 0
        self.action_applied = False
        self.clients_selected = False

    # Implement methods for communication between RL agent and env
    async def send_agent_in_chunks(self, data, sid) -> None:
        """ Sending a bytes object in fixed-sized chunks to the RL Agent client. """
        step = 1024 ^ 2
        chunks = [data[i:i + step] for i in range(0, len(data), step)]

        for chunk in chunks:
            await self.sio.emit('chunk', {'data': chunk}, room=sid)

        await self.sio.emit('update', {'agent': self.rl_agent['agent']},
                            room=sid)

    async def send_update(self, sid, update) -> None:
        """ Sending a new data update to the RL Agent client using socket.io. """
        data_size = 0

        if isinstance(update, list):
            for data in update:
                _data = pickle.dumps(data)
                await self.send_agent_in_chunks(_data, sid)
                data_size += sys.getsizeof(_data)

        else:
            _data = pickle.dumps(update)
            await self.send_agent_in_chunks(_data, sid)
            data_size = sys.getsizeof(_data)

        await self.sio.emit('update_done', {'agent': self.rl_agent['agent']},
                            room=sid)

        logging.info("[Server #%d] Sent %s B of update data to RL Agent.",
                     os.getpid(), round(data_size, 2))

    async def agent_update_to_arrive(self, sid, response):
        """ New update is about to arrive from the RL Agent. """
        self.process_agent_response(response)

        logging.info("[Server #%d] Updated by the RL Agent.", os.getpid())

    async def agent_chunk_arrived(self, sid, data) -> None:
        """ Upon receiving a chunk of data from a client. """
        self.agent_chunks.append(data)

    async def agent_update_arrived(self, sid, agent):
        """ Upon receiving a portion of the update from RL Agent. """
        assert len(self.agent_chunks) > 0

        update = b''.join(self.agent_chunks)
        _data = pickle.loads(update)
        self.agent_chunks = []

        if self.agent_update is None:
            self.agent_update = _data
        elif isinstance(self.agent_update, list):
            self.agent_update.append(_data)
        else:
            self.agent_update = [self.agent_update]
            self.agent_update.append(_data)

    async def agent_update_done(self, sid, agent):
        """ Upon receiving all the update from RL Agent. """
        update_size = 0

        if isinstance(self.agent_update, list):
            for _data in self.agent_update:
                update_size += sys.getsizeof(pickle.dumps(_data))
        elif isinstance(self.agent_update, dict):
            for key, value in self.agent_update.items():
                update_size += sys.getsizeof(pickle.dumps({key: value}))
        else:
            update_size = sys.getsizeof(pickle.dumps(self.agent_update))

        logging.info(
            "[Server #%d] Received %s B of update data from RL Agent.",
            os.getpid(), round(update_size, 2))

        # Apply action received from RL Agent
        await self.process_agent_update()

    def process_agent_response(self, response):
        """ Additional RL-specific processing upon the RL Agent response. """
        if 'current_step' in response:
            assert self.current_round + 1 == response['current_step']
        if 'current_episode' in response:
            assert self.current_rl_episode == response['current_episode']

    async def process_agent_update(self):
        """ Process action update to FL Env. """
        logging.info("[Server #%d] Applying RL action...")
        self.apply_action()
        self.action_applied = True
        self.agent_update = None

        # Carry on with the current round of FL training using RL action
        await self.step()

    async def step(self):
        """ Carry on with the current round of FL training using RL action. """
        if self.action_applied and not self.clients_selected and len(
                self.clients) >= self.clients_per_round:
            logging.info(
                "[Server #%d] Starting training for episode %d timestep %d.",
                os.getpid(), self.current_rl_episode, self.current_round + 1)
            await self.select_clients()
            self.clients_selected = True

        # Continue the round after client updates are prepared
        if self.action_applied and self.clients_selected and len(
                self.updates) > 0 and len(self.updates) >= len(
                    self.selected_clients):
            logging.info(
                "[Server #%d] All %d client reports received. Processing.",
                os.getpid(), len(self.updates))
            await self.process_reports()
            await self.wrap_up()
            self.action_applied = False
            self.clients_selected = False

    async def prep_env_update(self):
        """ Update the FL Env for the next time step. """
        env_response = {'current_round': self.current_round}
        env_response['current_rl_episode'] = self.current_rl_episode
        env_response = await self.customize_env_response(env_response)

        # Sending the response as metadata to the RL Agent (update to follow)
        await self.sio.emit('update_to_arrive', {'env_response': env_response},
                            room=self.rl_agent['sid'])

        # Sending the env update to RL Agent
        state = self.prep_state()

        logging.info(
            "[Server #%d] Sending the current state at episode %d timestep %d to RL Agent.",
            os.getpid(), self.current_rl_episode, self.current_round)
        await self.send_update(self.rl_agent['sid'], state)

    async def customize_env_response(self, response):
        """ Wrap up generating the env response with any additional information. """
        return response

    # Implement RL-related methods of simple RL server
    async def register_agent(self, sid, agent, current_rl_episode):
        """ Add an RL agent to the server's contacts. """
        if not self.rl_agent:
            self.rl_agent = {
                'agent': agent,
                'sid': sid,
                'last_contacted': time.perf_counter()
            }
        self.current_rl_episode = current_rl_episode
        logging.info("[Server #%d] RL Agent arrived.", os.getpid())

        # Initial state for the 1st RL control episode
        await self.prep_env_update()

    async def reset_env(self, sid, current_episode):
        """ Reboot for the following episodes. """
        self.current_rl_episode = current_episode
        self.configure()
        # Wrap up current env
        await self.wrap_up()

    def prep_state(self):
        """ Wrap up the state update to RL Agent. """
        return None

    @abstractmethod
    def apply_action(self):
        """ Apply action update from RL Agent to FL Env. """

    # Override the response for clients.
    # Let RL agent take control of the training rounds.
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
        # Rewrite the namespace
        self.sio.register_namespace(
            RLServerEvents(namespace='/', plato_server=self))

        if hasattr(Config().server, 's3_endpoint_url'):
            self.s3_client = s3.S3()

        app = web.Application()
        self.sio.attach(app)
        web.run_app(app, host=Config().server.address, port=port)

    async def register_client(self, sid, client_id):
        """ Adding a newly arrived client to the list of clients. """
        if not client_id in self.clients:
            # The last contact time is stored for each client
            self.clients[client_id] = {
                'sid': sid,
                'last_contacted': time.perf_counter()
            }
            logging.info("[Server #%d] New client with id #%d arrived.",
                         os.getpid(), client_id)
        else:
            self.clients[client_id]['last_contacted'] = time.perf_counter()
            logging.info("[Server #%d] New contact from Client #%d received.",
                         os.getpid(), client_id)

        await self.step()

    async def client_payload_done(self, sid, client_id, object_key):
        """ Upon receiving all the payload from a client, either via S3 or socket.io. """
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

        self.reporting_clients.append(client_id)
        del self.training_clients[client_id]

        await self.step()

    async def client_disconnected(self, sid):
        """ When a client disconnected it should be removed from its internal states. """
        for client_id, client in dict(self.clients).items():
            if client['sid'] == sid:
                del self.clients[client_id]

                if client_id in self.training_clients:
                    del self.training_clients[client_id]

                logging.info(
                    "[Server #%d] Client #%d disconnected and removed from this server.",
                    os.getpid(), client_id)

                if client_id in self.selected_clients:
                    self.selected_clients.remove(client_id)

                    await self.step()

    async def wrap_up(self):
        """ Wrapping up when each round of training is done. """
        # Loop is controlled by RL Agent instead
        # Update state to RL agent at end of each round
        await self.prep_env_update()

    async def close_connections(self):
        """ Closing all socket.io connections after training completes. """
        for client_id, client in dict(self.clients).items():
            logging.info("Closing the connection to client #%d.", client_id)
            await self.sio.emit('disconnect', room=client['sid'])
        logging.info("Closing the connection to RL Agent.")
        await self.sio.emit('disconnect', room=self.rl_agent['sid'])

    def configure(self):
        """ Booting the federated learning server by setting up 
        the data, model, and creating the clients. """
        logging.info("[Server #%d] Configuring the server for episode %d",
                     os.getpid(), self.current_rl_episode)

        self.current_round = 0

        self.load_trainer()

        if not Config().clients.do_test:
            dataset = datasources_registry.get(client_id=0)
            self.testset = dataset.get_test_set()

        # Initialize the csv file which will record results
        if self.current_rl_episode == 1 and hasattr(Config(), 'results'):
            result_csv_file = Config().result_dir + 'result.csv'
            csv_processor.initialize_csv(result_csv_file, self.recorded_items,
                                         Config().result_dir)

    def load_trainer(self):
        """ Setting up the global model to be trained via federated learning. """
        if self.trainer is None:
            self.trainer = trainers_registry.get(model=self.model)

        self.trainer.set_client_id(0)

        # Reset model for new episode
        self.trainer.model = models_registry.get()

        self.algorithm = algorithms_registry.get(self.trainer)

    async def periodic_task(self):
        """ A periodic task that is executed from time to time, determined by
        'server:periodic_interval' in the configuration. """
        # Call the async function that defines a customized periodic task, if any
        _task = getattr(self, "customize_periodic_task", None)
        if callable(_task):
            await self.customize_periodic_task()

        # If we are operating in asynchronous mode, aggregate the model updates received so far.
        if hasattr(Config().server,
                   'synchronous') and not Config().server.synchronous:
            if len(self.updates) > 0:
                logging.info(
                    "[Server #%d] %d client reports received in asynchronous mode. Processing.",
                    os.getpid(), len(self.updates))
                if self.action_applied and not self.clients_selected:
                    await self.select_clients()
                    self.clients_selected = True
                if self.action_applied and self.clients_selected:
                    await self.process_reports()
                    await self.wrap_up()
                    self.action_applied = False
                    self.clients_selected = False
            else:
                logging.info(
                    "[Server #%d] No client reports have been received. Nothing to process."
                )
