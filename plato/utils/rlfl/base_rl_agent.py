"""
The base class for RL Agent for FL server.
"""

import asyncio
import logging
import os
import pickle
import sys
from abc import abstractmethod

import socketio

from plato.config import Config


class RLAgentEvents(socketio.AsyncClientNamespace):
    """ A custom namespace for socketio.AsyncServer. """
    def __init__(self, namespace, plato_rl_agent):
        super().__init__(namespace)
        self.plato_rl_agent = plato_rl_agent

    #pylint: disable=unused-argument
    async def on_connect(self):
        """ Upon a new connection to the server. """
        logging.info("[RL Agent] Connected to the server.")

    # pylint: disable=protected-access
    async def on_disconnect(self):
        """ Upon a disconnection event. """
        logging.info("[RL Agent] The server disconnected the connection.")
        await self.plato_rl_agent.close()
        os._exit(0)

    async def on_connect_error(self, data):
        """ Upon a failed connection attempt to the server. """
        logging.info("[RL Agent] A connection attempt to the server failed.")

    async def on_update_to_arrive(self, data):
        """ New update is about to arrive from the server. """
        await self.plato_rl_agent.update_to_arrive(data['env_response'])

    async def on_chunk(self, data):
        """ A chunk of data from the server arrived. """
        await self.plato_rl_agent.chunk_arrived(data['data'])

    async def on_update(self, data):
        """ A portion of the new update from the server arrived. """
        await self.plato_rl_agent.update_arrived(data['agent'])

    async def on_update_done(self, data):
        """ All of the new update sent from the server arrived. """
        await self.plato_rl_agent.update_done(data['agent'])


class RLAgent:
    """ A basic RL Agent for federated learning. """
    def __init__(self) -> None:
        self.agent = 'base'
        self.sio = None
        self.chunks = []
        self.server_update = None

    async def start_agent(self) -> None:
        """ Startup function for agent. """

        await asyncio.sleep(5)
        logging.info("[RL Agent] Contacting the central server.")

        self.sio = socketio.AsyncClient(reconnection=True)
        self.sio.register_namespace(
            RLAgentEvents(namespace='/', plato_rl_agent=self))

        uri = ""
        if hasattr(Config().server, 'use_https'):
            uri = 'https://{}'.format(Config().server.address)
        else:
            uri = 'http://{}'.format(Config().server.address)

        uri = '{}:{}'.format(uri, Config().server.port)

        logging.info("[RL Agent] Connecting to the server at %s.", uri)
        await self.sio.connect(uri)
        await self.sio.emit('agent_alive', {
            'agent': self.agent,
            'current_rl_episode': self.current_episode
        })

        logging.info("[RL Agent] Waiting to be updated with new state.")
        await self.sio.wait()

    async def update_to_arrive(self, response) -> None:
        """ Upon receiving a response from the server. """
        self.process_env_response(response)

        logging.info("[RL Agent] Updated by the server.")

    async def chunk_arrived(self, data) -> None:
        """ Upon receiving a chunk of data from the server. """
        self.chunks.append(data)

    async def update_arrived(self, data) -> None:
        """ Upon receiving a portion of the update from the server. """

        update = b''.join(self.chunks)
        _data = pickle.loads(update)
        self.chunks = []

        if self.server_update is None:
            self.server_update = _data
        elif isinstance(self.server_update, list):
            self.server_update.append(_data)
        else:
            self.server_update = [self.server_update]
            self.server_update.append(_data)

    async def update_done(self, agent) -> None:
        """ Upon receiving all the update from the server. """
        update_size = 0

        if isinstance(self.server_update, list):
            for _data in self.server_update:
                update_size += sys.getsizeof(pickle.dumps(_data))
        elif isinstance(self.server_update, dict):
            for key, value in self.server_update.items():
                update_size += sys.getsizeof(pickle.dumps({key: value}))
        else:
            update_size = sys.getsizeof(pickle.dumps(self.server_update))

        logging.info(
            "[RL Agent] Received %s B of update data from the server.",
            round(update_size, 2))

        # Update state, reward, done, info... from FL env
        self.process_env_update()
        self.server_update = None

        await self.prep_agent_update()

    async def send_in_chunks(self, data) -> None:
        """ Sending a bytes object in fixed-sized chunks to the server. """
        step = 1024 ^ 2
        chunks = [data[i:i + step] for i in range(0, len(data), step)]

        for chunk in chunks:
            await self.sio.emit('agent_chunk', {'data': chunk})

        await self.sio.emit('agent_update', {'agent': self.agent})

    async def send_update(self, update) -> None:
        """ Sending the agent update to the server using socket.io. """
        if isinstance(update, list):
            data_size: int = 0

            for data in update:
                _data = pickle.dumps(data)
                await self.send_in_chunks(_data)
                data_size += sys.getsizeof(_data)
        else:
            _data = pickle.dumps(update)
            await self.send_in_chunks(_data)
            data_size = sys.getsizeof(_data)

        await self.sio.emit('agent_update_done', {'agent': self.agent})

        logging.info("[RL Agent] Sent %s B of update data to the server.",
                     round(data_size, 2))

    @abstractmethod
    def process_env_response(self, response) -> None:
        """ Additional RL-specific processing upon the server response. """

    @abstractmethod
    def process_env_update(self) -> None:
        """ Process update from FL Env to RL Agent. """

    @abstractmethod
    async def prep_agent_update(self):
        """ Update RL Agent. """
