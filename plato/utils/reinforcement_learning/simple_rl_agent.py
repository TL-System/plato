"""
A basic RL environment for FL server using Gym for RL control.
"""
import asyncio
import logging
import os
import pickle
import random
import sys
from abc import abstractmethod

import numpy as np
import socketio
from gym import spaces
from plato.config import Config
from plato.utils import csv_processor


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
    def on_disconnect(self):
        """ Upon a disconnection event. """
        logging.info("[RL Agent] The server disconnected the connection.")
        self.plato_rl_agent.close()
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


class RLAgentBase(object):
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

        if hasattr(Config().server, 'use_https'):
            uri = f'https://{Config().server.address}'
        else:
            uri = f'http://{Config().server.address}'

        uri = f'{uri}:{Config().server.port}'

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

    async def update_arrived(self, agent) -> None:
        """ Upon receiving a portion of the update from the server. """
        assert agent == self.agent

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
        assert agent == self.agent

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


class RLAgent(RLAgentBase):
    """ A basic RL environment for FL server using Gym for RL control. """
    def __init__(self):
        super().__init__()
        self.agent = 'simple'
        self.n_actions = Config().clients.per_round
        self.n_states = Config().clients.per_round * Config(
        ).algorithm.n_features

        if Config().algorithm.discrete_action_space:
            self.action_space = spaces.Discrete(self.n_actions)
        else:
            self.action_space = spaces.Box(low=int(
                Config().algorithm.min_action),
                                           high=Config().algorithm.max_action,
                                           shape=(self.n_actions, ),
                                           dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(self.n_states, ),
                                            dtype=np.float32)

        self.state = None
        self.next_state = None
        self.action = None
        self.next_action = None
        self.reward = None
        self.episode_reward = 0
        self.current_step = 0
        self.total_steps = 0
        self.current_episode = 0
        self.is_done = False

    def step(self):
        """ Update the followings using server update. """
        self.next_state = self.get_state()
        self.is_done = self.get_done()
        self.reward = self.get_reward()
        info = self.get_info()

        return self.next_state, self.reward, self.is_done, info

    async def reset(self):
        """ Reset RL environment. """
        # Start a new training session
        logging.info("[RL Agent] Reseting RL environment.")

        # Reset the episode-related variables
        self.current_step = 0
        self.is_done = False
        self.episode_reward = 0
        self.current_episode += 1

        logging.info("[RL Agent] Starting RL episode #%d.",
                     self.current_episode)

        # Reboot/reconfigure the FL server
        await self.sio.emit('env_reset',
                            {'current_episode': self.current_episode})

        return

    async def prep_action(self):
        """ Get action from RL policy. """
        logging.info("[RL Agent] Selecting action...")
        self.action = self.policy.select_action(self.state)
        return self.action

    def get_state(self):
        """ Get state for agent. """
        if self.server_update:
            return self.server_update
        # Initial state is random when env resets
        return [round(random.random(), 4) for i in range(self.n_states)]

    def get_reward(self):
        """ Get reward for agent. """
        return 0.0

    def get_done(self):
        """ Get done condition for agent. """
        if Config().algorithm.mode == 'train' and self.current_step >= Config(
        ).algorithm.steps_per_episode:
            logging.info("[RL Agent] Episode #%d ended.", self.current_episode)
            return True
        return False

    def get_info(self):
        """ Get info used for benchmarking. """
        return {}

    def close(self):
        """ Closing the RL Agent. """
        logging.info("[RL Agent] RL control concluded.")

    async def wrap_up(self):
        """ Wrap up when RL control is concluded. """
        # Close FL environment
        await self.sio.emit('agent_dead', {'agent': self.agent})

    # Implement methods for communication between RL agent and env
    def process_env_response(self, response):
        """ Additional RL-specific processing upon the server response. """
        if 'current_round' in response:
            assert self.current_step == response['current_round']
        if 'current_rl_episode' in response:
            assert self.current_episode == response['current_rl_episode']

    def process_env_update(self):
        """ Process state update to RL Agent. """
        if self.current_step == 0:
            self.state = self.get_state()
        else:
            self.step()
            if Config().algorithm.mode == 'train':
                self.process_experience()
            self.state = self.next_state
            self.episode_reward += self.reward

            step_result_csv_file = Config().results_dir + 'step_result.csv'
            csv_processor.write_csv(step_result_csv_file,
                                    [self.current_episode, self.current_step] +
                                    list(self.state) + list(self.action))

    async def prep_agent_update(self):
        """ Update RL Agent. """
        if self.is_done and Config().algorithm.mode == 'train':
            self.update_policy()

            # Break the loop when RL training is concluded
            if self.current_episode >= Config().algorithm.max_episode:
                await self.wrap_up()
            else:
                await self.reset()
        elif self.current_step >= Config().algorithm.test_step:
            # Break the loop when RL testing is concluded
            await self.wrap_up()
        else:
            self.current_step += 1
            self.total_steps += 1
            logging.info("[RL Agent] Preparing action...")
            agent_response = {'current_step': self.current_step}
            agent_response['current_episode'] = self.current_episode
            agent_response = await self.customize_agent_response(agent_response
                                                                 )

            # Sending the response as metadata to the server (update to follow)
            await self.sio.emit('update_to_arrive',
                                {'agent_response': agent_response})

            # Sending the agent update to server
            action = await self.prep_action()

            logging.info(
                "[RL Agent] Sending the current action at episode %d timestep %d to server.",
                self.current_episode, self.current_step)

            await self.send_update(action)

    async def customize_agent_response(self, response):
        """ Wrap up generating the agent response with any additional information. """
        return response

    @abstractmethod
    def update_policy(self):
        """ Update policy if needed in training mode. """

    @abstractmethod
    def process_experience(self):
        """ Process step experience if needed in training mode. """
