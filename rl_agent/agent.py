"""
Starting point for a reinforcement learning agent.
This agent will tune a parameter for federated learning.
"""

import logging
import json
import asyncio
import sys
import time
import pickle
import websockets

from config import Config
import servers
import utils.plot_figures as plot_figures


class RLAgent:
    """A basic reinforcement learning agent."""
    def __init__(self, port):
        self.port = port
        self.episode_num = 0
        self.time_step = 1
        self.central_server = None
        self.central_server_socket = None
        self.env = None
        self.rl_tuned_para_name = None
        self.rl_tuned_para_value = None
        self.rl_state = None
        self.is_tuned_para_got = False
        self.is_rl_episode_done = False

    def configure(self, env):
        """Booting the reinforcement learning agent."""
        logging.info('Configuring the RL agent...')

        self.env = env

        self.rl_tuned_para_name = {
            'edge_agg_num': 'number of aggregations on edge servers',
        }[Config().rl.tuned_para]
        logging.info("This RL agent will tune the %s.",
                     self.rl_tuned_para_name)

        total_episodes = Config().rl.episodes
        target_reward = Config().rl.target_reward

        if target_reward:
            logging.info('RL Training: %s episodes or %s%% reward\n',
                         total_episodes, 100 * target_reward)
        else:
            logging.info('RL Training: %s episodes\n', total_episodes)

    async def start_central_server(self):
        """Starting a Fl central server as a separate process."""

        logging.info("Starting a FL central server on port %s\n",
                     Config().server.port)

        loop = asyncio.get_event_loop()
        coroutines = []

        try:
            # Running the FL central server and clients concurrently
            server = {
                "fedavg": servers.fedavg.FedAvgServer
            }[Config().server.type]()

            server.configure()

            self.central_server = server

            coroutines.append(self.central_server.start_server_by_rl(
                self.port))

            start_server = websockets.serve(server.serve,
                                            Config().server.address,
                                            Config().server.port,
                                            ping_interval=None,
                                            max_size=2**30)

            coroutines.append(start_server)

            coroutines.append(self.start_clients(server))
            #loop.run_until_complete(self.start_clients(server))

            asyncio.gather(*coroutines)
            #loop.run_until_complete(asyncio.gather(*coroutines))

        except websockets.ConnectionClosed:
            logging.info("FL server: connection to the RL agent is closed.")
            sys.exit()

    async def start_clients(self, server):
        """Start FL clients."""
        # Allowing some time for the central server to start
        time.sleep(5)

        if Config().cross_silo:
            # Start clients that are edge servers
            server.start_clients(as_server=True)
            # Allowing some time for the edge servers to start
            await asyncio.sleep(5)

        # Start clients that are not edge servers
        server.start_clients()

    async def serve(self, websocket, path):
        """Running the reinforcement learning agent."""
        try:
            async for message in websocket:
                data = json.loads(message)
                logging.info("RL Agent: Data received from central server.")

                if 'rl_time_step' in data:
                    if data['rl_time_step'] == self.time_step:
                        # The received message contains info of FL training.
                        finished_time_step = data['rl_time_step']
                        self.time_step += 1

                        # A time step (one round of FL global training is finished)
                        server_update = await websocket.recv()
                        fl_info = pickle.loads(server_update)
                        logging.info(
                            "RL Agent: Update from central server of time step %s received.",
                            finished_time_step)

                        self.rl_state = fl_info.rl_state
                        self.is_rl_episode_done = fl_info.is_rl_episode_done

                        self.env.get_state_from_rl_agent(
                            self.rl_state, self.is_rl_episode_done)

                        # Wait until env comes up with the tuned para
                        while not self.is_tuned_para_got:
                            await asyncio.sleep(1)
                        self.is_tuned_para_got = False

                        # Send the tuned parameter to the central server
                        await self.send_tuned_para()

                        if self.is_rl_episode_done:
                            self.time_step = 1
                            self.episode_num += 1

                            # Break the loop when the target number of episodes is achieved
                            if self.episode_num >= Config().rl.episodes:
                                logging.info(
                                    'RL Agent: Target number of training episodes reached.'
                                )
                                self.plot_rl_figures()
                                await self.close_connection_with_central_server(
                                )
                                sys.exit()

                            # Start a new RL episode
                            await self.reset_env()

                else:
                    # The begining of the FL training
                    # The central server arrives
                    self.register_central_server(websocket)
                    logging.info(
                        'RL Agent: starting a new episode of RL training...')
                    await self.send_tuned_para()

                    #await self.reset_env()

        except websockets.ConnectionClosed as exception:
            logging.info("RL Agent: WebSockets connection closed abnormally.")
            logging.error(exception)
            sys.exit()

    def register_central_server(self, websocket):
        """Adding the FL central server."""
        self.central_server_socket = websocket

    async def close_connection_with_central_server(self):
        """Closing WebSocket connection after RL training completes."""
        await self.central_server_socket.close()

    def get_tuned_para(self, rl_tuned_para_value):
        """
        Get tuned parameter from env.
        This function is called by env.
        """
        self.rl_tuned_para_value = rl_tuned_para_value
        self.is_tuned_para_got = True

    async def send_tuned_para(self):
        """Send the tuned parameter to the central server."""
        socket = self.central_server_socket
        rl_agent_message = {
            'rl_time_step': self.time_step,
            'rl_tuned_para_value': self.rl_tuned_para_value
        }
        logging.info(
            "RL Agend: Sending the tuned parameter to the central server...")
        await socket.send(json.dumps(rl_agent_message))

    async def reset_env(self):
        """Reset the RL environment when an episode is finished."""
        self.central_server = None
        self.central_server_socket = None
        self.episode_num += 1
        self.time_step = 1

        await self.start_central_server()

    def plot_rl_figures(self):
        """Plot figures showing the results of reinforcement learning."""
        # To be implemented
        pass
