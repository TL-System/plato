"""
A federated learning training session using RL control.
"""
import asyncio
import logging
import multiprocessing as mp
import os

import simple_rl_agent
import simple_rl_server
from config import RLConfig
from plato.clients import simple


def run():
    """ Starting an RL Agent to connect to the server. """
    logging.info("Starting an RL Agent.")
    config = RLConfig()
    agent = simple_rl_agent.RLAgent(config)
    asyncio.run(agent.start_agent())


def main():
    """ A Plato federated learning training session using a custom RL algorithm. """

    logging.info("Starting RL Agent's process.")
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

    proc = mp.Process(target=run)
    proc.start()

    logging.info("Starting RL Environment's process.")
    server = simple_rl_server.RLServer()
    server.run()


if __name__ == "__main__":
    main()
