"""
A federated learning training session using fei.
"""
import asyncio
import logging
import multiprocessing as mp
import os

import fei_agent
import fei_client
import fei_server

def run():
    """Starting an RL Agent (client) to connect to the server."""

    logging.info("Starting an RL Agent.")
    agent = fei_agent.RLAgent()
    asyncio.run(agent.start_agent())


def main():
    """ A Plato federated learning training session using the FEI algorithm. """

    logging.info("Starting RL Agent's process.")
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

    proc = mp.Process(target=run)
    proc.start()

    logging.info("Starting RL Environment's process.")
    client = fei_client.Client()
    server = fei_server.RLServer()
    server.run(client)


if __name__ == "__main__":
    main()
