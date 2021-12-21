"""
A federated learning training session using DRL for global aggregation.
"""
import asyncio
import logging
import multiprocessing as mp
import os

import fei_agent
import fei_client
import fei_server
import fei_trainer

os.environ['config_file'] = 'examples/fei/fei_FMNIST_lenet5.yml'


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
    trainer = fei_trainer.Trainer()
    client = fei_client.Client(trainer=trainer)
    server = fei_server.RLServer(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
