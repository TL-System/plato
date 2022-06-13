"""
A federated learning training session using FEI.
"""
import logging

import fei_agent
import fei_client
import fei_server
import fei_trainer


def main():
    """ A Plato federated learning training session using the FEI algorithm. """
    logging.info("Starting RL Environment's process.")
    trainer = fei_trainer.Trainer
    client = fei_client.Client(trainer=trainer)
    agent = fei_agent.RLAgent()
    server = fei_server.RLServer(agent=agent, trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
