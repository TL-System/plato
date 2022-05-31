"""
A federated learning training session using td3
"""
import logging

import td3_learning_client
import td3_learning_agent
from plato.servers import fedavg
import td3_trainer


def main():
    """ A Plato federated learning training session using the FEI algorithm. """
    logging.info("Starting RL Environment's process.")
    trainer = td3_trainer.Trainer
    client = td3_learning_client.RLClient(trainer=trainer)
    agent = td3_learning_agent.RLAgent()
    server = fedavg.Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()