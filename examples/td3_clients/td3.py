"""
A federated learning training session using td3
"""
import logging

import td3_learning_client
import td3_learning_agent
from plato.servers import fedavg
import td3_trainer


def main():
    """ A Plato federated learning training session using TD3. """
    logging.info("Starting RL Environment's process.")
    trainer = td3_trainer.Trainer
    client = td3_learning_client.RLClient(trainer=trainer)
    agent = td3_learning_agent.RLAgent()
    server = fedavg.Server(trainer=trainer, model=agent)
    server.run(client)

if __name__ == "__main__":
    main()