"""
A federated learning training session using td3
"""
import logging

import td3_learning_client
import td3_trainer
import td3_learning_server

import globals

from torch import nn

#to run
#python examples/td3_learning/td3.py -c examples/td3_learning/td3_FashionMNIST_lenet5.yml

trainer = td3_trainer.Trainer(globals.state_dim, globals.action_dim)

evaluations = [td3_trainer.Trainer.evaluate_policy(trainer)]

def main():
    """ A Plato federated learning training session using TD3. """
    logging.info("Starting RL Environment's process.")

    """ A Plato federated learning training session using a custom model. """
    model = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    client = td3_learning_client.RLClient(trainer=trainer, model=model)
    server = td3_learning_server.TD3Server(model=model)
    server.run(client)

if __name__ == "__main__":
    main()
