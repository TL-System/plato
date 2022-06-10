"""
A federated learning training session with clients running td3
"""
import logging

import td3_learning_client
import td3_learning_trainer
import td3_learning_server
import td3_learning_model
import td3_learning_algorithm
import globals

from torch import nn

#to run
#python examples/td3_learning/td3.py -c examples/td3_learning/td3_FashionMNIST_lenet5.yml


def main():
    """ A Plato federated learning training session with clients running TD3. """
    logging.info("Starting RL Environment's process.")

    """ A Plato federated learning training session using a custom model. """

    model = td3_learning_model.Model(globals.state_dim, globals.action_dim, globals.max_action)
    #TODO Fix error model is not callable
    # If I use number 1 I get that model has no attirbute actor if I pass it into client(line 34) or server (line 36)
    # Number 2 is the same, tried using both but same error no attirbute actor
    #model = td3_learning_model.Model (number 1)
    #(number 2 way)
    #instanceModel = td3_learning_model.Model(globals.state_dim, globals.action_dim, globals.max_action)
    trainer = td3_learning_trainer.Trainer
    algorithm = td3_learning_algorithm.Algorithm
    client = td3_learning_client.RLClient(model = model, trainer=trainer, algorithm = algorithm)
    client.configure()
    server = td3_learning_server.TD3Server(model=model, algorithm = algorithm)

    server.run(client)

if __name__ == "__main__":
    main()
