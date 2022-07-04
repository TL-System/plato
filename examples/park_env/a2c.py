"""
A federated learning training session with clients running td3
"""
import logging

from plato.config import Config

import a2c_learning_algorithm
import a2c_learning_client
import a2c_learning_model
import a2c_learning_server
import a2c_learning_trainer
#to run
#python examples/park_env/a2c.py -c examples/park_env/a2c_FashionMNIST_lenet5.yml



def main():
    """ A Plato federated learning training session with clients running TD3. """
    logging.info("Starting RL Environment's process.")


    model = a2c_learning_model.Model
    trainer = a2c_learning_trainer.Trainer
    algorithm = a2c_learning_algorithm.Algorithm
    client = a2c_learning_client.RLClient(model=model,trainer=trainer,algorithm=algorithm)
    server = a2c_learning_server.A2CServer(Config().algorithm.algorithm_name, Config().algorithm.env_name,
    model=model, algorithm=algorithm, trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
