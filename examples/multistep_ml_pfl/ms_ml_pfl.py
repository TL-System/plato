"""
The implementation of my recent work:

Personalized Federated Learning with Multi-step Model-Agnostic 
    Meta-Learning Approach

"""

import os

os.environ[
    'config_file'] = 'examples/multistep_ml_pfl/ms_ml_pfl_MNIST_lenet5.yml'

import ms_ml_pfl_server
import ms_ml_pfl_client
import ms_ml_pfl_trainer


def main():
    """ A Plato federated learning training session using the one-step MAML algorithm. """

    trainer = ms_ml_pfl_trainer.Trainer()
    client = ms_ml_pfl_client.Client(trainer=trainer)
    server = ms_ml_pfl_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
