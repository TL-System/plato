"""
A federated learning training session using split learning.

Reference:

Vepakomma, et al., "Split learning for health: Distributed deep learning without sharing
raw patient data," in Proc. AI for Social Good Workshop, affiliated with ICLR 2018.

https://arxiv.org/pdf/1812.00564.pdf
"""
import os

os.environ['config_file'] = 'split_learning_MNIST_lenet5.yml'

import split_learning_server
import split_learning_algorithm
import split_learning_client
import split_learning_trainer


def main():
    """ A Plato federated learning training session using the split learning algorithm. """
    trainer = split_learning_trainer.Trainer()
    algorithm = split_learning_algorithm.Algorithm(trainer=trainer)
    client = split_learning_client.Client(algorithm=algorithm, trainer=trainer)
    server = split_learning_server.Server(algorithm=algorithm, trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
