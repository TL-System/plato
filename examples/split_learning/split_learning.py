"""
A federated learning training session using FedAtt.

Reference:

Ji et al., "Learning Private Neural Language Modeling with Attentive Aggregation,"
in the Proceedings of the 2019 International Joint Conference on Neural Networks (IJCNN).

https://arxiv.org/pdf/1812.07108.pdf
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
