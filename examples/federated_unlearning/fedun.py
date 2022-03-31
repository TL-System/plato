"""
A federated unlearning model to enables data holders to proactively erase their data from a trained model.

Reference: Liu et al., "The Right to be Forgotten in Federated Learning: An Efficient Realization with Rapid Retraining." in Proc. INFOCOM, 2022 https://arxiv.org/abs/2203.07320

"""

import os

os.environ['config_file'] = 'fedun_MNIST_lenet5.yml'

import fedun_client
import fedun_server
import fedun_trainer


def main():
    """ A Plato federated learning training session using the FedSarah algorithm. """
    trainer = fedun_trainer.Trainer()
    client = fedun_client.Client(trainer=trainer)
    server = fedun_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()