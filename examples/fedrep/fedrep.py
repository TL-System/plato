"""
Exploiting Shared Representations for Personalized Federated Learning.

Reference:

Collins, Liam, et al., "Exploiting Shared Representations for Personalized Federated Learning", in the Proceedings of ICML 2021.

Paper address: https://arxiv.org/abs/2102.07078

Official source code: https://github.com/lgcollins/FedRep

"""
import os

os.environ['config_file'] = 'fedrep_MNIST_lenet5.yml'

import fedrep_trainer
import fedrep_client
import fedrep_server


def main():
    """ A Plato federated learning training session using the FedRep algorithm. """
    trainer = fedrep_trainer.Trainer
    client = fedrep_client.Client(trainer=trainer)
    server = fedrep_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
