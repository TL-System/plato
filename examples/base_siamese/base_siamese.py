"""
The implementation for the basic siamese network.

Reference:

[1]. https://github.com/AceEviliano/Siamese-network-on-MNIST-PyTorch

"""

import siamese_mnist_net

import base_siamese_trainer
import base_siamese_client

from plato.servers import fedavg


def main():
    """ A Plato federated learning training session using the FedRep algorithm. """
    trainer = base_siamese_trainer.Trainer
    base_siamese_model = siamese_mnist_net.SiameseBase
    client = base_siamese_client.Client(model=base_siamese_model,
                                        trainer=trainer)
    server = fedavg.Server(model=base_siamese_model, trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
