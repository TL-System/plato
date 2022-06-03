"""
The implementation for the SimCLR [1] method.

The official code: https://github.com/google-research/simclr

The third-party code: https://github.com/PatrickHua/SimSiam

Our implementation relys on:
 https://github.com/spijkervet/SimCLR.git

Reference:

[1]. https://arxiv.org/abs/2002.05709


This demo implementation has the following properties.

- It is specifically designed for the MNIST dataset.
- It is used to compare with the central self-supervised learning
implemented by the work https://github.com/giakou4/MNIST_classification
- It is used to verify the correcness of our method.

"""

import simclr_net
from mnist_encoder_net import Encoder

from plato.trainers import self_sl as ssl_trainer
from plato.clients import ssl_simple as ssl_client
from plato.servers import fedavg


def main():
    """ A Plato federated learning training session using the SimCLR algorithm.

        This implementation of simclr utilizes the specifc encoder for the MNIST dataset.
        https://github.com/giakou4/MNIST_classification.
    """
    mnist_encoder = Encoder()
    trainer = ssl_trainer.Trainer
    simclr_model = simclr_net.SimCLR(
        encoder=mnist_encoder, encoder_dim=mnist_encoder.get_encoding_dim())
    client = ssl_client.Client(model=simclr_model, trainer=trainer)
    server = fedavg.Server(model=simclr_model, trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
