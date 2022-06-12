"""
The implementation for our idea of applying the moving average
for local-gloabl model.

This idea is similar with the one proposed by FedEMA [1].

However, our idea provides a more general case of calculating
the weights based on the representation quality guided by
informatio theory.

Reference:

[1]. https://arxiv.org/abs/2204.04385
Source code: None

But, we directly call this idea to be the same name of FedEMA
without losing generality.

"""

import fedema_net
import fedema_trainer
import fedema_server

from plato.clients import ssl_simple as ssl_client
from plato.algorithms import fedavg_ssl


def main():
    """ A Plato federated learning training session using the SimCLR algorithm.
        This implementation of simclr utilizes the general setting, i.e.,
        removing the final fully-connected layers of model defined by
        the 'model_name' in config file.
    """
    trainer = fedema_trainer.Trainer
    algorithm = fedavg_ssl.Algorithm
    byol_model = fedema_net.FedEMA()
    client = ssl_client.Client(model=byol_model,
                               trainer=trainer,
                               algorithm=algorithm)
    server = fedema_server.Server(model=byol_model,
                                  trainer=trainer,
                                  algorithm=algorithm)

    server.run(client)


if __name__ == "__main__":
    main()
