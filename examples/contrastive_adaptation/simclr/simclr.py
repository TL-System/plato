"""
The implementation for the SimCLR [1] method.

The official code: https://github.com/google-research/simclr

The third-party code: https://github.com/PatrickHua/SimSiam

The structure of our SimCLR and the classifier is the same as the ones used in
the work https://github.com/spijkervet/SimCLR.git.

Reference:

[1]. https://arxiv.org/abs/2002.05709

"""

import simclr_net

from plato.trainers import contrastive_ssl as ssl_trainer
from plato.clients import ssl_simple as ssl_client
from plato.servers import fedavg_ssl_base as ssl_server
from plato.algorithms import fedavg_ssl


def main():
    """ A Plato federated learning training session using the SimCLR algorithm.
        This implementation of simclr utilizes the general setting, i.e.,
        removing the final fully-connected layers of model defined by
        the 'model_name' in config file.
    """
    algorithm = fedavg_ssl.Algorithm
    trainer = ssl_trainer.Trainer
    simclr_model = simclr_net.SimCLR()
    client = ssl_client.Client(model=simclr_model,
                               trainer=trainer,
                               algorithm=algorithm)
    server = ssl_server.Server(model=simclr_model,
                               trainer=trainer,
                               algorithm=algorithm)

    server.run(client)


if __name__ == "__main__":
    main()
