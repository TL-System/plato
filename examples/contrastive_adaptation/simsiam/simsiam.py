"""
The implementation for the SimSiam [1] method.

The official code: https://github.com/facebookresearch/simsiam

The third-party code: https://github.com/PatrickHua/SimSiam


Reference:

[1]. https://arxiv.org/pdf/2011.10566.pdf

Source code: https://github.com/facebookresearch/simsiam
"""

import simsiam_net
import simsiam_trainer

from plato.clients import ssl_simple as ssl_client
from plato.servers import fedavg_ssl_base as ssl_server
from plato.algorithms import fedavg_ssl


def main():
    """ A Plato federated learning training session using the SimCLR algorithm.
        This implementation of simclr utilizes the general setting, i.e.,
        removing the final fully-connected layers of model defined by
        the 'model_name' in config file.
    """
    trainer = simsiam_trainer.Trainer
    algorithm = fedavg_ssl.Algorithm
    byol_model = simsiam_net.SimSiam()
    client = ssl_client.Client(model=byol_model,
                               trainer=trainer,
                               algorithm=algorithm)
    server = ssl_server.Server(model=byol_model,
                               trainer=trainer,
                               algorithm=algorithm)

    server.run(client)


if __name__ == "__main__":
    main()
