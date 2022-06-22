"""
The implementation for the MoCo [1] method.

The official code: https://github.com/facebookresearch/moco

The third-party code: https://github.com/PatrickHua/SimSiam


Reference:

[1]. https://arxiv.org/abs/1911.05722

Source code: https://github.com/facebookresearch/simsiam
"""

import moco_net
import moco_trainer

from plato.clients import ssl_simple as ssl_client
from plato.servers import fedavg_pers as ssl_server
from plato.algorithms import fedavg_ssl


def main():
    """ A Plato federated learning training session using the MoCo algorithm.

    """
    trainer = moco_trainer.Trainer
    algorithm = fedavg_ssl.Algorithm
    byol_model = moco_net.MoCo()
    client = ssl_client.Client(model=byol_model,
                               trainer=trainer,
                               algorithm=algorithm)
    server = ssl_server.Server(model=byol_model,
                               trainer=trainer,
                               algorithm=algorithm)

    server.run(client)


if __name__ == "__main__":
    main()
