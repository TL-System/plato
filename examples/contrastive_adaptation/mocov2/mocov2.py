"""
The implementation for the MoCo [1] method.

The official code: https://github.com/facebookresearch/moco

The third-party code: https://github.com/PatrickHua/SimSiam


Reference:

[1]. https://arxiv.org/abs/1911.05722

Source code: https://github.com/facebookresearch/simsiam
"""

import mocov2_net
import mocov2_trainer

from plato.clients import ssl_simple as ssl_client
from plato.servers import fedavg_pers as ssl_server
from plato.algorithms import fedavg_pers


def main():
    """ A Plato federated learning training session using the MoCo algorithm.

    """
    trainer = mocov2_trainer.Trainer
    algorithm = fedavg_pers.Algorithm
    moco_model = mocov2_net.MoCo
    client = ssl_client.Client(model=moco_model,
                               trainer=trainer,
                               algorithm=algorithm)
    server = ssl_server.Server(model=moco_model,
                               trainer=trainer,
                               algorithm=algorithm)

    server.run(client)


if __name__ == "__main__":
    main()
