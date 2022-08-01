"""
The interface of training the local models based on each client's dataset from
script.

"""

import backbone_cls
import script_client

from plato.servers import fedavg_pers
from plato.trainers import pers_basic
from plato.algorithms import fedavg_pers as fedavg_algo


def main():
    """ An interface for running the fedavg method under the
        supervised learning setting.
    """

    algo = fedavg_algo.Algorithm
    trainer = pers_basic.Trainer
    backbone_cls_model = backbone_cls.BackBoneCls
    client = script_client.Client(algorithm=algo,
                                  model=backbone_cls_model,
                                  trainer=trainer)
    server = fedavg_pers.Server(algorithm=algo,
                                model=backbone_cls_model,
                                trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
