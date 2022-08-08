"""
The interface of applying the fedavg on different datasets as the
baseline.

The fedavg with finetune.

"""

import backbone_cls
import fedavg_client
import fedavg_trainer

from plato.servers import fedavg_pers
from plato.algorithms import fedavg_pers as fedavg_algo


def main():
    """ An interface for running the fedavg method under the
        supervised learning setting.
    """

    algo = fedavg_algo.Algorithm
    trainer = fedavg_trainer.Trainer
    backbone_cls_model = backbone_cls.BackBoneCls
    client = fedavg_client.Client(algorithm=algo,
                                  model=backbone_cls_model,
                                  trainer=trainer)
    server = fedavg_pers.Server(algorithm=algo,
                                model=backbone_cls_model,
                                trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
