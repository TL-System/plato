"""
The implementation of LG-FedAvg method based on the plato's
pFL code.

Paul Pu Liang, et.al, Think Locally, Act Globally: Federated Learning with Local and Global Representations

paper address: https://arxiv.org/abs/2001.01523

Official code: https://github.com/pliang279/LG-FedAvg

"""

import lgfedavg_net
import lgfedavg_trainer
import lgfedavg_client

from plato.servers import fedavg_pers
from plato.algorithms import fedavg_pers as algo_fedavg_pers


def main():
    """ An interface for running the LG-FedAvg method.
    """

    algo = algo_fedavg_pers.Algorithm
    trainer = lgfedavg_trainer.Trainer
    backbone_cls_model = lgfedavg_net.BackBoneEnc
    client = lgfedavg_client.Client(algorithm=algo,
                                    model=backbone_cls_model,
                                    trainer=trainer)
    server = fedavg_pers.Server(algorithm=algo,
                                model=backbone_cls_model,
                                trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
