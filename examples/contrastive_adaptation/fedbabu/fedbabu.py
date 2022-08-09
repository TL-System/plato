"""
The implementation of FedBABU method based on the plato's pFL code.

Jaehoon Oh, et.al, FedBABU: Toward Enhanced Representation for Federated Image Classification.
in the Proceedings of ICML 2021.

Paper address: https://openreview.net/pdf?id=HuaYQfggn5u
Official code: https://github.com/jhoon-oh/FedBABU

"""

import fedbabu_net
import fedbabu_trainer
import fedbabu_client

from plato.servers import fedavg_pers
from plato.algorithms import fedavg_pers as algo_fedavg_pers


def main():
    """ An interface for running the FedBABU method under the
        supervised learning setting.
    """

    algo = algo_fedavg_pers.Algorithm
    trainer = fedbabu_trainer.Trainer
    backbone_cls_model = fedbabu_net.BackBoneEnc
    client = fedbabu_client.Client(algorithm=algo,
                                   model=backbone_cls_model,
                                   trainer=trainer)
    server = fedavg_pers.Server(algorithm=algo,
                                model=backbone_cls_model,
                                trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
