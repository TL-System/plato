"""
The implementation of Ditto method based on the plato's pFL code.

Tian Li, et.al, Ditto: Fair and robust federated learning through personalization, 2021:
 https://proceedings.mlr.press/v139/li21h.html

Official code: https://github.com/s-huu/Ditto
Third-part code: https://github.com/lgcollins/FedRep

In this implementation, we follow the the code in third-part
Third-part code:
- https://github.com/lgcollins/FedRep


This implementation version of APFL is the one that most related to the algorithm mentioned
in the original paper.

"""

import ditto_net
import ditto_trainer
import ditto_client

from plato.servers import fedavg_pers
from plato.algorithms import fedavg_pers as algo_fedavg_pers


def main():
    """ An interface for running the APFL method.
    """

    algo = algo_fedavg_pers.Algorithm
    trainer = ditto_trainer.Trainer
    backbone_cls_model = ditto_net.BackBoneEnc
    client = ditto_client.Client(algorithm=algo,
                                 model=backbone_cls_model,
                                 trainer=trainer)
    server = fedavg_pers.Server(algorithm=algo,
                                model=backbone_cls_model,
                                trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
