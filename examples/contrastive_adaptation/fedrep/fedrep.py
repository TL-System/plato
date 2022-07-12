"""
The implementation of FedRep method based on the plato's
pFL code.

"""

import fedrep_net
import fedrep_trainer

from plato.servers import fedavg_pers
from plato.algorithms import fedavg_ssl
from plato.clients import pers_simple


def main():
    """ An interface for running the fedavg method under the
        supervised learning setting.
    """

    algo = fedavg_ssl.Algorithm
    trainer = fedrep_trainer.Trainer
    backbone_cls_model = fedrep_net.BackBoneEnc
    client = pers_simple.Client(algorithm=algo,
                                model=backbone_cls_model,
                                trainer=trainer)
    server = fedavg_pers.Server(algorithm=algo,
                                model=backbone_cls_model,
                                trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
