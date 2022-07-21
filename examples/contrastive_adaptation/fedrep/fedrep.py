"""
The implementation of FedRep method based on the plato's
pFL code.

"""

import fedrep_net
import fedrep_trainer
import fedrep_client

from plato.servers import fedavg_pers
from plato.algorithms import fedavg_pers as algo_fedavg_pers


def main():
    """ An interface for running the fedavg method under the
        supervised learning setting.
    """

    algo = algo_fedavg_pers.Algorithm
    trainer = fedrep_trainer.Trainer
    backbone_cls_model = fedrep_net.BackBoneEnc
    client = fedrep_client.Client(algorithm=algo,
                                  model=backbone_cls_model,
                                  trainer=trainer)
    server = fedavg_pers.Server(algorithm=algo,
                                model=backbone_cls_model,
                                trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
