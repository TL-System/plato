"""
The implementation of FedPer method based on the plato's pFL code.

Manoj Ghuhan Arivazhagan, et.al, Federated learning with personalization layers.

paper address: https://arxiv.org/abs/1912.00818

Official code: None
Third-part code: https://github.com/jhoon-oh/FedBABU


"""

import backbone_cls
import fedper_client
import fedper_trainer

from plato.servers import fedavg_pers

from plato.algorithms import fedavg_pers as fedavg_algo


def main():
    """ An interface for running the FedPer method.
    """

    algo = fedavg_algo.Algorithm
    trainer = fedper_trainer.Trainer
    backbone_cls_model = backbone_cls.BackBoneCls
    client = fedper_client.Client(algorithm=algo,
                                  model=backbone_cls_model,
                                  trainer=trainer)
    server = fedavg_pers.Server(algorithm=algo,
                                model=backbone_cls_model,
                                trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
