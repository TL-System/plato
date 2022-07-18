"""
PyTorch Implementation of Personalized federated learning with theoretical guarantees: A model-agnostic meta-learning approach
NIPS 2020


https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html



"""

import perfedavg_net
import perfedavg_trainer

from plato.servers import fedavg_pers
from plato.algorithms import fedavg_pers as algo_fedavg_pers
from plato.clients import pers_simple


def main():
    """ An interface for running the fedavg method under the
        supervised learning setting.
    """

    algo = algo_fedavg_pers.Algorithm
    trainer = perfedavg_trainer.Trainer
    backbone_cls_model = perfedavg_net.BackBoneEnc
    client = pers_simple.Client(algorithm=algo,
                                model=backbone_cls_model,
                                trainer=trainer)
    server = fedavg_pers.Server(algorithm=algo,
                                model=backbone_cls_model,
                                trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
