"""
The implementation of APFL method based on the plato's pFL code.

Yuyang Deng, et.al, Adaptive Personalized Federated Learning

paper address: https://arxiv.org/abs/2001.01523

Official code: No official code
Third-part code: https://github.com/lgcollins/FedRep

In this implementation, we follow the the code in third-part
Third-part code:
- https://github.com/lgcollins/FedRep
- https://github.com/MLOPTPSU/FedTorch/blob/main/main.py
in which the parameter operations of apfl are performed in
the batch within each epoch.

Our implementation of APFL relies on the APFL code of:
https://github.com/MLOPTPSU/FedTorch/blob/main/fedtorch/comms/trainings/federated/apfl.py

This implementation version of APFL is the one that most related to the algorithm mentioned
in the original paper.


There is another type of implementation named 'apflepo'. Please access the corresponding
implementation under our contrastive adaptation.






"""

import apfl_net
import apfl_trainer
import apfl_client

from plato.servers import fedavg_pers
from plato.algorithms import fedavg_pers as algo_fedavg_pers


def main():
    """ An interface for running the APFL method.
    """

    algo = algo_fedavg_pers.Algorithm
    trainer = apfl_trainer.Trainer
    backbone_cls_model = apfl_net.BackBoneEnc
    client = apfl_client.Client(algorithm=algo,
                                model=backbone_cls_model,
                                trainer=trainer)
    server = fedavg_pers.Server(algorithm=algo,
                                model=backbone_cls_model,
                                trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
