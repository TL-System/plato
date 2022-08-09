"""
The implementation of Scaffold method based on the plato's pFL code.

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

Paper address: https://arxiv.org/pdf/1910.06378.pdf
Official code: https://github.com/lgcollins/FedRep


Our implementation of Scaffold heavily depends on the Plato's previous implementation
placed under examples/scaffold. In general, we directly utilized the previous code but only
introduced the properties of the personalized federated learning.


"""

import scaffold_net
import scaffold_trainer
import scaffold_client
import scaffold_server

from plato.algorithms import fedavg_pers as algo_fedavg_pers


def main():
    """ A Plato personalized federated learning training session using the SCAFFOLD algorithm. .
    """

    algo = algo_fedavg_pers.Algorithm
    trainer = scaffold_trainer.Trainer
    backbone_cls_model = scaffold_net.BackBoneEnc
    client = scaffold_client.Client(algorithm=algo,
                                    model=backbone_cls_model,
                                    trainer=trainer)
    server = scaffold_server.Server(algorithm=algo,
                                    model=backbone_cls_model,
                                    trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
