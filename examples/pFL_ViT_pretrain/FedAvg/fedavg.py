"""
The interface of applying the fedavg on different datasets as the
baseline.

The fedavg with finetune.

"""

import fedavg_client
import fedavg_trainer

import fedavg_pers
import fedavg_partial as fedavg_algo
import vitplatomodel


def main():
    """An interface for running the fedavg method under the
    supervised learning setting.
    """

    algo = fedavg_algo.Algorithm
    trainer = fedavg_trainer.Trainer
    # backbone_cls_model = backbone_cls.BackBoneCls
    model = vitplatomodel.get_vits()
    client = fedavg_client.Client(algorithm=algo, model=model, trainer=trainer)
    server = fedavg_pers.Server(algorithm=algo, model=model, trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
