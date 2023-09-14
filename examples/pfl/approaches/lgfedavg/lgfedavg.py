"""
The implementation of LG-FedAvg method based on the plato's
pFL code.

Paul Pu Liang, et.al, Think Locally, Act Globally: Federated Learning with Local and Global Representations
https://arxiv.org/abs/2001.01523

Official code: https://github.com/pliang279/LG-FedAvg

"""

from pflbases import fedavg_personalized_server
from pflbases import fedavg_partial
from pflbases.client_callbacks import personalized_completion_callbacks
from pflbases.trainer_callbacks import mixing_trainer_callbacks

import lgfedavg_client
import lgfedavg_trainer


def main():
    """
    A Plato personalized federated learning sesstion for LG-FedAvg approach.
    """
    trainer = lgfedavg_trainer.Trainer
    client = lgfedavg_client.Client(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        callbacks=[
            personalized_completion_callbacks.ClientModelPersonalizedCompletionCallback,
        ],
        trainer_callbacks=[
            mixing_trainer_callbacks.PersonalizedModelMetricCallback,
            mixing_trainer_callbacks.PersonalizedModelStatusCallback,
        ],
    )
    server = fedavg_personalized_server.Server(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
