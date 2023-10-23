"""
The implementation of LG-FedAvg method based on the plato's
pFL code.

Paul Pu Liang, et al., Think Locally, Act Globally: Federated Learning with Local and Global Representations
https://arxiv.org/abs/2001.01523

Official code: https://github.com/pliang279/LG-FedAvg

"""

from pflbases import fedavg_personalized_server
from pflbases import personalized_client
from pflbases import fedavg_partial
from pflbases.client_callbacks import personalized_completion_callbacks

import lgfedavg_trainer


def main():
    """
    A Plato personalized federated learning sesstion for LG-FedAvg approach.
    """
    trainer = lgfedavg_trainer.Trainer
    client = personalized_client.Client(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        callbacks=[
            personalized_completion_callbacks.ClientModelPersonalizedCompletionCallback,
        ],
    )
    server = fedavg_personalized_server.Server(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
