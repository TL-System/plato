"""
An implementation of the personalized learning variant of FedAvg.

The core idea is to achieve personalized FL in two stages:
First, it trains a global model using conventional FedAvg until convergence. 
Second, each client fine-tunes the trained global model using its local data by several epochs.

Due to its simplicity, no work has been proposed that specifically discusses this algorithm.

Therefore, the performance of this algorithm works as the baseline for personalized federated learning.
"""


from pflbases import fedavg_personalized_server
from pflbases import fedavg_partial

from pflbases.client_callbacks import base_callbacks

from plato.clients import simple

import fedavg_finetune_trainer


def main():
    """
    A Plato personalized federated learning sesstion for FedAvg with fine-tuning.
    """
    trainer = fedavg_finetune_trainer.Trainer
    client = simple.Client(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        callbacks=[base_callbacks.ClientPayloadCallback],
    )
    server = fedavg_personalized_server.Server(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
