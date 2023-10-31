"""
An implementation of the personalized learning variant of FedAvg.

The core idea is to achieve personalized FL in two stages:
First, it trains a global model using conventional FedAvg until convergence. 
Second, each client freezes the trained global model and optimizes the other 
parts.

Due to its simplicity, no work has been proposed that specifically discusses 
this algorithm but only utilizes it as the baseline for personalized federated 
learning.
"""


from pflbases import fedavg_personalized
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
    server = fedavg_personalized.Server(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
