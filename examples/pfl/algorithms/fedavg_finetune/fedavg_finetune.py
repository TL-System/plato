"""
An implementation of the personalized learning variant of FedAvg.

The core idea is to achieve personalized FL in two stages:
First, it trains a global model using conventional FedAvg until convergence. 
Second, each client fine-tunes the trained global model using its local data by several epochs.

Due to its simplicity, no work has been proposed that specifically discusses this algorithm.

Therefore, the performance of this algorithm works as the baseline for personalized federated learning.

"""


from pflbases import personalized_trainer
from pflbases import personalized_client
from pflbases import fedavg_personalized_server
from pflbases import fedavg_partial

from pflbases.trainer_callbacks import separate_trainer_callbacks
from pflbases.client_callbacks import base_callbacks


def main():
    """
    A Plato personalized federated learning sesstion for FedAvg with fine-tuning.
    """
    trainer = personalized_trainer.Trainer
    client = personalized_client.Client(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        callbacks=[base_callbacks.ClientPayloadCallback],
        trainer_callbacks=[
            separate_trainer_callbacks.PersonalizedModelMetricCallback,
            separate_trainer_callbacks.PersonalizedModelStatusCallback,
        ],
    )
    server = fedavg_personalized_server.Server(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
