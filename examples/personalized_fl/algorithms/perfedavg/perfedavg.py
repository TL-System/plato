"""
The implementation of Per-FedAvg method based on the plato's
pFL code.

Alireza Fallah, et.al, Personalized federated learning with theoretical guarantees:
A model-agnostic meta-learning approach, NeurIPS 2020.
https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html

Official code: None
Third-part code: https://github.com/jhoon-oh/FedBABU

"""

from pflbases import fedavg_personalized_server
from pflbases import fedavg_partial
from pflbases import personalized_client
from pflbases.trainer_callbacks import separate_trainer_callbacks
from pflbases.client_callbacks import base_callbacks

import perfedavg_trainer


def main():
    """
    A personalized federated learning sesstion for PerFedAvg approach.
    """
    trainer = perfedavg_trainer.Trainer
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
