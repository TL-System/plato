"""
An implementation of the FedBABU algorithm.

J. Oh, et al., "FedBABU: Toward Enhanced Representation for Federated Image Classification,"
in the Proceedings of ICLR 2022.

https://openreview.net/pdf?id=HuaYQfggn5u

Source code: https://github.com/jhoon-oh/FedBABU
"""

from pflbases import fedavg_personalized_server
from pflbases import fedavg_partial
from pflbases.trainer_callbacks import separate_trainer_callbacks
from pflbases.client_callbacks import personalized_completion_callbacks

import fedbabu_client
import fedbabu_trainer


def main():
    """
    A personalized federated learning sesstion for FedBABU algorithm under the supervised setting.
    """
    trainer = fedbabu_trainer.Trainer
    client = fedbabu_client.Client(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        callbacks=[
            personalized_completion_callbacks.ClientModelPersonalizedCompletionCallback,
        ],
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
