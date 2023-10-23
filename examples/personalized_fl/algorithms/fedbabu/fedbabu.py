"""
An implementation of the FedBABU algorithm.

J. Oh, et al., "FedBABU: Toward Enhanced Representation for Federated Image Classification,"
in the Proceedings of ICLR 2022.

https://openreview.net/pdf?id=HuaYQfggn5u

Source code: https://github.com/jhoon-oh/FedBABU
"""

from pflbases import fedavg_personalized_server
from pflbases import personalized_client
from pflbases import fedavg_partial
from pflbases.client_callbacks import personalized_completion_callbacks


import fedbabu_trainer


def main():
    """
    A personalized federated learning sesstion for FedBABU algorithm under the supervised setting.
    """
    trainer = fedbabu_trainer.Trainer
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
