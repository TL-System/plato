"""
An implementation of the FedBABU algorithm.

Oh, et al., "FedBABU: Toward Enhanced Representation for Federated Image Classification,"
in the Proceedings of ICLR 2022.

https://openreview.net/pdf?id=HuaYQfggn5u

Source code: https://github.com/jhoon-oh/FedBABU
"""

import fedbabu_trainer

from plato.clients import fedavg_personalized as personalized_client
from plato.servers import fedavg_personalized as personalized_server


def main():
    """
    A personalized federated learning session with FedBABU.
    """
    trainer = fedbabu_trainer.Trainer
    client = personalized_client.Client(trainer=trainer)
    server = personalized_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
