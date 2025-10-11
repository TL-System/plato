"""
An implementation of the Ditto personalized federated learning algorithm.

T. Li, et al., "Ditto: Fair and robust federated learning through personalization,"
in the Proceedings of ICML 2021.

https://proceedings.mlr.press/v139/li21h.html

Source code: https://github.com/litian96/ditto
"""

import ditto_trainer

from plato.clients import fedavg_personalized as personalized_client
from plato.servers import fedavg_personalized as personalized_server


def main():
    """
    A personalized federated learning session with Ditto.
    """
    trainer = ditto_trainer.Trainer
    client = personalized_client.Client(trainer=trainer)
    server = personalized_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
