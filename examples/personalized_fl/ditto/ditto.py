"""
An implementation of the Ditto personalized federated learning algorithm.

Reference:
T. Li, et al., "Ditto: Fair and robust federated learning through personalization," 2021.

URL: https://proceedings.mlr.press/v139/li21h.html

Official code: https://github.com/litian96/ditto
"""

import ditto_trainer

from plato.servers import fedavg_personalized as personalized_server
from plato.clients import fedavg_personalized as personalized_client


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
