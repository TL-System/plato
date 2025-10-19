"""
An implementation of the attack-defence scenario.

"""

import detector_server

from plato.clients import simple
from plato.trainers.basic import Trainer


def main():
    """
    A Plato federated learning training session with attackers existing under the
    supervised learning setting.
    """

    trainer = Trainer
    client = simple.Client(trainer=trainer)
    server = detector_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
