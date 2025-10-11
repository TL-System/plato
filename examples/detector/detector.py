"""
An implementation of the attack-defence scenario.

"""

from detector_server import Server

from plato.clients.simple import Client
from plato.trainers.basic import Trainer


def main():
    """
    A Plato federated learning training session with attackers existing under the
    supervised learning setting.
    """

    client = Client(trainer=Trainer)
    server = Server(trainer=Trainer)

    server.run(client)


if __name__ == "__main__":
    main()
