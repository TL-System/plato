"""
A federated learning training session using Hermes
"""

import hermes_trainer
import hermes_server
from plato.clients import simple


def main():
    """A Plato federated learning training session using the Hermes algorithm."""
    trainer = hermes_trainer.Trainer
    server = hermes_server.Server(trainer=trainer)
    client = simple.Client(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
