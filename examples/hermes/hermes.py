"""
A federated learning training session using Hermes
"""

import hermes_trainer
import hermes_server
import hermes_client


def main():
    """A Plato federated learning training session using the Hermes algorithm."""
    trainer = hermes_trainer.Trainer
    server = hermes_server.Server(trainer=trainer)
    client = hermes_client.Client(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
