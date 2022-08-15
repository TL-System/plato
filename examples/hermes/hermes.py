"""
A federated learning training session using Hermes
"""

import hermes_client
import hermes_trainer
import hermes_server


def main():
    """A Plato federated learning training session using the Hermes algorithm."""
    trainer = hermes_trainer.Trainer
    client = hermes_client.Client(trainer=trainer)
    server = hermes_server.Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
