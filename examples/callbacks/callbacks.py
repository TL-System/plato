"""
This example shows how to use callbacks to customize server, client, and trainer.
"""

from callback_examples import *

from plato.clients import simple
from plato.servers import fedavg


def main():
    """
    A Plato federated learning training session customized by callbacks.
    """
    # Pass callbacks as arguments
    client = simple.Client(
        callbacks=[argumentClientCallback], trainer_callbacks=[customTrainerCallback]
    )
    server = fedavg.Server(callbacks=[argumentServerCallback])

    # Add callbacks after initialization
    client.add_callbacks(callbacks=[dynamicClientCallback])
    server.add_callbacks(callbacks=[dynamicServerCallback])

    # Run the session
    server.run(client)


if __name__ == "__main__":
    main()
