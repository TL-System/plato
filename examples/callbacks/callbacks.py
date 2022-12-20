"""
This example shows how to use callbacks to customize server, client, and trainer.
"""

from plato.clients import simple
from plato.servers import fedavg
from callback_examples import *


def main():
    """
    A Plato federated learning training session customized by callbacks.
    """
    # Pass the callbacks to client and server as arguments
    client = simple.Client(callbacks=[argumentClientCallback])
    server = fedavg.Server(callbacks=[argumentServerCallback])

    # Dynamically add callbacks after instantiated
    client.add_callbacks(callbacks=[dynamicClientCallback])
    server.add_callbacks(callbacks=[dynamicServerCallback])
    client.add_trainer_callbacks(trainer_callbacks=[dynamicTrainerCallback])

    # Run the session
    server.run(client)


if __name__ == "__main__":
    main()
