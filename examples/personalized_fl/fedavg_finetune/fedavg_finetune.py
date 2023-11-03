"""
This implementation first trains a global model using conventional FedAvg until
a target number of rounds has been reached. In the final `personalization`
round, each client will use its local data samples to further fine-tune the
shared global model for a number of epochs, and then the server will compute the
average client test accuracy.

Due to its simplicity, no papers specifically discussed or proposed this
algorithm; they only utilized it as their baseline for comparisons.
"""

from plato.clients import fedavg_personalized as personalized_client
from plato.servers import fedavg_personalized as personalized_server


def main():
    """
    A Plato personalized federated learning session for FedAvg with fine-tuning.
    """
    client = personalized_client.Client()
    server = personalized_server.Server()
    server.run(client)


if __name__ == "__main__":
    main()
