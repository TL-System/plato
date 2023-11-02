"""
An implementation of the personalized learning variant of FedAvg.

The core idea is to achieve personalized FL in two stages:
First, it trains a global model using conventional FedAvg until convergence. 
Second, each client freezes the trained global model and optimizes the other 
parts.

Due to its simplicity, no work has been proposed that specifically discusses 
this algorithm but only utilizes it as the baseline for personalized federated 
learning.
"""

from plato.clients import simple
from plato.trainers import basic

from plato.servers import fedavg_personalized as personalized_server


def main():
    """
    A Plato personalized federated learning session for FedAvg with fine-tuning.
    """
    trainer = basic.Trainer
    client = simple.Client(trainer=trainer)
    server = personalized_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
