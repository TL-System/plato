"""
A federated learning training session using Per-FedAvg.

Reference
Alireza Fallah, et al., 
“Personalized Federated Learning with Theoretical Guarantees:
A Model-Agnostic Meta-Learning Approach,”
in Proc. Advances in Neural Information Processing Systems (NeurIPS), 2020.

Third-party code: https://github.com/jhoon-oh/FedBABU
"""

import perfedavg_trainer

from plato.servers import fedavg_personalized as personalized_server
from plato.clients import fedavg_personalized as personalized_client


def main():
    """
    A personalized federated learning session using the PerFedAvg algorithm.
    """
    trainer = perfedavg_trainer.Trainer
    client = personalized_client.Client(trainer=trainer)
    server = personalized_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
