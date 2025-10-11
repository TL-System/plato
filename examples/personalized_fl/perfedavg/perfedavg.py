"""
A federated learning training session using Per-FedAvg.

A. Fallah, et al., “Personalized Federated Learning with Theoretical Guarantees:
A Model-Agnostic Meta-Learning Approach,” in Proc. Advances in Neural
Information Processing Systems (NeurIPS), 2020.

https://dl.acm.org/doi/abs/10.5555/3495724.3496024

Third-party code: https://github.com/jhoon-oh/FedBABU
"""

import perfedavg_trainer

from plato.clients import fedavg_personalized as personalized_client
from plato.servers import fedavg_personalized as personalized_server


def main():
    """
    A personalized federated learning session using the Per-FedAvg algorithm.
    """
    trainer = perfedavg_trainer.Trainer
    client = personalized_client.Client(trainer=trainer)
    server = personalized_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
