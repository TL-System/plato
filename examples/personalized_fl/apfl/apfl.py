"""
An implementation of Adaptive Personalized Federated Learning (APFL).

Y. Deng, et al., "Adaptive Personalized Federated Learning"

URL: https://arxiv.org/pdf/2003.13461.pdf

Third-party code:
https://github.com/MLOPTPSU/FedTorch/blob/main/main.py
https://github.com/MLOPTPSU/FedTorch/blob/main/fedtorch/comms/trainings/federated/apfl.py
"""

import apfl_trainer

from plato.clients import fedavg_personalized as personalized_client
from plato.servers import fedavg_personalized as personalized_server


def main():
    """
    A personalized federated learning session using APFL.
    """
    trainer = apfl_trainer.Trainer
    client = personalized_client.Client(trainer=trainer)
    server = personalized_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
