"""
The implementation of APFL method.

Yuyang Deng, et al., Adaptive Personalized Federated Learning

paper address: https://arxiv.org/pdf/2003.13461.pdf

Official code: None
Third-party code: 
- https://github.com/lgcollins/FedRep
- https://github.com/MLOPTPSU/FedTorch/blob/main/main.py
- https://github.com/MLOPTPSU/FedTorch/blob/main/fedtorch/comms/trainings/federated/apfl.py

"""

import apfl_trainer

from plato import servers
from plato import clients


def main():
    """
    A personalized federated learning session for APFL approach.
    """
    trainer = apfl_trainer.Trainer
    client = clients.fedavg_personalized.Client(trainer=trainer)
    server = servers.fedavg_personalized.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
