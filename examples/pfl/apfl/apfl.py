"""
The implementation of APFL method based on the plato's pFL code.

Yuyang Deng, et.al, Adaptive Personalized Federated Learning

paper address: https://arxiv.org/abs/2001.01523

Official code: None
Third-part code: 
- https://github.com/lgcollins/FedRep
- https://github.com/MLOPTPSU/FedTorch/blob/main/main.py
- https://github.com/MLOPTPSU/FedTorch/blob/main/fedtorch/comms/trainings/federated/apfl.py


"""

import apfl_trainer
import apfl_client
from examples.pfl.bases import fedavg_personalized


def main():
    """An interface for running the APFL method."""

    trainer = apfl_trainer.Trainer
    client = apfl_client.Client(trainer=trainer)
    server = fedavg_personalized.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
