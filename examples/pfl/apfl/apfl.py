"""
The implementation of APFL method based on the plato's pFL code.

Yuyang Deng, et.al, Adaptive Personalized Federated Learning

paper address: https://arxiv.org/pdf/2003.13461.pdf

Official code: None
Third-part code: 
- https://github.com/lgcollins/FedRep
- https://github.com/MLOPTPSU/FedTorch/blob/main/main.py
- https://github.com/MLOPTPSU/FedTorch/blob/main/fedtorch/comms/trainings/federated/apfl.py


"""
import os
import sys

# Get the current directory of module1.py
pfl_bases = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(pfl_bases))


from bases import fedavg_personalized_server
from bases import fedavg_partial

from bases.trainer_callbacks import (
    PersonalizedLogMetricCallback,
    PersonalizedLogProgressCallback,
)

import apfl_trainer_callbacks
import apfl_client
import apfl_trainer


def main():
    """
    A Plato personalized federated learning sesstion for FedBABU approach.
    """
    trainer = apfl_trainer.Trainer
    client = apfl_client.Client(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        trainer_callbacks=[
            PersonalizedLogMetricCallback,
            PersonalizedLogProgressCallback,
            apfl_trainer_callbacks.LearningStatusCallback,
        ],
    )
    server = fedavg_personalized_server.Server(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
