"""
An implementation of the personalized learning variant of FedAvg.

Such an variant of FedAvg is recently mentioned and discussed in work [1].

[1] Liam Collins, et al., "Exploiting shared representations for personalized federated learning,"
in the Proceedings of ICML 2021.

    Address: https://proceedings.mlr.press/v139/collins21a.html

    Code: https://github.com/lgcollins/FedRep

"""

import os
import sys

# Add `bases` to the path
pfl_bases = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(pfl_bases))


from pflbases import personalized_trainer
from pflbases import personalized_client
from pflbases import fedavg_personalized_server
from pflbases import fedavg_partial

from pflbases.trainer_callbacks import separate_trainer_callbacks
from pflbases.client_callbacks import base_callbacks


def main():
    """
    A Plato personalized federated learning sesstion for FedAvg with fine-tuning.
    """
    trainer = personalized_trainer.Trainer
    client = personalized_client.Client(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        callbacks=[base_callbacks.ClientPayloadCallback],
        trainer_callbacks=[
            separate_trainer_callbacks.PersonalizedModelMetricCallback,
            separate_trainer_callbacks.PersonalizedModelStatusCallback,
        ],
    )
    server = fedavg_personalized_server.Server(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
