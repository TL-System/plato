"""
A personalized federated learning training session using FedRep.

Reference:

Collins et al., "Exploiting Shared Representations for Personalized Federated
Learning," in the Proceedings of ICML 2021.

https://arxiv.org/abs/2102.07078

Source code: https://github.com/lgcollins/FedRep
"""
from pflbases import fedavg_personalized_server
from pflbases import fedavg_partial
from pflbases.client_callbacks import personalized_completion_callbacks
from pflbases.trainer_callbacks import semi_mixing_trainer_callbacks

import fedrep_client
import fedrep_trainer


def main():
    """
    A Plato personalized federated learning sesstion for FedBABU approach.
    """
    trainer = fedrep_trainer.Trainer
    client = fedrep_client.Client(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        callbacks=[
            personalized_completion_callbacks.ClientModelPersonalizedCompletionCallback,
        ],
        trainer_callbacks=[
            semi_mixing_trainer_callbacks.PersonalizedModelMetricCallback,
            semi_mixing_trainer_callbacks.PersonalizedModelStatusCallback,
        ],
    )
    server = fedavg_personalized_server.Server(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
