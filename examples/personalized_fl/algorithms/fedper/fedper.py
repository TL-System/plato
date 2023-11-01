"""
The implementation of FedPer method based on the plato's pFL code.

Manoj Ghuhan Arivazhagan, et al., Federated learning with personalization layers, 2019.
https://arxiv.org/abs/1912.00818

Official code: None
Third-part code: https://github.com/jhoon-oh/FedBABU
"""

import fedper_trainer

from plato.clients import simple

from pflbases import fedavg_personalized
from pflbases import fedavg_partial
from pflbases.client_callbacks import local_completion_callbacks


def main():
    """
    A personalized federated learning session for FedPer approach.
    """
    trainer = fedper_trainer.Trainer
    client = simple.Client(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        callbacks=[
            local_completion_callbacks.PayloadCompletionCallback,
        ],
    )
    server = fedavg_personalized.Server(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
