"""
An implementation of the FedBABU algorithm.

J. Oh, et al., "FedBABU: Toward Enhanced Representation for Federated Image Classification,"
in the Proceedings of ICLR 2022.

https://openreview.net/pdf?id=HuaYQfggn5u

Source code: https://github.com/jhoon-oh/FedBABU
"""


from plato.clients import simple

from pflbases import fedavg_personalized
from pflbases import fedavg_partial

import fedbabu_trainer


def main():
    """
    A personalized federated learning session for FedBABU algorithm under the supervised setting.
    """
    trainer = fedbabu_trainer.Trainer
    client = simple.Client(trainer=trainer, algorithm=fedavg_partial.Algorithm)
    server = fedavg_personalized.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
