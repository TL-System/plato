"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""

import scaffold_client
import scaffold_server
import scaffold_trainer
from scaffold_callback import ScaffoldCallback


def main():
    """A Plato federated learning training session using the SCAFFOLD algorithm."""
    trainer = scaffold_trainer.Trainer
    client = scaffold_client.Client(trainer=trainer, callbacks=[ScaffoldCallback])
    server = scaffold_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
