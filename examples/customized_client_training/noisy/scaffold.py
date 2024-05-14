"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
from scaffold import scaffold_client
from scaffold import scaffold_server
from scaffold import scaffold_trainer
from scaffold.scaffold_callback import ScaffoldCallback
from plato.config import Config

from  noisy_datasource import NoisyDataSource


def main():
    """A Plato federated learning training session using the SCAFFOLD algorithm."""
    if hasattr(Config().data, "noise"):
        datasource = NoisyDataSource
    else:
        datasource = None

    trainer = scaffold_trainer.Trainer
    client = scaffold_client.Client(datasource=datasource,trainer=trainer, callbacks=[ScaffoldCallback])
    server = scaffold_server.Server(datasource=datasource,trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
