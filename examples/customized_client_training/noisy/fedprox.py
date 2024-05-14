"""
A federated learning training session using FedProx.

To better handle system heterogeneity, the FedProx algorithm introduced a
proximal term in the optimizer used by local training on the clients. It has
been quite widely cited and compared with in the federated learning literature.

Reference:
Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020).
"Federated optimization in heterogeneous networks." Proceedings of Machine
Learning and Systems, 2, 429-450.

https://proceedings.mlsys.org/paper/2020/file/38af86134b65d0f10fe33d30dd76442e-Paper.pdf
"""
from plato.config import Config
from fedprox import fedprox_trainer
from  noisy_datasource import NoisyDataSource

from plato.servers import fedavg
from plato.clients import simple


def main():
    """A Plato federated learning training session using FedProx."""
    if hasattr(Config().data, "noise"):
        datasource = NoisyDataSource
    else:
        datasource = None

    trainer = fedprox_trainer.Trainer
    client = simple.Client(datasource=datasource, trainer=trainer)
    server = fedavg.Server(datasource=datasource, trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
