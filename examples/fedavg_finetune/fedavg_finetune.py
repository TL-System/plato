"""
An implementation of the personalized learning variant of FedAvg.

Such an variant of FedAvg is recently mentioned and discussed in work [1].

[1] Liam Collins, et al., "Exploiting shared representations for personalized federated learning,"
in the Proceedings of ICML 2021.

    Address: https://proceedings.mlr.press/v139/collins21a.html

    Code: https://github.com/lgcollins/FedRep

"""

from fedavg_finetune_client import Client
from plato.servers import registry as server_registry


def main():
    """
    A Plato personalized federated learning training session using the FedAvg algorithm under the
    supervised learning setting.
    """
    client = Client()
    server = server_registry.get()

    server.run(client)


if __name__ == "__main__":
    main()
