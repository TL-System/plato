"""
An implementation of the SwAV algorithm.

M. Caron, et al., "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments," NeurIPS 2020.

https://arxiv.org/abs/2006.09882

Source code: https://github.com/facebookresearch/swav
"""

import swav_model

from plato.clients import self_supervised_learning as ssl_client
from plato.datasources import self_supervised_learning as ssl_datasource
from plato.servers import fedavg_personalized as personalized_server


def main():
    """
    A self-supervised federated learning session with SwAV.
    """
    client = ssl_client.Client(
        model=swav_model.SwaV,
        datasource=ssl_datasource.SSLDataSource,
    )
    server = personalized_server.Server(model=swav_model.SwaV)

    server.run(client)


if __name__ == "__main__":
    main()
