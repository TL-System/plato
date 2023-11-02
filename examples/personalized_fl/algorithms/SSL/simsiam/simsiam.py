"""
The implementation for the SimSiam [1] method.

[1]. Xinlei Chen, et al., Exploring Simple Siamese Representation Learning.
https://arxiv.org/pdf/2011.10566.pdf

Reference:
Source code: https://github.com/facebookresearch/simsiam
Third-party code: https://github.com/PatrickHua/SimSiam
"""

from pflbases import fedavg_personalized

from pflbases import ssl_client
from pflbases import ssl_datasources

import simsiam_trainer
import simsiam_model


def main():
    """
    A personalized federated learning session for SimSiam approach.
    """
    client = ssl_client.Client(
        model=simsiam_model.SimSiam,
        datasource=ssl_datasources.SSLDataSource,
        trainer=simsiam_trainer.Trainer,
    )
    server = fedavg_personalized.Server(
        model=simsiam_model.SimSiam, trainer=simsiam_trainer.Trainer
    )

    server.run(client)


if __name__ == "__main__":
    main()
