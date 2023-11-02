"""
The implementation for the SMoG [1] method.

[1]. Bo Pang, et al., Unsupervised Visual Representation Learning by Synchronous Momentum Grouping.
ECCV, 2022. https://arxiv.org/pdf/2006.07733.pdf.

Source code: None
"""


from pflbases import fedavg_personalized
from pflbases import fedavg_partial

from pflbases import ssl_client
from pflbases import ssl_datasources


import smog_trainer
import smog_model


def main():
    """
    A personalized federated learning session for SMoG approach.
    """
    trainer = smog_trainer.Trainer
    client = ssl_client.Client(
        model=smog_model.SMoG,
        datasource=ssl_datasources.SSLDataSource,
        trainer=trainer,
    )
    server = fedavg_personalized.Server(
        model=smog_model.SMoG,
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
