"""
The implementation for the SMoG [1] method.

[1]. Bo Pang, et al., Unsupervised Visual Representation Learning by Synchronous Momentum Grouping.
ECCV, 2022. https://arxiv.org/pdf/2006.07733.pdf.

Source code: None
"""


from pflbases import fedavg_personalized_server
from pflbases import fedavg_partial

from pflbases.client_callbacks import local_completion_callbacks
from pflbases.models import SSL
from pflbases import ssl_client
from pflbases import ssl_datasources


from smog_trainer import Trainer


def main():
    """
    A personalized federated learning sesstion for SMoG approach.
    """
    trainer = Trainer
    client = ssl_client.Client(
        model=SSL.SMoG,
        datasource=ssl_datasources.TransformedDataSource,
        personalized_datasource=ssl_datasources.TransformedDataSource,
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        callbacks=[
            local_completion_callbacks.ClientModelLocalCompletionCallback,
        ],
    )
    server = fedavg_personalized_server.Server(
        model=SSL.SMoG,
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
