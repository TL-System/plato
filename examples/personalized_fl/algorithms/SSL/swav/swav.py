"""
The implementation for the SwAV [1] method.

[1]. Mathilde Caron, et al., Unsupervised Learning of Visual Features by Contrasting Cluster Assignments.
https://arxiv.org/abs/2006.09882, NeurIPS 2020.

Source code: https://github.com/facebookresearch/swav
"""

from pflbases import fedavg_personalized_server
from pflbases import fedavg_partial

from pflbases.client_callbacks import local_completion_callbacks
from pflbases.models import SSL
from pflbases import ssl_client
from pflbases import ssl_trainer
from pflbases import ssl_datasources


def main():
    """
    A pFL sesstion for SwaV approach.
    """
    trainer = ssl_trainer.Trainer
    client = ssl_client.Client(
        model=SSL.SwaV,
        datasource=ssl_datasources.TransformedDataSource,
        personalized_datasource=ssl_datasources.TransformedDataSource,
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        callbacks=[
            local_completion_callbacks.ClientModelLocalCompletionCallback,
        ],
    )
    server = fedavg_personalized_server.Server(
        model=SSL.SwaV,
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
