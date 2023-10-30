"""
The implementation for the SimSiam [1] method.

[1]. Xinlei Chen, et al., Exploring Simple Siamese Representation Learning.
https://arxiv.org/pdf/2011.10566.pdf

Source code: https://github.com/facebookresearch/simsiam
Third-party code: https://github.com/PatrickHua/SimSiam
"""

from plato.trainers import loss_criterion

from pflbases import fedavg_personalized_server
from pflbases import fedavg_partial

from pflbases.client_callbacks import local_completion_callbacks
from pflbases.models import SSL
from pflbases import ssl_client
from pflbases import ssl_trainer
from pflbases import ssl_datasources


class Trainer(ssl_trainer.Trainer):
    """A trainer for SimSiam to rewrite the loss wrapper."""

    def plato_ssl_loss_wrapper(self):
        """A wrapper to connect ssl loss with plato."""
        defined_ssl_loss = loss_criterion.get()

        def compute_plato_loss(outputs, labels):
            if isinstance(outputs, (list, tuple)):
                loss = 0.5 * (
                    defined_ssl_loss(*outputs[0]) + defined_ssl_loss(*outputs[1])
                )
                return loss
            else:
                return defined_ssl_loss(outputs)

        return compute_plato_loss


def main():
    """
    A personalized federated learning sesstion for SimSiam approach.
    """
    trainer = Trainer
    client = ssl_client.Client(
        model=SSL.SimSiam,
        datasource=ssl_datasources.TransformedDataSource,
        personalized_datasource=ssl_datasources.TransformedDataSource,
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        callbacks=[
            local_completion_callbacks.ClientModelLocalCompletionCallback,
        ],
    )
    server = fedavg_personalized_server.Server(
        model=SSL.SimSiam,
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
