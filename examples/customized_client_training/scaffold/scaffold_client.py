"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""

from plato.clients import simple
from plato.clients.strategies.defaults import DefaultLifecycleStrategy


class ScaffoldLifecycleStrategy(DefaultLifecycleStrategy):
    """The trainer strategy loads controls when configure assigns a logical ID.

    It is the sole control-state owner, including exact same-client legacy file
    migration. Inbound controls arrive after configure through the processor.
    """


def create_client(
    *,
    model=None,
    datasource=None,
    algorithm=None,
    trainer=None,
    callbacks=None,
):
    """Build a SCAFFOLD client configured with the lifecycle strategy."""
    client = simple.Client(
        model=model,
        datasource=datasource,
        algorithm=algorithm,
        trainer=trainer,
        callbacks=callbacks,
    )

    payload_strategy = client.payload_strategy
    training_strategy = client.training_strategy
    reporting_strategy = client.reporting_strategy
    communication_strategy = client.communication_strategy

    client._configure_composable(
        lifecycle_strategy=ScaffoldLifecycleStrategy(),
        payload_strategy=payload_strategy,
        training_strategy=training_strategy,
        reporting_strategy=reporting_strategy,
        communication_strategy=communication_strategy,
    )

    return client


# Backwards compatibility for previous imports expecting a Client class-like callable.
Client = create_client
