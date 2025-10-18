"""
References:

Liu et al., "FedEraser: Enabling Efficient Client-Level Data Removal from Federated Learning Models,"
in IWQoS 2021.

Shokri et al., "Membership Inference Attacks Against Machine Learning Models," in IEEE S&P 2017.

https://ieeexplore.ieee.org/document/9521274
https://arxiv.org/pdf/1610.05820.pdf
"""

from plato.clients import simple
from plato.clients.strategies.base import ClientContext
from plato.clients.strategies.defaults import DefaultReportingStrategy


class FedUnlearningReportingStrategy(DefaultReportingStrategy):
    """Reporting strategy that records sampler indices for unlearning."""

    def build_report(self, context: ClientContext, report):
        report = super().build_report(context, report)

        sampler = getattr(context, "sampler", None)

        if sampler is not None and hasattr(sampler, "subset_indices"):
            report.indices = sampler.subset_indices
            report.deleted_indices = []
            if hasattr(sampler, "deleted_subset_indices"):
                report.deleted_indices = sampler.deleted_subset_indices
        else:
            report.indices = []
            report.deleted_indices = []

        return report


def create_client(
    *,
    model=None,
    datasource=None,
    algorithm=None,
    trainer=None,
    callbacks=None,
):
    """Build a federated unlearning client with sampler metadata reporting."""
    client = simple.Client(
        model=model,
        datasource=datasource,
        algorithm=algorithm,
        trainer=trainer,
        callbacks=callbacks,
    )

    client._configure_composable(
        lifecycle_strategy=client.lifecycle_strategy,
        payload_strategy=client.payload_strategy,
        training_strategy=client.training_strategy,
        reporting_strategy=FedUnlearningReportingStrategy(),
        communication_strategy=client.communication_strategy,
    )

    return client


# Maintain compatibility for imports expecting a Client callable/class.
Client = create_client
