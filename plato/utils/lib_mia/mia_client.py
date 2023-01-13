"""
References:

Liu et al., "FedEraser: Enabling Efficient Client-Level Data Removal from Federated Learning Models,"
in IWQoS 2021.

Shokri et al., "Membership Inference Attacks Against Machine Learning Models," in IEEE S&P 2017.

https://ieeexplore.ieee.org/document/9521274
https://arxiv.org/pdf/1610.05820.pdf
"""
from types import SimpleNamespace

from plato.clients import simple


class Client(simple.Client):
    """A federated learning client of federated unlearning with local PGA."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=None,
        )

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        """Customizes the report with assigned sample indices."""
        report.indices = self.sampler.subset_indices
        report.deleted_indices = []
        if hasattr(self.sampler, "deleted_subset_indices"):
            report.deleted_indices = self.sampler.deleted_subset_indices
        return report
