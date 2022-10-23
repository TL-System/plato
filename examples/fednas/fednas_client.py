from types import SimpleNamespace

from Darts.model_search_local import MaskedNetwork

from plato.clients import simple
from plato.config import Config


class Client(simple.Client):
    """A personalized federated learning client using the FedRep algorithm."""

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(model, datasource, algorithm, trainer)

        # parameter names of the representation
        #   As mentioned by Eq. 1 and Fig. 2 of the paper, the representation
        #   behaves as the global model.
        self.representation_param_names = []

    def process_server_response(self, server_response) -> None:
        self.algorithm.mask_normal = server_response["mask_normal"]
        self.algorithm.mask_reduce = server_response["mask_reduce"]
        self.algorithm.model = MaskedNetwork(
            Config().parameters.model.C,
            Config().parameters.model.num_classes,
            Config().parameters.model.layers,
            self.algorithm.mask_normal,
            self.algorithm.mask_reduce,
        )
        self.trainer.model = self.algorithm.model

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        """Customizes the report with any additional information."""
        report.mask_normal = self.algorithm.mask_normal
        report.mask_reduce = self.algorithm.mask_reduce
        return report
