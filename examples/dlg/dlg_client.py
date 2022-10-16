import pickle
from types import SimpleNamespace

from plato.clients import simple
from plato.config import Config


class Client(simple.Client):
    def __init__(self, model, trainer):
        super().__init__(model=model, trainer=trainer)

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        """Customizes the report with additional information about
        gt data and target gradients for attack validation at server."""
        file_path = f"{Config().params['model_path']}/{self.client_id}.pickle"
        with open(file_path, "rb") as handle:
            report.gt_data, report.gt_labels, report.target_grad = pickle.load(handle)

        return report
