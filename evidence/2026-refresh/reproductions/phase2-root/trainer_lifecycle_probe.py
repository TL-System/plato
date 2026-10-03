import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from plato.config import Config
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms.ditto_strategy import DittoUpdateStrategy
from plato.trainers.strategies.algorithms.fedprox_strategy import FedProxLossStrategy
from tests.integration.utils import build_minimal_config, configure_environment

def model():
    result = torch.nn.Linear(2, 2)
    with torch.no_grad():
        for parameter in result.parameters():
            parameter.fill_(1)
    return result

class ObserveLoss(FedProxLossStrategy):
    def __init__(self):
        super().__init__(mu=0.1)
        self.observations = []

    def compute_loss(self, outputs, labels, context):
        total = super().compute_loss(outputs, labels, context)
        base = torch.nn.functional.cross_entropy(outputs, labels)
        self.observations.append({"base": base.item(), "total": total.item(), "proximal": (total-base).item(), "snapshot_weight": self.global_weights["weight"].tolist(), "current_weight": context.model.weight.detach().tolist()})
        return total

config = build_minimal_config(model_name="toy")
config["trainer"].update(batch_size=2, epochs=1)
with tempfile.TemporaryDirectory(prefix="plato-trainer-lifecycle-") as temporary:
    with configure_environment(config, runtime_root=Path(temporary)):
        torch.manual_seed(17)
        dataset = TensorDataset(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1]))
        paths = []
        for client_id in (1, 2):
            strategy = DittoUpdateStrategy(model_fn=model, personalization_epochs=1)
            trainer = ComposableTrainer(model=model, model_update_strategy=strategy)
            trainer.set_client_id(client_id)
            trainer.device = torch.device("cpu")
            trainer.context.device = trainer.device
            trainer.train_model({**config["trainer"], "run_id": "probe"}, dataset, [0, 1])
            path = Path(strategy.personalized_model_path)
            paths.append({"client_id": client_id, "path": path.name, "exists": path.is_file()})
        loss = ObserveLoss()
        trainer = ComposableTrainer(model=model, loss_strategy=loss)
        trainer.set_client_id(1)
        trainer.device = torch.device("cpu")
        trainer.context.device = trainer.device
        with torch.no_grad():
            for parameter in trainer.model.parameters():
                parameter.fill_(2)
        trainer.train_model({**config["trainer"], "run_id": "probe"}, dataset, [0, 1])
        print("PROBE_RESULT", json.dumps({"ditto_paths": paths, "fedprox_first_loss": loss.observations[0]}))
