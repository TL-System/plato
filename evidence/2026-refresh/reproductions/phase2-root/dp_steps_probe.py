import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from plato.callbacks.trainer import TrainerCallback
from plato.trainers.diff_privacy import Trainer
from tests.integration.utils import build_minimal_config, configure_environment

class Observe(TrainerCallback):
    def __init__(self):
        self.events = []

    def on_train_step_end(self, trainer, config, batch, loss, **kwargs):
        self.events.append({"batch": batch, "optimizer_skipped": trainer.optimizer._is_last_step_skipped})

config = build_minimal_config(model_name="toy", trainer_type="diff_privacy")
config["trainer"].update(batch_size=8, epochs=1, max_physical_batch_size=4)
with tempfile.TemporaryDirectory(prefix="plato-dp-steps-probe-") as temporary:
    with configure_environment(config, runtime_root=Path(temporary)):
        torch.manual_seed(15)
        observer = Observe()
        trainer = Trainer(model=lambda: torch.nn.Linear(2, 2), callbacks=[observer])
        trainer.device = torch.device("cpu")
        trainer.context.device = trainer.device
        trainer.set_client_id(1)
        dataset = TensorDataset(torch.randn(32, 2), torch.randint(0, 2, (32,)))
        trainer.train_model({**config["trainer"], "run_id": "probe"}, dataset, list(range(32)))
        print("PROBE_RESULT", json.dumps({"step_end_events": observer.events, "accountant_history": trainer.optimizer_strategy.privacy_engine.accountant.history}))
