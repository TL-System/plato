import json
import tempfile
import traceback
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from plato.trainers.diff_privacy import Trainer
from tests.integration.utils import build_minimal_config, configure_environment

config = build_minimal_config(model_name="toy", trainer_type="diff_privacy")
config["trainer"].update(batch_size=8, epochs=1, max_physical_batch_size=4)
with tempfile.TemporaryDirectory(prefix="plato-dp-probe-") as temporary:
    with configure_environment(config, runtime_root=Path(temporary)):
        torch.manual_seed(15)
        trainer = Trainer(model=lambda: torch.nn.Linear(2, 2))
        trainer.device = torch.device("cpu")
        trainer.context.device = trainer.device
        trainer.set_client_id(1)
        dataset = TensorDataset(torch.randn(32, 2), torch.randint(0, 2, (32,)))
        outcomes = []
        for number in (1,):
            try:
                trainer.train_model(
                    {**config["trainer"], "run_id": "probe"},
                    dataset,
                    list(range(32)),
                )
                outcomes.append({"round": number, "outcome": "passed", "model_type": type(trainer.model).__name__, "keys": list(trainer.model.state_dict())})
            except Exception as error:
                traceback.print_exc()
                outcomes.append({"round": number, "outcome": "failed", "type": type(error).__name__, "message": str(error)})
                break
        trainer.save_model(filename="dp.safetensors")
        try:
            trainer.load_model(filename="dp.safetensors")
            outcomes.append({"checkpoint_roundtrip": "passed"})
        except Exception as error:
            outcomes.append({"checkpoint_roundtrip": "failed", "type": type(error).__name__, "message": str(error)})
        print("PROBE_RESULT", json.dumps(outcomes))
