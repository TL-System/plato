import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from plato.trainers.composable import ComposableTrainer
from tests.integration.utils import build_minimal_config, configure_environment

config = build_minimal_config(model_name='lenet5')
with tempfile.TemporaryDirectory(prefix='plato-urgent-checkpoint-') as temporary:
    with configure_environment(config, runtime_root=Path(temporary)):
        trainer = ComposableTrainer()
        trainer.set_client_id(7)
        trainer.device = torch.device('cpu')
        trainer.context.device = trainer.device
        for epoch in (9,10):
            with torch.no_grad():
                for parameter in trainer.model.parameters():
                    parameter.fill_(epoch)
            trainer.save_model(f'7_{epoch}_{epoch}.0.safetensors')
        try:
            with patch.object(torch, "load", wraps=torch.load) as observed_load:
                historical=trainer.obtain_model_at_time(7,11.0)
            result={'selected_weight':float(next(historical.parameters()).flatten()[0])}
        except Exception as error:
            result={'error':type(error).__name__+': '+str(error)}
        result['attempted_reads']=[Path(call.args[0]).name for call in observed_load.call_args_list]
        print('PROBE_RESULT', json.dumps(result))
