import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from plato.trainers.split_learning import Trainer
from tests.integration.utils import build_minimal_config, configure_environment

config = build_minimal_config(model_name='toy', trainer_type='split_learning')
config['trainer'].update(batch_size=2, epochs=1)
results=[]
with tempfile.TemporaryDirectory(prefix='plato-split-evaluation-') as temporary:
    with configure_environment(config, runtime_root=Path(temporary)):
        model = torch.nn.Linear(2, 2)
        trainer = Trainer(model=model)
        trainer.device = torch.device('cpu')
        trainer.context.device = trainer.device
        dataset = TensorDataset(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1]))
        for label, data in [('fresh_public_test', dataset), ('empty_after_run_start', TensorDataset(torch.empty(0, 2), torch.empty(0,dtype=torch.long)))]:
            if label.startswith('empty'):
                trainer.callback_handler.call_event('on_train_run_start', trainer, config['trainer'])
            try:
                result=trainer.test(data)
                results.append({'case':label, 'value':result})
            except Exception as error:
                results.append({'case':label, 'error':type(error).__name__+': '+str(error)})
print('PROBE_RESULT', json.dumps(results))
