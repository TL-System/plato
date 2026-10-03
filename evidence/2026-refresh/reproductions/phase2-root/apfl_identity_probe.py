import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms.apfl_strategy import APFLStepStrategy, APFLUpdateStrategy
from tests.integration.utils import build_minimal_config, configure_environment


def model():
    result = torch.nn.Linear(2, 2)
    with torch.no_grad():
        for parameter in result.parameters():
            parameter.fill_(1)
    return result


config = build_minimal_config(model_name='toy')
config['trainer'].update(batch_size=2, epochs=1)
with tempfile.TemporaryDirectory(prefix='plato-apfl-identity-') as temporary:
    with configure_environment(config, runtime_root=Path(temporary)):
        torch.manual_seed(17)
        dataset = TensorDataset(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1]))
        paths = []
        for client_id in (1, 2):
            strategy = APFLUpdateStrategy(model_fn=model)
            trainer = ComposableTrainer(model=model, model_update_strategy=strategy, training_step_strategy=APFLStepStrategy())
            trainer.set_client_id(client_id)
            trainer.device = torch.device('cpu')
            trainer.context.device = trainer.device
            trainer.train_model({**config['trainer'], 'run_id': 'probe'}, dataset, [0, 1])
            paths.append({'client_id': client_id, 'model_path': Path(strategy.personalized_model_path).name, 'alpha_path': Path(strategy.alpha_path).name, 'alpha': float(strategy.alpha)})
        print('PROBE_RESULT', json.dumps({'apfl_paths': paths}))
