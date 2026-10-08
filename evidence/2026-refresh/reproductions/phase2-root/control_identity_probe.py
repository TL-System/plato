import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms.feddyn_strategy import FedDynLossStrategy, FedDynUpdateStrategy
from plato.trainers.strategies.algorithms.scaffold_strategy import SCAFFOLDUpdateStrategy
from tests.integration.utils import build_minimal_config, configure_environment


def model():
    result = torch.nn.Linear(2, 2)
    with torch.no_grad():
        for parameter in result.parameters():
            parameter.fill_(1)
    return result


config = build_minimal_config(model_name='toy')
config['trainer'].update(batch_size=2, epochs=1)
results=[]
with tempfile.TemporaryDirectory(prefix='plato-control-identity-') as temporary:
    with configure_environment(config, runtime_root=Path(temporary)):
        torch.manual_seed(17)
        dataset = TensorDataset(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1]))
        for cls, attr in [(FedDynUpdateStrategy, 'grad_vector_path'), (SCAFFOLDUpdateStrategy, 'client_control_variate_path')]:
            for client_id in (1,2):
                strategy = cls()
                extra={'loss_strategy': FedDynLossStrategy()} if cls is FedDynUpdateStrategy else {}
                trainer = ComposableTrainer(model=model, model_update_strategy=strategy, **extra)
                trainer.set_client_id(client_id)
                trainer.device = torch.device('cpu')
                trainer.context.device = trainer.device
                try:
                    trainer.train_model({**config['trainer'], 'run_id': 'probe'}, dataset, [0, 1])
                    path=Path(getattr(strategy,attr))
                    results.append({'strategy':cls.__name__,'client_id':client_id,'path':path.name,'exists':path.is_file()})
                except Exception as error:
                    results.append({'strategy':cls.__name__,'client_id':client_id,'error':type(error).__name__+': '+str(error)})
        print('PROBE_RESULT', json.dumps(results))
