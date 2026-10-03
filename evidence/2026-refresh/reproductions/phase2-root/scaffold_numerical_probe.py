import json
import tempfile
from pathlib import Path

import torch

from plato.trainers.strategies.base import TrainingContext
from plato.trainers.strategies.algorithms.scaffold_strategy import SCAFFOLDUpdateStrategy
from tests.integration.utils import build_minimal_config, configure_environment

with tempfile.TemporaryDirectory(prefix='plato-scaffold-numerical-') as temporary:
    with configure_environment(build_minimal_config(model_name='toy'), runtime_root=Path(temporary)):
        model = torch.nn.Linear(1,1,bias=False,dtype=torch.float64)
        with torch.no_grad():
            model.weight.fill_(1)
        context=TrainingContext()
        context.model=model
        context.device=torch.device('cpu')
        context.client_id=1
        context.state.update(learning_rate=0.1,server_control_variate={'weight':torch.tensor([[2.]],dtype=torch.float64)})
        strategy=SCAFFOLDUpdateStrategy()
        strategy.setup(context)
        strategy.client_control_variate={'weight':torch.tensor([[1.]],dtype=torch.float64)}
        strategy.on_train_start(context)
        optimizer=torch.optim.SGD(model.parameters(),lr=0.1)
        optimizer.zero_grad()
        (model.weight.sum()*0).backward()
        optimizer.step()
        strategy.after_step(context)
        strategy.on_train_end(context)
        result={'initial_weight':1.,'server_c':2.,'old_client_c':1.,'local_gradient':0.,'lr':0.1,'actual_weight':model.weight.item(),'actual_new_client_c':strategy.client_control_variate['weight'].item(),'expected_algorithm1_weight':0.9,'expected_equation4_new_client_c':0.0}
        print('PROBE_RESULT',json.dumps(result))
