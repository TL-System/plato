"""Actual CPU GradScaler plus public clipping wrapper numerical reference."""
import json
import tempfile
import warnings
from pathlib import Path
import torch
from torch.utils.data import TensorDataset
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.loss_criterion import DefaultLossCriterionStrategy
from plato.trainers.strategies.optimizer import DefaultOptimizerStrategy, GradientClippingOptimizerStrategy
from plato.trainers.strategies.training_step import MixedPrecisionStepStrategy
from tests.integration.utils import build_minimal_config, configure_environment

records=[]
for fused in (False, True):
 config=build_minimal_config()
 config['parameters']['optimizer'].update(lr=0.1,fused=fused)
 config['trainer'].update(batch_size=1,epochs=1)
 with tempfile.TemporaryDirectory(prefix='amp-clipping-',dir='/tmp/plato-phase2-root') as tmp:
  with configure_environment(config,runtime_root=Path(tmp)):
   model=torch.nn.Linear(1,1,bias=False)
   model.weight.data.fill_(1)
   step=MixedPrecisionStepStrategy(enabled=False)
   trainer=ComposableTrainer(model=model,training_step_strategy=step,
    loss_strategy=DefaultLossCriterionStrategy(lambda outputs,labels:10*outputs.mean()),
    optimizer_strategy=GradientClippingOptimizerStrategy(DefaultOptimizerStrategy(),max_norm=1))
   step.enabled=True
   step.scaler=torch.amp.GradScaler('cpu')
   data=TensorDataset(torch.ones(1,1),torch.zeros(1,1))
   with warnings.catch_warnings(record=True) as seen:
    warnings.simplefilter('always')
    trainer.train_model({**config['trainer'],'run_id':'ampclip'},data,[0])
   expected=1-0.1*10/(10+1e-6)
   records.append({'fused':fused,'initial_scale':65536,'actual':model.weight.item(),'expected':expected,'error':abs(model.weight.item()-expected),'warnings':[str(w.message) for w in seen]})
print(json.dumps(records,indent=2))
