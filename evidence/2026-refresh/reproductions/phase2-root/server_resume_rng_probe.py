"""Actual generic checkpoint writer/reader through Server.run selection ordering."""
import asyncio
import copy
import json
import random
import tempfile
from pathlib import Path
import numpy as np
import torch
from plato.config import Config
from tests.integration.utils import build_minimal_config, configure_environment
from tests.integration.test_checkpoint_runtime_agreement import make_server

config=build_minimal_config(total_clients=10)
config['clients']['per_round']=2
config['server'].update(random_seed=17,disable_clients=True)
with tempfile.TemporaryDirectory(prefix='resume-rng-',dir='/tmp/plato-phase2-root') as tmp:
 with configure_environment(config,runtime_root=Path(tmp)):
  server=make_server()
  server.current_round=2
  server.disable_clients=True
  random.seed(17)
  np.random.seed(19)
  server.prng_state=random.getstate()
  population=list(range(1,11))
  prior=[server._select_clients_with_strategy(population,2) for _ in range(2)]
  committed_prng=server.prng_state
  expected_rng=random.Random();expected_rng.setstate(committed_prng)
  expected=expected_rng.sample(population,2)
  weights=copy.deepcopy(server.trainer.model.state_dict())
  server.save_to_checkpoint()
  expected_np=np.random.random()
  random.seed(99);np.random.seed(98)
  with torch.no_grad():server.trainer.model.weight.zero_()
  Config().args.resume=True
  # Keep the actual save, resume, run and random-selection methods. These
  # fixtures avoid datasource construction, client processes and sockets only.
  server.configure=lambda:None
  async def periodic(interval):pass
  server._periodic=periodic
  observed={}
  def start():
   observed.update(actual_next=server._select_clients_with_strategy(population,2),
    resumed=server.resumed_session,current_round=server.current_round,
    numpy_restored=np.random.random()==expected_np,
    weights_restored=all(torch.equal(server.trainer.model.state_dict()[k],v) for k,v in weights.items()))
  server.start=start
  server.run()
  print(json.dumps({'prior_selections':prior,'expected_next':expected,'actual':observed,'checkpoint_files':[p.name for p in Path(Config.params['checkpoint_path']).iterdir()]},indent=2))
