"""Run this file to experiment with the Breaching code.

Note that parameter files to change are:
1. examples/fishing/breaching/config/attack/clsattack.yaml
2. examples/fishing/breaching/cases/users.py
    * Modify the number of local updates by FedAvg with self.num_local_updates
    * We do this because this parameter doesn't seem to be set when we change
    it in the YAML file...
"""

try:
    import breaching
except ModuleNotFoundError:
    # You only really need this safety net if you want to run these notebooks directly in the examples directory
    # Don't worry about this if you installed the package or moved the notebook to the main directory.
    import os; os.chdir("..")
    import breaching
    
import torch
import matplotlib.pyplot as plt

####

# Redirects logs directly into the jupyter notebook
import logging, sys
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
logger = logging.getLogger()

####

cfg = breaching.get_config(overrides=["case/server=malicious-fishing", "attack=clsattack", "case/user=multiuser_aggregate"])
          
device = torch.device(f'cuda:1') if torch.cuda.is_available() else torch.device('cpu')
torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
print(setup)

####

cfg.case.user.user_range = [0, 1]

cfg.case.data.partition = "random" # This is the average case
cfg.case.user.num_data_points = 256
cfg.case.data.default_clients = 32

cfg.case.user.provide_labels = True # Mostly out of convenience
cfg.case.server.target_cls_idx = 0 # Which class to attack?

####

user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)

# Print the server, user, and attacker
breaching.utils.overview(server, user, attacker)

####

[shared_data], [server_payload], true_user_data = server.run_protocol(user)

####

user.plot(true_user_data)

####

reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], 
                                                      server.secrets, dryrun=cfg.dryrun)

####

fished_data = dict(data=reconstructed_user_data["data"][server.secrets["ClassAttack"]["target_indx"]], 
                   labels=None)
user.plot(fished_data)

####

# metrics = breaching.analysis.report(reconstructed_user_data, true_user_data, [server_payload],
#                                     server.model, order_batch=True, compute_full_iip=False,
#                                     cfg_case=cfg.case, setup=setup)

####

user.plot(reconstructed_user_data)

####

# from breaching.cases.malicious_modifications.classattack_utils import print_gradients_norm, cal_single_gradients
# single_gradients, single_losses = cal_single_gradients(user.model, loss_fn, true_user_data, setup=setup)
# print_gradients_norm(single_gradients, single_losses)