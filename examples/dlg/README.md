# Gradient Leakage in Production Federated Learning

Research on gradient leakage attack and defense using Plato framework.

--- 
# Ported Algorithms

## Attack baselines
- {DLG} [Zhu et al., "Deep Leakage from Gradients," NeurIPS 2019.](https://papers.nips.cc/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html) – [[Code available]](https://github.com/mit-han-lab/dlg)

- {iDLG} [Zhao et al., "iDLG: Improved Deep Leakage from Gradients," arXiv 2020.](https://arxiv.org/pdf/2001.02610.pdf) – [[Code available]](https://github.com/PatrickZH/Improved-Deep-Leakage-from-Gradients)

- {csDLG} [Geiping et al., "Inverting Gradients - How easy is it to break privacy in federated learning?," NeurIPS 2020.](https://proceedings.neurips.cc/paper/2020/file/c4ede56bbd98819ae6112b20ac6bf145-Paper.pdf) – [[Code available]](https://github.com/JonasGeiping/invertinggradients)

## Defense baselines

- {Soteria} [Sun et al., "Soteria: Provable Defense against Privacy Leakage in Federated Learning from Representation Perspective," CVPR 2021.](https://openaccess.thecvf.com/content/CVPR2021/papers/Sun_Soteria_Provable_Defense_Against_Privacy_Leakage_in_Federated_Learning_From_CVPR_2021_paper.pdf) – [[Code available]](https://github.com/jeremy313/Soteria)

- {GradDefense} [Wang et al., "Protect Privacy from Gradient Leakage Attack in Federated Learning," INFOCOM 2022.](https://infocom.info/day/2/track/Track%20A#A-3) – [[Code available]](https://github.com/wangjunxiao/GradDefense)

- {DP} differential privacy (referred to Soteria's code) 

- {GC} gradient compression (referred to Soteria's code) 
---

# Running Plato with DLG attack
Try replacing `rec_emnist.yml` below with custom config files:
```
python examples/dlg.py -c examples/dlg/rec_emnist.yml
```

## DLG Related Configurations
Try tuning the following hyperparameters in .yml configuration files.

### under `algorithm`

- `attack_method: [DLG, iDLG, csDLG]` — DLG attack variants.

- `cost_fn: [l2, l1, max, sim, simlocal]` — different way to calculate the loss function used in DLG attack variants.

- `total_variation: [float]` — weight for adding total data variation to the loss used in DLG attack variants.

- `lr: float` — learning rate of the optimizer for data reconstruction

- `attack_round: [int]` — which particular communication round to reconstruct data from gradients/weights obtained; 1st round is what literature usually assumes.

- `victim_client: [int]` — which particular client to reconstruct data from, e.g., the 1st selected client.

- `num_iters: [int]` — the number of iterations for matching gradients/weights.

- `log_interval: [int]` — how often the matching loss, performance, etc., are logged.

- `init_params: [boolean]` — whether or not use explicit initialization for model weights.

- `share_gradients: [boolean]` — whether sharing gradients or model weights.

- `match_weights: [boolean]` — when `share_gradients` is set to `false`, whether reconstructing data by directly matching weights or by matching gradients calculated from model updates.

- `use_updates: [boolean]` — when `share_gradients` is set to `false` and `match_weights` is set to `true`, whether matching weights (absolute) or model updates (delta).

- `trials: [int]` - the number of times the attack will be performed, with different initializations for dummy data.

- `random_seed: [int]` — the random seed for generating dummy data and label.

- `defense: [no, GC, DP, Soteria, GradDefense, Outpost(ours)]` — different defenses against DLG attacks.

- (for GC) `prune_pct: 80` 
- (for DP) `epsilon: 0.1`
- (for Soteria) `threshold: 50`
- (for GradDefense) `clip: true`
- (for GradDefense) `slices_num: 10`
- (for GradDefense) `perturb_slices_num: 5`
- (for GradDefense) `scale: 0.01`
- (for GradDefense) `Q: 6`
- (for Outpost) `beta: 0.125` — iteration decay speed
- (for Outpost) `phi: 40` — the amount (%) of perturbation
- (for Outpost) `prune_base: 80` — pruning percentage
- (for Outpost) `noise_base: 0.8` — scale for gaussian noise


### under `results`

- `subprocess: [int]` - the name of the subdirectory (subprocess) from which the file should be loaded from. Takes the latest one if not specified

- `trial: [int]` - the trial number to be plotted. Plots the best reconstruction based off MSE if not specified

## Plot Instructions

Run ```python plot.py -c config_file``` where ```config_file``` is the same one used to run the DLG attack.


