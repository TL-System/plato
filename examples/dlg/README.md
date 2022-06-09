# DLG related configuration parameters

`attack_round: 1` — reconstruct data from gradients/weights obtained in a particular communication round; 1st round by default.

`victim_client: 0` — reconstruct data from one particular client, e.g., the first selected client.

`num_iters: 100` — the number of iterations for matching gradients/weights.

`log_interval: 10` — determines how often the matching loss, performance, etc., are logged.

`init_params: true` — whether or not use customized initialization for model weights.

`share_gradients: false` — whether sharing gradients or model weights.

`match_weight: true` — reconstruct data by directly matching weights when it's true; reconstruct data by matching gradients calculated from updates otherwise; setting `share_gradients: false` is a prerequisite.

`random_seed: 50` — the random seed for generating dummy data and label.

## Plot instructions

Run ```python plot.py -c configuration_file``` where ```configuration_file``` is the same one used to run the DLG attack.