# DLG related configuration parameters

`attack_round: 1` 每 reconstruct data from gradients/weights obtained in a particular communication round; 1st round by default.

`victim_client: 0` 每 reconstruct data from one particular client, e.g., the first selected client.

`num_iters: 100` 每 the number of iterations for matching gradients/weights.

`log_interval: 10` 每 determines how often the matching loss, performance, etc., are logged.

`init_params: true` 每 whether or not use customized initialization for model weights.

`share_gradients: false` 每 whether sharing gradients or model weights.

`match_weight: true` 每 reconstruct data by directly matching weights when it's true; reconstruct data by matching gradients calculated from updates otherwise; setting `share_gradients: false` is a prerequisite.

`random_seed: 50` 每 the random seed for generating dummy data and label.