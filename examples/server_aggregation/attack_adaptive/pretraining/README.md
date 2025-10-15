## Pretraining the Attack-Adaptive Attention Module

Follow the steps below to obtain a checkpoint that matches the attack-adaptive
aggregation released with *Wan & Chen (2021)*.

### 1. Capture rounds from a simulation

Edit your experiment YAML so the attack-adaptive strategy records each round.
Add the following keys under `algorithm` (directories are created if missing):

```yaml
algorithm:
  type: fedavg
  attention_model_path: examples/server_aggregation/attack_adaptive/attention_model.pt
  scaling_factor: 10
  threshold: 0.005
  attention_loops: 5
  attention_hidden: 32
  pca_components: 10
  dataset_capture_dir: ./attack_adaptive_dataset
```

Run the experiment as usual:

```bash
uv run python examples/server_aggregation/attack_adaptive/attack_adaptive.py \
  -c examples/server_aggregation/attack_adaptive/attack_adaptive_MNIST_lenet5.yml
```

Each run produces a timestamped folder under `./attack_adaptive_dataset/` (for
example `run_20251015-105500`) containing one `round_XXXXX.pt` file per round
and a `metadata.json` summary.

### 2. Train the attention network

Point the pretraining script to the captured directory and choose where to save
the checkpoint:

```bash
uv run examples/server_aggregation/attack_adaptive/pretraining/train_attention.py \
  --dataset-dir ./attack_adaptive_dataset/run_20251015-105500 \
  --save-path examples/server_aggregation/attack_adaptive/attention_model.pt \
  --epochs 200
```

Additional options:

- `--batch-size` (default `16`)
- `--learning-rate` (default `1e-4`)
- `--val-ratio` (default `0.1`, fraction of rounds used for validation)
- `--epsilon`, `--scale`, `--attention-loops`, `--hidden-size` to match paper
  hyperparameters.

The command prints the training and validation losses each epoch and reports the
best checkpoint once training finishes.

### 3. Re-run with the trained model

Ensure `algorithm.attention_model_path` in the experiment YAML points to the
new checkpoint, then launch a fresh run. The server will load the pretrained
weights automatically.

> **Tip:** Capture rounds from several scenarios (with/without attacks) and
> merge them into a single directory before training to improve robustness.
