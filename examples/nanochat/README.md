# Nanochat Integration Workspace

This workspace hosts Nanochat-focused experiments within Plato.

## Quick Start

1. Install dependencies (including the vendored tokenizer build requirements):

   ```bash
   uv sync --extra nanochat
   uv run --with ./external/nanochat maturin develop --release
   ```

2. Run the synthetic smoke configuration:

   ```bash
   uv run --extra nanochat python plato.py --config configs/Nanochat/synthetic_micro.toml
   ```

   This launches a single-client training round using the Nanochat trainer, synthetic
   token streams, and a downsized GPT configuration for CPU debugging.

## CORE Evaluation

The Nanochat trainer can invoke the upstream CORE benchmark by adding the section
below to your TOML configuration:

```toml
[evaluation]
type = "nanochat_core"
max_per_task = 128  # optional; limits evaluation samples per task
# bundle_dir = "/custom/path/to/nanochat"  # defaults to ~/.cache/nanochat
```

Make sure the official evaluation bundle has been downloaded so the following files
exist (the default location is `~/.cache/nanochat/eval_bundle`):

- `core.yaml`
- `eval_data/*.jsonl`
- `eval_meta_data.csv`

The provided `configs/Nanochat/synthetic_micro.toml` can be extended with the
`[evaluation]` block once those assets are present.

## Roadmap

- Integrate real Nanochat tokenized datasets and publish download helpers.
- Add baseline evaluation scripts leveraging `nanochat/core_eval.py`.
- Capture reproducible metrics and hardware notes for larger-scale runs.
