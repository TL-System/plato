# Nanochat Integration Workspace

This workspace hosts Nanochat-focused experiments within Plato.

## Quick Start

For the full working setup, see:

- [Nanochat in Plato](../../docs/docs/examples/case-studies/5. Nanochat in Plato.md)

The short version is:

1. Initialize the nanochat submodule:

   ```bash
   git submodule update --init --recursive
   ```

2. Install dependencies:

   ```bash
   uv sync --extra nanochat
   ```

3. Install `maturin` if needed, then build the Rust tokenizer extension for Plato's environment:

   ```bash
   uv tool install maturin
   uv run --extra nanochat maturin develop --release --manifest-path external/nanochat/rustbpe/Cargo.toml
   ```

4. Run the synthetic configuration **after either** (a) preparing the tokenizer for CORE evaluation or (b) disabling the `[evaluation]` block in a local config copy:

   ```bash
   uv run --extra nanochat python plato.py --config configs/Nanochat/synthetic_micro.toml --cpu
   ```

## Important note about CORE evaluation

`configs/Nanochat/synthetic_micro.toml` already enables:

```toml
[evaluation]
type = "nanochat_core"
```

That means a tokenizer must exist under `~/.cache/nanochat/tokenizer/` before the
run can finish successfully.

To prepare it, use the commands documented in:

- `docs/docs/examples/case-studies/5. Nanochat in Plato.md`

In particular, the current working flow is:

1. download at least 2 Nanochat data shards
2. train a tokenizer with `external/nanochat/scripts/tok_train.py`
3. then run the Plato config

The CORE evaluation bundle itself is downloaded automatically on first use.

## Roadmap

- Integrate real Nanochat tokenized datasets and publish download helpers.
- Add baseline evaluation scripts leveraging `nanochat/core_eval.py`.
- Capture reproducible metrics and hardware notes for larger-scale runs.
