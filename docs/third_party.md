# Third-Party Assets

This page records external projects that are vendored into the Plato repository to support specific integrations. Please update the relevant entry whenever the upstream source, commit hash, or licensing information changes.

## Nanochat
- **Upstream:** [karpathy/nanochat](https://github.com/karpathy/nanochat)
- **Location:** `external/nanochat` (git submodule)
- **License:** MIT (included in `external/nanochat/LICENSE`)

### Updating the Submodule
1. `git submodule update --remote external/nanochat`
2. Inspect upstream changes for compatibility with Plato.
3. Commit the submodule pointer update and note any required integration work in the checklist.

### Notes
- After cloning Plato, run `git submodule update --init --recursive` to populate all external dependencies.
- The Rust tokenizer (`rustbpe`) builds via `maturin`. Ensure `uv run --with ./external/nanochat maturin develop --release` succeeds before pushing updates.
- Avoid local modifications inside the submodule; contribute fixes upstream when possible.
