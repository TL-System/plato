# Third-Party Assets

This page records external projects that are vendored into the Plato repository to support specific integrations. Please update the relevant entry whenever the upstream source, commit hash, or licensing information changes.

## Nanochat
- **Upstream:** [karpathy/nanochat](https://github.com/karpathy/nanochat)
- **Vendored location:** `runtime/third_party/nanochat`
- **Snapshot commit:** `c75fe54aa7c1fa881701c246f9427bcbe4eee5a4` (captured 2025-03-04)
- **License:** MIT (included in `runtime/third_party/nanochat/LICENSE`)

### Updating the Snapshot
1. `cd runtime/third_party/nanochat`
2. `git fetch origin && git checkout <new_commit>`
3. Review upstream changes and confirm compatibility with Plato.
4. Record the new commit hash and date in this document, and call out notable changes in the integration checklist before landing.

### Notes
- The Rust tokenizer (`rustbpe`) builds via `maturin`. Ensure `uv run --with ./runtime/third_party/nanochat maturin develop --release` succeeds before pushing updates.
- Keep the vendored tree free of local modifications unless backporting fixes; prefer upstream contributions when feasible.
