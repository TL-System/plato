# Nanochat Integration Checklist

This checklist coordinates the work to incorporate the Nanochat stack into Plato. Owners are placeholder roles until specific engineers are assigned.

## Third-Party Submodule
- **Owner:** Infrastructure
- **Deliverables:** Maintain the `external/nanochat` git submodule; document update procedure in `docs/third_party.md`.
- **Dependencies:** None.

## Model Registry
- **Owner:** Modeling
- **Deliverables:** Implement `plato/models/nanochat.py` mirroring Nanochat GPT config, register entry in `plato/models/registry.py`, supply weight-loading utilities.
- **Status:** In progress – factory module and registry wiring landed.
- **Dependencies:** Third-party submodule.

## Tokenizer & Processor
- **Owner:** Infrastructure
- **Deliverables:** Package Rust BPE via Maturin optional extra; wrap as `plato/processors/nanochat_tokenizer.py` with lazy import and fallbacks; document build steps in README.
- **Status:** Prototype processor and optional dependency group landed; CI build integration remains TODO.
- **Dependencies:** Third-party submodule, build tooling prototype.

## Datasource
- **Owner:** Data
- **Deliverables:** Create `plato/datasources/nanochat.py` handling dataset acquisition and sharding; register in datasource registry; store license metadata.
- **Status:** In progress – streaming dataset with synthetic fallback available.
- **Dependencies:** Tokenizer availability.

## Trainer & Algorithm
- **Owner:** Training
- **Deliverables:** Port Nanochat engine into `plato/trainers/nanochat.py`; add algorithm glue if federated coordination diverges; ensure checkpoint compatibility.
- **Status:** In progress – composable trainer wrapper with Nanochat-specific optimiser/loader strategies in place.
- **Dependencies:** Model registry entry, datasource.

## Evaluation Strategy
- **Owner:** Evaluation
- **Deliverables:** Translate `nanochat/core_eval.py` into reusable evaluator hooked into Plato testing strategy; add pytest coverage with synthetic data.
- **Status:** CORE evaluation adapter hooked into trainer testing strategy; follow-up coverage to use real eval bundles outstanding.
- **Dependencies:** Model, tokenizer.

## Configuration & Examples
- **Owner:** Product
- **Deliverables:** Author `configs/Nanochat/*.toml` scenarios and `examples/nanochat/` workspace; include reference scripts and documentation.
- **Status:** Synthetic micro config and workspace README published; larger-scale scenarios pending.
- **Dependencies:** Model, datasource, trainer.

## Documentation & Release
- **Owner:** Docs
- **Deliverables:** Publish `docs/models/nanochat.md`, extend root README tables, add integration notes and changelog entry; outline hardware requirements.
- **Dependencies:** All prior tracks.

## Validation
- **Owner:** QA
- **Deliverables:** Expand CI to compile tokenizer, run smoke train/eval, and enforce import order checks; record expected metrics in evaluation baselines.
- **Status:** Initial pytest smoke checks for tokenizer/trainer added; CI enablement still pending.
- **Dependencies:** Evaluation strategy, trainer.
