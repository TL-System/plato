# Getting Started

In `examples/`, we included a wide variety of examples that showed how federated learning algorithms in the research literature can be implemented using Plato by customizing the `client`, `server`, `algorithm`, and `trainer` classes.

### Prepare the environment and assets

Use Python 3.13 and the checked-in lockfile. From the repository root, the
base environment is provisioned with:

```bash
uv sync --locked --python 3.13
```

Most examples inherit the root dependencies. Some are workspace members with
additional packages in their local `pyproject.toml`; shared optional features
use root extras. Follow [Installation](../install.md) for workspace selection
and separate environments for incompatible extras such as `ssl` and `llm_eval`.
Keep the selected extra on syncing `uv run --locked` commands.

Follow the particular example's recipe for its working directory, entrypoint,
config, datasets, and model checkpoints. Some data sources download assets on
first use; others require preparation or external access before training.
See the [Qwen3 reference](case-studies/6. Qwen3 Federated LoRA.md) and
[Lighteval preparation](../install.md#optional-server-side-llm-evaluation-with-lighteval)
for those specific workloads.

CPU, CUDA, and Apple Silicon execution depend on the selected backend and
workload. See [Quick Start](../quickstart.md) for device flags and
[Native MLX](../mlx.md) for the bounded Apple CPU/Metal reference and checks.
A chapter or config in this index does not establish validated support for
every dataset, algorithm, or device combination; qualification applies to the
specific workload and environment recorded in its validation evidence.

---

## Algorithms Using Plato

- [Server Aggregation Algorithms](algorithms/1. Server Aggregation Algorithms.md)

- [Secure Aggregation with Homomorphic Encryption](algorithms/2. Secure Aggregation with Homomorphic Encryption.md)

- [Asynchronous Federated Learning Algorithms](algorithms/3. Asynchronous Federated Learning Algorithms.md)

- [Federated Unlearning](algorithms/4. Federated Unlearning.md)

- [Algorithms with Customized Client Training Loops](algorithms/5. Algorithms with Customized Client Training Loops.md)

- [Client Selection Algorithms](algorithms/6. Client Selection Algorithms.md)

- [Split Learning Algorithms](algorithms/7. Split Learning Algorithms.md)

- [Personalized Federated Learning Algorithms](algorithms/8. Personalized Federated Learning Algorithms.md)

- [Personalized Federated Learning Algorithms based on Self-Supervised Learning](algorithms/9. Personalized Federated Learning Algorithms based on Self-Supervised Learning.md)

- [Algorithms based on Neural Architecture Search and Model Search](algorithms/10. Algorithms based on Neural Architecture Search and Model Search.md)

- [Three-layer Federated Learning Algorithms](algorithms/11. Three-layer Federated Learning Algorithms.md)

- [Poisoning Detection Algorithms](algorithms/12. Poisoning Detection Algorithms.md)

- [Model Pruning Algorithms](algorithms/13. Model Pruning Algorithms.md)

- [Gradient Leakage Attacks and Defences](algorithms/14. Gradient Leakage Attacks and Defences.md)

## Archived Research

- [Archived research examples](archived.md)

## Case Studies

- [Qwen3 Federated LoRA](case-studies/6. Qwen3 Federated LoRA.md)

- [Federated LoRA Fine-Tuning](case-studies/1. LoRA.md)

- [Composable Trainer API](case-studies/2. Composable Trainer.md)

- [Server-side Lighteval for SmolLM2](case-studies/4. Server-side Lighteval for SmolLM2.md)
- [Time-Series Forecasting with TimesFM](case-studies/6. Time-Series Forecasting with TimesFM.md)
