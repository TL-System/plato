# Getting Started

In `examples/`, we included a wide variety of examples that showed how federated learning algorithms in the research literature can be implemented using Plato by customizing the `client`, `server`, `algorithm`, and `trainer` classes.

### Dataset Preparation

When you run an example for the first time, Plato downloads the required datasets automatically before training begins. Depending on the dataset size and your connection speed, the first round may take a little longer while the assets are prepared.

Plato uses [uv](https://docs.astral.sh/uv/) for hierarchical dependency management. Example-specific packages are defined in local `pyproject.toml` files rather than in the top-level directory.

To run an example with its dependencies, you need to run `uv sync` first in the top-level directory, navigate to the directory containing the example, and then use `uv run` to run the example.

!!! tip "Note"
    To make sure that all dependencies are properly loaded, always run `uv run` from within the directory containing the example.

Plato supports both Linux with NVIDIA GPUs and macOS with M1/M2/M4/M4 GPUs. It will automatically detect and use these GPUs when they are present.

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

## Case Studies

- [Federated LoRA Fine-Tuning](case-studies/1. LoRA.md)

- [Composable Trainer API](case-studies/2. Composable Trainer.md)

- [Server-side Lighteval for SmolLM2](case-studies/4. Server-side Lighteval for SmolLM2.md)

- [SmolVLA Trainer with LeRobot](case-studies/3. SmolVLA Trainer with LeRobot.md)
