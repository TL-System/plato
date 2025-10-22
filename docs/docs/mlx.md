A reasonable roadmap now that the client/server layers no longer depend on PyTorch is:

  - Define MLX runtime primitives
      - Implement plato/trainers/mlx.py (or similar) wrapping MLX optimizers, loss, device management, model save/load,
        metric evaluation.
      - Mirror the existing composable trainer APIs so strategies can swap in MLX variants without touching higher
        layers.
  - Add MLX-aware algorithms
      - Create MLX counterparts for the shared algorithm utilities (e.g., mlx_fedavg.Algorithm) implementing weight
        extraction/loading, delta math, and any specialised helpers (MaskCrypt, FedAtt, etc.) using MLX tensors.
  - Model & datasource integration
      - Extend plato/models/registry to register MLX models or conversion shims.
      - Ensure datasources/samplers can deliver data in formats MLX expects (potentially via NumPy) and update processors
        if batch conversion is required.
  - Configuration & registry wiring
      - Update the trainer/algorithm registries so configs can select type = "mlx" (both for trainer and algorithm).
      - Add new TOML examples demonstrating MLX usage, including any MLX-specific hyperparameters.
      - Provide configuration shortcuts such as `framework = "mlx"` for trainer/algorithm/model sections.
  - Testing & validation
      - Build unit tests for the MLX trainer and algorithms, covering save/load, aggregation, and feature-specific logic.
      - Add smoke tests that run small MLX training loops end-to-end.
  - Tooling & documentation
      - Document installation prerequisites (e.g., MLX, Apple Silicon requirements) and update README/docs with MLX
        instructions.
      - Provide migration notes for contributors on how to implement MLX versions of custom trainers/algorithms.
  - Performance & parity checks
      - Benchmark key examples against their PyTorch counterparts to confirm comparable behaviour, and identify any API
        gaps that require follow-up enhancements.

### Configuration shortcuts

With the MLX trainer and algorithm registered, configs can opt in by setting a
`framework = "mlx"` key in the `[trainer]`, `[algorithm]`, or `[parameters.model]`
sections. For example:

```toml
[trainer]
type = "mlx"
framework = "mlx"
rounds = 5

[algorithm]
type = "mlx_fedavg"
framework = "mlx"

[parameters.model]
framework = "mlx"
model_name = "lenet5"

[parameters.optimizer]
learning_rate = 0.001
```

A complete reference configuration is available at
`configs/MNIST/fedavg_lenet5_mlx.toml`, which pairs the MLX trainer, MLX FedAvg
algorithm, numpy conversion processors, and the MLX LeNet-5 model.
