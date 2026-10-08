# DiLoCo Design Contract

This note defines what Plato calls faithful DiLoCo in the current
implementation.

Faithful DiLoCo in Plato means algorithm-faithful execution of the DiLoCo
training loop inside Plato's federated runtime. It does not mean reproducing
the paper's exact C4 dataset, model scale, tokenizer, hardware topology,
pretraining duration, or final benchmark numbers.

## Example Configurations

Plato includes MNIST/LeNet and CIFAR-10/ResNet-18 comparison configurations
for checking DiLoCo against matched FedAvg runs:

```bash
uv run python plato.py --config configs/MNIST/diloco_lenet5.toml
uv run python plato.py --config configs/MNIST/fedavg_lenet5_diloco_comparison.toml
uv run python plato.py --config configs/CIFAR10/diloco_resnet18.toml
uv run python plato.py --config configs/CIFAR10/fedavg_resnet18_diloco_comparison.toml
```

These examples validate Plato's DiLoCo mechanics without reproducing the C4
dataset, tokenizer, language-model scale, hardware topology, pretraining
duration, or final benchmark numbers from the paper.

## Algorithm Contract

DiLoCo has two optimizer levels:

- The client-local inner optimizer trains each selected logical client for
  exactly `H` local optimizer steps between synchronizations.
- The server-side outer optimizer updates the global model from the averaged
  outer gradient.

Plato's FedAvg-style model delta is:

```text
plato_delta = client_after - global_before
```

DiLoCo's outer gradient is:

```text
outer_gradient = global_before - client_after = -plato_delta
```

The DiLoCo server must still return a Plato-compatible model delta because
`algorithm.update_weights()` adds the returned delta to the current global
model. For example, outer SGD with learning rate `1.0` returns the averaged
Plato delta and is equivalent to FedAvg only when the same averaging rule is
used.

The outer optimizer runs on the server. Clients run only the inner optimizer
and send model weights or weight-equivalent updates. Client-local optimizer and
scheduler state persists per logical client and is never sent to the server.

## Local Work `H`

`H` means client-local optimizer steps between synchronizations. It is not:

- epochs,
- raw dataloader batches, or
- gradient-accumulation micro-batches.

When gradient accumulation is enabled, `H` counts completed optimizer steps.
Raw batches that do not trigger `optimizer.step()` do not increment `H`.

`H` may be smaller than one epoch. Faithful DiLoCo must therefore stop local
training mid-epoch after exactly `H` optimizer steps. This early stop must
still run normal trainer cleanup, state persistence, callback completion, and
reporting paths. It must not perform an extra final optimizer step.

Small-`H` training must not repeatedly replay the same first `H` batches only
because the train loader is recreated each round. The implementation must use
round-aware resampling or an equivalent persistent sampling stream so each
logical client's local data stream advances across rounds in a reproducible
way.

## State Ownership

Server-owned state:

- the global model,
- outer optimizer momentum or other outer optimizer state,
- aggregation metadata needed to update the global model.

Client-owned state:

- inner optimizer state, such as AdamW first and second moments,
- scheduler state and global/local optimizer-step counters,
- sampler or dataloader stream position needed for small-`H` continuity.

Client-owned optimizer and scheduler state must not appear in client-server
payloads. It must remain local to the logical client, including when training
uses subprocesses.

## Parameter And Buffer Policy

By default, the outer optimizer applies only to trainable floating parameters.
This matches the algorithm definition, which optimizes model parameters.

Floating buffers, such as batch normalization running statistics, are
synchronized without outer momentum by default. They use the selected averaging
rule but do not receive server-side momentum or Nesterov treatment.

Non-floating buffers use conservative FedAvg-style behavior, including casting
or rounding as needed to preserve the buffer's dtype-compatible semantics.

The implementation may offer `apply_outer_optimizer_to = "all_floating"` for
experiments, but the default must remain `parameters`.

## Configuration Contract

The faithful initial mode uses these configuration names and defaults:

```toml
[server]
type = "diloco"

[algorithm]
type = "fedavg"

[trainer]
local_steps_per_round = H
preserve_optimizer_state = true
optimizer = "AdamW"

[server.diloco]
outer_optimizer = "nesterov"
outer_learning_rate = 0.7
outer_momentum = 0.9
aggregation_weighting = "uniform" # or "num_samples"
apply_outer_optimizer_to = "parameters" # or "all_floating"
```

`algorithm.type = "fedavg"` is intentional. Plato should reuse the existing
FedAvg weight extraction, delta computation, and global model loading path,
while `server.type = "diloco"` selects the server-side DiLoCo aggregation and
outer optimizer behavior.

`aggregation_weighting = "uniform"` matches the balanced worker setting most
closely. `aggregation_weighting = "num_samples"` matches Plato's traditional
sample-weighted FedAvg behavior. FedAvg equivalence for outer SGD with learning
rate `1.0` is valid only when both runs use the same weighting rule.

Unsupported modes must fail clearly. They must not silently fall back to an
approximate DiLoCo variant. Examples include trainer backends that cannot count
local optimizer steps exactly, execution paths that cannot preserve
client-local optimizer and scheduler state, samplers that cannot advance the
small-`H` local data stream across rounds, or payload paths that would send
optimizer state to the server. Experimental combinations that are allowed but
not faithful must warn clearly.

## Implementation Sequence

Dependency graph:

```text
D1
|-- D2 --> D3
|-- D4 --> D5
|-- D6 --> D7
|-- D8 --> D9
`-- D10 --> D11

D3, D5, D7, D9, D11 --> D12 --> D13
```

Tasks:

```yaml
- id: D1
  depends_on: []
  task: Document the exact DiLoCo contract and unsupported modes.

- id: D2
  depends_on: [D1]
  task: Add red tests for server-side outer gradient sign, weighting, and
    FedAvg equivalence under matching weighting.

- id: D3
  depends_on: [D2]
  task: Implement DiLoCo server aggregation and outer optimizer state for SGD,
    momentum SGD, and Nesterov.

- id: D4
  depends_on: [D1]
  task: Add red tests for exact local optimizer-step counting and `H` smaller
    than one epoch.

- id: D5
  depends_on: [D4]
  task: Implement `trainer.local_steps_per_round` with mid-epoch termination
    after exactly `H` optimizer steps.

- id: D6
  depends_on: [D1]
  task: Add red tests for per-client optimizer and scheduler state
    persistence.

- id: D7
  depends_on: [D6]
  task: Persist client-local optimizer and scheduler state without sending it
    to the server.

- id: D8
  depends_on: [D1]
  task: Add red tests for round-aware small-`H` sampling.

- id: D9
  depends_on: [D8]
  task: Implement round-aware resampling or an equivalent persistent sampling
    stream for each logical client.

- id: D10
  depends_on: [D1]
  task: Add red tests for parameter and buffer eligibility.

- id: D11
  depends_on: [D10]
  task: Implement the default trainable-parameter-only outer optimizer policy
    and conservative buffer synchronization.

- id: D12
  depends_on: [D3, D5, D7, D9, D11]
  task: Wire exact DiLoCo configuration, examples, and user-facing
    documentation.

- id: D13
  depends_on: [D12]
  task: Add end-to-end faithful-mode validation coverage.
```

Every implementation task should use red/green test-driven development. Add
the failing tests that describe the contract first, then implement the smallest
runtime change that makes those tests pass.
