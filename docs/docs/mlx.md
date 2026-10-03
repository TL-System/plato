# Native MLX on Apple Silicon

Plato's native MLX path runs LeNet-5 on MNIST with sample-weighted FedAvg.
The bounded native validation covers float32 training on Apple CPU and Metal,
evaluation, parameter transport, and weight restoration. Use Python 3.13, the
project's default and CI interpreter. Linux CI covers the core suite; native
MLX qualification requires an Apple Silicon Mac with working Metal.

## Install and run

Run from the repository root:

```bash
uv sync --python 3.13 --extra mlx
uv run --extra mlx python plato.py --config configs/MNIST/fedavg_lenet5_mlx.toml
```

The first run downloads MNIST. Add `--cpu` for Apple CPU execution. With no
device flag, the trainer preserves MLX's native default device, normally Metal
on Apple Silicon; it does not infer a CPU override from PyTorch's device
selection. `--mps` explicitly requests Metal and fails if it is unavailable.

The supplied config selects `trainer.type = "mlx"`, `algorithm.type =
"mlx_fedavg"`, and `framework = "mlx"` in the trainer, algorithm, and
`parameters.model` sections. Start by copying that config and changing only the
settings you need. Changing a PyTorch model's framework field does not convert
its architecture or weights into MLX.

The example samples two logical clients per round from ten, with
`trainer.max_concurrency = 1`: one physical client worker is reused for their
local training. Payload communication is simulated by default. To send actual
socket payloads, set this in your copied config:

```toml
[clients]
comm_simulation = false
```

Keep the supplied `safetensor_encode` client outbound processor and
`safetensor_decode` server inbound processor. The native socket test exercises
the actual entrypoint and codec with two unequal client sample counts and two
physical clients. The shipped MNIST entrypoint has separate bounded validation
with one reused worker. These are local tests, not distributed fleet or
multi-worker performance qualification. `server.simulate_wall_time` controls
simulated timing separately from payload transport.

## Reproducibility and training controls

These optional values belong in `[trainer]`:

```toml
model_seed = 17
training_seed = 29
clip_grad_norm = 2.0
```

`model_seed` scopes model construction so independent trainer factories can
start from the same parameters. It does not reinitialize a model instance
already supplied by the caller. `training_seed` derives separate random streams
from the logical client ID, federated round, and local epoch, independent of
the physical worker's process ID or assignment order. The seeded scopes restore
the caller's Python, NumPy, PyTorch CPU, and MLX random state when they finish.
Keep data partitioning and server selection seeds fixed as well when replaying
an experiment. Replay assumes the same environment; CPU and Metal comparisons
use numerical tolerances rather than a bitwise guarantee.

Omitting either seed preserves that operation's legacy unseeded behavior.
Clipping is also off unless requested: `clip_grad_norm` applies global gradient
norm clipping before the optimizer update. Its value must be finite and
nonnegative; zero is valid. When enabled, nonfinite gradients or norms raise an
error before the update. Custom training-step strategies own their clipping
behavior.

Set the optimizer name in `trainer.optimizer` and its arguments in
`[parameters.optimizer]`. Names are case-insensitive. The MLX 0.32.3 constructor
options used by this backend are:

| Optimizer | Options in addition to required `learning_rate` |
| --- | --- |
| `adam` | `betas`, `eps`, `bias_correction` |
| `adamw` | `betas`, `eps`, `weight_decay`, `bias_correction` |
| `sgd` | `momentum`, `weight_decay`, `dampening`, `nesterov` |
| `momentum` | Same as `sgd`; an explicit `momentum` value is required |
| `rmsprop` | `alpha`, `eps` |
| `lion` | `betas`, `weight_decay` |

Arguments pass directly to MLX; PyTorch option names and defaults are not
translated. For example:

```toml
[trainer]
optimizer = "momentum"

[parameters.optimizer]
learning_rate = 0.01
momentum = 0.9
```

The reference config uses Adam with `learning_rate = 0.001`. Training is eager:
each step materializes the loss, model state, and optimizer state. Requesting
`DefaultMLXTrainingStepStrategy(jit=True)` raises an error because compiled
execution is not qualified. The default loss is cross entropy, evaluation
returns classification accuracy, and the default scheduler performs no update.
The PyTorch scheduler, mixed-precision, and gradient-accumulation settings do
not imply equivalent MLX support.

## Models, transport, and extension hooks

`ComposableMLXTrainer` accepts a native `mlx.nn.Module` or a factory returning
one, with MLX-specific strategy interfaces. Factory construction, training,
evaluation, and parameter restoration use the trainer's selected stream.
Prefer a factory when the trainer must control initialization and placement;
already constructed lazy work retains its original placement.

Training switches the model to training mode. Evaluation temporarily switches
to evaluation mode and restores each submodule's previous mode, including when
evaluation raises an exception. BatchNorm and dropout state handling has
focused native coverage, but this does not qualify additional production
architectures beyond LeNet-5.

Parameter extraction produces owned NumPy snapshots. Trees may contain nested
dictionaries, lists, tuples, array leaves, and `None`. MLX FedAvg checks the
whole tree's container types, keys, sequence lengths, and array shapes before
arithmetic or model restoration. Reports and payloads must have matching counts,
and sample weights must be finite and nonnegative. Zero total weight preserves
the existing no-update behavior. Floating aggregation follows client sample
counts, with result dtypes aligned to the reference tree.

Server hooks have a **positional contract**: a replacement weight tree at index
`i` must still correspond to the report and logical client at index `i`.
`weights_received`, `on_weights_received`, and selection-strategy
`on_reports_received` extensions must preserve this association. The MLX path
validates again after hooks, rejects observable reordering of original payloads
or report objects and changes to client IDs, and stops before aggregation when
validation fails. Unlabeled deep copies cannot prove semantic correspondence:
same-schema replacements remain the hook author's responsibility. Validation
does not roll back arbitrary side effects performed by a hook.

The qualified training dtype is float32. Float16 has host-transport coverage
only; native bfloat16 host transport is explicitly rejected without a silent
cast. Checkpoints contain model parameters in safetensors and run history in a
`.pkl` sidecar. Optimizers are recreated for each local training run; optimizer,
RNG, and scheduler state are not saved for full or mid-training resume. Wall-clock
model retrieval is unimplemented. MLX-LM, LoRA, broad model/algorithm parity,
and mixed-framework weight conversion are outside this support scope.

## Run native checks

Install the native extra and the mandatory test dependencies:

```bash
uv sync --python 3.13 --extra mlx --group test
uv run --extra mlx --group test python -m pytest tests/mlx_native --test-profile=mlx-native -ra
```

This strict command requires the complete native case inventory and a successful
run with no native skips, xfails, or xpasses. Prerequisites include macOS arm64,
matching installed `mlx` and `mlx-metal` versions, and working CPU and Metal
arithmetic. Missing prerequisites fail instead of silently skipping tests.
The profile also requires the `dp`, `mpc`, and `rl` dependencies supplied by the
`test` group. `--collect-only` checks prerequisites and the inventory but does
not qualify runtime behavior.

For combined core and native qualification, use `test-model-search`. This group
includes `test` and supplies `ptflops` for the retained model-search tests:

```bash
uv run --extra mlx --group test-model-search python -m pytest tests --test-profile=mlx-native -ra
```

For a focused debugging run, specify a native filesystem path and omit the
profile:

```bash
uv run --extra mlx --group test python -m pytest tests/mlx_native/test_phase3_controls.py -k clipping -ra
```

Focused native runs retain the native prerequisite and no-skip checks but do
not establish complete qualification. They do not independently require the
mandatory extras. The strict profile rejects filters such as `-k`, `-m`,
`--deselect`, `--ignore`, and `--ignore-glob`; native `--pyargs` selection is
unsupported. Native paths are incompatible with `base` and `mandatory`
profiles.

Ordinary `pytest tests`, configured test-root collection, and the `base` and
`mandatory` profiles cover the core suite and exclude `tests/mlx_native` before
import. Installing MLX does not opt those commands into native testing. A core
pass therefore does not establish MLX qualification. Native validation results
apply to their recorded source and environment; rerun the strict command after
runtime changes.

For the full core suite, including retained model-search tests, use:

```bash
uv run --group test-model-search python -m pytest tests --test-profile=mandatory -ra
```

To run only the retained model-search checks:

```bash
uv run --group test-model-search python -m pytest tests/integration/test_retained_model_search.py --test-profile=mandatory -m retained_model_search -ra
```

Historical timing receipts measure their recorded source revisions and bounded
workloads. Later validation and import-loading changes mean those wall times
do not measure the current checkout. They establish neither a general MLX
speedup nor a performance threshold for ordinary pytest runs. Archived research
snapshots likewise require their external historical setup and are not current
MLX support targets.
