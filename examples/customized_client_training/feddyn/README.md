# FedDyn

This example couples a dynamic local regularizer with a dedicated FedDyn server.
It uses cumulative parameter displacements to express the history in Acar et al.,
“[Federated Learning Based on Dynamic Regularization][paper],” ICLR 2021,
Algorithm 1, following the [authors' displacement formulation][reference].
The server evaluates and broadcasts the corrected cloud model. The reference
also reports separate selected-client and all-client model averages.

## Run the example

From the repository root:

```bash
uv sync
uv run python examples/customized_client_training/feddyn/feddyn.py \
  -c examples/customized_client_training/feddyn/feddyn_MNIST_lenet5.toml \
  --cpu -b ./runtime/feddyn
```

The shipped TOML downloads MNIST and trains LeNet-5 with 1,000 logical clients,
10 selected per round, 20 local epochs, and up to three rounds. It uses
`algorithm.alpha_coef = 0.01`, `algorithm.feddyn_weighting = "uniform"`, and SGD
with learning rate `0.03`, zero momentum, and zero weight decay.
`trainer.max_concurrency = 3` enables spawned local workers. Copy the TOML before
changing an experiment; use a new base directory for independent runs.

Use this entrypoint: it installs the FedDyn client, trainer, and server.
The retained `algorithm.type = "fedavg"` selects underlying model exchange;
the dedicated server performs the FedDyn aggregation below.

## Objective and aggregation

Let $x$ be the received cloud parameters, $h_i$ client $i$'s cumulative
displacement, initially zero, and $y_i$ its successful local endpoint. Over
trainable parameters, the local objective is

$$
J_i(w) = F_i(w) + \alpha_i\langle h_i,w\rangle
         + \frac{\alpha_i}{2}\|w-x\|^2.
$$

Its gradient is $\nabla F_i(w)+\alpha_i(w-x+h_i)$. The received $x$, history,
and coefficient remain fixed throughout the local attempt. History is a
cumulative displacement, not a measured gradient. For accepted participants
$S$ in a fixed population of $N$ clients,

$$
h_i^{\mathrm{new}} = h_i+y_i-x \quad (i\in S),
\qquad
x_{\mathrm{new}} =
\frac{1}{|S|}\sum_{i\in S}y_i+
\frac{1}{N}\sum_{i=1}^{N}h_i^{\mathrm{new}}.
$$

Inactive clients retain their histories and still contribute to the population
history mean. Frozen parameters and static buffers retain the server baseline.

The weighting modes change the local coefficient, not these two means:

- **Uniform** (default): $\alpha_i=\alpha$, for an equal-client objective.
  Do not set `algorithm.feddyn_sample_counts` in this mode.
- **Sample**: set `algorithm.feddyn_weighting = "sample"` and supply
  `algorithm.feddyn_sample_counts`, a complete vector of positive integer
  partition counts for all $N$ logical clients, ordered by client ID starting
  at 1. With $q_i=Nn_i/\sum_{j=1}^{N}n_j$, the coefficient is
  $\alpha_i=\alpha/q_i$. This does not use selected-client sample-weighted
  FedAvg.

Counts must match the actual partition sampler cardinalities. They are not
label values, minibatch sizes, epoch totals, the backing dataset size, or merely
the requested `data.partition_size`. The loader requires an explicit nonempty
sampler, agrees with its declared count, and does not drop the last batch.
In sample mode it checks the configured count for that client. In uniform
mode the server records each accepted client's count and requires it to remain
the same on later participation.

`algorithm.alpha_coef` takes precedence over `algorithm.feddyn_alpha`; the
default is `0.01`. The full example requires finite, positive alpha.
The reusable `FedDynLossStrategy(alpha=0)` separately reduces to task loss;
it does not make the full example a FedAvg mode.

The objective has the same gradient as the authors' [linear-term formulation][loss]
with quadratic SGD weight decay when clipping is inactive. This implementation
puts the entire regularizer in the loss and requires zero optimizer weight
decay. It does not reproduce the reference's active clipping order or claim
its convergence, accuracy, or communication budget: the server stores all
client histories and sends the assigned history with each model.

## Supported training and state ownership

The example supports synchronous full or partial participation in a fixed
population. Every selected client must return a valid result before the round
can commit. It requires ordinary PyTorch training with finite, dense float32
or float64 trainable parameters and plain `torch.optim.SGD`: a fixed positive
finite learning rate, zero momentum, dampening, and weight decay, and no
Nesterov or maximization. Each trainable parameter must belong to the optimizer
exactly once. The trainable set, shapes, dtypes, and ownership must stay fixed;
parameter aliases, changing frozen parameters, and mutable buffers such as
BatchNorm running statistics are rejected.

Schedulers, AMP, gradient clipping, differential privacy, asynchronous or
cross-silo rounds, and custom multiple-update training strategies are
unsupported and rejected. `trainer.gradient_accumulation_steps` may be a
positive integer; accumulation averages microbatch gradients and normalizes
a final partial window by its actual size. At least one optimizer update must
complete. Empty or entirely skipped attempts cannot produce a valid result.

The server owns all histories. Direct training and spawned workers return
provisional results tied to the current run, round, logical client, and
dispatch. The parent validates the model/history handoff before allowing an
outbound result; neither parent client nor worker writes a live history file.
Reusing a worker for another logical client does not transfer that client's
history.

The server validates the whole batch, including the model state used after
receive callbacks. The model and histories commit together only after
aggregation callbacks and final validation succeed. Failures before that
boundary preserve the previous committed values; evaluation or reporting
failures afterward retain the completed commit. External callback side effects
are outside this guarantee.

## Resume or start from existing weights

To continue from the last saved committed round, use the same base directory
and compatible configuration:

```bash
uv run python examples/customized_client_training/feddyn/feddyn.py \
  -c examples/customized_client_training/feddyn/feddyn_MNIST_lenet5.toml \
  --cpu -b ./runtime/feddyn --resume
```

For the shipped configuration, the bundle is
`./runtime/feddyn/models/feddyn/mnist/feddyn_lenet5.pth`.
It is atomically replaced when the server saves a completed round. It contains
the full model, all population histories, observed counts, algorithm settings,
model schema, run identity, committed round, accepted dispatch tokens, and
random-number states. The latter include separate Python global and
client-selection states, NumPy global state, and Torch's default CPU state.

`--resume` requires this complete versioned bundle and validates its settings
and schema. It restores the saved committed round and installs the pending RNG
states once before server registration and selection resume. Keep the same
population, partitions, and compatible algorithm/model settings; when extending
a completed run, set `trainer.rounds` to the desired total round count.
A model file alone is insufficient. A commit that has not yet been saved is
not a durable restart point. This is committed-round continuation, not replay
of an unfinished round or recovery of optimizer, GPU RNG, arbitrary datasource,
or external callback state.

Older per-client histories are available only through the explicit read-only
`FedDynUpdateStrategy.read_legacy_history(context)` API. `save_path` supplies an
inspection root and is rejected for live training. For a positive logical
client ID, inspection prefers `<root>/feddyn_grad_<client_id>.pth`, then the
exact old concatenated `<root>_feddyn_grad_<client_id>.pth`. Client 0 supplies
no history for another client. The loader validates finite tensors against
trainable shapes and dtypes, permits validated frozen-parameter extras, and
rejects unknown keys. Inspection does not adopt, rewrite, or migrate the file.

For a model-only warm start, explicitly call the dedicated server's
`warm_start_model(weights)` after model initialization and before dispatch.
This starts a new run at round zero with zero histories and cleared count/token
records. There is no CLI warm-start flag. Old runs used different loss and
aggregation rules; their histories do not implicitly continue the corrected
trajectory. Use a consistent fresh run for comparisons.

[paper]: https://arxiv.org/pdf/2111.04263v1
[reference]: https://github.com/alpemreacar/FedDyn/blob/48a19fac440ef079ce563da8e0c2896f8256fef9/utils_methods.py#L286-L402
[loss]: https://github.com/alpemreacar/FedDyn/blob/48a19fac440ef079ce563da8e0c2896f8256fef9/utils_general.py#L185-L227
