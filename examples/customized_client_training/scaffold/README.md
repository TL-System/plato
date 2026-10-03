# SCAFFOLD

This example uses server and client control variates to correct local updates.
It implements the corrected Option II update from Karimireddy et al.,
“[SCAFFOLD: Stochastic Controlled Averaging for Federated Learning][paper],”
ICML 2020, Algorithm 1 and equations (3)–(5).

From the repository root, install the dependencies and run the MNIST example:

```bash
uv sync
uv run python examples/customized_client_training/scaffold/scaffold.py \
  -c examples/customized_client_training/scaffold/scaffold_MNIST_lenet5.toml
```

The configuration downloads MNIST and trains LeNet-5 with five clients, two
selected per round, three local epochs, and up to five rounds. It uses SGD with
`lr = 0.01`, `momentum = 0.9`, and `weight_decay = 0.0`.
`trainer.max_concurrency = 2` selects spawned local training workers. Add `--cpu`
to use the CPU, or `-b /path/to/new-run` to choose a separate base directory for
data, models, checkpoints, and results. Copy the TOML before changing an
experiment's settings.

For a round starting from model parameters $x$, let $y$ be the local parameters,
$c$ the received server control, and $c_i$ the client's previous control. With
vanilla SGD (no momentum or weight decay, and `maximize = false`), the local
update is

$$
y \leftarrow y - \eta(g - c_i + c).
$$

The strategy applies $-\eta(c-c_i)$ after the optimizer update. After $K > 0$
completed optimizer updates, it computes

$$
c_i^{\mathrm{new}} = c_i - c + \frac{x-y}{K\eta},
\qquad \Delta c_i = c_i^{\mathrm{new}} - c_i.
$$

These signs follow the main algorithm and equations (3)–(5) in the
[versioned paper][paper], rather than the reversed control terms in appendix
equation (19). For constant-rate vanilla SGD, summing the local updates makes
$c_i^{\mathrm{new}}$ the mean of the gradients used in those updates.
`SCAFFOLDUpdateStrategyV2` is a compatibility name for this same corrected
Option II implementation. True Option I, which evaluates gradients at $x$ in
an additional pass, is not implemented.

The learning rate comes from the actual optimizer at each update. Every group
containing trainable model parameters must use the same finite, positive scalar
rate, and that rate must remain constant throughout a local round. A different
constant rate is allowed in the next round. Unequal group rates and within-round
rate changes are rejected before the affected update. A later scheduler or
optimizer hook cannot replace the rate already used for a completed update.

$K$ counts optimizer updates, not minibatches or epochs. With gradient
accumulation, a final partial window contributes one update and one correction;
skipped updates contribute neither. If $K=0$, the client retains its control and
returns a fresh zero delta. Corrections apply to optimizer-owned trainable
parameters, including those with no gradient. Excluded trainable parameters
retain their controls and emit zero deltas; frozen parameters and model buffers
receive no control correction.

`ScaffoldCallback` installs processors that extract `[weights, server_controls]`
before training and attach `[weights, delta_ci]` afterward. The federated example
requires a valid current server control payload. Direct training uses the
strategy's resulting state in the same process. Spawned training returns the
new client controls, delta, update count, and rate to the parent alongside the
saved model; a missing or stale worker handoff cannot supply a successful
outbound delta.

For participating clients $S$ out of a total population $N$, the server updates
its control as

$$
c \leftarrow c + \frac{1}{N}\sum_{i\in S}\Delta c_i.
$$

Here $N$ is `clients.total_clients`, not the number selected or a sample count.
Model aggregation separately retains FedAvg's sample weights: for positive
total sample count, $x_{\mathrm{new}} = \sum_{i\in S}n_i y_i / \sum_{i\in S}n_i$.
With unequal counts this differs from the paper's uniform-client model update.
The shipped momentum of `0.9` also makes the post-optimizer correction an
extension of vanilla SCAFFOLD. Other non-vanilla optimizers use this additive
correction within the same rate constraints. The paper's vanilla-SGD,
uniform-client convergence guarantees are not claimed for these extensions.

Client controls are owned by `SCAFFOLDUpdateStrategy` and saved as
`scaffold_cv_<client_id>.pkl` inside `Config().params["model_path"]`, or the
strategy's explicit `save_path`. Changing a trainer's logical client ID clears
its previous client's state and loads only the assigned client's controls.
Construction-time client 0 state is never a substitute for another client.

The canonical file takes precedence. Only when it is absent, a nonzero client
ID may import one of these exact historical pickle dictionaries, in this order:

1. `<root>/<model_name>_<client_id>_control_variate.pth`, if the resolved path
   stays within the configured root, including for names with subdirectories.
2. The old concatenated path `<root>scaffold_cv_<client_id>.pkl`, with no added
   separator. For a root without a trailing separator, this historical file
   sits beside the root directory.

The loader does not search for other clients' files. Invalid canonical state
raises an error instead of falling back to legacy state. Loaded dictionaries
must contain finite floating-point tensors of the right shapes for every
trainable parameter; known buffer and frozen-parameter entries are ignored,
and unknown keys are rejected. Subsequent writes use the canonical path and
leave distinct legacy files unchanged.

This correction intentionally changes numerical evolution from older runs,
even when their control dictionaries remain structurally loadable. For
comparisons, start a fresh run in a separate base directory with consistent
server and client controls, initially all zero. Client control persistence and
worker handoff do not provide a complete experiment restart: they do not
restore the server control, optimizer history, random-number state, or privacy
accounting state as a coordinated checkpoint.

[paper]: https://arxiv.org/pdf/1910.06378v4
