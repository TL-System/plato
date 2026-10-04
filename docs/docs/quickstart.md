# Quick Start

## Running Plato Directly Using `uv`

From the repository root, provision the locked Python 3.13 environment and run
a reference workload on the CPU:

```bash
uv sync --locked --python 3.13
uv run --locked python plato.py --config configs/MNIST/fedavg_lenet5.toml --cpu
```

The following command-line parameters are supported:

- `-c`: the path to the configuration file to be used. The default is `config.toml` in the project's home directory.

- `-b`: the base path, to be used to contain all models, datasets, checkpoints, and results (defaults to `./runtime`).

- `-r`: resume a previously interrupted training session (only works correctly in synchronous training sessions).

- `--cpu`: use the CPU as the device only.

For this MNIST reference, the datasource downloads the dataset on first use
and reuses it under the selected base path. Other datasets and pretrained models
may require separate files or access; follow the selected example's instructions.
Omit `--cpu` to let the configured trainer select an available device.

_Plato_ uses the TOML format for its configuration files to manage runtime configuration parameters. Example configuration files have been provided in the `configs/` directory.

In `examples/`, a number of federated learning algorithms have been included. To run them, just run the main Python program in each of the directories with a suitable configuration file. For example, to run the `basic` example located at `examples/basic/`, run the command:

```bash
uv run --locked python examples/basic/basic.py -c configs/MNIST/fedavg_lenet5.toml --cpu
```

## Running Server-side Lighteval Evaluation

If your config uses:

```toml
[evaluation]
type = "lighteval"
```

install the optional evaluator stack first:

```bash
uv sync --locked --python 3.13 --extra llm_eval
uv run --locked --extra llm_eval python -m nltk.downloader punkt punkt_tab
```

Then run the reference SmolLM2 configuration:

```bash
uv run --locked --extra llm_eval python plato.py --config configs/HuggingFace/fedavg_smol_smoltalk_smollm2_135m.toml
```

This configuration trains SmolLM2 while the server evaluates the aggregated
global model after each round. It downloads external models and datasets and
sets `evaluation.device = "cuda:0"`. For CPU evaluation, change that field to
`"cpu"` in a copied config and pass `--cpu` for the trainer. Keep `--extra llm_eval`
on syncing commands; see [installation](install.md#optional-server-side-llm-evaluation-with-lighteval)
for dependency isolation and NLTK resources.

See [Evaluation](configurations/evaluation.md) for the available evaluator options and [Server-side Lighteval for SmolLM2](examples/case-studies/4. Server-side Lighteval for SmolLM2.md) for the full example.

## Running Qwen3 Federated LoRA

The bounded Qwen3 example fine-tunes LoRA adapters on a small local text fixture:

```bash
uv run python plato.py --config configs/HuggingFace/fedavg_qwen3_06b_lora.toml --cpu
```

Run from the repository root. The model and tokenizer download from a fixed
Hugging Face revision on first use. See
[Qwen3 Federated LoRA](examples/case-studies/6. Qwen3 Federated LoRA.md)
for setup, memory considerations, and the distinction between offline tests
and pretrained-model qualification.

## Using MLX as a Backend

Run the native LeNet-5/MNIST FedAvg reference on Apple Silicon with Python 3.13:

```bash
uv sync --python 3.13 --extra mlx
uv run --extra mlx python plato.py --config configs/MNIST/fedavg_lenet5_mlx.toml
```

Add `--cpu` for Apple CPU execution; without a device flag the trainer preserves
MLX's native default, normally Metal. The shipped config uses simulated payload
communication and one physical client worker. See [Native MLX](mlx.md) for
socket opt-in, seeds, clipping, optimizer options, checkpoint limits, and the
explicit native qualification command.

## Running Plato in a Docker Container

The supplied development image uses Ubuntu 24.04, CUDA 13.0.3 and managed Python
3.13. It installs the locked base runtime dependencies into `/opt/plato/.venv`,
with Python itself under `/opt/plato/python`. Both locations sit outside the
checkout mount at `/root/plato`, so mounting your source does not hide the
container's interpreter or environment. Optional extras and development tools
are installed separately when needed.

Build the image from the repository root on a Linux Docker host:

```bash
docker build -t plato -f Dockerfile .
```

Run the CPU reference using the Linux launcher:

```bash
./dockerrun.sh python plato.py --config configs/MNIST/fedavg_lenet5.toml --cpu
```

To open a shell instead, run `./dockerrun.sh` without arguments. The launcher
forwards supplied command arguments and allocates a terminal only when running
from a terminal. It uses host networking and the host's `/dev/shm`.

Both launchers quote the current directory in the bind mount
(`-v "$PWD:/root/plato"`) and use `--rm` to remove the container when it exits.
Run them from the repository root. Source edits and outputs written under this
mount, including the default `runtime` directory, remain on the host after exit.
The container runs as root; with ordinary rootful Docker on Linux, files it
creates in the mount can be root-owned on the host. Changes elsewhere in the
container, including packages added to `/opt/plato/.venv`, disappear when the
container is removed.

Build from the checkout you intend to mount. If its dependency lock changes,
rebuild the image or run a locked sync inside the container before using it:

```bash
uv sync --locked --python 3.13 --no-default-groups
```

For an optional workload, add its compatible extra to the sync and subsequent
syncing run commands; the [Lighteval setup](install.md#optional-server-side-llm-evaluation-with-lighteval)
requires `--extra llm_eval` throughout. Rebuild an appropriately provisioned image
to retain added dependencies across disposable container runs.

## Running Plato in a Docker Container with GPU Support

On a Linux host with a supported NVIDIA GPU, install the driver and follow the
current [NVIDIA Container Toolkit installation and Docker configuration guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The image's CUDA libraries do not install or configure the host driver/runtime.

With the `plato` image built and Docker configured, inspect GPU visibility:

```bash
./dockerrun_gpu.sh nvidia-smi
```

Then run the reference workload with GPU access:

```bash
./dockerrun_gpu.sh python plato.py --config configs/MNIST/fedavg_lenet5.toml
```

The GPU launcher adds `--gpus all`; `--cpu` still forces CPU training if supplied.
`nvidia-smi` checks device visibility. A successful CPU run or visibility check
does not establish GPU training compatibility for every model, driver or device.

## Formatting the Code and Fixing Linter Errors

Use the locked development tools from the repository root. To format the active
Python code and apply the configured Ruff fixes:

```bash
uv run --locked --group dev ruff format .
uv run --locked --group dev ruff check . --fix
```

## Type Checking

Run the locked type checker against the core package:

```bash
uv run --locked --group dev ty check plato
```

See [Development](development.md) for the test profiles and contribution checks.
