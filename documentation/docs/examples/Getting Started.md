# Getting Started
In `examples/`, we included a wide variety of examples that showcased how third-party deep learning frameworks, such as [Catalyst](https://catalyst-team.github.io/catalyst/), can be used, and how a collection of federated learning algorithms in the research literature can be implemented using Plato by customizing the `client`, `server`, `algorithm`, and `trainer`. We also included detailed tutorials on how Plato can be run on Google Colab. Here is a list of the examples we included.


[TODO: Add all available examples]
---


## Getting Started

### Prerequisites

Before running any examples, you'll need to complete two important steps:

#### 1. Download the Dataset

On your first run, you must download the required dataset using the `-d` flag:

```shell
uv run examples/personalized_fl/fedbabu/fedbabu.py \
  -c examples/personalized_fl/configs/fedbabu_CIFAR10_resnet18.yml -d
```

Wait for the confirmation message:

```shell
The dataset has been successfully downloaded. Re-run the experiment without '-d' or '--download'.
```

Then run the command again **without** the `-d` flag:

```shell
uv run examples/personalized_fl/fedbabu/fedbabu.py \
  -c examples/personalized_fl/configs/fedbabu_CIFAR10_resnet18.yml
```

#### 2. Install Dependencies

Plato uses **uv** for hierarchical dependency management. Example-specific packages are defined in local `pyproject.toml` files rather than in the root directory.

To run an example with its dependencies:

1. Navigate to the example directory
2. Execute using `uv run`

**Example:**

```shell
cd examples/ssl/smog
uv run smog.py -c ../../../examples/ssl/configs/smog_CIFAR10_resnet18.yml
```

**Important:**
    *Always run `uv run` from within the specific example directory to ensure all dependencies are properly loaded.*

---
