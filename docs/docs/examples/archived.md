# Archived research examples

Archived examples preserve source, configuration, and provenance for historical
study. Their commands require a separate historical checkout and environment;
they are outside current runtime support. Read the per-family README before
attempting restoration. Original documents inside an archive retain their old
wording, including paths and commands that are no longer active.

## Retired integrations and model-search variants

| Family | Preserved scope |
| --- | --- |
| [Legacy ViT factory](https://github.com/TL-System/plato/tree/main/archives/retired/legacy-vit) | The former `plato.models.vit` factory and four ViT configs, including DeepViT, T2T-ViT, SwinV2, and LeViT. |
| [FedTP](https://github.com/TL-System/plato/tree/main/archives/retired/fedtp) | The complete FedTP experiment, including the server hypernetwork, algorithm, and both ViT/T2T-ViT configs. |
| [pFedRLNAS / PerFedRLNAS](https://github.com/TL-System/plato/tree/main/archives/retired/pfedrlnas) | All NASViT, MobileNetV3, and DARTS modes, their shared configs, and vendored NASViT support. |
| [FedRLNAS](https://github.com/TL-System/plato/tree/main/archives/retired/fedrlnas) | The FedRLNAS experiment, its MNIST config, and its DARTS search-space code. |
| [AnyCostFL local ViT](https://github.com/TL-System/plato/tree/main/archives/retired/anycostfl-vit) | The local `vit.py` implementation and `example_ViT.toml`; shared runtime files are copied only as historical context. |
| [FedRolex local ViT](https://github.com/TL-System/plato/tree/main/archives/retired/fedrolex-vit) | The local `vit.py` implementation and `example_ViT.toml`; shared runtime files are copied only as historical context. |
| [HeteroFL custom MobileNetV3](https://github.com/TL-System/plato/tree/main/archives/retired/heterofl-mobilenetv3) | The custom `mobilenetv3.py` branch; shared runtime files and the ResNet config are copied only as historical context. |
| [Nanochat](https://github.com/TL-System/plato/tree/main/archives/retired/nanochat) | Original integration, runbook, and pinned upstream source snapshot. |
| [LeRobot / SmolVLA](https://github.com/TL-System/plato/tree/main/archives/retired/lerobot) | Original robotics integration and runbook; upstream policy/data revisions were not pinned. |

The [repository archive index](https://github.com/TL-System/plato/blob/main/archives/retired/README.md)
links manifests, immutable source identities, historical restoration notes, and
license provenance. Files marked `copied_context` explain a retired branch;
they do not retire every current implementation with the same filename.

The active [model-search examples](<algorithms/10. Algorithms based on Neural Architecture Search and Model Search.md>)
retain AnyCostFL, FedRolex, and HeteroFL ResNet paths plus the separate SysHeteroFL
ResNet example. The legacy `model_type = "vit"` factory is retired, while generic
Hugging Face and Torchvision families remain separate. The current Hugging Face
causal-LM factory is not an image-classification ViT replacement.

[Qwen3 Federated LoRA](<case-studies/6. Qwen3 Federated LoRA.md>) is the maintained
text-model reference. Its pinned base model and tokenizer do not make Nanochat
artifacts interchangeable, and it does not replace robotics policy training.
For Apple Silicon workloads, consult the separate [native MLX guide](../mlx.md).

## Earlier historical examples and utilities

The four experiments below remain in `examples/outdated/`. Their 20 source and
configuration files are listed individually so that an entrypoint, helper module
or model definition is not mistaken for an independently supported example.
The notes describe inspected source, not successful execution on current Python
3.13 dependencies. These files do not pin a complete working historical
environment; dependency versions and the matching Plato checkout must be
established separately before attempting a restoration.

### Norm bounding

The server uses NumPy to calculate update norms and overrides FedAvg's
`aggregate_deltas` hook. Its config selects Torchvision MNIST, LeNet-5 and SGD.
The supplied threshold is 5, but omitting it leaves `None` in the denominator;
threshold validation would need attention. Tensor-to-NumPy conversion also
assumes tensors can be converted directly, so device handling needs explicit
checks. The loop sums clipped updates without sample weighting or a final
average. A restoration must check that aggregation rule against the cited
research before changing it. The override hook still exists in current Plato;
that alone does not qualify this implementation.

| Preserved file | Role |
| --- | --- |
| [norm_bounding.py](https://github.com/TL-System/plato/blob/main/examples/outdated/norm_bounding/norm_bounding.py) | Experiment entrypoint. |
| [norm_bounding_MNIST_lenet5.toml](https://github.com/TL-System/plato/blob/main/examples/outdated/norm_bounding/norm_bounding_MNIST_lenet5.toml) | MNIST/LeNet-5 configuration; threshold set to 5. |
| [norm_bounding_server.py](https://github.com/TL-System/plato/blob/main/examples/outdated/norm_bounding/norm_bounding_server.py) | Update norm calculation and clipped-update aggregation. |

### FjORD

The entrypoint imports both local model modules before selecting one. The
preserved code uses PyTorch, NumPy, `ptflops` for complexity measurement and
`einops` in its local ViT; even the ResNet entrypoint imports that ViT module.
This is not evidence of a dependency on the separately retired legacy ViT
factory. The supplied config selects CIFAR-10/ResNet-18 with stochastic width
selection and resource limits disabled.

Restoration work would need to check model-factory/trainer interactions, subnet
slicing, the distillation loss and aggregation of partially covered parameters.
The optional resource-limit branch computes FLOPs but compares model size with
both configured limits, which needs separate review. Existing client strategy
code does not establish that these training and aggregation paths work together.
The local ViT branch needs its own scope and validation; it is not covered by the
ResNet planning estimate below.

| Preserved file | Role |
| --- | --- |
| [fjord.py](https://github.com/TL-System/plato/blob/main/examples/outdated/model_search/fjord/fjord.py) | Entrypoint selecting a local ResNet or ViT model. |
| [fjord_algorithm.py](https://github.com/TL-System/plato/blob/main/examples/outdated/model_search/fjord/fjord_algorithm.py) | Width selection, parameter slicing and aggregation. |
| [fjord_client.py](https://github.com/TL-System/plato/blob/main/examples/outdated/model_search/fjord/fjord_client.py) | Client lifecycle handling for the server-selected width. |
| [fjord_resnet18_dynamic.toml](https://github.com/TL-System/plato/blob/main/examples/outdated/model_search/fjord/fjord_resnet18_dynamic.toml) | CIFAR-10/ResNet-18 configuration; resource limits disabled. |
| [fjord_server.py](https://github.com/TL-System/plato/blob/main/examples/outdated/model_search/fjord/fjord_server.py) | Per-client width responses and aggregation dispatch. |
| [fjord_trainer.py](https://github.com/TL-System/plato/blob/main/examples/outdated/model_search/fjord/fjord_trainer.py) | Server model construction and client subnet training. |
| [resnet.py](https://github.com/TL-System/plato/blob/main/examples/outdated/model_search/fjord/resnet.py) | Local variable-width ResNet model definitions. |
| [vit.py](https://github.com/TL-System/plato/blob/main/examples/outdated/model_search/fjord/vit.py) | Local variable-width ViT model definition. |

### FL-MAML

This experiment uses PyTorch/NumPy, Torchvision MNIST and a custom trainer with
separate inner and outer SGD optimizers, personalized testing and checkpoint
paths. Several concrete source inconsistencies block a straightforward run:
the trainer reads momentum and weight decay from `Config().trainer`, while the
supplied config puts them under `parameters.optimizer`; scheduler branches use
undefined `optimizers` and `lr_scheduler` names; and the two
`training_per_stage` calls omit an argument required by its signature.

Fixing those interfaces would not establish MAML correctness. The connection
between the copied inner-stage model and outer-stage gradients needs a numerical
reference checked against the cited algorithm. Personalization isolation,
report handling and checkpoint behavior also need end-to-end validation.

| Preserved file | Role |
| --- | --- |
| [fl_maml.py](https://github.com/TL-System/plato/blob/main/examples/outdated/fl_maml/fl_maml.py) | Experiment entrypoint wiring client, server and trainer. |
| [fl_maml_MNIST_lenet5.toml](https://github.com/TL-System/plato/blob/main/examples/outdated/fl_maml/fl_maml_MNIST_lenet5.toml) | MNIST/LeNet-5 configuration and meta learning rate. |
| [fl_maml_client.py](https://github.com/TL-System/plato/blob/main/examples/outdated/fl_maml/fl_maml_client.py) | Personalization request handling and accuracy reporting. |
| [fl_maml_server.py](https://github.com/TL-System/plato/blob/main/examples/outdated/fl_maml/fl_maml_server.py) | Training/personalization round and report coordination. |
| [fl_maml_trainer.py](https://github.com/TL-System/plato/blob/main/examples/outdated/fl_maml/fl_maml_trainer.py) | Two-stage training, personalization and testing loops. |

### CS-MAML

This experiment reuses the sibling FL-MAML client and trainer through
`sys.path.append("../fl_maml/")`, making the import dependent on the working
directory. It adds central/edge coordination, personalization events and
accuracy reports on top of the same MNIST/PyTorch stack. It therefore inherits
the FL-MAML trainer blockers above.

The entrypoint's four positional `server.run` arguments still match the current
method signature; they are not, by themselves, a demonstrated API failure.
Restoration would nevertheless need a real central/edge round to check model
and trainer construction, report representation, event completion and
personalization before claiming cross-silo compatibility.

| Preserved file | Role |
| --- | --- |
| [cs_maml.py](https://github.com/TL-System/plato/blob/main/examples/outdated/cs_maml/cs_maml.py) | Cross-silo entrypoint importing sibling FL-MAML modules. |
| [cs_maml_MNIST_lenet5.toml](https://github.com/TL-System/plato/blob/main/examples/outdated/cs_maml/cs_maml_MNIST_lenet5.toml) | MNIST/LeNet-5 configuration with one edge silo. |
| [cs_maml_edge.py](https://github.com/TL-System/plato/blob/main/examples/outdated/cs_maml/cs_maml_edge.py) | Edge-client personalization strategy and event waiting. |
| [cs_maml_server.py](https://github.com/TL-System/plato/blob/main/examples/outdated/cs_maml/cs_maml_server.py) | Central/edge round, report and personalization coordination. |

### Restoration planning estimates

These are preliminary estimates in focused implementation/review sessions for
bounded local regression fixtures. They are not commitments or promises of a
working restoration. Recovering an appropriate dependency environment, resolving
research semantics, obtaining external assets and reproducing paper-scale
results can require additional work.

| Experiment | Planning estimate | Work needed before claiming a restored example |
| --- | --- | --- |
| Norm bounding | 1–2 sessions | Threshold handling, a reference for the clipping/aggregation rule, and numerical/device regressions. |
| FjORD ResNet | 3–5 sessions | Width selection, real subnet training and aggregation checks; local ViT restoration is separate. |
| FL-MAML | 3–5 sessions | Trainer/config repairs, a two-stage gradient reference, personalization isolation and checkpoint qualification. |
| CS-MAML | 2–3 additional sessions after FL-MAML | A real central/edge numerical round, personalization reporting and lifecycle checks. |

### Other preserved utilities

[Multimodal dataset utilities](https://github.com/TL-System/plato/tree/main/archives/legacy_datalib)
and the [legacy Gym adapter](https://github.com/TL-System/plato/tree/main/archives/legacy_rl_env)
remain preserved utilities with their original limitations. They are separate
from the four experiment inventories above; individual modules are not
maintained experiment entrypoints. Consult their archive READMEs for original
locations, dependencies and known limitations.

These are research references, not recommended current strategy templates.
Start new client extensions from the active examples in the
[client reference](../references/clients.md). Preserved lockfiles and captured
source hashes establish provenance, not successful execution on current
Python 3.13 dependencies. External datasets, checkpoints, and toolchains may
need separate historical versions; unknown upstream revisions remain unknown.
