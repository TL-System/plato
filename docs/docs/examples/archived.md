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

The following sources remain at their existing historical locations:

- [Norm bounding](https://github.com/TL-System/plato/tree/main/examples/outdated/norm_bounding):
  legacy aggregation and threshold handling.
- [FjORD](https://github.com/TL-System/plato/tree/main/examples/outdated/model_search/fjord):
  legacy width sampling with local ResNet/ViT implementations.
- [FL-MAML](https://github.com/TL-System/plato/tree/main/examples/outdated/fl_maml) and
  [CS-MAML](https://github.com/TL-System/plato/tree/main/examples/outdated/cs_maml):
  historical meta-training and cross-silo personalization; CS-MAML depends on
  the sibling FL-MAML directory.
- [Multimodal dataset utilities](https://github.com/TL-System/plato/tree/main/archives/legacy_datalib)
  and [legacy Gym adapter](https://github.com/TL-System/plato/tree/main/archives/legacy_rl_env):
  preserved utilities with their original limitations.

These are research references, not recommended current strategy templates.
Start new client extensions from the active examples in the
[client reference](../references/clients.md). Preserved lockfiles and captured
source hashes establish provenance, not successful execution on current
Python 3.13 dependencies. External datasets, checkpoints, and toolchains may
need separate historical versions; unknown upstream revisions remain unknown.
