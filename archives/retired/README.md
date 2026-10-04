# Archived research and integrations

These sources preserve research history and provenance. They are outside the
current supported runtime and are not current copy-and-run examples. Use the
family README and manifest to reconstruct a separate historical checkout;
current `uv sync` does not requalify old dependencies, datasets, checkpoints, or
training recipes. Archived source and original documentation retain their old
wording and paths.

## Retired families

| Archive | Historical scope |
| --- | --- |
| [Legacy ViT factory](legacy-vit/README.md) | The former `plato.models.vit` factory and four ViT configs, including DeepViT, T2T-ViT, SwinV2, and LeViT. |
| [FedTP](fedtp/README.md) | The complete FedTP experiment, including the server hypernetwork, algorithm, and both ViT/T2T-ViT configs. |
| [pFedRLNAS / PerFedRLNAS](pfedrlnas/README.md) | All NASViT, MobileNetV3, and DARTS modes, their shared configs, and vendored NASViT support. |
| [FedRLNAS](fedrlnas/README.md) | The FedRLNAS experiment, its MNIST config, and its DARTS search-space code. |
| [AnyCostFL local ViT](anycostfl-vit/README.md) | The local `vit.py` implementation and `example_ViT.toml`; shared runtime files are copied only as historical context. |
| [FedRolex local ViT](fedrolex-vit/README.md) | The local `vit.py` implementation and `example_ViT.toml`; shared runtime files are copied only as historical context. |
| [HeteroFL custom MobileNetV3](heterofl-mobilenetv3/README.md) | The custom `mobilenetv3.py` branch; shared runtime files and the ResNet config are copied only as historical context. |
| [Nanochat](nanochat/README.md) | Plato integration and exact upstream snapshot; current text reference is Qwen3 LoRA. |
| [LeRobot / SmolVLA](lerobot/README.md) | Original robotics integration and runbook; no maintained robotics replacement. |
| [FEI](fei/README.md) | Original reinforcement-learning experiment and both configs; unresolved research semantics and an unaccepted repair are documented. |

Each manifest distinguishes moved source from copied context and records its
own pre-retirement commit. Upstream tarballs are inert snapshots. Vendored code
without a recorded upstream revision is labeled unknown; supplemental license
retrieval revisions do not identify the original code revision. See
[third-party provenance](../../docs/third_party.md) for exact ViT pins and
mixed-license notices.

## Earlier historical examples

These files remain in their existing locations; no restoration or new runtime
qualification is implied by this index.

| Location | Historical assumptions |
| --- | --- |
| [Norm bounding](../../examples/outdated/norm_bounding) | Legacy aggregation override, tensor/NumPy conversion, and an explicit norm threshold. |
| [FjORD](../../examples/outdated/model_search/fjord) | Local ResNet/ViT factories, width sampling, legacy training overrides, `ptflops`, and `einops`. |
| [FL-MAML](../../examples/outdated/fl_maml) | Meta-training and personalized testing, legacy lifecycle and checkpoint assumptions. |
| [CS-MAML](../../examples/outdated/cs_maml) | Cross-silo personalization with a relative-path dependency on sibling FL-MAML. Restore both together. |
| [Multimodal dataset utilities](../legacy_datalib/README.md) | Old video/audio/annotation tooling and external datasets; known defects are preserved. |
| [Legacy Gym adapter](../legacy_rl_env/README.md) | Obsolete environment/lifecycle assumptions; separate from the active reinforcement-learning package. |

## Current examples

The [model-search guide](<../../docs/docs/examples/algorithms/10. Algorithms based on Neural Architecture Search and Model Search.md>)
retains AnyCostFL, FedRolex, and HeteroFL ResNet paths and the separate SysHeteroFL
ResNet example. Generic Hugging Face and Torchvision model families are not
retired by these implementation-specific changes.

[Qwen3 Federated LoRA](<../../docs/docs/examples/case-studies/6. Qwen3 Federated LoRA.md>)
is the maintained text-model reference, and [native MLX](../../docs/docs/mlx.md)
has its own bounded Apple Silicon support and qualification commands.
