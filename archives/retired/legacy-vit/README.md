# Retired Legacy ViT factory

The former `plato.models.vit` factory and four ViT configs, including DeepViT, T2T-ViT, SwinV2, and LeViT. This is a historical research snapshot, outside the supported runtime,
workspace, package distributions, and normal test collection. Historical run
commands have not been requalified on current dependencies. Python 3.13 remains
the maintained project's default and CI interpreter.

## Contents and provenance

The [manifest](manifest.json) is the complete path map. It records each original
path, archive destination, Git blob and mode, SHA256, byte size, and whether the
file was moved or copied as context. The source is Plato commit
`477f2f1bb52a1b5a58c1f9f2d30f640039f83a8e`, tree
`82af1367f106218d0de98f46775c011e0afd930c`. The captured
[package manifest](original/pyproject.toml), [lockfile](original/uv.lock), and
[Plato license](original/LICENSE) preserve the environment and license context;
they do not establish successful historical execution.

The [original model-search chapter](<original/docs/docs/examples/algorithms/10. Algorithms based on Neural Architecture Search and Model Search.md>) retains its historical
wording. Its paths resolve in the original checkout layout, not necessarily
within this archive.

## Historical restoration

From a clone containing the recorded commit, create a separate worktree:

```bash
git worktree add --detach ../plato-history-legacy-vit 477f2f1bb52a1b5a58c1f9f2d30f640039f83a8e
```

Investigate dependencies in that separate checkout using the preserved manifests
and lockfile. External datasets, model checkpoints, and toolchain requirements
must be supplied separately. Source preservation does not bundle those assets
or establish compatibility with the current environment. Do not import archived
modules from the maintained runtime.

The original factory and configs are under
[`original/plato/models/vit.py`](original/plato/models/vit.py) and
[`original/configs/ViT`](original/configs/ViT). In the historical checkout,
initialize the two recorded submodule paths, or reconstruct them from the exact
inert tarballs below:

```bash
cd ../plato-history-legacy-vit
git submodule update --init plato/models/dvit plato/models/t2tvit
```

A historical config target is `configs/ViT/fedavg_cifar10_dvit.toml`; it is not a
current launch target. The old `@` model-name convention and `pretrained` option
belong to this factory. A retained [local checkpoint probe](original/evidence/2026-refresh/vit-initialization-probe.json)
found that `pretrained=false` still loaded pretrained weights. That defect is
preserved; retirement does not repair initialization.

## Pinned upstream snapshots

- **DViT:** [1ccb152cea43fcbc3cd517a45c12c65734f8ace3](https://github.com/zhoudaquan/dvit_repo/tree/1ccb152cea43fcbc3cd517a45c12c65734f8ace3), preserved in [dvit-upstream-1ccb152cea43fcbc3cd517a45c12c65734f8ace3.tar.gz](dvit-upstream-1ccb152cea43fcbc3cd517a45c12c65734f8ace3.tar.gz); original path `plato/models/dvit`.
- **T2T-ViT:** [0f63dc9558f4d192de926504dbddfa1b3f5db6ca](https://github.com/yitu-opensource/T2T-ViT/tree/0f63dc9558f4d192de926504dbddfa1b3f5db6ca), preserved in [t2tvit-upstream-0f63dc9558f4d192de926504dbddfa1b3f5db6ca.tar.gz](t2tvit-upstream-0f63dc9558f4d192de926504dbddfa1b3f5db6ca.tar.gz); original path `plato/models/t2tvit`.

DViT's [MIT notice](licenses/dvit/LICENSE), nested
[DeiT Apache-2.0 notice](licenses/dvit/lib_deit/LICENSE), and
[progress ISC-style notice](licenses/dvit/utils/progress/LICENSE) are retained.
T2T-ViT retains its [Clear BSD notice](licenses/t2tvit/LICENSE). The manifest
records every tracked upstream file and each tarball hash; these are inert
snapshots, not active submodules.

This retirement is specific to Plato's legacy factory. Generic Hugging Face and
Torchvision model families remain separate, and current causal-LM support does
not provide an image-classification ViT replacement.

See the [archive index](../README.md) and [third-party provenance](../../../docs/third_party.md).
