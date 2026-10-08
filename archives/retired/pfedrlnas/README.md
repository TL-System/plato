# Retired pFedRLNAS / PerFedRLNAS

All NASViT, MobileNetV3, and DARTS modes, their shared configs, and vendored NASViT support. This is a historical research snapshot, outside the supported runtime,
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
git worktree add --detach ../plato-history-pfedrlnas 477f2f1bb52a1b5a58c1f9f2d30f640039f83a8e
```

Investigate dependencies in that separate checkout using the preserved manifests
and lockfile. External datasets, model checkpoints, and toolchain requirements
must be supplied separately. Source preservation does not bundle those assets
or establish compatibility with the current environment. Do not import archived
modules from the maintained runtime.

Restore the complete [`pfedrlnas` directory](original/examples/model_search/pfedrlnas)
with its original sibling paths. The MobileNetV3 implementation imports support
from `VIT/nasvit_wrapper`; restoring it alone is insufficient. Its DARTS copy and the
[separate FedRLNAS copy](../fedrlnas/README.md) are preserved independently
without deduplication.

The historical chapter lists these entrypoints, all **unverified on current
dependencies**. Each directory is relative to the reconstructed repository root:

| Working directory | Historical command |
| --- | --- |
| `examples/model_search/pfedrlnas/VIT` | `uv run fednas.py -c ../configs/PerFedRLNAS_CIFAR10_NASVIT_NonIID01.toml` |
| `examples/model_search/pfedrlnas/MobileNetV3` | `uv run fednas.py -c ../configs/PerFedRLNAS_CIFAR10_Mobilenet_NonIID03.toml` |
| `examples/model_search/pfedrlnas/MobileNetV3` | `uv run fednas.py -c ../configs/MobileNetV3_CIFAR10_03_async.toml` |
| `examples/model_search/pfedrlnas/DARTS` | `uv run fednas.py -c ../configs/PerFedRLNAS_CIFAR10_DARTS_NonIID_03.toml` |

The vendored NASViT [README](original/examples/model_search/pfedrlnas/VIT/nasvit_wrapper/NASViT/README.md)
and [CC-BY-NC-4.0 license](original/examples/model_search/pfedrlnas/VIT/nasvit_wrapper/NASViT/LICENSE)
are preserved, along with code headers and project notices. DARTS, Swin,
Once-for-All, timm, TensorFlow, MobileNetV3, and SENet references have
[supplemental notices](licenses). Their retrieval provenance is in the manifest.
The original vendored revisions are unknown; supplemental notice revisions must
not be presented as the revisions of the archived code. These components are
not uniformly covered by Plato's Apache-2.0 license.

See the [archive index](../README.md) and [third-party provenance](../../../docs/third_party.md).
