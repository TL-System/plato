# Retired HeteroFL custom MobileNetV3

The custom `mobilenetv3.py` branch; shared runtime files and the ResNet config are copied only as historical context. This is a historical research snapshot, outside the supported runtime,
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
git worktree add --detach ../plato-history-heterofl-mobilenetv3 477f2f1bb52a1b5a58c1f9f2d30f640039f83a8e
```

Investigate dependencies in that separate checkout using the preserved manifests
and lockfile. External datasets, model checkpoints, and toolchain requirements
must be supplied separately. Source preservation does not bundle those assets
or establish compatibility with the current environment. Do not import archived
modules from the maintained runtime.

The [custom MobileNetV3 source](original/examples/model_search/heterofl/mobilenetv3.py)
and historical entrypoint are preserved. There was no dedicated MobileNetV3
TOML in this source tree, so this archive does not supply a verified launch
recipe. The copied ResNet config is historical context, not a MobileNet config.

The source's MobileNetV3 and SENet references retain
[supplemental notices](licenses), including MIT and Apache-2.0 texts. Their
retrieval URLs and revisions are in the manifest; the original vendored source
revision remains unknown.

The current [HeteroFL ResNet entrypoint](../../../examples/model_search/heterofl/heterofl.py)
and [dynamic ResNet config](../../../examples/model_search/heterofl/heterofl_resnet18_dynamic.toml)
remain active. This retirement affects the custom HeteroFL branch, not generic
Torchvision MobileNet models.

See the [archive index](../README.md) and [third-party provenance](../../../docs/third_party.md).
