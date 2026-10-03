# Retired AnyCostFL local ViT

The local `vit.py` implementation and `example_ViT.toml`; shared runtime files are copied only as historical context. This is a historical research snapshot, outside the supported runtime,
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
git worktree add --detach ../plato-history-anycostfl-vit 477f2f1bb52a1b5a58c1f9f2d30f640039f83a8e
```

Investigate dependencies in that separate checkout using the preserved manifests
and lockfile. External datasets, model checkpoints, and toolchain requirements
must be supplied separately. Source preservation does not bundle those assets
or establish compatibility with the current environment. Do not import archived
modules from the maintained runtime.

The archive includes the original entrypoint and shared algorithm, client,
server, trainer, and ResNet files under
[`original/examples/model_search/anycostfl`](original/examples/model_search/anycostfl)
so the old ViT branch can be studied in context. Those copied shared files are
not declarations that the corresponding active ResNet path is retired.

The original entrypoint accepts this archived ViT config. This reconstruction
command is **unverified on current dependencies**:

```bash
cd ../plato-history-anycostfl-vit/examples/model_search/anycostfl
uv run anycostfl.py -c example_ViT.toml
```

This local ViT implementation is separate from `plato.models.vit` and its old
submodules. Its header and Plato license are preserved; a precise external
source attribution and original upstream revision are not established. Do not
infer either from code similarity.

For current use, the [AnyCostFL ResNet entrypoint](../../../examples/model_search/anycostfl/anycostfl.py)
and [`example_ResNet.toml`](../../../examples/model_search/anycostfl/example_ResNet.toml)
remain active. The [model-search guide](<../../../docs/docs/examples/algorithms/10. Algorithms based on Neural Architecture Search and Model Search.md>) gives that command.

See the [archive index](../README.md) and [third-party provenance](../../../docs/third_party.md).
