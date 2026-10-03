# Retired FedTP

The complete FedTP experiment, including the server hypernetwork, algorithm, and both ViT/T2T-ViT configs. This is a historical research snapshot, outside the supported runtime,
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
git worktree add --detach ../plato-history-fedtp 477f2f1bb52a1b5a58c1f9f2d30f640039f83a8e
```

Investigate dependencies in that separate checkout using the preserved manifests
and lockfile. External datasets, model checkpoints, and toolchain requirements
must be supplied separately. Source preservation does not bundle those assets
or establish compatibility with the current environment. Do not import archived
modules from the maintained runtime.

FedTP depends on the [legacy ViT factory](../legacy-vit/README.md). Restore
that original layout and its exact upstream pins too. The preserved sources are
under [`original/examples/model_search/fedtp`](original/examples/model_search/fedtp).
The following command comes from the historical chapter and is **unverified on
current dependencies**; run it only after reconstructing the historical setup:

```bash
cd ../plato-history-fedtp/examples/model_search/fedtp
uv run fedtp.py -c FedTP_CIFAR10_ViT_NonIID03_scratch.toml
```

The second supplied config is `FedTP_CIFAR10_T2TVIT14_NonIID03_scratch.toml`.
The [supplemental FedTP MIT notice](licenses/fedtp-LICENSE) records its retrieval
source in the manifest. The original vendored upstream revision is unknown;
the license retrieval revision is not a source-code pin.

See the [archive index](../README.md) and [third-party provenance](../../../docs/third_party.md).
