# Retired PyTorch Hub model adapter

The unpinned network-backed pytorch/vision adapter is retired. The maintained torchvision factory uses torchvision.models.get_model from the installed dependency. Model architectures and constructor defaults remain available.

Maintained replacement: [configs/CIFAR10/fedavg_resnet18_torchvision.toml](../../../configs/CIFAR10/fedavg_resnet18_torchvision.toml).

## Preserved source

- [plato/models/torch_hub.py](original/plato/models/torch_hub.py)
- [configs/CIFAR10/fedavg_resnet18_torchhub.toml](original/configs/CIFAR10/fedavg_resnet18_torchhub.toml)

The [manifest](manifest.json) records every original path, Git blob and mode, SHA256, byte count and moved-versus-copied disposition. Source commit: c1359992f70533c0f2a47d30742f9b14295da882; tree: 9e75b3b2f306677fabc6f9214658bfeacd152dc5. Copied runtime files and the pre-change qualification ledger are historical context.

The [license](original/LICENSE), [package manifest](original/pyproject.toml) and [lockfile](original/uv.lock) preserve the historical environment, without establishing compatibility or successful execution.

## Historical restoration

Create a separate checkout:

```bash
git worktree add --detach ../plato-history-torch-hub c1359992f70533c0f2a47d30742f9b14295da882
```

Use original paths in that checkout; do not import archived modules into the maintained runtime. External data/checkpoints must be supplied separately. Original documentation and source retain their wording, bibliography and historical commands; those commands have not been requalified.

See the [archive index](../README.md).
