# Retired Legacy modality-mask samplers

These samplers return modality names, whereas maintained training loaders consume dataset indices. No maintained datasource or example uses their modality API. Ordinary data partition samplers remain active.

## Preserved source

- [plato/samplers/modality_iid.py](original/plato/samplers/modality_iid.py)
- [plato/samplers/modality_quantity_noniid.py](original/plato/samplers/modality_quantity_noniid.py)

The [manifest](manifest.json) records every original path, Git blob and mode, SHA256, byte count and moved-versus-copied disposition. Source commit: c1359992f70533c0f2a47d30742f9b14295da882; tree: 9e75b3b2f306677fabc6f9214658bfeacd152dc5. Copied runtime files and the pre-change qualification ledger are historical context.

The [license](original/LICENSE), [package manifest](original/pyproject.toml) and [lockfile](original/uv.lock) preserve the historical environment, without establishing compatibility or successful execution.

## Historical restoration

Create a separate checkout:

```bash
git worktree add --detach ../plato-history-modality-samplers c1359992f70533c0f2a47d30742f9b14295da882
```

Use original paths in that checkout; do not import archived modules into the maintained runtime. External data/checkpoints must be supplied separately. Original documentation and source retain their wording, bibliography and historical commands; those commands have not been requalified.

See the [archive index](../README.md).
