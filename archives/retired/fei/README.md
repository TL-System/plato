# Retired FEI

The FEI experiment and its FashionMNIST and CIFAR10 configurations are retired.
This archive preserves the original six files byte-for-byte, outside the
supported examples, package distributions, and normal test collection. Shared
`plato.utils.reinforcement_learning` utilities and the `rl` extra remain active.

The [manifest](manifest.json) records every original path, archive path, source
commit, Git blob and mode, SHA256, byte size, and disposition. The six original
files come from Plato commit `3d4e08b26972e114d87ef39795e01f4dfa554b94`, tree
`16602a4bd057c30ec2bf195b926d6f2d405774dd`. The copied
[license](original/LICENSE), [package manifest](original/pyproject.toml), and
[lockfile](original/uv.lock) retain that checkout's context. The copied
[Phase4 ledger](original/tests/examples_phase4/cases.json) comes separately from
integration commit `e81248dae1844fb5bc66a8d880c9a17b485606ac`. Its seven frozen FEI
cases are historical inventory, not current qualification or runnable tests.

## Research and execution status

The [independent rejected-repair review](evidence/rejected-repair-review.json)
records unresolved differences between the code's state correlations and the
local/global gradient definitions in *Quality-Oriented Federated Learning on
the Fly* (DOI `10.1109/MNET.001.2200235`). It demonstrated a state discrepancy of
569.2927. The attempted repair's strict profile passed six cases and failed one
because its rounded-state oracle rejected a difference of 0.0001. These are
separate findings. Neither the repair nor its CI changes were accepted or
integrated, and the archived source contains neither candidate change.

No accuracy improvement, benchmark reproduction, or successful current FEI
qualification is claimed. The [retirement plan](evidence/retirement-plan.json)
and its [independent approval](evidence/retirement-plan-review.json) explain the
bounded retirement. The raw review archive remains external; its SHA256 is in
the manifest. It contains a third-party paper PDF and is not redistributed here.
Absolute temporary paths in historical reports describe the original audit
environment and need not exist in a later checkout.

## Historical restoration

From a clone containing the original commit, create a separate checkout:

```bash
git worktree add --detach ../plato-history-fei 3d4e08b26972e114d87ef39795e01f4dfa554b94
```

Use that checkout's shared RL code and preserved dependency manifests to
investigate the historical experiment. External datasets, checkpoints, and
toolchains must be supplied separately. Historical run commands remain
unverified on current dependencies. Source preservation establishes provenance,
not compatibility or execution success. Do not import archived modules from the
maintained runtime. A future revival requires an explicit research disposition,
author corrections, and fresh independent review.
