# Retired LeRobot and SmolVLA integration

This integration was retired from the supported Plato runtime in October 2026.
The archive preserves research history; it is excluded from supported runtime
imports, package distributions, workspace membership, and normal test collection.
Historical dependencies and commands have not been requalified on the refreshed
stack. Python 3.13 is the current full qualification/CI target; retirement does
not establish a universal Python 3.14 incompatibility.

## Contents and provenance

[manifest.json](manifest.json) records original paths, byte sizes, SHA256 hashes,
Git blob identities, and whether each file was moved or copied as historical
context. The pre-retirement Plato commit is
`9ee92c52307cd3ecdce0c7416708c504bdb52137`. The original package manifest and
lockfile are included under `original/`, along with the project’s
[Apache-2.0 license](original/LICENSE). Environment provenance records Python
3.13.16 and uv 0.12.22; this records the capture environment, not successful
historical execution.

The [original runbook](<original/docs/docs/examples/case-studies/3. SmolVLA Trainer with LeRobot.md>)
retains its original commands and wording. References inside archived documents
are historical paths and may require the historical checkout to resolve.

## Isolated historical restoration

Create a separate worktree from a clone with the recorded commit available:

```bash
git worktree add --detach ../plato-historical-lerobot 9ee92c52307cd3ecdce0c7416708c504bdb52137
cd ../plato-historical-lerobot
```

Use the preserved runbook and lockfile to investigate an isolated historical
environment. Do not copy the retired sources into the maintained runtime and
assume current dependencies reproduce the original experiment. Restoration and
old training commands are unverified unless accompanied by a separate receipt.

## Upstream references and migration

The original integration imports APIs from
[Hugging Face LeRobot](https://github.com/huggingface/lerobot), references the
[`lerobot/smolvla_base` policy](https://huggingface.co/lerobot/smolvla_base), and
uses the [`lerobot/pusht_image` dataset](https://huggingface.co/datasets/lerobot/pusht_image).
Their revisions were not pinned in the original integration. The manifest
preserves those identifiers and upstream license references without claiming
that current upstream artifacts reproduce the historical setup.

This archive contains Plato integration code, not vendored LeRobot source,
model weights, or datasets. Source, model, and dataset licenses must be checked
separately at their respective upstream sources.

There is no maintained robotics replacement in Plato. The
[Qwen3 Federated LoRA guide](<../../../docs/docs/examples/case-studies/6. Qwen3 Federated LoRA.md>)
is the maintained text-model example and does not implement robotics policy
training.
