# Retired Nanochat integration

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

The [original runbook](<original/docs/docs/examples/case-studies/5. Nanochat in Plato.md>)
retains its original commands and wording. References inside archived documents
are historical paths and may require the historical checkout to resolve.

## Isolated historical restoration

Create a separate worktree from a clone with the recorded commit available:

```bash
git worktree add --detach ../plato-historical-nanochat 9ee92c52307cd3ecdce0c7416708c504bdb52137
cd ../plato-historical-nanochat
git submodule update --init external/nanochat
```

Use the preserved runbook and lockfile to investigate an isolated historical
environment. Do not copy the retired sources into the maintained runtime and
assume current dependencies reproduce the original experiment. Restoration and
old training commands are unverified unless accompanied by a separate receipt.

## Upstream snapshot and migration

The former submodule points to
[karpathy/nanochat](https://github.com/karpathy/nanochat/tree/c75fe54aa7c1fa881701c246f9427bcbe4eee5a4)
at `c75fe54aa7c1fa881701c246f9427bcbe4eee5a4`. The inert
[upstream tarball](upstream-c75fe54aa7c1fa881701c246f9427bcbe4eee5a4.tar.gz)
contains its 54 tracked files. [UPSTREAM_LICENSE](UPSTREAM_LICENSE) retains the
MIT license and Andrej Karpathy copyright. The manifest hashes the tarball and
each upstream file; the snapshot does not bundle model weights or datasets.

The maintained text-model replacement is
[Qwen3 Federated LoRA](<../../../docs/docs/examples/case-studies/6. Qwen3 Federated LoRA.md>).
It uses native Transformers and PEFT with a pinned base checkpoint. Nanochat
configurations, tokenizer artifacts, and CORE results are not interchangeable
with this replacement.
