# Third-party provenance

## Qwen3 text-model reference

The maintained federated LoRA reference uses
[Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base/tree/da87bfb608c14b7cf20ba1ce41287e8de496c0cd),
pinned for both model and tokenizer to
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Its upstream model license is
[Apache-2.0](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/LICENSE).
Weights are downloaded from the official repository, not vendored here.

The original local text fixture is distributed under Plato’s Apache-2.0
repository license, with separate provenance in
[tests/fixtures/qwen3/PROVENANCE.txt](../tests/fixtures/qwen3/PROVENANCE.txt).
Model licensing does not determine the license of a dataset supplied by a user.
See the [Qwen3 guide](<docs/examples/case-studies/6. Qwen3 Federated LoRA.md>)
for execution and validation scope.

## Retired Nanochat integration

The [Nanochat archive](../archives/retired/nanochat/README.md) preserves the
original Plato integration and a source snapshot from
[karpathy/nanochat](https://github.com/karpathy/nanochat/tree/c75fe54aa7c1fa881701c246f9427bcbe4eee5a4)
at `c75fe54aa7c1fa881701c246f9427bcbe4eee5a4`. Its inert upstream tarball preserves
all 54 tracked files. The accompanying `UPSTREAM_LICENSE` retains the MIT notice
and Andrej Karpathy copyright; the manifest records hashes and original paths.
There is no active Nanochat submodule or supported tokenizer build step.

## Retired LeRobot and SmolVLA integration

The [LeRobot archive](../archives/retired/lerobot/README.md) preserves Plato’s
integration and its original runbook. It records references to upstream
[LeRobot](https://github.com/huggingface/lerobot), `lerobot/smolvla_base`, and
`lerobot/pusht_image`. The original integration did not pin model or dataset
revisions. No upstream LeRobot source, model weights, or datasets were vendored.
Consult the upstream source, model, and dataset notices separately when studying
the historical setup.
