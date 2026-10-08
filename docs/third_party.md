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

## Retired legacy ViT upstream snapshots

The [legacy ViT archive](../archives/retired/legacy-vit/README.md) replaces the
former active gitlinks with inert snapshots at their exact recorded commits:

| Component | Exact gitlink revision | Preserved notices |
| --- | --- | --- |
| DViT | `1ccb152cea43fcbc3cd517a45c12c65734f8ace3` | [MIT](../archives/retired/legacy-vit/licenses/dvit/LICENSE), nested [DeiT Apache-2.0](../archives/retired/legacy-vit/licenses/dvit/lib_deit/LICENSE), and [progress ISC-style](../archives/retired/legacy-vit/licenses/dvit/utils/progress/LICENSE) |
| T2T-ViT | `0f63dc9558f4d192de926504dbddfa1b3f5db6ca` | [Clear BSD](../archives/retired/legacy-vit/licenses/t2tvit/LICENSE) |

The repositories are [zhoudaquan/dvit_repo](https://github.com/zhoudaquan/dvit_repo/tree/1ccb152cea43fcbc3cd517a45c12c65734f8ace3)
and [yitu-opensource/T2T-ViT](https://github.com/yitu-opensource/T2T-ViT/tree/0f63dc9558f4d192de926504dbddfa1b3f5db6ca).
The [manifest](../archives/retired/legacy-vit/manifest.json) records original
paths, all tracked upstream files, tarball hashes, and copied notices. Source
snapshots do not bundle the external checkpoints and datasets used by the old
configs.

## Retired model-search vendored code

The seven model-search/factory archives preserve original Plato source bytes,
headers, and license context. Their manifests identify the exact Plato commit,
tree, and blobs. Except for the two gitlinks above, an exact original upstream
revision for the vendored code is **unknown**. Supplemental notices under each
archive's `licenses/` directory record retrieval URL, observed upstream revision,
date, and hash. Those revisions identify the retrieved notices, not the original
vendored source.

- **NASViT and pFedRLNAS:** the vendored
  [NASViT license](../archives/retired/pfedrlnas/original/examples/model_search/pfedrlnas/VIT/nasvit_wrapper/NASViT/LICENSE)
  is CC-BY-NC-4.0. Its
  [README](../archives/retired/pfedrlnas/original/examples/model_search/pfedrlnas/VIT/nasvit_wrapper/NASViT/README.md)
  and headers retain component attribution. Supplemental notices cover Swin
  Transformer and Once-for-All (MIT), timm and TensorFlow model helpers
  (Apache-2.0), and MobileNetV3/SENet references. See the
  [pFedRLNAS manifest](../archives/retired/pfedrlnas/manifest.json) for every
  notice source and provenance caveat.
- **DARTS:** both [FedRLNAS](../archives/retired/fedrlnas/README.md) and
  [pFedRLNAS](../archives/retired/pfedrlnas/README.md) retain their separate
  historical code copies and supplemental DARTS Apache-2.0 notices.
- **FedTP:** its archived hypernetwork integration retains a
  [supplemental upstream MIT notice](../archives/retired/fedtp/licenses/fedtp-LICENSE).
- **Custom MobileNetV3:** the HeteroFL and pFedRLNAS copies retain references to
  d-li14 MobileNetV3 (MIT), kuan-wang MobileNetV3 (Apache-2.0), and SENet (MIT).
  The [HeteroFL manifest](../archives/retired/heterofl-mobilenetv3/manifest.json)
  and [supplemental notices](../archives/retired/heterofl-mobilenetv3/licenses)
  record the captured evidence.
- **Local AnyCostFL/FedRolex ViT:** existing headers and Plato license context
  are preserved. An exact external source attribution or upstream revision is
  not established; no origin is inferred from similarity to other ViT code.

These archives have mixed provenance and notices; Plato's Apache-2.0 license
does not replace third-party terms. Model weights and datasets have separate
provenance from source code. Consult the preserved notices and manifests when
studying or restoring a historical setup. The [archive index](../archives/retired/README.md)
links all retired families and the earlier historical utilities.
