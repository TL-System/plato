"""Explicit real pinned pretrained qualification; fails on unavailable artifacts.

Run from the repository root:
    uv run python examples/huggingface/qualify_qwen3.py --output /tmp/qwen3-proof

HF_HOME may select an isolated cache. This performs serial CPU FP32 training
through the production datasource/trainer/PEFT/server aggregation interfaces.
It does not launch sockets or measure language-model benchmark quality.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import resource
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from plato.config import Config
from tests.integration.utils import configure_environment
from tests.test_utils.qwen3 import reference_config, validate_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = reference_config()
    with configure_environment(config, runtime_root=args.output):
        expected = config["trainer"]["model_revision"]
        snapshot = Path(
            snapshot_download(
                config["trainer"]["model_name"],
                revision=expected,
                cache_dir=Config().params["model_path"] + "/huggingface",
                allow_patterns=["config.json", "model.safetensors"],
            )
        )
        if snapshot.name != expected or not (snapshot / "model.safetensors").is_file():
            raise RuntimeError(
                "Pretrained checkpoint did not resolve to the approved pin."
            )
        result = validate_runtime(args.output / "adapter-checkpoint")
    result.update(
        resolved_model_revision=snapshot.name,
        resolved_checkpoint_directory=str(snapshot),
        status="PASS",
        scope="Real official pretrained Qwen3; serial CPU 2/3-client update, production "
        "server aggregation, held-out perplexity, adapter checkpoint/reload.",
        python=sys.version,
        platform=platform.platform(),
        package_versions={
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "peft",
                "datasets",
                "huggingface-hub",
                "safetensors",
            )
        },
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        peak_rss_scope="Single qualification process with one shared frozen base, "
        "serial logical clients and the production server object.",
    )
    destination = args.output / "qualification.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
