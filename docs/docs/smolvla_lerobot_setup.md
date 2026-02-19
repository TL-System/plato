# SmolVLA + LeRobot Optional Setup

This setup path is optional. Core Plato federated workloads continue to use the
default dependency set from `uv sync`.

## Install the robotics extra

From the repository root:

```bash
uv sync --extra robotics
```

This installs `lerobot[smolvla]` and the associated training stack only when the
`robotics` extra is requested.

## Environment gating

When adding LeRobot-backed modules, keep imports guarded so non-robotics
environments fail with a clear action instead of a hard crash at import time.

```python
try:
    import lerobot
except ImportError as exc:
    raise ImportError(
        "LeRobot support is optional. Install with: uv sync --extra robotics"
    ) from exc
```

## Runtime notes for SmolVLA/LeRobot

- CUDA-capable GPUs are recommended for practical SmolVLA fine-tuning; CPU is
  mainly suitable for smoke checks.
- Install `ffmpeg` on hosts that read video-backed LeRobot datasets.
- Authenticate with Hugging Face (`huggingface-cli login`) when accessing
  private dataset repositories.
- LeRobot currently constrains the Torch stack used by this optional path;
  if you need different Torch constraints for non-robotics research, keep a
  separate virtual environment.

## Quick verification

```bash
uv run python -c "import lerobot; print(lerobot.__version__)"
```
