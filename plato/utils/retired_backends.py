"""Diagnostics for explicitly retired backend configuration names."""


def raise_if_retired(name: str, *, category: str) -> None:
    """Reject a retired backend and point to its archived provenance."""
    backend = {
        "nanochat": "nanochat",
        "nanochat_core": "nanochat",
        "lerobot": "lerobot",
        "smolvla": "lerobot",
    }.get(name.lower())
    if category == "model" and name.lower() == "vit":
        backend = "legacy-vit"
    if backend is None:
        return
    message = (
        f"The {category} '{name}' was retired. Historical source and restoration "
        f"instructions: archives/retired/{backend}/README.md."
    )
    if backend == "nanochat":
        message += (
            " For supported text training, use "
            "configs/HuggingFace/fedavg_qwen3_06b_lora.toml."
        )
    raise ValueError(message)
