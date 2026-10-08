"""Diagnostics for explicitly retired backend configuration names."""


def raise_if_retired(name: str, *, category: str) -> None:
    """Reject a retired backend and point to its archived provenance."""
    backend = {
        "nanochat": "nanochat",
        "nanochat_core": "nanochat",
        "lerobot": "lerobot",
        "smolvla": "lerobot",
    }.get(name.lower())
    retired_categories = {
        ("model", "vit"): "legacy-vit",
        ("model", "torch_hub"): "torch-hub",
        ("optimizer", "adahessian"): "adahessian",
        ("sampler", "modality_iid"): "modality-samplers",
        ("sampler", "modality_quantity_noniid"): "modality-samplers",
    }
    backend = retired_categories.get((category, name.lower()), backend)
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
    if backend == "torch-hub":
        message += " Use model_type='torchvision' for installed torchvision models."
    raise ValueError(message)


def raise_if_retired_config(filename: str) -> None:
    """Explain known historical recipe names when their files are missing."""
    from pathlib import Path

    archive = {
        "fedunlearning_adahessian_MNIST_lenet5.toml": "adahessian",
        "fedavg_resnet18_torchhub.toml": "torch-hub",
        "fedavg_shakespeare_bert.toml": "bert-shakespeare",
        "split_learning_wikitext2_llama2.toml": "older-split-llm-recipes",
        "split_learning_wikitext2_opt350m.toml": "older-split-llm-recipes",
    }.get(Path(filename).name)
    if archive is not None:
        raise ValueError(
            f"The configuration '{filename}' was retired. Historical source and "
            f"replacement guidance: archives/retired/{archive}/README.md."
        )
