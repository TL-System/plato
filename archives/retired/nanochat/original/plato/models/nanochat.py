"""
Factory for Nanochat GPT models integrated with Plato's registry.
"""

from __future__ import annotations

import logging
from dataclasses import fields
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Nanochat model integration requires PyTorch. "
        "Install torch via the project's optional dependencies."
    ) from exc

from plato.utils.third_party import ThirdPartyImportError, ensure_nanochat_importable

DEFAULT_MODEL_CONFIG: dict[str, int] = {
    "sequence_len": 2048,
    "vocab_size": 50304,
    "n_layer": 12,
    "n_head": 6,
    "n_kv_head": 6,
    "n_embd": 768,
}


def _import_nanochat_modules():
    ensure_nanochat_importable()
    from nanochat.checkpoint_manager import (
        load_model_from_dir,
    )
    from nanochat.gpt import GPT, GPTConfig

    return GPT, GPTConfig, load_model_from_dir


def _sanitize_kwargs(kwargs: dict[str, Any], valid_fields: set[str]) -> dict[str, Any]:
    """Filter kwargs to those accepted by GPTConfig."""
    config_kwargs = DEFAULT_MODEL_CONFIG.copy()
    for key, value in kwargs.items():
        if key in valid_fields:
            config_kwargs[key] = value
    return config_kwargs


def _load_from_checkpoint(
    load_dir: str,
    *,
    device: str | torch.device = "cpu",
    phase: str = "train",
    model_tag: str | None = None,
    step: int | None = None,
):
    """Load a Nanochat checkpoint via checkpoint_manager."""
    GPT, _, load_model_from_dir = _import_nanochat_modules()
    torch_device = torch.device(device)
    model, tokenizer, metadata = load_model_from_dir(
        load_dir,
        device=torch_device,
        phase=phase,
        model_tag=model_tag,
        step=step,
    )
    # Attach helpful metadata to the model for downstream use.
    setattr(model, "nanochat_tokenizer", tokenizer)
    setattr(model, "nanochat_metadata", metadata)
    if not isinstance(model, GPT):
        raise TypeError(
            "Checkpoint loader returned an unexpected model type. "
            "Ensure the checkpoint directory points to Nanochat artifacts."
        )
    return model


class Model:
    """Nanochat GPT factory compatible with Plato's model registry."""

    @staticmethod
    def get(model_name: str | None = None, **kwargs: Any):
        """
        Instantiate a Nanochat GPT model.

        Keyword Args:
            sequence_len: Context length (tokens).
            vocab_size: Token vocabulary size.
            n_layer: Number of transformer blocks.
            n_head: Attention heads for queries.
            n_kv_head: Attention heads for keys/values (MQA/GQA).
            n_embd: Hidden dimension width.
            init_weights: Whether to run Nanochat's weight initialisation (default True).
            load_checkpoint_dir: Optional checkpoint directory produced by Nanochat.
            load_checkpoint_tag: Optional subdirectory/model tag within checkpoint dir.
            load_checkpoint_step: Optional numeric step to load (defaults to latest).
            device: Torch device string for checkpoint loading.
            phase: "train" or "eval" when loading checkpoints.
        """
        try:
            GPT, GPTConfig, _ = _import_nanochat_modules()
        except ThirdPartyImportError as exc:  # pragma: no cover - defensive branch
            raise ImportError(
                "Nanochat submodule not found. "
                "Run `git submodule update --init --recursive` to populate external/nanochat."
            ) from exc

        init_weights = kwargs.pop("init_weights", True)
        load_dir = kwargs.pop("load_checkpoint_dir", None)
        checkpoint_tag = kwargs.pop("load_checkpoint_tag", None)
        checkpoint_step = kwargs.pop("load_checkpoint_step", None)
        checkpoint_phase = kwargs.pop("phase", "train")
        checkpoint_device = kwargs.pop("device", "cpu")

        # GPTConfig only accepts specific fields; filter unknown kwargs.
        config_fields = {field.name for field in fields(GPTConfig)}
        config_kwargs = _sanitize_kwargs(kwargs, config_fields)

        if load_dir:
            model = _load_from_checkpoint(
                load_dir,
                device=checkpoint_device,
                phase=checkpoint_phase,
                model_tag=checkpoint_tag,
                step=checkpoint_step,
            )
            return model

        # Model vocab_size MUST match tokenizer vocab_size to avoid IndexError
        try:
            from nanochat.tokenizer import get_tokenizer

            tokenizer = get_tokenizer()
            actual_vocab_size = tokenizer.get_vocab_size()

            # Override vocab_size with tokenizer's actual vocab_size
            if "vocab_size" in config_kwargs:
                configured_vocab = config_kwargs["vocab_size"]
                if configured_vocab != actual_vocab_size:
                    logging.warning(
                        f"[Nanochat Model] Config specifies vocab_size={configured_vocab}, "
                        f"but tokenizer has vocab_size={actual_vocab_size}. "
                        f"Using tokenizer's vocab_size={actual_vocab_size} to match tokenizer."
                    )
            config_kwargs["vocab_size"] = actual_vocab_size

            logging.info(
                f"[Nanochat Model] Using vocab_size={actual_vocab_size} from tokenizer"
            )
        except Exception as e:
            logging.warning(
                f"[Nanochat Model] Could not auto-detect vocab_size from tokenizer: {e}. "
                f"Using configured or default vocab_size."
            )

        config = GPTConfig(**config_kwargs)
        model = GPT(config)
        if init_weights:
            model.init_weights()

        # This allows CORE evaluation and other components to access the tokenizer
        try:
            from nanochat.tokenizer import get_tokenizer

            tokenizer = get_tokenizer()
            setattr(model, "nanochat_tokenizer", tokenizer)
        except Exception:
            pass
        # Set max_seq_len for CORE evaluation truncation
        setattr(model, "max_seq_len", config.sequence_len)
        setattr(model, "nanochat_config", config_kwargs)
        return model
