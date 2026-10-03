"""Shared artifact identity for Plato's Hugging Face integrations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from plato.config import Config


def artifact_identity(model_name: str | None = None) -> dict[str, str]:
    """Resolve model/tokenizer repositories and their independent revisions."""
    trainer = Config().trainer
    model_name = model_name or trainer.model_name
    model_revision = getattr(trainer, "model_revision", None) or "main"
    tokenizer_name = getattr(trainer, "tokenizer_name", None) or model_name
    tokenizer_revision = getattr(trainer, "tokenizer_revision", None)
    if not tokenizer_revision:
        tokenizer_revision = model_revision if tokenizer_name == model_name else "main"
    return {
        "model_name": model_name,
        "model_revision": model_revision,
        "tokenizer_name": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
    }


def pretrained_kwargs(*, revision: str, cache_dir: str | None) -> dict[str, Any]:
    """Use supported Transformers arguments without permitting remote code."""
    kwargs: dict[str, Any] = {
        "revision": revision,
        "cache_dir": cache_dir,
        "trust_remote_code": False,
    }
    token = getattr(getattr(Config(), "parameters", None), "huggingface_token", None)
    if isinstance(token, str) and token:
        kwargs["token"] = token
    return kwargs


def dataset_kwargs(data_config: Any) -> dict[str, Any]:
    """Resolve optional dataset config, revision and local split files."""
    kwargs: dict[str, Any] = {}
    for config_key, argument in (
        ("dataset_config", "name"),
        ("dataset_revision", "revision"),
    ):
        value = getattr(data_config, config_key, None)
        if value is not None:
            kwargs[argument] = value
    data_files = getattr(data_config, "data_files", None)
    if data_files is not None:
        if not isinstance(data_files, Mapping):
            raise TypeError("data.data_files must map split names to local files.")
        kwargs["data_files"] = dict(data_files)
    return kwargs


def adapter_save_embeddings(model: Any) -> bool:
    """Preserve resized/targeted embeddings without PEFT's unpinned hub lookup."""
    if getattr(model, "plato_save_embedding_layers", False):
        return True
    config = getattr(model, "active_peft_config", None)
    targets = getattr(config, "target_modules", None)
    if not targets:
        return False
    # This is the same local target test PEFT uses for its embedding export.
    from peft.tuners.tuners_utils import match_target_against_key
    from peft.utils.other import EMBEDDING_LAYER_NAMES

    if not isinstance(targets, str):
        return any(name in targets for name in EMBEDDING_LAYER_NAMES)
    return any(
        name.rsplit(".", 1)[-1] in EMBEDDING_LAYER_NAMES
        and match_target_against_key(targets, name)
        for name, _ in model.named_modules()
    )
