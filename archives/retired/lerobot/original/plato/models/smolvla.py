"""
Factory for SmolVLA policies integrated with Plato's model registry.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any

import torch.nn as nn

from plato.config import Config

DEFAULT_POLICY_PATH = "lerobot/smolvla_base"
DEFAULT_FINETUNE_MODE = "full"
SUPPORTED_FINETUNE_MODES = {"full", "adapter"}
DEFAULT_ADAPTER_PATTERNS = ("adapter", "lora", "peft")


def _node_to_dict(node: Any) -> dict[str, Any]:
    """Convert a config section into a plain dictionary."""
    if node is None:
        return {}
    if isinstance(node, dict):
        return dict(node)
    if hasattr(node, "_asdict"):
        return dict(node._asdict())
    if hasattr(node, "__dict__"):
        return {
            key: value
            for key, value in node.__dict__.items()
            if not key.startswith("_")
        }
    raise TypeError("Unsupported policy configuration format.")


def _import_smolvla_policy() -> type[Any]:
    """Import SmolVLAPolicy lazily to keep robotics dependencies optional."""
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "SmolVLA requires optional LeRobot robotics dependencies. "
            "Install the robotics stack in the active environment before using `model_type = \"smolvla\"`."
        ) from exc
    return SmolVLAPolicy


def _resolve_policy_config() -> dict[str, Any]:
    """Read [parameters.policy] config values, if present."""
    parameters = getattr(Config(), "parameters", None)
    policy = getattr(parameters, "policy", None)
    return _node_to_dict(policy)


def _resolve_policy_path(
    model_name: str | None,
    policy_config: dict[str, Any],
    overrides: dict[str, Any],
) -> str:
    """Resolve pretrained source from kwargs, config, or defaults."""
    candidate = (
        overrides.get("policy_path")
        or overrides.get("path")
        or policy_config.get("path")
    )

    if candidate is None and isinstance(model_name, str) and model_name:
        if model_name.lower() not in {"smolvla", "smolvla_base"}:
            candidate = model_name

    if candidate is None:
        candidate = DEFAULT_POLICY_PATH

    if not isinstance(candidate, str):
        raise TypeError("SmolVLA policy path must be provided as a string.")

    resolved = candidate.strip()
    if not resolved:
        raise ValueError("SmolVLA policy path cannot be empty.")

    if resolved.lower() == "smolvla_base":
        return DEFAULT_POLICY_PATH

    return resolved


def _resolve_finetune_mode(
    policy_config: dict[str, Any],
    overrides: dict[str, Any],
) -> str:
    """Resolve the finetune mode with validation."""
    mode = overrides.get("finetune_mode", policy_config.get("finetune_mode"))
    if mode is None:
        mode = DEFAULT_FINETUNE_MODE

    if not isinstance(mode, str):
        raise TypeError("`finetune_mode` must be a string.")

    normalized_mode = mode.strip().lower()
    if normalized_mode not in SUPPORTED_FINETUNE_MODES:
        raise ValueError(
            "Unsupported SmolVLA finetune mode "
            f"'{mode}'. Expected one of {sorted(SUPPORTED_FINETUNE_MODES)}."
        )
    return normalized_mode


def _resolve_adapter_patterns(
    policy_config: dict[str, Any],
    overrides: dict[str, Any],
) -> list[str]:
    """Resolve adapter parameter patterns from kwargs or config."""
    raw_patterns = overrides.get(
        "adapter_parameter_patterns",
        policy_config.get("adapter_parameter_patterns"),
    )

    if raw_patterns is None:
        return list(DEFAULT_ADAPTER_PATTERNS)

    if isinstance(raw_patterns, str):
        parsed = [token.strip() for token in raw_patterns.split(",") if token.strip()]
        return parsed or list(DEFAULT_ADAPTER_PATTERNS)

    if isinstance(raw_patterns, Iterable):
        parsed = [
            token.strip()
            for token in raw_patterns
            if isinstance(token, str) and token.strip()
        ]
        return parsed or list(DEFAULT_ADAPTER_PATTERNS)

    raise TypeError(
        "`adapter_parameter_patterns` must be a comma-separated string "
        "or list of strings."
    )


def _count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def _apply_finetune_mode(
    policy: nn.Module,
    finetune_mode: str,
    adapter_patterns: list[str],
) -> dict[str, Any]:
    """Set requires_grad according to the requested finetune mode."""
    named_parameters = list(policy.named_parameters())
    if not named_parameters:
        raise ValueError("Loaded SmolVLA policy has no named parameters.")

    if finetune_mode == "full":
        for _, parameter in named_parameters:
            parameter.requires_grad = True

        trainable_names = [name for name, _ in named_parameters]
        return {
            "trainable_names": trainable_names,
            "fallback_mode": None,
        }

    original_trainable_names = {
        name for name, parameter in named_parameters if parameter.requires_grad
    }

    for _, parameter in named_parameters:
        parameter.requires_grad = False

    lowered_patterns = [pattern.lower() for pattern in adapter_patterns]
    matched_names: set[str] = set()
    for name, parameter in named_parameters:
        lowered_name = name.lower()
        if any(pattern in lowered_name for pattern in lowered_patterns):
            parameter.requires_grad = True
            matched_names.add(name)

    fallback_mode: str | None = None
    if not matched_names and original_trainable_names:
        for name, parameter in named_parameters:
            if name in original_trainable_names:
                parameter.requires_grad = True
                matched_names.add(name)
        fallback_mode = "original_requires_grad"
        logging.warning(
            "SmolVLA adapter mode found no parameter names matching patterns %s; "
            "falling back to model-defined requires_grad flags.",
            adapter_patterns,
        )

    if _count_trainable_parameters(policy) == 0:
        raise ValueError(
            "SmolVLA adapter mode selected, but no trainable parameters were "
            "resolved. Configure `adapter_parameter_patterns` or set "
            "`finetune_mode` to 'full'."
        )

    return {
        "trainable_names": sorted(matched_names),
        "fallback_mode": fallback_mode,
    }


def _ensure_checkpoint_compatibility(policy: nn.Module) -> None:
    """Validate required methods for Plato aggregation and checkpoint flow."""
    required_methods = ("state_dict", "load_state_dict", "save_pretrained")
    missing = [name for name in required_methods if not hasattr(policy, name)]
    if missing:
        joined = ", ".join(missing)
        raise TypeError(
            "Loaded SmolVLA policy is incompatible with Plato checkpoints; "
            f"missing method(s): {joined}."
        )


class Model:
    """Factory for LeRobot SmolVLA policies."""

    @staticmethod
    def get(model_name: str | None = None, **kwargs: Any) -> nn.Module:
        """
        Build a SmolVLA policy.

        Keyword Args:
            policy_path/path: Hugging Face repo id or local path for pretrained policy.
            token: HF token for private repositories.
            strict: Strictness flag forwarded to SmolVLAPolicy.from_pretrained().
            finetune_mode: "full" or "adapter".
            adapter_parameter_patterns: Names/patterns used to select adapter params.
        """
        policy_config = _resolve_policy_config()
        policy_path = _resolve_policy_path(model_name, policy_config, kwargs)

        token = kwargs.get("token", policy_config.get("token", os.getenv("HF_TOKEN")))
        strict = kwargs.get("strict", policy_config.get("strict", True))

        SmolVLAPolicy = _import_smolvla_policy()

        try:
            policy = SmolVLAPolicy.from_pretrained(
                policy_path,
                token=token,
                strict=strict,
            )
        except Exception as exc:  # pragma: no cover - exercised via integration
            raise RuntimeError(
                "Failed to load SmolVLA policy from "
                f"'{policy_path}'. Check `parameters.policy.path` and access token."
            ) from exc

        if not isinstance(policy, nn.Module):
            raise TypeError(
                "SmolVLA policy loader returned a non-module object. "
                "Expected a torch.nn.Module-compatible policy."
            )

        finetune_mode = _resolve_finetune_mode(policy_config, kwargs)
        adapter_patterns = _resolve_adapter_patterns(policy_config, kwargs)
        trainable_metadata = _apply_finetune_mode(
            policy,
            finetune_mode=finetune_mode,
            adapter_patterns=adapter_patterns,
        )

        _ensure_checkpoint_compatibility(policy)

        trainable_count = _count_trainable_parameters(policy)
        setattr(policy, "plato_policy_path", policy_path)
        setattr(policy, "plato_finetune_mode", finetune_mode)
        setattr(policy, "plato_adapter_patterns", tuple(adapter_patterns))
        setattr(policy, "plato_adapter_fallback_mode", trainable_metadata["fallback_mode"])
        setattr(policy, "plato_trainable_parameter_count", trainable_count)
        setattr(
            policy,
            "plato_trainable_parameter_names",
            tuple(trainable_metadata["trainable_names"]),
        )

        return policy
