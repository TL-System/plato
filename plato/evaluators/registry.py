"""Registry for structured evaluator implementations."""

from __future__ import annotations

from typing import Any, Callable

from plato.config import Config

EvaluatorFactory = Callable[[dict[str, Any] | Any], Any]

_registered_evaluators: dict[str, EvaluatorFactory] = {}


def register(name: str, factory: EvaluatorFactory) -> None:
    """Register an evaluator factory under a short name."""
    _registered_evaluators[name] = factory


def unregister(name: str) -> None:
    """Remove an evaluator factory when present."""
    _registered_evaluators.pop(name, None)


def registered_names() -> list[str]:
    """Return registered evaluator names sorted for stable diagnostics."""
    return sorted(_registered_evaluators)


def _config_to_dict(config: Any) -> dict[str, Any] | Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "_asdict"):
        return config._asdict()
    return config


def get(config: Any | None = None, *, allow_missing: bool = False):
    """Resolve the configured evaluator instance, or ``None`` when disabled."""
    resolved = config
    if resolved is None:
        resolved = getattr(Config(), "evaluation", None)

    if resolved is None:
        return None

    resolved = _config_to_dict(resolved)

    evaluator_type = None
    if isinstance(resolved, dict):
        evaluator_type = resolved.get("type")
    else:
        evaluator_type = getattr(resolved, "type", None)

    if not isinstance(evaluator_type, str) or not evaluator_type:
        raise ValueError("Evaluation config must define a non-empty 'type'.")

    if evaluator_type not in _registered_evaluators:
        if allow_missing:
            return None
        raise ValueError(
            f"No such evaluator: {evaluator_type}. Registered evaluators: {registered_names()}"
        )

    factory = _registered_evaluators[evaluator_type]
    return factory(resolved)
