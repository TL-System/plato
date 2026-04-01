"""Base abstractions for structured evaluator integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from plato.trainers.strategies.base import TrainingContext


@dataclass(slots=True)
class EvaluationInput:
    """Inputs provided to evaluator implementations."""

    model: Any
    trainer: Any | None = None
    tokenizer: Any | None = None
    context: TrainingContext | None = None
    config: dict[str, Any] | None = None
    testset: Any | None = None
    sampler: Any | None = None
    local_metric: float | None = None


@dataclass(slots=True)
class EvaluationResult:
    """Structured result returned by evaluator implementations."""

    evaluator: str
    primary_metric: str
    metrics: dict[str, float]
    higher_is_better: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.primary_metric not in self.metrics:
            raise ValueError(
                f"Primary metric '{self.primary_metric}' missing from evaluator metrics."
            )

    @property
    def primary_value(self) -> float:
        """Return the numeric value for the primary metric."""
        return self.metrics[self.primary_metric]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary payload."""
        payload = asdict(self)
        payload["primary_value"] = self.primary_value
        return payload


class Evaluator(ABC):
    """Interface for structured evaluation integrations."""

    def __init__(self, config: dict[str, Any] | Any):
        self.config = config

    @abstractmethod
    def evaluate(self, request: EvaluationInput) -> EvaluationResult:
        """Evaluate a model and return a structured result."""
