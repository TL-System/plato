"""
Client strategies for the composable client architecture.

This package exposes the shared context, base strategy interfaces, and the
default implementations that mirror the behaviour of the legacy client stack.
"""

from plato.clients.strategies.base import (
    ClientContext,
    ClientStrategy,
    CommunicationStrategy,
    LifecycleStrategy,
    PayloadStrategy,
    ReportingStrategy,
    TrainingStrategy,
)
from plato.clients.strategies.defaults import (
    DefaultCommunicationStrategy,
    DefaultLifecycleStrategy,
    DefaultPayloadStrategy,
    DefaultReportingStrategy,
    DefaultTrainingStrategy,
)
from plato.clients.strategies.edge import EdgeLifecycleStrategy, EdgeTrainingStrategy
from plato.clients.strategies.fedavg_personalized import (
    FedAvgPersonalizedPayloadStrategy,
)
from plato.clients.strategies.split_learning import SplitLearningTrainingStrategy

__all__ = [
    "ClientContext",
    "ClientStrategy",
    "LifecycleStrategy",
    "PayloadStrategy",
    "TrainingStrategy",
    "ReportingStrategy",
    "CommunicationStrategy",
    "FedAvgPersonalizedPayloadStrategy",
    "DefaultLifecycleStrategy",
    "DefaultPayloadStrategy",
    "DefaultTrainingStrategy",
    "DefaultReportingStrategy",
    "DefaultCommunicationStrategy",
    "EdgeLifecycleStrategy",
    "EdgeTrainingStrategy",
    "SplitLearningTrainingStrategy",
]
