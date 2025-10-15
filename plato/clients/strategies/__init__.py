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
from plato.clients.strategies.mistnet import MistNetTrainingStrategy

__all__ = [
    "ClientContext",
    "ClientStrategy",
    "LifecycleStrategy",
    "PayloadStrategy",
    "TrainingStrategy",
    "ReportingStrategy",
    "CommunicationStrategy",
    "DefaultLifecycleStrategy",
    "DefaultPayloadStrategy",
    "DefaultTrainingStrategy",
    "DefaultReportingStrategy",
    "DefaultCommunicationStrategy",
    "EdgeLifecycleStrategy",
    "EdgeTrainingStrategy",
    "MistNetTrainingStrategy",
]
