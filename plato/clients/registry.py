"""
Client registry for instantiating configured clients.

The registry coordinates known client implementations and offers a best-effort
compatibility shim for externally supplied subclasses. Each instantiated client
is expected to build its composable strategy stack during initialisation; if a
custom subclass does not, we fall back to the legacy strategy adapters to keep
behaviour consistent with the pre-refactor architecture.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any, Callable, Dict, Optional, Type

from plato.clients import (
    edge,
    fedavg_personalized,
    mpc,
    self_supervised_learning,
    simple,
    split_learning,
)
from plato.clients.base import Client
from plato.clients.strategies.legacy import (
    LegacyCommunicationStrategy,
    LegacyLifecycleStrategy,
    LegacyPayloadStrategy,
    LegacyReportingStrategy,
    LegacyTrainingStrategy,
)
from plato.config import Config

ClientFactory = Callable[..., Client]


def _instantiate_with_signature(cls: Type[Client], **kwargs) -> Client:
    """Instantiate a client class using only parameters supported by its signature."""
    signature = inspect.signature(cls.__init__)
    supported_kwargs = {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return cls(**supported_kwargs)


def _ensure_strategy_configuration(instance: Client) -> Client:
    """Ensure the client has an active composable strategy stack."""
    if getattr(instance, "_composable_configured", False):
        return instance

    instance._configure_composable(  # pylint: disable=protected-access
        lifecycle_strategy=LegacyLifecycleStrategy(instance),
        payload_strategy=LegacyPayloadStrategy(instance),
        training_strategy=LegacyTrainingStrategy(instance),
        reporting_strategy=LegacyReportingStrategy(instance),
        communication_strategy=LegacyCommunicationStrategy(instance),
    )

    return instance


def _simple_like_factory(cls: Type[Client]) -> ClientFactory:
    """Factory wrapper for clients following the simple client signature."""

    def factory(
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        trainer_callbacks=None,
        **kwargs,
    ) -> Client:
        instance = _instantiate_with_signature(
            cls,
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            trainer_callbacks=trainer_callbacks,
            **kwargs,
        )
        return _ensure_strategy_configuration(instance)

    return factory


def _edge_factory() -> ClientFactory:
    """Factory for edge clients requiring a server instance."""

    def factory(
        server: Optional[Any] = None,
        **kwargs,
    ) -> Client:
        if server is None:
            raise ValueError("Edge client instantiation requires a `server` argument.")

        instance = _instantiate_with_signature(edge.Client, server=server, **kwargs)
        return _ensure_strategy_configuration(instance)

    return factory


def _legacy_factory(cls: Type[Client]) -> ClientFactory:
    """Factory for legacy-style clients that retain bespoke lifecycle overrides."""

    def factory(**kwargs) -> Client:
        instance = _instantiate_with_signature(cls, **kwargs)
        return _ensure_strategy_configuration(instance)

    return factory


registered_clients: Dict[str, ClientFactory] = {
    "simple": _simple_like_factory(simple.Client),
    "fedavg_personalized": _simple_like_factory(fedavg_personalized.Client),
    "mpc": _simple_like_factory(mpc.Client),
    "self_supervised_learning": _simple_like_factory(self_supervised_learning.Client),
    "split_learning": _simple_like_factory(split_learning.Client),
    "edge": _edge_factory(),
}


def _resolve_external_class(path: str) -> Type[Client]:
    """Resolve a dotted-path client class for custom configurations."""
    module_path, _, class_name = path.rpartition(".")
    if not module_path:
        raise ValueError(
            "Custom client types must be provided as a fully qualified class path."
        )

    module = importlib.import_module(module_path)
    client_cls = getattr(module, class_name)
    if not inspect.isclass(client_cls) or not issubclass(client_cls, Client):
        raise ValueError(f"{path} is not a valid Client subclass.")

    return client_cls


def get(
    model=None,
    datasource=None,
    algorithm=None,
    trainer=None,
    callbacks=None,
    trainer_callbacks=None,
    **kwargs,
) -> Client:
    """Instantiate a client configured by `Config().clients.type`."""
    client_type = getattr(Config().clients, "type", None)
    if client_type is None:
        client_type = getattr(Config().algorithm, "type", "simple")

    factory = registered_clients.get(client_type)

    if factory is None:
        client_cls = _resolve_external_class(client_type)
        supports_trainer_callbacks = (
            "trainer_callbacks" in inspect.signature(client_cls.__init__).parameters
        )
        factory = (
            _simple_like_factory(client_cls)
            if supports_trainer_callbacks
            else _legacy_factory(client_cls)
        )
        registered_clients[client_type] = factory

    logging.info("Client: %s", client_type)

    return factory(
        model=model,
        datasource=datasource,
        algorithm=algorithm,
        trainer=trainer,
        callbacks=callbacks,
        trainer_callbacks=trainer_callbacks,
        **kwargs,
    )
