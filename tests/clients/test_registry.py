"""Tests for client registry instantiation behavior."""

from __future__ import annotations

from plato.clients import registry as clients_registry
from plato.clients import simple
from plato.config import Config


class ClientAcceptingKwargs(simple.Client):
    """Custom client used to verify registry kwargs propagation."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        trainer_callbacks=None,
        **kwargs,
    ):
        self.extra_kwargs = dict(kwargs)
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            trainer_callbacks=trainer_callbacks,
        )


def test_registry_preserves_kwargs_for_external_clients(temp_config):
    """External clients with ``**kwargs`` should receive custom registry kwargs."""
    client_type = f"{__name__}.ClientAcceptingKwargs"
    Config.clients.type = client_type
    clients_registry.registered_clients.pop(client_type, None)

    try:
        client = clients_registry.get(custom_value=42)
    finally:
        clients_registry.registered_clients.pop(client_type, None)

    assert isinstance(client, ClientAcceptingKwargs)
    assert client.extra_kwargs["custom_value"] == 42
