"""The default inbound client pipeline bounds incomplete/malformed transfers."""

import asyncio
import pickle
from types import SimpleNamespace

import pytest

from plato.clients.strategies.base import ClientContext
from plato.clients.strategies.defaults import DefaultPayloadStrategy
from plato.config import Config


@pytest.mark.parametrize(
    "kind", ["valid", "oversize", "malformed", "incomplete", "wrong_id", "timeout"]
)
def test_default_client_payload_cleanup_and_retry(temp_config, kind):
    Config().server.max_chunk_bytes = 64
    Config().server.payload_timeout = 0.03

    async def scenario():
        context = ClientContext()
        context.client_id = 7
        context.comm_simulation = False
        context.owner = SimpleNamespace(chunks=[], server_payload=None)
        strategy = DefaultPayloadStrategy()
        strategy.reset_payload(context)
        if kind == "valid":
            await strategy.accumulate_chunk(context, pickle.dumps({"w": 1}))
            await strategy.commit_chunk_group(context, 7)
            assert await strategy.finalise_inbound_payload(context, 7) == {"w": 1}
        elif kind == "timeout":
            await strategy.accumulate_chunk(context, b"x")
            await asyncio.sleep(0.06)
            assert context.server_payload is None
            assert context.owner.chunks == []
        else:
            with pytest.raises((ValueError, pickle.UnpicklingError)):
                if kind == "oversize":
                    await strategy.accumulate_chunk(context, b"x" * 65)
                elif kind == "malformed":
                    await strategy.accumulate_chunk(context, b"broken")
                    await strategy.commit_chunk_group(context, 7)
                else:
                    await strategy.accumulate_chunk(context, pickle.dumps({"w": 1}))
                    await strategy.finalise_inbound_payload(
                        context, 2 if kind == "wrong_id" else 7
                    )
            assert context.chunks == []
            assert context.server_payload is None
        # A failed transfer must leave the next valid round usable.
        strategy.reset_payload(context)
        await strategy.accumulate_chunk(context, pickle.dumps({"w": 2}))
        await strategy.commit_chunk_group(context, 7)
        assert await strategy.finalise_inbound_payload(context, 7) == {"w": 2}
        assert context.chunks == []

    asyncio.run(scenario())
