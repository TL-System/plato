"""Bounded runtime transfers for trusted peers and trusted S3 object writers.

These resource limits do not make Python pickle safe for untrusted input.
"""

from __future__ import annotations

import asyncio
import io
import math
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import aiohttp

from plato.config import Config


@dataclass(frozen=True)
class TransportLimits:
    """Limits configured in the server section, shared by both socket peers."""

    max_report_bytes: int = 1024**2
    max_chunk_bytes: int = 1024**2
    max_payload_bytes: int = 512 * 1024**2
    max_payload_chunks: int = 4096
    max_payload_parts: int = 1024
    max_buffered_bytes: int = 1024**3
    payload_timeout: float = 120.0

    @classmethod
    def from_config(cls) -> TransportLimits:
        defaults = cls()
        values = {}
        for name in cls.__dataclass_fields__:
            value = getattr(Config().server, name, getattr(defaults, name))
            if name == "payload_timeout":
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"server.{name} must be finite and positive.")
            elif not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"server.{name} must be a positive integer.")
            if value <= 0:
                raise ValueError(f"server.{name} must be positive.")
            values[name] = value
        return cls(**values)

    @property
    def max_message_bytes(self) -> int:
        """Allow Socket.IO framing/base64 overhead on one report or chunk."""
        return 2 * max(self.max_report_bytes, self.max_chunk_bytes) + 65536


def load_pickle(data: bytes) -> Any:
    """Decode one complete pickle from a trusted peer, rejecting trailing bytes."""
    stream = io.BytesIO(data)
    result = pickle.Unpickler(stream).load()
    if stream.read(1):
        raise ValueError("Trailing bytes after payload pickle.")
    return result


class InboundTransfer:
    """Count wire bytes across parts and expire incomplete transfers."""

    def __init__(self, limits: TransportLimits, expire: Callable[[], None]):
        self.limits = limits
        self.chunks: list[bytes] = []
        self.payload: Any = None
        self.byte_count = 0
        self.report_bytes = 0
        self.chunk_count = 0
        self.part_count = 0
        self.closed = False
        loop = asyncio.get_running_loop()
        self.deadline = loop.time() + limits.payload_timeout
        self.timer = loop.call_later(limits.payload_timeout, expire)

    @property
    def buffered_bytes(self) -> int:
        return self.byte_count + self.report_bytes

    def check_active(self) -> None:
        if self.closed or asyncio.get_running_loop().time() >= self.deadline:
            raise ValueError("Payload transfer expired or already completed.")

    def reserve(self, count: int) -> None:
        self.check_active()
        if self.byte_count + count > self.limits.max_payload_bytes:
            raise ValueError("Payload exceeds maximum byte limit.")
        self.byte_count += count

    def append(self, data: bytes) -> None:
        self.check_active()
        if not isinstance(data, bytes) or not data:
            raise ValueError("Payload chunk must be nonempty bytes.")
        if len(data) > self.limits.max_chunk_bytes:
            raise ValueError("Payload chunk exceeds maximum byte limit.")
        if self.chunk_count >= self.limits.max_payload_chunks:
            raise ValueError("Payload exceeds maximum chunk count.")
        self.reserve(len(data))
        self.chunk_count += 1
        self.chunks.append(data)

    def commit(self) -> Any:
        self.check_active()
        if not self.chunks:
            raise ValueError("Payload part has no chunks.")
        if self.part_count >= self.limits.max_payload_parts:
            raise ValueError("Payload exceeds maximum part count.")
        data = load_pickle(b"".join(self.chunks))
        self.chunks.clear()
        self.part_count += 1
        if self.payload is None:
            self.payload = data
        elif isinstance(self.payload, list):
            self.payload.append(data)
        else:
            self.payload = [self.payload, data]
        return self.payload

    def finish(self) -> Any:
        self.check_active()
        if self.chunks or self.payload is None:
            raise ValueError("Payload completion arrived with incomplete data.")
        return self.payload

    def close(self) -> None:
        self.closed = True
        self.timer.cancel()
        self.chunks.clear()
        self.payload = None


async def receive_s3_payload(
    storage: Any,
    key: str,
    transfer: InboundTransfer,
    reserve: Callable[[int], None],
) -> Any:
    """Read the existing presigned S3 object with a total deadline and byte cap.

    Keep S3's key-prefix and pickle format contract. The synchronous utility's
    unbounded receive_from_s3 is not used by runtime ingress.
    """
    transfer.check_active()
    object_key = storage.key_prefix + "/" + key
    url = storage.s3_client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": storage.bucket, "Key": object_key},
        ExpiresIn=300,
    )
    remaining = transfer.deadline - asyncio.get_running_loop().time()
    transfer.check_active()
    timeout = aiohttp.ClientTimeout(total=remaining)
    # Preserve exact presigned URL escaping; S3 stores uncompressed pickle bytes.
    from yarl import URL

    async with (
        asyncio.timeout_at(transfer.deadline),
        aiohttp.ClientSession(timeout=timeout, auto_decompress=False) as session,
    ):
        async with session.get(URL(url, encoded=True)) as response:
            if response.status != 200:
                raise ValueError(
                    f"S3 payload request failed: status {response.status}."
                )
            length = response.content_length
            if length is not None and (
                length > transfer.limits.max_payload_bytes - transfer.byte_count
            ):
                raise ValueError("S3 payload exceeds maximum byte limit.")
            buffer = bytearray()
            async for chunk in response.content.iter_chunked(64 * 1024):
                reserve(len(chunk))
                buffer.extend(chunk)
            transfer.check_active()
            return load_pickle(bytes(buffer))
