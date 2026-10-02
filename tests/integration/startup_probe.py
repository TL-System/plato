"""Importable, main-guarded probes for the actual synchronous entrypoints."""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import signal
import sys
from pathlib import Path

ROOT = Path(os.environ["PLATO_STARTUP_ROOT"])
CASE = os.environ["PLATO_STARTUP_CASE"]
OPTIONS = json.loads(os.environ["PLATO_STARTUP_OPTIONS"])
LOOPS = []


def emit(event: str, **fields):
    """Keep observations even when the production server calls os._exit."""
    data = {"event": event, "pid": os.getpid(), **fields}
    descriptor = os.open(
        ROOT / f"events-{os.getpid()}.jsonl",
        os.O_CREAT | os.O_APPEND | os.O_WRONLY,
        0o600,
    )
    try:
        os.write(descriptor, (json.dumps(data) + "\n").encode())
    finally:
        os.close(descriptor)
    print(json.dumps(data), flush=True)


class SentinelError(Exception):
    """An unmistakable entrypoint failure."""


async def pending_work():
    try:
        await asyncio.Future()
    finally:
        emit("task_finalizer_entered")
        if OPTIONS.get("stall") == "task":
            await asyncio.Future()
        await asyncio.sleep(0.02)
        emit("task_finalized")


async def resource_generator():
    try:
        yield 1
    finally:
        emit("generator_finalizer_entered")
        await asyncio.sleep(0.02)
        emit("generator_finalized")


def client_lifecycle():
    from plato import client as entry

    context = contextvars.ContextVar("configured", default="unconfigured")
    borrowed = OPTIONS.get("borrowed", False)
    if borrowed:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        LOOPS.append(loop)
        gate = loop.create_future()
        caller_future = loop.create_future()
        caller_task = loop.create_task(asyncio.sleep(3600))

    class Client:
        def configure(self):
            self.loop = asyncio.get_event_loop()
            LOOPS.append(self.loop)
            self.gate = gate if borrowed else self.loop.create_future()
            self.loop.call_soon(self.gate.set_result, "configured-future")
            context.set("configure-context")
            self.background = self.loop.create_task(pending_work())
            emit("client_configure", loop=id(self.loop))
            if OPTIONS.get("cleanup_fail"):

                async def shutdown_failure():
                    raise SentinelError("runner-cleanup-sentinel")

                self.loop.shutdown_asyncgens = shutdown_failure
            if OPTIONS.get("configure_fail"):
                raise SentinelError("configure-sentinel")

        async def generator(self):
            try:
                yield 1
            finally:
                emit("generator_finalizer_entered")
                if OPTIONS.get("stall") == "generator":
                    await asyncio.Future()
                await asyncio.sleep(0.02)
                emit("generator_finalized")

        async def start_client(self):
            assert asyncio.get_running_loop() is self.loop
            assert await self.gate == "configured-future"
            assert context.get() == "configure-context"
            emit("client_body", loop=id(self.loop), context=context.get())
            self.stream = self.generator()
            await anext(self.stream)

            def job():
                import time

                emit("executor_entered")
                if OPTIONS.get("stall") == "executor":
                    while True:
                        time.sleep(1)
                time.sleep(0.02)
                emit("executor_finished")

            self.loop.run_in_executor(None, job)
            await asyncio.sleep(0)
            if OPTIONS.get("control") == "interrupt":
                raise KeyboardInterrupt()
            if OPTIONS.get("control") == "exit":
                raise SystemExit(7)
            if OPTIONS.get("fail") or OPTIONS.get("stall"):
                emit("original_failure", message="client-sentinel")
                raise SentinelError("client-sentinel")

    client = Client()
    primary = None
    try:
        for _ in range(2 if OPTIONS.get("repeat") else 1):
            entry.run(1, None, client=client)
    except SentinelError as exc:
        primary = exc
    if borrowed:
        assert asyncio.get_event_loop() is loop
        assert not loop.is_closed()
        assert not caller_task.cancelled() and not caller_task.done()
        assert not caller_future.done()
        loop.call_soon(caller_future.set_result, "caller-owned")
        assert loop.run_until_complete(caller_future) == "caller-owned"
        assert loop.run_until_complete(loop.run_in_executor(None, lambda: 42)) == 42
        emit("borrowed_preserved")
        caller_task.cancel()
        client.background.cancel()
        loop.run_until_complete(
            asyncio.gather(caller_task, client.background, return_exceptions=True)
        )
        loop.run_until_complete(client.stream.aclose())
    if primary:
        raise primary


class ProbeClient:
    def __init__(self, server=None):
        self.server = server
        self.client_id = 0

    def configure(self):
        emit("client_configure")
        self.loop = asyncio.get_event_loop()
        self.future = self.loop.create_future()
        self.loop.call_soon(self.future.set_result, "configure-queued")

    async def start_client(self):
        loop = asyncio.get_running_loop()
        assert loop is self.loop
        assert await self.future == "configure-queued"
        LOOPS.append(loop)
        emit("client_body", loop=id(loop), client_id=self.client_id)
        if OPTIONS.get("secondary") and CASE.startswith("edge"):
            try:
                await asyncio.Future()
            finally:
                emit("secondary_failure", message="client-finalizer")
                raise SentinelError("client-finalizer")
        await asyncio.sleep(0)
        if OPTIONS.get("fail"):
            raise SentinelError("client-sentinel")


def bootstrap():
    from plato import client as entry
    from plato.config import Config
    from plato.servers import base

    class ProbeServer(base.Server):
        def __init__(self, trainer=None):
            self.disable_clients = True
            self.periodic_interval = 0.01
            self.ping_interval = 20
            self.ping_timeout = 20
            if OPTIONS.get("borrowed"):
                self.captured_loop = asyncio.get_event_loop()
                self.gate = self.captured_loop.create_future()
            emit("server_construct", trainer=trainer)

        def configure(self):
            emit("server_configure")
            loop = asyncio.get_event_loop()
            loop.call_soon(emit, "configure_queued_work")
            if (
                OPTIONS.get("real")
                or OPTIONS.get("borrowed")
                or OPTIONS.get("fallback_resources")
            ):
                self.loop = asyncio.get_event_loop()
                LOOPS.append(self.loop)
                if OPTIONS.get("borrowed"):
                    assert self.loop is self.captured_loop
                    self.loop.call_soon(self.gate.set_result, "server-future")
                if OPTIONS.get("secondary"):
                    self.loop.create_task(pending_work())
                if OPTIONS.get("fallback_resources"):
                    self.loop.create_task(pending_work())
                    self.stream = resource_generator()

                    async def open_stream():
                        await anext(self.stream)

                    self.loop.create_task(open_stream())

                    def executor_job():
                        import time

                        emit("executor_entered")
                        time.sleep(0.05)
                        emit("executor_finished")

                    self.loop.run_in_executor(None, executor_job)

        async def _periodic(self, interval):
            loop = asyncio.get_running_loop()
            LOOPS.append(loop)
            emit("periodic_body", loop=id(loop))
            if OPTIONS.get("self_cancel"):
                task = asyncio.current_task()
                loop.call_later(0.001, task.cancel)
                try:
                    await asyncio.Future()
                finally:
                    emit("task_finalizer_entered")
                    await asyncio.sleep(0.04)
                    emit("task_finalized")
            if OPTIONS.get("secondary"):
                try:
                    await asyncio.Future()
                finally:
                    emit("secondary_failure", message="periodic-finalizer")
                    raise SentinelError("periodic-finalizer")
            await asyncio.sleep(0)
            if OPTIONS.get("fail"):
                raise SentinelError("periodic-sentinel")

        def start(self, port=None):
            emit("server_start", port=port)
            if CASE == "direct" or OPTIONS.get("real"):
                super().start(port=Config().server.port if OPTIONS.get("bind") else 0)
            else:
                loop = asyncio.get_event_loop()
                LOOPS.append(loop)
                emit("server_work", loop=id(loop))
                try:
                    if OPTIONS.get("borrowed"):
                        assert loop.run_until_complete(self.gate) == "server-future"
                    loop.run_until_complete(asyncio.sleep(0.02))
                finally:
                    if OPTIONS.get("same_message_error"):
                        raise RuntimeError(
                            "Event loop stopped before Future completed."
                        )
                    if OPTIONS.get("start_fail"):
                        raise SentinelError("start-sentinel")

    class BorrowedServer(ProbeServer):
        def start(self):
            # The actual run entrypoint must retain no-argument dispatch.
            super().start()

    if OPTIONS.get("borrowed"):
        borrowed_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(borrowed_loop)
        LOOPS.append(borrowed_loop)
        caller_task = borrowed_loop.create_task(asyncio.sleep(3600))

    if CASE == "direct" or OPTIONS.get("real"):
        from aiohttp import web

        application = web.Application

        def app_factory():
            app = application()

            async def startup(app):
                loop = asyncio.get_running_loop()
                LOOPS.append(loop)
                emit("server_body", loop=id(loop))
                await asyncio.sleep(0.03)
                if OPTIONS.get("startup_fail"):
                    emit("original_failure", message="setup-sentinel")
                    raise SentinelError("setup-sentinel")
                if not OPTIONS.get("fail") and not OPTIONS.get("bind"):
                    loop.call_later(0.05, os.kill, os.getpid(), signal.SIGTERM)

            app.on_startup.append(startup)
            return app

        web.Application = app_factory
    blocker = None
    if OPTIONS.get("bind"):
        import socket

        blocker = socket.socket()
        blocker.bind(("127.0.0.1", Config().server.port))
        blocker.listen()

    if CASE in ("client_default", "client_custom"):
        entry.client_registry.get = lambda **kwargs: ProbeClient()
        entry.run(1, None, client=ProbeClient() if CASE.endswith("custom") else None)
    elif CASE in ("ordinary", "central"):
        base.Server._start_clients = staticmethod(
            lambda **kwargs: emit("launch", as_server=kwargs.get("as_server", False))
        )
        server = BorrowedServer() if OPTIONS.get("borrowed") else ProbeServer()
        try:
            server.run()
        finally:
            if OPTIONS.get("borrowed"):
                if OPTIONS.get("real"):
                    assert borrowed_loop.is_closed()
                    emit("borrowed_consumed")
                else:
                    assert not borrowed_loop.is_closed()
                    assert asyncio.get_event_loop() is borrowed_loop
                    assert not caller_task.done()
                    emit("borrowed_preserved")
                    caller_task.cancel()
                    borrowed_loop.run_until_complete(
                        asyncio.gather(caller_task, return_exceptions=True)
                    )
    elif CASE == "direct":
        ProbeServer().start()
    elif CASE in ("edge_default", "edge_custom"):
        if CASE.endswith("default"):
            from plato.clients import edge
            from plato.servers import fedavg_cs

            edge.Client = ProbeClient
            fedavg_cs.Server = ProbeServer
            entry.run(3, Config().server.port)
        else:
            entry.run(
                3,
                Config().server.port,
                edge_server=ProbeServer,
                edge_client=None if OPTIONS.get("missing_client") else ProbeClient,
                trainer=lambda: "trainer-factory",
            )
    else:
        raise ValueError(CASE)
    if blocker:
        blocker.close()


def running_loop_rejection():
    from plato import client as entry
    from plato.servers import base

    async def driver():
        def unexpected_factory(**kwargs):
            emit("unexpected_resource")
            return ProbeClient()

        entry.client_registry.get = unexpected_factory
        server = base.Server.__new__(base.Server)
        server.configure = lambda: emit("unexpected_resource")
        for name, call in (
            ("client", lambda: entry.run(1, None)),
            ("server_run", server.run),
            ("server_start", server.start),
        ):
            try:
                call()
            except RuntimeError as error:
                assert str(error) == "Plato startup requires a non-running event loop."
                emit("running_rejected", entrypoint=name)
            else:
                raise AssertionError(f"{name} allowed a running loop")
        assert "sio" not in vars(server)

    asyncio.run(driver())


def main():
    emit(
        "fresh_process",
        version=sys.version,
        precreated_loop=OPTIONS.get("borrowed", False),
    )
    if OPTIONS.get("absent"):
        asyncio.set_event_loop(None)
    if OPTIONS.get("closed"):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.close()
    try:
        if CASE.startswith("socket"):
            from tests.integration.startup_socket import real_round

            real_round()
        elif CASE == "client_lifecycle":
            client_lifecycle()
        elif CASE == "running":
            running_loop_rejection()
        else:
            bootstrap()
        emit("returned")
    except BaseException as exc:
        emit(
            "caller_error",
            type=type(exc).__name__,
            message=str(exc),
            notes=getattr(exc, "__notes__", []),
        )
        raise
    finally:
        for loop in set(LOOPS):
            emit("loop_after", closed=loop.is_closed())
            # Fresh 3.13's policy loop is conservatively borrowed by the runtime.
            # Release it only after recording ownership, in the test subprocess.
            if not loop.is_closed():
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()
        asyncio.set_event_loop(None)


if __name__ == "__main__":
    main()
