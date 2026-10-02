"""
Starting point for a Plato federated learning client.
"""

import asyncio
import logging
import os
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import copy_context
from typing import Any, cast

from plato.clients import registry as client_registry
from plato.config import Config


def _current_startup_loop() -> asyncio.AbstractEventLoop | None:
    """Reuse a usable synchronous caller loop, rejecting nested entrypoints."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Plato startup requires a non-running event loop.")
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        return None
    return None if loop.is_closed() else loop


def _secondary_error(
    error: BaseException, primary: BaseException | None, *, task_name: str | None = None
) -> None:
    """Retain cleanup diagnostics without replacing the original failure."""
    logging.error(
        "Plato startup failure (%s): %s",
        task_name or "loop",
        error,
        exc_info=(type(error), error, error.__traceback__),
    )
    if primary is not None:
        primary.add_note(
            f"Plato startup also failed ({task_name or 'loop'}): {error!r}"
        )


def _close_startup_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Finalize a loop we created when an external start did not consume it."""
    errors = []
    try:
        tasks = asyncio.all_tasks(loop)
        for task in tasks:
            if not task.cancelling():
                task.cancel()
        if tasks:
            results = loop.run_until_complete(
                asyncio.gather(*tasks, return_exceptions=True)
            )
            errors.extend(
                result
                for result in results
                if isinstance(result, BaseException)
                and not isinstance(result, asyncio.CancelledError)
            )
        # These stages rely on cooperative extension code. The test watchdog
        # owns the hard process deadline; the executor timeout only bounds wait.
        for shutdown in (
            loop.shutdown_asyncgens,
            lambda: cast(asyncio.BaseEventLoop, loop).shutdown_default_executor(
                timeout=5
            ),
        ):
            if not loop.is_closed():
                try:
                    loop.run_until_complete(shutdown())
                except BaseException as error:
                    errors.append(error)
    finally:
        if not loop.is_closed():
            loop.close()
    if errors:
        for error in errors[1:]:
            _secondary_error(error, errors[0])
        raise errors[0]


@contextmanager
def _startup_loop(loop: asyncio.AbstractEventLoop | None, *, client_runner=False):
    """Install a missing loop; ordinary clients and aiohttp have distinct owners."""
    borrowed = loop is not None
    runner = None
    if not borrowed:
        if client_runner:
            runner = asyncio.Runner()
            loop = runner.get_loop()
        else:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    primary = None
    try:
        yield loop, runner
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            if runner is not None:
                runner.close()
            elif not borrowed and not loop.is_closed():
                _close_startup_loop(loop)
        except BaseException as error:
            _secondary_error(error, primary)
            if primary is None:
                raise
        finally:
            asyncio.set_event_loop(loop if borrowed and not loop.is_closed() else None)


@contextmanager
def _startup_task(loop, coroutine, *, stop_on_failure=True):
    """Retain only the periodic/edge task, or a borrowed client's main task."""
    task = loop.create_task(coroutine)
    active_error = None
    cleanup_errors = []
    armed = True
    observed = False
    stopped = False

    def completed(done):
        nonlocal active_error, observed, stopped
        if observed or done.cancelled():
            return
        observed = True
        error = done.exception()
        if error is None:
            return
        # aiohttp cancels tasks before returning from run_app. A finalizer may
        # replace CancelledError, so cancelled() alone cannot detect teardown.
        if not armed or done.cancelling():
            cleanup_errors.append(error)
        else:
            active_error = error
            if stop_on_failure and loop.is_running():
                stopped = True
                loop.stop()

    task.add_done_callback(completed)
    primary = None
    try:
        yield task
    except BaseException as error:
        primary = error
    finally:
        if task.done():
            completed(task)
        armed = False
        if not loop.is_closed() and not task.done():
            if not task.cancelling():
                task.cancel()
            try:
                loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
            except BaseException as error:
                cleanup_errors.append(error)
        if task.done():
            completed(task)

    # A synchronous override can independently raise the same RuntimeError text.
    # Attribute replacement to asyncio's actual stop check, not just its message.
    traceback = primary.__traceback__ if primary is not None else None
    while traceback is not None and traceback.tb_next is not None:
        traceback = traceback.tb_next
    stopped_in_run_until_complete = (
        traceback is not None
        and traceback.tb_frame.f_code
        is asyncio.BaseEventLoop.run_until_complete.__code__
    )
    if (
        stopped
        and stopped_in_run_until_complete
        and isinstance(primary, RuntimeError)
        and str(primary) == "Event loop stopped before Future completed."
    ):
        primary = active_error
    selected = primary if primary is not None else active_error
    if primary is not None and active_error is not None and primary is not active_error:
        _secondary_error(active_error, primary, task_name=task.get_name())
    for error in cleanup_errors:
        _secondary_error(error, selected, task_name=task.get_name())
    if selected is not None:
        raise selected
    if cleanup_errors:
        raise cleanup_errors[0]


def run(
    client_id: int,
    port: int | None,
    client: Any = None,
    edge_server: Callable[..., Any] | None = None,
    edge_client: Callable[..., Any] | None = None,
    trainer: Callable[[], Any] | None = None,
    client_kwargs: dict[str, Any] | None = None,
) -> None:
    """Starting a client to connect to the server."""
    current_loop = _current_startup_loop()
    Config().args.id = client_id
    if port is not None:
        Config().args.port = port

    with _startup_loop(current_loop, client_runner=not Config().is_edge_server()) as (
        loop,
        runner,
    ):
        # If a server needs to be running concurrently
        if Config().is_edge_server():
            Config().trainer = Config().trainer._replace(
                rounds=Config().algorithm.local_rounds
            )

            if edge_server is None:
                from plato.clients import edge
                from plato.servers import fedavg_cs

                server = fedavg_cs.Server()
                client = edge.Client(server)
            else:
                # A customized edge server
                if trainer is not None:
                    server = edge_server(trainer=trainer())
                else:
                    server = edge_server()
                if edge_client is None:
                    raise ValueError(
                        "edge_client must be provided when edge_server is set."
                    )
                client = edge_client(server=server)

            server.configure()
            client.configure()

            logging.info("Starting an edge server as client #%d", Config().args.id)
            with _startup_task(loop, client.start_client()):
                logging.info(
                    "Starting an edge server as server #%d on port %d",
                    os.getpid(),
                    Config().args.port,
                )
                server.start(port=Config().args.port)

        else:
            if client is None:
                client_kwargs = client_kwargs or {}
                client = client_registry.get(**client_kwargs)

                logging.info(
                    "Starting a %s client #%d.", Config().clients.type, client_id
                )
            else:
                client.client_id = client_id

                # Keep the shared context aligned with the explicit client ID.
                if hasattr(client, "_sync_to_context"):
                    try:
                        client._sync_to_context(("client_id",))
                    except Exception:
                        if hasattr(client, "_context"):
                            client._context.client_id = client_id
                elif hasattr(client, "_context"):
                    client._context.client_id = client_id

                logging.info("Starting a custom client #%d.", client_id)

            client.configure()

            if runner is not None:
                runner.run(client.start_client(), context=copy_context())
            else:
                with _startup_task(
                    loop, client.start_client(), stop_on_failure=False
                ) as task:
                    loop.run_until_complete(task)


if __name__ == "__main__":
    run(Config().args.id, Config().args.port)
