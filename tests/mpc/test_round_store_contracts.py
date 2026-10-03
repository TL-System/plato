"""MPC storage errors and distributed-lock cleanup through public store calls."""

from types import SimpleNamespace

import pytest

from plato.config import Config
from plato.mpc import RoundInfoStore
from plato.mpc import round_store as module
from plato.utils import s3


@pytest.fixture
def lock_timeout_type(request):
    """Use the real Kazoo exception for mandatory library qualification."""
    if request.config.getoption("test_profile") == "mandatory":
        from kazoo.exceptions import LockTimeout

        return LockTimeout
    return TimeoutError


@pytest.fixture
def remote_store(temp_config, monkeypatch):
    monkeypatch.setattr(
        Config, "server", SimpleNamespace(zk_address="local", zk_port=1)
    )
    state = SimpleNamespace(stopped=False, released=False, saved=None)

    class Client:
        def __init__(self, **_kwargs):
            pass

        def start(self, timeout=15):
            state.start_timeout = timeout

        def stop(self):
            state.stopped = True

    class ObjectStorage:
        def put_to_s3(self, _key, payload):
            state.saved = payload

        def receive_from_s3(self, _key):
            raise ValueError("S3 service unavailable")

    class AcquiredLock:
        def __init__(self, *_args):
            pass

        def acquire(self, **_kwargs):
            return True

        def release(self):
            state.released = True

    monkeypatch.setattr(module, "KazooClient", Client)
    monkeypatch.setattr(module, "Lock", AcquiredLock)
    monkeypatch.setattr(s3, "S3", ObjectStorage)
    return RoundInfoStore(use_s3=True), state


def test_lock_failure_stops_client_and_never_writes(
    remote_store, monkeypatch, lock_timeout_type
):
    store, state = remote_store

    class FailedLock:
        def __init__(self, *_args):
            pass

        def acquire(self, timeout=None):
            raise lock_timeout_type("Service lock stalled")

        def release(self):
            state.released = True

    monkeypatch.setattr(module, "Lock", FailedLock)
    with pytest.raises(lock_timeout_type, match="stalled"):
        store.initialise_round(1, [1])
    assert state.stopped
    assert not state.released
    assert state.saved is None


def test_finite_lock_wait_prevents_unlocked_write(remote_store, monkeypatch):
    store, state = remote_store

    class UnavailableLock:
        def __init__(self, *_args):
            pass

        def acquire(self, timeout=None):
            # Kazoo's timeout-aware contract: unsuccessful acquisition is false.
            state.lock_timeout = timeout
            return False

        def release(self):
            state.released = True

    monkeypatch.setattr(module, "Lock", UnavailableLock)
    with pytest.raises(TimeoutError, match="lock"):
        store.initialise_round(1, [1])
    assert 0 < state.lock_timeout < 60
    assert state.stopped
    assert not state.released
    assert state.saved is None


def test_s3_read_error_is_not_reported_as_an_uninitialized_round(remote_store):
    store, state = remote_store

    with pytest.raises(ValueError, match="service unavailable"):
        store.load_state()
    assert state.stopped
    assert state.released
