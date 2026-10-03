"""
Shared state coordination for multiparty computation (MPC) rounds.

The original MPC prototype stored coordination metadata for each round in either
local files or an S3 bucket guarded by ZooKeeper. This module modernises that
behaviour behind a consistent interface so that both clients and servers create
and mutate shared state without duplicating locking logic.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import pickle
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from plato.config import Config

try:  # Optional dependency – only required when S3/ZooKeeper mode is enabled.
    from kazoo.client import KazooClient
    from kazoo.recipe.lock import Lock
except ImportError:  # pragma: no cover - fallback when kazoo is unavailable.
    KazooClient = None
    Lock = None

logger = logging.getLogger(__name__)


@dataclass
class RoundInfoState:
    """
    Serializable state stored for the current MPC round.

    Attributes:
        round_number: The communication round this state belongs to.
        selected_clients: Client IDs participating in the round.
        client_samples: Mapping client ID -> number of local samples.
        additive_shares: Mapping client ID -> accumulated additive shares destined
            for that client (corresponds to PR316's ``client_X_info["data"]``).
        pairwise_shares: Mapping (to_id, from_id) -> share payload used by the
            Shamir secret sharing flow.
    """

    round_number: int
    selected_clients: list[int] = field(default_factory=list)
    client_samples: dict[int, float | None] = field(default_factory=dict)
    additive_shares: dict[int, dict[str, Any] | None] = field(default_factory=dict)
    additive_contributors: dict[int, set[int]] = field(default_factory=dict)
    pairwise_shares: dict[tuple[int, int], dict[str, Any] | None] = field(
        default_factory=dict
    )

    def initialise_clients(self, clients: Iterable[int]) -> None:
        """Ensure bookkeeping dictionaries contain all selected clients."""
        clients_list = list(clients)
        if len(clients_list) != len(set(clients_list)):
            raise ValueError("MPC participants must be unique.")
        self.selected_clients = clients_list
        for client_id in clients_list:
            self.client_samples.setdefault(client_id, None)
            self.additive_shares.setdefault(client_id, None)
            self.additive_contributors.setdefault(client_id, set())
            for peer_id in clients_list:
                self.pairwise_shares.setdefault((client_id, peer_id), None)


class RoundInfoStore:
    """
    Persist and synchronise ``RoundInfoState`` between MPC participants.

    The store supports two storage backends:
        * Local filesystem (default) guarded by an optional multiprocessing lock.
        * S3 object storage guarded by a ZooKeeper distributed lock.
    """

    ROUND_INFO_FILENAME = "round_info"
    ZK_LOCK_PATH = "/plato/mpc/round_info_lock"
    ZK_START_TIMEOUT = 10.0
    ZK_LOCK_TIMEOUT = 30.0

    def __init__(
        self,
        *,
        lock=None,
        storage_dir: str | None = None,
        use_s3: bool = False,
        s3_key_prefix: str = "mpc",
    ):
        self._lock = lock
        if storage_dir is None:
            resolved_storage_dir = self._default_storage_dir()
        else:
            resolved_storage_dir = os.fspath(storage_dir)
        self._storage_dir = resolved_storage_dir
        self._use_s3 = use_s3
        self._s3_key_prefix = s3_key_prefix.rstrip("/")

        self._s3_client = None
        self._zk_client = None
        self._zk_lock = None

        if use_s3:
            from plato.utils import s3

            if KazooClient is None or Lock is None:
                raise RuntimeError(
                    "ZooKeeper/Kazoo is required for MPC S3 coordination but is not installed."
                )

            self._s3_client = s3.S3()
            self._zk_client = KazooClient(
                hosts=f"{Config().server.zk_address}:{Config().server.zk_port}"
            )

        # Ensure local path exists on demand.
        if not use_s3 and not os.path.isdir(self._storage_dir):
            os.makedirs(self._storage_dir, exist_ok=True)

    @classmethod
    def from_config(cls, lock=None) -> RoundInfoStore:
        """
        Factory building a store from the active configuration.

        Automatically enables S3/ZooKeeper mode when ``server.s3_endpoint_url`` is set.
        """
        use_s3 = hasattr(Config().server, "s3_endpoint_url")
        storage_dir = cls._default_storage_dir()
        return cls(lock=lock, storage_dir=storage_dir, use_s3=use_s3)

    @staticmethod
    def _default_storage_dir() -> str:
        """Resolve the storage directory from the current configuration."""
        params = getattr(Config, "params", {}) or {}
        if isinstance(params, dict):
            mpc_path = params.get("mpc_data_path")
            if mpc_path:
                return mpc_path

            base_path = params.get("base_path")
            if base_path:
                return os.path.join(base_path, "mpc_data")

        return os.path.join("./runtime", "mpc_data")

    @contextlib.contextmanager
    def _acquire(self):
        """Acquire the appropriate lock for the configured backend."""
        if self._use_s3:
            assert self._zk_client is not None and Lock is not None
            acquired = False
            try:
                self._zk_client.start(timeout=self.ZK_START_TIMEOUT)
                self._zk_lock = Lock(self._zk_client, self.ZK_LOCK_PATH)
                acquired = self._zk_lock.acquire(timeout=self.ZK_LOCK_TIMEOUT)
                if not acquired:
                    raise TimeoutError("Unable to acquire the MPC round store lock.")
                logger.debug("Acquired ZooKeeper lock for MPC round store.")
                yield
            finally:
                try:
                    if acquired:
                        assert self._zk_lock is not None
                        self._zk_lock.release()
                        logger.debug("Released ZooKeeper lock for MPC round store.")
                finally:
                    self._zk_client.stop()
        else:
            if self._lock is not None:
                self._lock.acquire()
            try:
                yield
            finally:
                if self._lock is not None:
                    self._lock.release()

    def _load_state(self) -> RoundInfoState | None:
        """Load the current round state from the configured backend."""
        if self._use_s3:
            assert self._s3_client is not None
            try:
                raw = self._s3_client.receive_from_s3(
                    f"{self._s3_key_prefix}/{self.ROUND_INFO_FILENAME}"
                )
            except FileNotFoundError:
                logger.debug("No existing round info found in S3.", exc_info=True)
                return None
            return raw

        path = os.path.join(self._storage_dir, self.ROUND_INFO_FILENAME)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as round_file:
            return pickle.load(round_file)

    def _save_state(self, state: RoundInfoState) -> None:
        """Persist the provided round state."""
        if self._use_s3:
            assert self._s3_client is not None
            self._s3_client.put_to_s3(
                f"{self._s3_key_prefix}/{self.ROUND_INFO_FILENAME}", state
            )
            return

        path = os.path.join(self._storage_dir, self.ROUND_INFO_FILENAME)
        # A failed write must leave the last complete state available to peers.
        fd, temporary = tempfile.mkstemp(dir=self._storage_dir, prefix=".round_info-")
        try:
            with os.fdopen(fd, "wb") as round_file:
                pickle.dump(state, round_file)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    @property
    def storage_dir(self) -> str:
        """Directory used for local storage (useful for debug artefacts)."""
        return self._storage_dir

    @property
    def uses_s3(self) -> bool:
        """Indicates whether the store persists via S3/ZooKeeper."""
        return self._use_s3

    def reset(self) -> None:
        """Remove any persisted round information."""
        with self._acquire():
            if self._use_s3:
                try:
                    assert self._s3_client is not None
                    self._s3_client.delete_from_s3(
                        f"{self._s3_key_prefix}/{self.ROUND_INFO_FILENAME}"
                    )
                except Exception:  # pragma: no cover - defensive cleanup for S3
                    logger.debug(
                        "Unable to delete MPC round info from S3.", exc_info=True
                    )
            else:
                path = os.path.join(self._storage_dir, self.ROUND_INFO_FILENAME)
                if os.path.exists(path):
                    os.remove(path)

    def initialise_round(
        self, round_number: int, selected_clients: Iterable[int]
    ) -> RoundInfoState:
        """Create and persist the bookkeeping structure for a new round."""
        with self._acquire():
            state = RoundInfoState(round_number=round_number)
            state.initialise_clients(selected_clients)
            self._save_state(state)
            return state

    def record_client_samples(
        self, client_id: int, num_samples: float, *, round_number: int | None = None
    ) -> RoundInfoState:
        """Update the stored sample count for a given client."""
        with self._acquire():
            state = self._ensure_state(round_number)
            self._check_client(state, client_id)
            if not math.isfinite(num_samples) or num_samples < 0:
                raise ValueError("MPC sample count must be finite and nonnegative.")
            state.client_samples[client_id] = num_samples
            self._save_state(state)
            return state

    def append_additive_share(
        self,
        target_client: int,
        share_payload: dict[str, Any],
        *,
        round_number: int | None = None,
        from_client: int | None = None,
    ) -> RoundInfoState:
        """Accumulate additive-share payloads destined for ``target_client``."""
        with self._acquire():
            state = self._ensure_state(round_number)
            self._check_client(state, target_client)
            contributors = state.additive_contributors.setdefault(target_client, set())
            if from_client is not None:
                self._check_client(state, from_client)
                if from_client == target_client or from_client in contributors:
                    raise ValueError("Duplicate or self-directed additive share.")
            existing = state.additive_shares.get(target_client)
            if existing is None:
                state.additive_shares[target_client] = share_payload
            else:
                if existing.keys() != share_payload.keys() or any(
                    existing[key].shape != value.shape
                    for key, value in share_payload.items()
                ):
                    raise ValueError("Mismatched additive share keys or shapes.")
                for key, value in share_payload.items():
                    existing[key] += value
            if from_client is not None:
                contributors.add(from_client)
            self._save_state(state)
            return state

    def store_pairwise_share(
        self,
        target_client: int,
        from_client: int,
        share_payload: dict[str, Any],
        *,
        round_number: int | None = None,
    ) -> RoundInfoState:
        """Store a pairwise share (used by Shamir secret sharing)."""
        with self._acquire():
            state = self._ensure_state(round_number)
            self._check_client(state, target_client)
            self._check_client(state, from_client)
            if state.pairwise_shares.get((target_client, from_client)) is not None:
                raise ValueError("Duplicate Shamir share.")
            state.pairwise_shares[(target_client, from_client)] = share_payload
            self._save_state(state)
            return state

    def load_state(self) -> RoundInfoState:
        """Retrieve the current round state. Raises if unset."""
        with self._acquire():
            return self._ensure_state()

    @staticmethod
    def _check_client(state: RoundInfoState, client_id: int) -> None:
        if client_id not in state.selected_clients:
            raise ValueError(f"Client {client_id} is not among the MPC participants.")

    def _ensure_state(self, round_number: int | None = None) -> RoundInfoState:
        state = self._load_state()
        if state is None:
            raise RuntimeError("Round information has not been initialised yet.")
        if round_number is not None and state.round_number != round_number:
            raise RuntimeError(
                f"MPC round changed from {round_number} to {state.round_number}."
            )
        return state
