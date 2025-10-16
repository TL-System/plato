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
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from plato.config import Config

try:  # Optional dependency – only required when S3/ZooKeeper mode is enabled.
    from kazoo.client import KazooClient
    from kazoo.recipe.lock import Lock
except ImportError:  # pragma: no cover - fallback when kazoo is unavailable.
    KazooClient = None  # type: ignore
    Lock = None  # type: ignore

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
    selected_clients: List[int] = field(default_factory=list)
    client_samples: Dict[int, Optional[int]] = field(default_factory=dict)
    additive_shares: Dict[int, Optional[Dict[str, Any]]] = field(default_factory=dict)
    pairwise_shares: Dict[Tuple[int, int], Optional[Dict[str, Any]]] = field(
        default_factory=dict
    )

    def initialise_clients(self, clients: Iterable[int]) -> None:
        """Ensure bookkeeping dictionaries contain all selected clients."""
        clients_list = list(clients)
        self.selected_clients = clients_list
        for client_id in clients_list:
            self.client_samples.setdefault(client_id, None)
            self.additive_shares.setdefault(client_id, None)
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

    def __init__(
        self,
        *,
        lock=None,
        storage_dir: str = "mpc_data",
        use_s3: bool = False,
        s3_key_prefix: str = "mpc",
    ):
        self._lock = lock
        self._storage_dir = storage_dir
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
    def from_config(cls, lock=None) -> "RoundInfoStore":
        """
        Factory building a store from the active configuration.

        Automatically enables S3/ZooKeeper mode when ``server.s3_endpoint_url`` is set.
        """
        use_s3 = hasattr(Config().server, "s3_endpoint_url")
        storage_dir = getattr(Config().params, "mpc_data_path", "mpc_data")
        return cls(lock=lock, storage_dir=storage_dir, use_s3=use_s3)

    @contextlib.contextmanager
    def _acquire(self):
        """Acquire the appropriate lock for the configured backend."""
        if self._use_s3:
            assert self._zk_client is not None and Lock is not None
            self._zk_client.start()
            self._zk_lock = Lock(self._zk_client, self.ZK_LOCK_PATH)
            self._zk_lock.acquire()
            logger.debug("Acquired ZooKeeper lock for MPC round store.")
            try:
                yield
            finally:
                assert self._zk_lock is not None
                self._zk_lock.release()
                logger.debug("Released ZooKeeper lock for MPC round store.")
                self._zk_client.stop()
        else:
            if self._lock is not None:
                self._lock.acquire()
            try:
                yield
            finally:
                if self._lock is not None:
                    self._lock.release()

    def _load_state(self) -> Optional[RoundInfoState]:
        """Load the current round state from the configured backend."""
        if self._use_s3:
            assert self._s3_client is not None
            try:
                raw = self._s3_client.receive_from_s3(
                    f"{self._s3_key_prefix}/{self.ROUND_INFO_FILENAME}"
                )
            except Exception:  # pragma: no cover - dependent on S3 behaviour.
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
        with open(path, "wb") as round_file:
            pickle.dump(state, round_file)

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

    def record_client_samples(self, client_id: int, num_samples: int) -> RoundInfoState:
        """Update the stored sample count for a given client."""
        with self._acquire():
            state = self._ensure_state()
            state.client_samples[client_id] = num_samples
            self._save_state(state)
            return state

    def append_additive_share(
        self, target_client: int, share_payload: Dict[str, Any]
    ) -> RoundInfoState:
        """Accumulate additive-share payloads destined for ``target_client``."""
        with self._acquire():
            state = self._ensure_state()
            existing = state.additive_shares.get(target_client)
            if existing is None:
                state.additive_shares[target_client] = share_payload
            else:
                for key, value in share_payload.items():
                    existing[key] += value
            self._save_state(state)
            return state

    def store_pairwise_share(
        self, target_client: int, from_client: int, share_payload: Dict[str, Any]
    ) -> RoundInfoState:
        """Store a pairwise share (used by Shamir secret sharing)."""
        with self._acquire():
            state = self._ensure_state()
            state.pairwise_shares[(target_client, from_client)] = share_payload
            self._save_state(state)
            return state

    def load_state(self) -> RoundInfoState:
        """Retrieve the current round state. Raises if unset."""
        with self._acquire():
            return self._ensure_state()

    def _ensure_state(self) -> RoundInfoState:
        state = self._load_state()
        if state is None:
            raise RuntimeError("Round information has not been initialised yet.")
        return state
