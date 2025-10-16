"""
Multiparty computation outbound processor using additive secret sharing.
"""

from __future__ import annotations

import copy
import logging
import os
import random
from typing import Any, Dict, List

from plato.config import Config
from plato.mpc import RoundInfoStore
from plato.processors import model


class Processor(model.Processor):
    """Encrypts model updates using additive secret sharing."""

    def __init__(
        self,
        *,
        client_id: int,
        round_store: RoundInfoStore,
        debug_artifacts: bool | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.client_id = client_id
        self.round_store = round_store
        if debug_artifacts is None:
            debug_artifacts = getattr(Config().clients, "mpc_debug_artifacts", False)
        self.debug_artifacts = debug_artifacts

    @staticmethod
    def _split_tensor(tensor, num_shares: int) -> List[Any]:
        """Randomly decomposes ``tensor`` into ``num_shares`` additive shares."""
        if num_shares == 1:
            return [tensor]

        shares = [tensor / num_shares for _ in range(num_shares)]
        perturbations = [random.uniform(-0.5, 0.5) for _ in range(num_shares - 1)]
        perturbations.append(-sum(perturbations))
        for idx, perturb in enumerate(perturbations):
            shares[idx] += perturb
        return shares

    def _write_debug_artifact(
        self, round_number: int, label: str, payload: Dict[str, Any]
    ) -> None:
        if self.debug_artifacts and not self.round_store.uses_s3:
            path = os.path.join(
                self.round_store.storage_dir,
                f"{label}_round{round_number}_client{self.client_id}",
            )
            try:
                with open(path, "w", encoding="utf8") as debug_file:
                    debug_file.write(str(payload))
            except OSError:
                logging.debug("Unable to persist MPC debug artefact at %s.", path)

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        state = self.round_store.load_state()
        if self.client_id not in state.selected_clients:
            raise RuntimeError(
                f"Client {self.client_id} was not selected for round {state.round_number}."
            )

        num_samples = state.client_samples.get(self.client_id)
        if num_samples is None:
            logging.warning(
                "Client %s is encrypting updates without recorded num_samples. Defaulting to 0.",
                self.client_id,
            )
            num_samples = 0

        num_clients = len(state.selected_clients)
        data_shares = [copy.deepcopy(data) for _ in range(num_clients)]

        self._write_debug_artifact(state.round_number, "raw_weights", data)

        for name, tensor in data.items():
            scaled = tensor * num_samples
            shares = self._split_tensor(scaled, num_clients)
            for idx in range(num_clients):
                data_shares[idx][name] = shares[idx]

        self_index = state.selected_clients.index(self.client_id)

        for idx, target_client in enumerate(state.selected_clients):
            if target_client == self.client_id:
                continue

            self.round_store.append_additive_share(target_client, data_shares[idx])

        self._write_debug_artifact(
            state.round_number, "encrypted_weights", data_shares[self_index]
        )

        return data_shares[self_index]
