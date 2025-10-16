"""
Multiparty computation outbound processor using Shamir secret sharing.
"""

from __future__ import annotations

import copy
import logging
import math
import os
from random import randint
from typing import Any, Dict, List

import torch

from plato.config import Config
from plato.mpc import RoundInfoStore
from plato.processors import model


class Processor(model.Processor):
    """Encrypts model updates using Shamir secret sharing."""

    def __init__(
        self,
        *,
        client_id: int,
        round_store: RoundInfoStore,
        threshold: int | None = None,
        debug_artifacts: bool | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.client_id = client_id
        self.round_store = round_store
        self.threshold = threshold
        if debug_artifacts is None:
            debug_artifacts = getattr(Config().clients, "mpc_debug_artifacts", False)
        self.debug_artifacts = debug_artifacts

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

    @staticmethod
    def _calculate_poly_value(x: int, coefficients: torch.Tensor) -> float:
        """Evaluate polynomial with the given coefficients at position ``x``."""
        y_val = 0.0
        power = 1.0
        for coeff in coefficients:
            y_val += coeff.item() * power
            power *= x
        return y_val

    def _secret_shares(
        self, secret: torch.Tensor, num_clients: int, threshold: int
    ) -> torch.Tensor:
        """Generate Shamir shares for a single scalar secret."""
        scaled_secret = round(secret.item() * 1_000_000)
        coefficients = torch.zeros(threshold)
        coefficients[0] = scaled_secret

        for idx in range(1, threshold):
            value = randint(1, 999)
            while value in coefficients:
                value = randint(1, 999)
            coefficients[idx] = value

        points = torch.zeros([num_clients, 2])
        for j in range(1, num_clients + 1):
            points[j - 1][0] = j
            points[j - 1][1] = self._calculate_poly_value(j, coefficients)

        return points

    def _split_tensor(
        self, tensor: torch.Tensor, num_clients: int, threshold: int
    ) -> torch.Tensor:
        """Encrypt tensor entries using Shamir secret sharing."""
        if num_clients == 1:
            size = list(tensor.size())
            size.insert(0, 1)
            size.append(2)
            coords = torch.zeros(size)
            coords[0, ..., 0] = 1
            coords[0, ..., 1] = tensor
            return coords

        flattened_size = math.prod(list(tensor.size()))
        flattened = tensor.view(flattened_size)

        coords = torch.empty([num_clients, flattened_size, 2])
        for idx in range(flattened_size):
            coords[:, idx] = self._secret_shares(flattened[idx], num_clients, threshold)

        encrypted_size = list(tensor.size())
        encrypted_size.insert(0, num_clients)
        encrypted_size.append(2)
        return coords.view(encrypted_size)

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

        selected_clients = state.selected_clients
        num_clients = len(selected_clients)
        threshold = self.threshold or max(num_clients - 2, 1)

        data_shares: List[Dict[str, Any]] = [
            copy.deepcopy(data) for _ in range(num_clients)
        ]

        self._write_debug_artifact(state.round_number, "raw_weights", data)

        for name, tensor in data.items():
            scaled = tensor * num_samples
            shares = self._split_tensor(scaled, num_clients, threshold)
            for idx in range(num_clients):
                data_shares[idx][name] = shares[idx]

        self_index = selected_clients.index(self.client_id)

        for idx, target_client in enumerate(selected_clients):
            if idx == self_index:
                continue
            self.round_store.store_pairwise_share(
                target_client, self.client_id, data_shares[idx]
            )

        self._write_debug_artifact(
            state.round_number, "encrypted_weights", data_shares[self_index]
        )
        return data_shares[self_index]
