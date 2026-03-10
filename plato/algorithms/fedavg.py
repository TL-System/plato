"""
The federated averaging algorithm for PyTorch.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any

import torch
from torch.nn import Module

from plato.algorithms import base


class Algorithm(base.Algorithm):
    """PyTorch-based federated averaging algorithm, used by both the client and the server."""

    @staticmethod
    def _as_state_mapping(weights: Any, context: str) -> Mapping[str, torch.Tensor]:
        """Validate and cast a state-dict-like payload."""
        if not isinstance(weights, Mapping):
            raise TypeError(f"{context} must be a mapping of parameter names to tensors.")
        return weights

    @staticmethod
    def _to_transport_tensor(
        tensor: torch.Tensor, tensor_name: str
    ) -> torch.Tensor:
        """
        Convert a tensor to a wire-safe representation for payload transport.

        Safetensor serialization in the current runtime path does not support
        `torch.bfloat16` conversion through numpy. Cast bf16 payload tensors to
        fp32 for transport, then cast back in `load_weights`.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Payload tensor '{tensor_name}' must be a torch.Tensor, "
                f"received {type(tensor).__name__}."
            )

        prepared = tensor.detach().cpu().clone()
        if prepared.dtype == torch.bfloat16:
            return prepared.to(torch.float32)
        return prepared

    @staticmethod
    def _cast_tensor_like(
        tensor: torch.Tensor, reference: torch.Tensor, tensor_name: str
    ) -> torch.Tensor:
        """Cast an incoming tensor to match the dtype expected by `reference`."""
        if tensor.shape != reference.shape:
            raise ValueError(
                f"Tensor shape mismatch for '{tensor_name}': "
                f"received {tuple(tensor.shape)}, expected {tuple(reference.shape)}."
            )

        if tensor.dtype == reference.dtype:
            return tensor.detach()

        if reference.dtype == torch.bool:
            if torch.is_floating_point(tensor):
                return (tensor >= 0.5).detach()
            return tensor.ne(0).detach()

        if torch.is_floating_point(reference) or torch.is_complex(reference):
            return tensor.to(reference.dtype).detach()

        if torch.is_floating_point(tensor):
            return torch.round(tensor).to(reference.dtype).detach()

        return tensor.to(reference.dtype).detach()

    @staticmethod
    def _compute_tensor_delta(
        current_weight: torch.Tensor,
        baseline_weight: torch.Tensor,
        tensor_name: str,
    ) -> torch.Tensor:
        """Compute a dtype-safe delta tensor for a parameter."""
        current_casted = Algorithm._cast_tensor_like(
            current_weight, baseline_weight, tensor_name
        )

        if baseline_weight.dtype == torch.bool:
            return current_casted.to(torch.int8) - baseline_weight.to(torch.int8)

        if torch.is_floating_point(baseline_weight) or torch.is_complex(baseline_weight):
            return current_casted.to(baseline_weight.dtype) - baseline_weight

        return current_casted.to(torch.int64) - baseline_weight.to(torch.int64)

    @staticmethod
    def _apply_tensor_delta(
        baseline_weight: torch.Tensor, delta: torch.Tensor, tensor_name: str
    ) -> torch.Tensor:
        """Apply a delta tensor to the baseline tensor with dtype safeguards."""
        if delta.shape != baseline_weight.shape:
            raise ValueError(
                f"Delta shape mismatch for '{tensor_name}': "
                f"received {tuple(delta.shape)}, expected {tuple(baseline_weight.shape)}."
            )

        if baseline_weight.dtype == torch.bool:
            if torch.is_floating_point(delta):
                delta_integral = torch.round(delta).to(torch.int8)
            else:
                delta_integral = delta.to(torch.int8)
            return (baseline_weight.to(torch.int8) + delta_integral).ne(0)

        if torch.is_floating_point(baseline_weight) or torch.is_complex(baseline_weight):
            return baseline_weight + delta.to(baseline_weight.dtype)

        if torch.is_floating_point(delta):
            delta_casted = torch.round(delta).to(baseline_weight.dtype)
        else:
            delta_casted = delta.to(baseline_weight.dtype)
        return baseline_weight + delta_casted

    def _resolve_payload_limit_mb(self) -> float | None:
        """Resolve an optional payload-size limit from model attrs or env vars."""
        model = self.require_model()
        configured_limit = getattr(model, "plato_max_payload_size_mb", None)
        if configured_limit is None:
            configured_limit = os.getenv("PLATO_FEDAVG_MAX_PAYLOAD_MB")

        if configured_limit is None:
            return None

        try:
            limit_mb = float(configured_limit)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid payload size limit: {configured_limit!r}. "
                "Expected a positive numeric value in megabytes."
            ) from exc

        if limit_mb <= 0:
            raise ValueError("Payload size limit must be greater than 0 MB.")

        return limit_mb

    @staticmethod
    def _estimate_payload_size_bytes(weights: Mapping[str, torch.Tensor]) -> int:
        """Estimate payload size by summing tensor storage bytes."""
        size_bytes = 0
        for name, tensor in weights.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"Payload tensor '{name}' must be a torch.Tensor, "
                    f"received {type(tensor).__name__}."
                )
            size_bytes += tensor.numel() * tensor.element_size()
        return size_bytes

    def _assert_payload_size(self, weights: Mapping[str, torch.Tensor], source: str) -> None:
        """Enforce an optional payload-size safeguard."""
        limit_mb = self._resolve_payload_limit_mb()
        if limit_mb is None:
            return

        payload_size_mb = self._estimate_payload_size_bytes(weights) / 1024**2
        if payload_size_mb > limit_mb:
            raise ValueError(
                f"{source} payload size {payload_size_mb:.2f} MB exceeds "
                f"configured limit {limit_mb:.2f} MB."
            )

    @staticmethod
    def _resolve_adapter_parameter_names(
        target_model: Module, state_dict: Mapping[str, torch.Tensor]
    ) -> list[str] | None:
        """Resolve parameter names to exchange for adapter-only finetuning."""
        finetune_mode = getattr(target_model, "plato_finetune_mode", None)
        if not isinstance(finetune_mode, str) or finetune_mode.strip().lower() != "adapter":
            return None

        trainable_names_attr = getattr(target_model, "plato_trainable_parameter_names", None)
        names_from_attr = (
            [
                name
                for name in trainable_names_attr
                if isinstance(name, str) and name in state_dict
            ]
            if isinstance(trainable_names_attr, Iterable)
            else []
        )

        if names_from_attr:
            return names_from_attr

        names_from_requires_grad = [
            name
            for name, parameter in target_model.named_parameters()
            if parameter.requires_grad and name in state_dict
        ]
        if names_from_requires_grad:
            return names_from_requires_grad

        raise ValueError(
            "Adapter finetune mode is enabled, but no trainable parameters "
            "were resolved for federated payload exchange."
        )

    def compute_weight_deltas(
        self,
        baseline_weights: MutableMapping[str, Any],
        weights_received,
    ):
        """Compute the deltas between baseline weights and weights received."""
        baseline_mapping = self._as_state_mapping(
            baseline_weights, context="baseline_weights"
        )

        # Calculate updates from the received weights
        deltas = []
        for weight in weights_received:
            weight_mapping = self._as_state_mapping(weight, context="received weights")
            self._assert_payload_size(weight_mapping, source="Received")

            unknown_keys = set(weight_mapping).difference(baseline_mapping)
            if unknown_keys:
                unknown = ", ".join(sorted(unknown_keys))
                raise KeyError(f"Received weights include unexpected parameter(s): {unknown}.")

            delta = OrderedDict()
            for name, current_weight in weight_mapping.items():
                if not isinstance(current_weight, torch.Tensor):
                    raise TypeError(
                        f"Received tensor '{name}' must be a torch.Tensor, "
                        f"received {type(current_weight).__name__}."
                    )

                baseline = baseline_mapping[name]
                if not isinstance(baseline, torch.Tensor):
                    raise TypeError(
                        f"Baseline tensor '{name}' must be a torch.Tensor, "
                        f"received {type(baseline).__name__}."
                    )

                # Calculate update
                _delta = self._compute_tensor_delta(current_weight, baseline, name)
                delta[name] = _delta
            deltas.append(delta)

        return deltas

    def update_weights(self, deltas):
        """Updates the existing model weights from the provided deltas."""
        baseline_weights = self.extract_weights()
        delta_mapping = self._as_state_mapping(deltas, context="deltas")

        updated_weights = OrderedDict()
        for name, weight in baseline_weights.items():
            updated_weights[name] = weight

        unknown_keys = set(delta_mapping).difference(baseline_weights)
        if unknown_keys:
            unknown = ", ".join(sorted(unknown_keys))
            raise KeyError(f"Delta includes unexpected parameter(s): {unknown}.")

        for name, delta in delta_mapping.items():
            baseline = baseline_weights[name]
            if not isinstance(delta, torch.Tensor):
                raise TypeError(
                    f"Delta tensor '{name}' must be a torch.Tensor, "
                    f"received {type(delta).__name__}."
                )
            updated_weights[name] = self._apply_tensor_delta(baseline, delta, name)

        self._assert_payload_size(updated_weights, source="Updated")

        return updated_weights

    def extract_weights(self, model: Module | None = None):
        """Extracts weights from the model."""
        target_model: Module
        if model is None:
            target_model = self.require_model()
        else:
            target_model = model

        state_dict = target_model.state_dict()
        adapter_names = self._resolve_adapter_parameter_names(target_model, state_dict)
        keys_to_exchange = adapter_names or list(state_dict.keys())

        payload = OrderedDict(
            (
                name,
                self._to_transport_tensor(state_dict[name], name),
            )
            for name in keys_to_exchange
        )
        self._assert_payload_size(payload, source="Extracted")
        return payload

    def load_weights(self, weights):
        """Loads the model weights passed in as a parameter."""
        weights_mapping = self._as_state_mapping(weights, context="weights")
        self._assert_payload_size(weights_mapping, source="Inbound")

        model: Module = self.require_model()
        current_state = model.state_dict()

        unknown_keys = set(weights_mapping).difference(current_state)
        if unknown_keys:
            unknown = ", ".join(sorted(unknown_keys))
            raise KeyError(f"Inbound weights include unexpected parameter(s): {unknown}.")

        merged_state = OrderedDict(current_state.items())
        for name, incoming_tensor in weights_mapping.items():
            if not isinstance(incoming_tensor, torch.Tensor):
                raise TypeError(
                    f"Inbound tensor '{name}' must be a torch.Tensor, "
                    f"received {type(incoming_tensor).__name__}."
                )
            merged_state[name] = self._cast_tensor_like(
                incoming_tensor, current_state[name], name
            )

        model.load_state_dict(merged_state, strict=True)
