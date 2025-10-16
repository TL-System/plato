"""
Server aggregation using the attack-adaptive aggregation strategy.

Reference:

Ching Pui Wan, Qifeng Chen, "Robust Federated Learning with Attack-Adaptive Aggregation"
Unpublished
(https://arxiv.org/pdf/2102.05257.pdf)

This implementation mirrors the released reference code: each round we
project client updates with PCA, feed them to the trained attention module,
truncate the attention weights, and apply the learned weighting to aggregate
the updates.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from plato.config import Config
from plato.servers.strategies.base import AggregationStrategy, ServerContext


def _get_float_parameter_names(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    """Return parameter names whose tensors are floating point."""
    return [name for name, tensor in state_dict.items() if tensor.is_floating_point()]


def _stack_state_dicts(
    deltas: Sequence[Dict[str, torch.Tensor]],
    allowed_names: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Stack state dicts into 2-D matrices of shape (num_features, num_clients).

    Only parameters listed in ``allowed_names`` are included. Non-floating tensors
    are ignored to stay consistent with the reference implementation.
    """
    if not deltas:
        raise ValueError("No client deltas provided for attack-adaptive aggregation.")

    reference = deltas[0]
    float_names = _get_float_parameter_names(reference)

    if allowed_names is not None:
        allowed_set = set(allowed_names)
        float_names = [name for name in float_names if name in allowed_set]

    stacked: Dict[str, torch.Tensor] = {}
    for name in float_names:
        values = [delta[name].detach().cpu() for delta in deltas]
        param_stack = torch.stack(values, dim=-1)
        stacked[name] = param_stack.reshape(-1, len(values))
    return stacked


def _net2vec(components: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Concatenate matrices (num_components, num_clients) across parameters.

    The result mirrors the behaviour of utils.convert_pca.net2vec in the
    reference repository, yielding a tensor of shape (total_components, num_clients).
    """
    stacked_components: List[torch.Tensor] = []
    for name in components:
        stacked_components.append(components[name])
    if not stacked_components:
        raise ValueError("No floating-point parameters available for PCA projection.")
    return torch.cat(stacked_components, dim=0)


def _apply_weights_to_state_dicts(
    deltas: Sequence[Dict[str, torch.Tensor]], weights: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Apply scalar weights to each client's delta and sum the result.

    Only floating-point tensors are combined; other tensors (e.g., buffers) are
    copied from the first client unchanged.
    """
    if len(deltas) != len(weights):
        raise ValueError(
            f"Number of deltas ({len(deltas)}) and weights ({len(weights)}) must match."
        )

    result: Dict[str, torch.Tensor] = {}
    float_names = _get_float_parameter_names(deltas[0])

    for name, tensor in deltas[0].items():
        if name not in float_names:
            result[name] = tensor.clone()
            continue

        accumulator = torch.zeros_like(tensor)
        for client_idx, delta in enumerate(deltas):
            accumulator = (
                accumulator + delta[name].to(accumulator.device) * weights[client_idx]
            )
        result[name] = accumulator

    return result


def _svd_with_fallback(
    matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run SVD with a fallback path for numerical issues."""
    try:
        # torch.linalg.svd returns U, S, Vh
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        v = vh.transpose(-2, -1)
    except RuntimeError:
        # Fall back to the legacy torch.svd for numerical robustness
        u, s, v = torch.svd(matrix)
    return u, s, v


def _pca(matrix: torch.Tensor, n_components: int) -> torch.Tensor:
    """
    Compute PCA projection using SVD, matching the reference implementation.

    Args:
        matrix: Tensor with shape (num_features, num_clients).
        n_components: Maximum number of principal components to keep.

    Returns:
        Tensor with shape (min(n_components, num_clients), num_clients).
    """
    matrix = matrix.float()
    num_features, num_clients = matrix.shape
    n_components = min(n_components, num_clients, num_features)

    if num_clients <= num_features:
        # Overdetermined case (more features than clients)
        _, s, v = _svd_with_fallback(matrix)
        projected = (v * s.unsqueeze(0))[:, :n_components]
    else:
        # Underdetermined case (more clients than features)
        u, s, _ = _svd_with_fallback(matrix.transpose(0, 1))
        projected = (u * s.unsqueeze(0))[:, :n_components]

    return projected.transpose(0, 1)


def _apply_pca_to_state_dict(
    stacked: Dict[str, torch.Tensor], n_components: int
) -> Dict[str, torch.Tensor]:
    """Apply PCA to each stacked parameter matrix."""
    projected: Dict[str, torch.Tensor] = {}
    for name, matrix in stacked.items():
        projected[name] = _pca(matrix, n_components)
    return projected


def _flatten_grad_dict(grad_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten and concatenate gradient tensors in a deterministic order."""
    if not grad_dict:
        return torch.tensor([])
    names = sorted(grad_dict.keys(), key=str.lower)
    flattened = [grad_dict[name].reshape(-1) for name in names]
    return torch.cat(flattened)


class _NonLinearity(nn.Module):
    """1x1 convolutional non-linearity used by the reference attention model."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, 2 * in_channels, kernel_size=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv1d(2 * in_channels, out_channels, kernel_size=1, bias=bias),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.main(inputs)


class _Affinity(nn.Module):
    """Scaled dot-product attention with temperature and truncation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        bias: bool = False,
        self_attention: bool = True,
        epsilon: float,
        scale: float,
    ):
        super().__init__()
        self.key_conv = _NonLinearity(in_channels, out_channels, bias=bias)
        if self_attention:
            self.query_conv = self.key_conv
        else:
            self.query_conv = _NonLinearity(in_channels, out_channels, bias=bias)

        self.scale = scale
        self.threshold = nn.Threshold(epsilon, 0.0)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_out = F.normalize(self.query_conv(query), dim=1)
        k_out = F.normalize(self.key_conv(key), dim=1)
        scores = torch.bmm(q_out.transpose(1, 2), k_out) * self.scale
        weights = F.softmax(scores, dim=-1)
        weights = self.threshold(weights)
        return weights, key


class _AttentionConv(nn.Module):
    """Single attention pass that mirrors the reference architecture."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        bias: bool = False,
        epsilon: float,
        scale: float,
    ):
        super().__init__()
        self.affinity = _Affinity(
            in_channels,
            out_channels,
            bias=bias,
            epsilon=epsilon,
            scale=scale,
        )

    def forward(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weights, value = self.affinity(query, key)
        # Einstein summation emulates the projection in the reference code.
        output = torch.einsum("bqi,bji->bjq", weights, value)
        return output, weights


class _AttentionLoop(nn.Module):
    """
    Multi-pass attention module with shared parameters.

    The module iteratively refines the attention weights exactly as implemented
    in the reference repository (nloop passes with tied weights).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        bias: bool = False,
        iterations: int,
        epsilon: float,
        scale: float,
    ):
        super().__init__()
        self.iterations = iterations
        self.attention = _AttentionConv(
            in_channels,
            out_channels,
            bias=bias,
            epsilon=epsilon,
            scale=scale,
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        output = query
        weights = None
        for _ in range(self.iterations):
            output, weights = self.attention(output, key)
        return output

    def get_weights(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        output = query
        weights = None
        for _ in range(self.iterations):
            output, weights = self.attention(output, key)
        if weights is None:
            raise RuntimeError("Attention weights were not computed.")
        return weights


def _load_attention_state_dict(model_path: Path) -> Optional[Dict[str, torch.Tensor]]:
    if not model_path.exists():
        logging.warning(
            "Attack-adaptive attention model not found at '%s'. "
            "A randomly initialised attention module will be used instead. "
            "For robust behaviour, pretrain the module with the authors' "
            "scripts and set algorithm.attention_model_path accordingly.",
            model_path,
        )
        return None

    state_dict = torch.load(model_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Expected a state_dict when loading attention model from '{model_path}', "
            f"received object of type {type(state_dict)}."
        )
    return state_dict


class AttackAdaptiveAggregationStrategy(AggregationStrategy):
    """Attack-adaptive aggregation strategy using the released attention model."""

    def __init__(
        self,
        *,
        attention_model_path: Optional[str] = None,
        pca_components: Optional[int] = None,
        epsilon: Optional[float] = None,
        scaling_factor: Optional[float] = None,
        attention_hidden: int = 32,
        attention_loops: int = 5,
        dataset_capture_dir: Optional[str] = None,
    ):
        super().__init__()
        self.attention_model_path = attention_model_path
        self.pca_components = pca_components
        self.epsilon = epsilon
        self.scaling_factor = scaling_factor
        self.attention_hidden = attention_hidden
        self.attention_loops = attention_loops
        self.dataset_capture_dir = (
            Path(dataset_capture_dir) if dataset_capture_dir else None
        )

        self._cached_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self._attention_model_path: Optional[Path] = (
            Path(attention_model_path) if attention_model_path else None
        )
        self._capture_run_dir: Optional[Path] = None
        self._capture_metadata_written = False
        self._local_angles: Dict[int, float] = {}

    def setup(self, context: ServerContext) -> None:
        """Load configuration defaults and the cached attention weights."""
        algorithm_cfg = getattr(Config(), "algorithm", None)

        if self.attention_model_path is None and algorithm_cfg is not None:
            if hasattr(algorithm_cfg, "attention_model_path"):
                self.attention_model_path = algorithm_cfg.attention_model_path

        if self.pca_components is None:
            self.pca_components = (
                algorithm_cfg.pca_components
                if hasattr(algorithm_cfg, "pca_components")
                else 10
            )

        if self.epsilon is None:
            self.epsilon = (
                algorithm_cfg.threshold
                if hasattr(algorithm_cfg, "threshold")
                else 0.005
            )

        if self.scaling_factor is None:
            self.scaling_factor = (
                algorithm_cfg.scaling_factor
                if hasattr(algorithm_cfg, "scaling_factor")
                else 10.0
            )

        if hasattr(algorithm_cfg, "attention_loops"):
            self.attention_loops = algorithm_cfg.attention_loops

        if hasattr(algorithm_cfg, "attention_hidden"):
            self.attention_hidden = algorithm_cfg.attention_hidden

        if self.dataset_capture_dir is None and algorithm_cfg is not None:
            if hasattr(algorithm_cfg, "dataset_capture_dir"):
                self.dataset_capture_dir = Path(algorithm_cfg.dataset_capture_dir)

        if self.attention_model_path is None:
            default_model_path = Path(
                "examples/server_aggregation/attack_adaptive/attention_model.pt"
            )
            logging.warning(
                "No attention model path configured; defaulting to '%s'.",
                default_model_path,
            )
            self.attention_model_path = str(default_model_path)

        model_path = Path(self.attention_model_path)
        self._cached_state_dict = _load_attention_state_dict(model_path)
        self._attention_model_path = model_path

        if self.dataset_capture_dir is not None:
            capture_root = Path(self.dataset_capture_dir)
            capture_root.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self._capture_run_dir = capture_root / f"run_{timestamp}"
            self._capture_run_dir.mkdir(parents=True, exist_ok=True)
            self._capture_metadata_written = False

    async def aggregate_deltas(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        """This method is intentionally not implemented."""
        raise NotImplementedError(
            "Attack-adaptive aggregation operates on model weights directly."
        )

    async def aggregate_weights(
        self,
        updates: List[SimpleNamespace],
        baseline_weights: Dict[str, torch.Tensor],
        weights_received: List[Dict[str, torch.Tensor]],
        context: ServerContext,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Aggregate client weights with the attack-adaptive attention mechanism."""
        deltas = self._compute_deltas(baseline_weights, weights_received)
        num_samples = [update.report.num_samples for update in updates]

        trainable_names = self._get_trainable_parameter_names(context)
        stacked = _stack_state_dicts(deltas, trainable_names)
        projected = _apply_pca_to_state_dict(stacked, self.pca_components)
        proj_vec = _net2vec(projected)

        reference_weights = self._compute_reference_weights(
            deltas=deltas,
            num_samples=num_samples,
            updates=updates,
            context=context,
        )

        input_channels = proj_vec.shape[0]

        attention_module = _AttentionLoop(
            input_channels,
            self.attention_hidden,
            iterations=self.attention_loops,
            epsilon=self.epsilon,
            scale=self.scaling_factor,
        )
        if self._cached_state_dict is not None:
            attention_module.load_state_dict(self._cached_state_dict)
        else:
            # Cache the randomly initialised weights so we reuse the same instance
            self._cached_state_dict = attention_module.state_dict()
            if self._attention_model_path is not None:
                try:
                    self._attention_model_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self._cached_state_dict, self._attention_model_path)
                    logging.warning(
                        "Saved randomly initialised attack-adaptive attention weights "
                        "to '%s'. Consider training the module for better robustness.",
                        self._attention_model_path,
                    )
                except OSError as err:
                    logging.warning("Unable to persist attention weights: %s", err)

        # Ensure we work with the cached parameters (trained or random)
        if self._cached_state_dict is not None:
            attention_module.load_state_dict(self._cached_state_dict)
        attention_module.eval()

        data = proj_vec.unsqueeze(0)  # Shape: (1, channels, num_clients)
        beta = data.median(dim=-1, keepdim=True).values
        weights_tensor = attention_module.get_weights(beta, data)

        # weights_tensor shape: (batch=1, queries=1, clients)
        weights_tensor = weights_tensor.squeeze(0).squeeze(0)
        if weights_tensor.numel() != len(deltas):
            raise RuntimeError(
                "Attention module produced weights of mismatched length: "
                f"{weights_tensor.numel()} vs {len(deltas)}."
            )

        normalized_weights = F.normalize(weights_tensor, p=1, dim=0)
        if torch.isnan(normalized_weights).any():
            normalized_weights = torch.full_like(
                weights_tensor, 1.0 / weights_tensor.numel()
            )
        aggregated_delta = _apply_weights_to_state_dicts(deltas, normalized_weights)

        self._capture_round(
            context=context,
            projection=proj_vec.detach().cpu(),
            reference_weights=reference_weights.detach().cpu()
            if reference_weights.numel() > 0
            else reference_weights,
            attention_weights=normalized_weights.detach().cpu(),
            num_samples=num_samples,
            client_ids=[update.client_id for update in updates],
        )

        updated_weights: Dict[str, torch.Tensor] = OrderedDict()
        for name, baseline in baseline_weights.items():
            updated_weights[name] = baseline + aggregated_delta.get(
                name, torch.zeros_like(baseline)
            )

        return updated_weights

    @staticmethod
    def _compute_deltas(
        baseline_weights: Dict[str, torch.Tensor],
        weights_received: Sequence[Dict[str, torch.Tensor]],
    ) -> List[Dict[str, torch.Tensor]]:
        """Compute client deltas relative to the baseline weights."""
        deltas: List[Dict[str, torch.Tensor]] = []
        for weights in weights_received:
            delta = OrderedDict()
            for name, baseline in baseline_weights.items():
                delta[name] = weights[name] - baseline
            deltas.append(delta)
        return deltas

    @staticmethod
    def _get_trainable_parameter_names(context: ServerContext) -> Optional[List[str]]:
        """Return names of trainable parameters from the server's trainer, if available."""
        trainer = getattr(context, "trainer", None)
        model = getattr(trainer, "model", None) if trainer is not None else None

        if model is None:
            logging.warning(
                "Attack-adaptive aggregation could not locate the trainer's model. "
                "Proceeding by assuming all floating-point parameters are trainable."
            )
            return None

        return [name for name, param in model.named_parameters() if param.requires_grad]

    def _compute_reference_weights(
        self,
        *,
        deltas: Sequence[Dict[str, torch.Tensor]],
        num_samples: Sequence[int],
        updates: Sequence[SimpleNamespace],
        context: ServerContext,
    ) -> torch.Tensor:
        """Compute heuristic weights using the pre-strategy FedAdp formulation."""
        if not deltas:
            return torch.tensor([])

        total_samples = float(sum(num_samples))
        if total_samples <= 0:
            return torch.full((len(deltas),), 1.0 / len(deltas), dtype=torch.float32)

        names = sorted(deltas[0].keys(), key=str.lower)
        global_grad_dict: Dict[str, torch.Tensor] = {}
        for name in names:
            accumulator = torch.zeros_like(deltas[0][name])
            for idx, delta in enumerate(deltas):
                weight = num_samples[idx] / total_samples
                accumulator = accumulator + delta[name] * weight
            global_grad_dict[name] = accumulator

        global_vec = _flatten_grad_dict(global_grad_dict)
        current_round = max(context.current_round, 1)
        alpha = Config().algorithm.alpha if hasattr(Config().algorithm, "alpha") else 5

        contribs: List[float] = []
        global_norm = torch.norm(global_vec)

        for idx, delta in enumerate(deltas):
            local_vec = _flatten_grad_dict(delta)
            local_norm = torch.norm(local_vec)
            if global_norm.item() == 0.0 or local_norm.item() == 0.0:
                angle = torch.tensor(0.0)
            else:
                cosine = torch.clamp(
                    torch.dot(global_vec, local_vec) / (global_norm * local_norm),
                    -1.0,
                    1.0,
                )
                angle = torch.acos(cosine)

            client_id = updates[idx].client_id
            previous = self._local_angles.get(client_id, angle.item())
            smoothed_angle = (current_round - 1) / current_round * previous + (
                1 / current_round
            ) * angle.item()
            self._local_angles[client_id] = smoothed_angle

            angle_tensor = torch.tensor(smoothed_angle, dtype=torch.float32)
            contrib_tensor = alpha * (
                1 - torch.exp(-torch.exp(-alpha * (angle_tensor - 1)))
            )
            contribs.append(float(contrib_tensor))

        contrib_tensor = torch.tensor(contribs, dtype=torch.float32)
        sample_tensor = torch.tensor(num_samples, dtype=torch.float32)
        numerators = sample_tensor * torch.exp(contrib_tensor)
        denom = numerators.sum()
        if denom.item() == 0.0:
            return torch.full((len(deltas),), 1.0 / len(deltas), dtype=torch.float32)
        return numerators / denom

    def _capture_round(
        self,
        *,
        context: ServerContext,
        projection: torch.Tensor,
        reference_weights: torch.Tensor,
        attention_weights: torch.Tensor,
        num_samples: Sequence[int],
        client_ids: Sequence[int],
    ) -> None:
        """Persist per-round projection and target weights for pretraining."""
        if self._capture_run_dir is None:
            return

        round_idx = context.current_round
        payload = {
            "round": round_idx,
            "projection": projection,
            "reference_weights": reference_weights,
            "attention_weights": attention_weights,
            "num_samples": torch.tensor(num_samples, dtype=torch.float32),
            "client_ids": torch.tensor(list(client_ids), dtype=torch.int64),
        }
        output_path = self._capture_run_dir / f"round_{round_idx:05d}.pt"
        torch.save(payload, output_path)

        if not self._capture_metadata_written:
            metadata = {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "pca_components": self.pca_components,
                "epsilon": self.epsilon,
                "scaling_factor": self.scaling_factor,
                "attention_hidden": self.attention_hidden,
                "attention_loops": self.attention_loops,
                "total_clients": len(client_ids),
            }
            metadata_path = self._capture_run_dir / "metadata.json"
            with metadata_path.open("w", encoding="utf-8") as meta_file:
                json.dump(metadata, meta_file, indent=2)
            self._capture_metadata_written = True
