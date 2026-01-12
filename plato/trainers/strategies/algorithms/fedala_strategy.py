"""
FedALA Strategy Implementation

Reference:
Zhang, C., Xie, X., Tian, H., Wang, J., & Xu, Y. (2022).
"FedALA: Adaptive Local Aggregation for Federated Learning."
arXiv preprint arXiv:2212.01197.

Paper: https://arxiv.org/abs/2212.01197

Description:
FedALA performs adaptive local aggregation (ALA) on the client side to
initialize the local model before each training round. It mixes the received
global model with the client's previous local model using learnable weights
that are optimized on a randomly sampled subset of local data.

The ALA update is applied to the higher layers of the model while preserving
lower-layer updates. The learnable weights are trained until convergence in
the first round, and then updated for one epoch per subsequent round.
"""

from __future__ import annotations

import copy
import logging
import os
import random
from collections.abc import Callable
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from plato.config import Config
from plato.trainers.strategies.base import ModelUpdateStrategy, TrainingContext

LOGGER = logging.getLogger(__name__)


class FedALAUpdateStrategy(ModelUpdateStrategy):
    """
    FedALA update strategy for adaptive local aggregation.

    This strategy:
    1. Keeps a cached copy of the previous local model on each client.
    2. Uses ALA to initialize the current local model by combining the
       received global model with the cached local model.
    3. Updates and persists ALA weights across rounds.

    Args:
        layer_idx: Number of parameter tensors (from the end) to apply ALA to.
            Use 0 to select all layers (default: 0).
        eta: Learning rate for ALA weights (default: 1.0).
        rand_percent: Percentage of local data used for ALA (default: 80).
        batch_size: Batch size for ALA weight learning. If None, uses
            trainer batch_size (default: None).
        threshold: Convergence threshold for weight learning (default: 0.1).
        num_pre_loss: Number of recent losses used for convergence check
            (default: 10).
        loss_fn: Loss function used for ALA weight learning. If None, uses
            CrossEntropyLoss.
        save_state: Persist local model and ALA weights to disk (default: True).
    """

    def __init__(
        self,
        layer_idx: int = 0,
        eta: float = 1.0,
        rand_percent: float = 80.0,
        batch_size: int | None = None,
        threshold: float = 0.1,
        num_pre_loss: int = 10,
        max_ala_epochs: int | None = 20,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        save_state: bool = True,
    ) -> None:
        if layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative, got {layer_idx}")
        if eta <= 0:
            raise ValueError(f"eta must be positive, got {eta}")
        if rand_percent < 0 or rand_percent > 100:
            raise ValueError(
                f"rand_percent must be in [0, 100], got {rand_percent}"
            )
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")
        if num_pre_loss < 1:
            raise ValueError(
                f"num_pre_loss must be >= 1, got {num_pre_loss}"
            )
        if max_ala_epochs is not None and max_ala_epochs < 1:
            raise ValueError(
                f"max_ala_epochs must be >= 1 or None, got {max_ala_epochs}"
            )
        if batch_size is not None and batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.layer_idx = int(layer_idx)
        self.eta = float(eta)
        self.rand_percent = float(rand_percent)
        self.batch_size = batch_size
        self.threshold = float(threshold)
        self.num_pre_loss = int(num_pre_loss)
        self.max_ala_epochs = (
            int(max_ala_epochs) if max_ala_epochs is not None else None
        )
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.save_state = save_state

        self.weights: list[torch.Tensor] | None = None
        self.start_phase = True
        self.local_model_state: dict[str, torch.Tensor] | None = None
        self._ala_applied = False
        self._rng = random.Random()

        self._local_model_path: str | None = None
        self._ala_state_path: str | None = None
        self._cached_client_id: int | None = None

    def on_train_start(self, context: TrainingContext) -> None:
        """Prepare FedALA state for this training round."""
        self._resolve_state_paths(context)
        self._load_state(context)
        self._ala_applied = False

        if hasattr(Config(), "clients") and hasattr(Config().clients, "random_seed"):
            seed = int(Config().clients.random_seed)
            seed += int(context.client_id)
            seed += int(context.current_round)
            self._rng.seed(seed)

    def before_step(self, context: TrainingContext) -> None:
        """Apply ALA once before the first training step of the round."""
        if self._ala_applied:
            return
        if context.current_epoch != 1:
            return
        if context.state.get("current_batch", 0) != 0:
            return

        applied = self._apply_adaptive_local_aggregation(context)
        self._ala_applied = True

        if applied:
            LOGGER.info(
                "[Client #%d] Applied FedALA initialization.", context.client_id
            )

    def on_train_end(self, context: TrainingContext) -> None:
        """Persist local model and ALA weights after training."""
        model = context.model
        if model is None:
            return

        self.local_model_state = {
            name: param.detach().cpu().clone()
            for name, param in model.state_dict().items()
        }

        if self.save_state:
            self._save_state(context)

    def _resolve_state_paths(self, context: TrainingContext) -> None:
        if self._cached_client_id == context.client_id and self._local_model_path:
            return

        model_name = (
            Config().trainer.model_name
            if hasattr(Config(), "trainer") and hasattr(Config().trainer, "model_name")
            else "model"
        )
        base_path = Config().params.get("model_path", ".")

        self._local_model_path = (
            f"{base_path}/{model_name}_{context.client_id}_fedala_local.pth"
        )
        self._ala_state_path = (
            f"{base_path}/{model_name}_{context.client_id}_fedala_state.pth"
        )
        self._cached_client_id = context.client_id

    def _load_state(self, context: TrainingContext) -> None:
        if self.save_state:
            self._load_local_model_state(context)
            self._load_ala_state(context)

    def _load_local_model_state(self, context: TrainingContext) -> None:
        if self._local_model_path is None:
            return
        if not os.path.exists(self._local_model_path):
            return

        try:
            self.local_model_state = torch.load(
                self._local_model_path,
                map_location=torch.device("cpu"),
            )
            LOGGER.info(
                "[Client #%d] Loaded FedALA local model state.",
                context.client_id,
            )
        except Exception as exc:  # pragma: no cover - best effort load
            LOGGER.warning(
                "[Client #%d] Failed to load FedALA local model state: %s",
                context.client_id,
                exc,
            )

    def _load_ala_state(self, context: TrainingContext) -> None:
        if self._ala_state_path is None:
            return
        if not os.path.exists(self._ala_state_path):
            return

        try:
            state = torch.load(
                self._ala_state_path,
                map_location=torch.device("cpu"),
                weights_only=False,
            )
            self.weights = state.get("weights")
            self.start_phase = state.get("start_phase", True)
            LOGGER.info(
                "[Client #%d] Loaded FedALA weight state.",
                context.client_id,
            )
        except Exception as exc:  # pragma: no cover - best effort load
            LOGGER.warning(
                "[Client #%d] Failed to load FedALA weight state: %s",
                context.client_id,
                exc,
            )

    def _save_state(self, context: TrainingContext) -> None:
        if self._local_model_path is None or self._ala_state_path is None:
            return
        local_state = self.local_model_state
        if local_state is not None:
            try:
                torch.save(local_state, self._local_model_path)
            except Exception as exc:  # pragma: no cover - best effort save
                LOGGER.warning(
                    "[Client #%d] Failed to save FedALA local model state: %s",
                    context.client_id,
                    exc,
                )

        if self.weights is not None:
            weights = [weight.detach().cpu().clone() for weight in self.weights]
        else:
            weights = None

        try:
            torch.save(
                {"weights": weights, "start_phase": self.start_phase},
                self._ala_state_path,
            )
        except Exception as exc:  # pragma: no cover - best effort save
            LOGGER.warning(
                "[Client #%d] Failed to save FedALA weight state: %s",
                context.client_id,
                exc,
            )

    def _apply_adaptive_local_aggregation(self, context: TrainingContext) -> bool:
        if self.local_model_state is None:
            return False

        if self.rand_percent <= 0:
            LOGGER.info(
                "[Client #%d] FedALA disabled (rand_percent=0).",
                context.client_id,
            )
            return False

        train_loader = context.state.get("train_loader")
        if train_loader is None:
            LOGGER.warning(
                "[Client #%d] No train_loader available for FedALA.",
                context.client_id,
            )
            return False

        model = context.model
        if model is None:
            raise ValueError("Training context must provide a model for FedALA.")

        device = context.device or torch.device("cpu")

        rand_loader = self._build_random_loader(train_loader, context)
        if rand_loader is None:
            LOGGER.warning(
                "[Client #%d] Failed to build FedALA weight loader.",
                context.client_id,
            )
            return False

        global_model = model
        local_model = copy.deepcopy(model)
        try:
            local_model.load_state_dict(self.local_model_state, strict=True)
        except Exception as exc:
            LOGGER.warning(
                "[Client #%d] Failed to load local model for FedALA: %s",
                context.client_id,
                exc,
            )
            return False

        global_model.to(device)
        local_model.to(device)
        local_model.train()

        applied = self._adaptive_local_aggregation(
            global_model, local_model, rand_loader, context
        )
        if applied:
            global_model.load_state_dict(local_model.state_dict(), strict=True)
            global_model.train()

        return applied

    def _build_random_loader(
        self, train_loader: torch.utils.data.DataLoader, context: TrainingContext
    ) -> torch.utils.data.DataLoader | None:
        dataset = getattr(train_loader, "dataset", None)
        if dataset is None or not hasattr(dataset, "__len__"):
            LOGGER.warning(
                "[Client #%d] Train dataset is not indexable for FedALA.",
                context.client_id,
            )
            return None

        indices = self._resolve_sampler_indices(train_loader)
        if not indices:
            try:
                indices = list(range(len(dataset)))
            except TypeError:
                return None

        rand_num = int(len(indices) * (self.rand_percent / 100.0))
        if rand_num < 1:
            return None

        max_start = max(len(indices) - rand_num, 0)
        start = self._rng.randint(0, max_start) if max_start > 0 else 0
        subset_indices = indices[start : start + rand_num]

        subset = torch.utils.data.Subset(dataset, subset_indices)
        batch_size = self.batch_size
        if batch_size is None:
            batch_size = context.config.get("batch_size", 1)

        return torch.utils.data.DataLoader(
            dataset=subset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

    @staticmethod
    def _resolve_sampler_indices(
        train_loader: torch.utils.data.DataLoader,
    ) -> list[int] | None:
        sampler = getattr(train_loader, "sampler", None)
        if sampler is None:
            return None
        try:
            indices = list(iter(sampler))
        except TypeError:
            return None
        return indices

    @staticmethod
    def _move_to_device(value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(
                FedALAUpdateStrategy._move_to_device(v, device) for v in value
            )
        if isinstance(value, list):
            return [FedALAUpdateStrategy._move_to_device(v, device) for v in value]
        if isinstance(value, dict):
            return {
                key: FedALAUpdateStrategy._move_to_device(val, device)
                for key, val in value.items()
            }
        return value

    def _adaptive_local_aggregation(
        self,
        global_model: nn.Module,
        local_model: nn.Module,
        rand_loader: torch.utils.data.DataLoader,
        context: TrainingContext,
    ) -> bool:
        params_g = list(global_model.parameters())
        params_l = list(local_model.parameters())

        if not params_g or not params_l:
            return False

        if torch.sum(params_g[0] - params_l[0]).item() == 0:
            return False

        layer_idx = self._resolve_layer_idx(len(params_l))
        if layer_idx < len(params_l):
            for param, param_g in zip(params_l[:-layer_idx], params_g[:-layer_idx]):
                param.data = param_g.data.clone()

        model_t = copy.deepcopy(local_model)
        model_t.to(next(local_model.parameters()).device)
        model_t.train()

        params_t = list(model_t.parameters())
        params_p = params_l[-layer_idx:]
        params_gp = params_g[-layer_idx:]
        params_tp = params_t[-layer_idx:]

        if layer_idx < len(params_t):
            for param in params_t[:-layer_idx]:
                param.requires_grad = False

        optimizer = torch.optim.SGD(params_tp, lr=0.0)

        self._ensure_weights(params_p)
        if self.weights is None:
            raise RuntimeError("FedALA weights were not initialized.")
        weights = [weight.to(params_p[0].device) for weight in self.weights]
        self.weights = weights

        for param_t, param, param_g, weight in zip(
            params_tp, params_p, params_gp, weights
        ):
            param_t.data = param.data + (param_g.data - param.data) * weight

        losses: list[float] = []
        cnt = 0
        while True:
            loss_value = None
            for batch in rand_loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    examples, labels = batch[0], batch[1]
                else:
                    examples, labels = batch

                examples = self._move_to_device(
                    examples, next(model_t.parameters()).device
                )
                labels = self._move_to_device(
                    labels, next(model_t.parameters()).device
                )

                optimizer.zero_grad()
                output = model_t(examples)
                loss_value = self.loss_fn(output, labels)
                loss_value.backward()

                for param_t, param, param_g, weight in zip(
                    params_tp, params_p, params_gp, weights
                ):
                    if param_t.grad is None:
                        continue
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)),
                        0,
                        1,
                    )

                for param_t, param, param_g, weight in zip(
                    params_tp, params_p, params_gp, weights
                ):
                    param_t.data = param.data + (param_g.data - param.data) * weight

            if loss_value is None:
                break

            losses.append(float(loss_value.item()))
            cnt += 1
            if self.max_ala_epochs is not None and cnt >= self.max_ala_epochs:
                LOGGER.info(
                    "[Client #%d] FedALA reached max_ala_epochs=%d; stopping ALA.",
                    context.client_id,
                    self.max_ala_epochs,
                )
                break

            if not self.start_phase:
                break

            if len(losses) > self.num_pre_loss:
                std = np.std(losses[-self.num_pre_loss :])
                if std < self.threshold:
                    LOGGER.info(
                        "[Client #%d] FedALA converged (std=%.4f) after %d epochs.",
                        context.client_id,
                        std,
                        cnt,
                    )
                    break

        self.start_phase = False

        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()

        return True

    def _resolve_layer_idx(self, num_params: int) -> int:
        if self.layer_idx <= 0 or self.layer_idx > num_params:
            return num_params
        return self.layer_idx

    def _ensure_weights(self, params: Iterable[torch.Tensor]) -> None:
        params_list = list(params)
        if not params_list:
            self.weights = None
            return

        if self.weights is None or len(self.weights) != len(params_list):
            self.weights = [torch.ones_like(param.data) for param in params_list]
            return

        for weight, param in zip(self.weights, params_list):
            if weight.shape != param.data.shape:
                self.weights = [
                    torch.ones_like(param.data) for param in params_list
                ]
                return


class FedALAUpdateStrategyFromConfig(FedALAUpdateStrategy):
    """
    FedALA update strategy that reads configuration from Config.

    Configuration:
        The strategy looks for the following keys under Config().algorithm:
        - eta (default: 1.0)
        - rand_percent (default: 80)
        - layer_idx (default: 0)
        - threshold (default: 0.1)
        - num_pre_loss (default: 10)
        - max_ala_epochs (default: 20)
        - ala_batch_size (optional)
    """

    def __init__(self) -> None:
        config = Config()
        algo = getattr(config, "algorithm", None)

        eta = self._get_config_value(algo, ["eta", "fedala_eta", "ala_eta"], 1.0)
        rand_percent = self._get_config_value(
            algo, ["rand_percent", "fedala_rand_percent", "ala_rand_percent"], 80
        )
        layer_idx = self._get_config_value(
            algo, ["layer_idx", "fedala_layer_idx", "ala_layer_idx"], 0
        )
        threshold = self._get_config_value(
            algo, ["threshold", "fedala_threshold", "ala_threshold"], 0.1
        )
        num_pre_loss = self._get_config_value(
            algo, ["num_pre_loss", "fedala_num_pre_loss", "ala_num_pre_loss"], 10
        )
        max_ala_epochs = self._get_config_value(
            algo,
            [
                "max_ala_epochs",
                "fedala_max_ala_epochs",
                "ala_max_epochs",
                "fedala_max_epochs",
            ],
            20,
        )
        batch_size = self._get_config_value(
            algo, ["ala_batch_size", "fedala_batch_size"], None
        )

        if batch_size is None and hasattr(config, "trainer"):
            batch_size = getattr(config.trainer, "batch_size", None)

        super().__init__(
            layer_idx=int(layer_idx),
            eta=float(eta),
            rand_percent=float(rand_percent),
            batch_size=batch_size,
            threshold=float(threshold),
            num_pre_loss=int(num_pre_loss),
            max_ala_epochs=None if max_ala_epochs is None else int(max_ala_epochs),
        )

    @staticmethod
    def _get_config_value(section: Any, keys: Iterable[str], default: Any) -> Any:
        if section is None:
            return default
        for key in keys:
            if hasattr(section, key):
                return getattr(section, key)
        return default
