"""Composable trainer for LeRobot policies such as SmolVLA."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data._utils.collate import default_collate

from plato.config import Config
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies import CustomCollateFnDataLoaderStrategy
from plato.trainers.strategies.base import (
    TestingStrategy,
    TrainingContext,
    TrainingStepStrategy,
)

_RESERVED_KEYS = frozenset({"plato_inputs", "plato_targets", "plato_metadata"})
_SUPPORTED_POLICY_PRECISIONS = frozenset({"fp32", "fp16", "bf16"})


def _config_node_to_dict(node: Any) -> dict[str, Any]:
    """Convert config sections to plain dictionaries."""
    if node is None:
        return {}
    if isinstance(node, dict):
        return dict(node)
    if hasattr(node, "_asdict"):
        return dict(node._asdict())
    if hasattr(node, "__dict__"):
        return {
            key: value
            for key, value in node.__dict__.items()
            if not key.startswith("_")
        }
    return {}


def _import_make_pre_post_processors() -> Callable[..., tuple[Callable, Callable]]:
    """Import LeRobot pre/post processors lazily to keep robotics deps optional."""
    try:
        from lerobot.policies.factory import make_pre_post_processors
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "LeRobot trainer requires optional LeRobot / SmolVLA robotics dependencies. "
            "Install the robotics stack in the active environment before running LeRobot workloads."
        ) from exc

    return make_pre_post_processors


def _move_to_device(value: Any, device: torch.device | str) -> Any:
    """Recursively move tensors inside nested containers to target device."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if hasattr(value, "to"):
        try:
            return value.to(device)
        except (TypeError, AttributeError):
            return value
    return value


def _resolve_batch_size(value: Any) -> int | None:
    """Infer batch size from the first tensor-like value in a nested structure."""
    if torch.is_tensor(value):
        if value.ndim == 0:
            return 1
        return int(value.shape[0])
    if isinstance(value, Mapping):
        for nested in value.values():
            size = _resolve_batch_size(nested)
            if size is not None:
                return size
    if isinstance(value, list):
        return len(value) if value else None
    return None


def _collate_values(values: list[Any]) -> Any:
    """Collate values with a robust fallback for heterogeneous metadata fields."""
    if not values:
        return values
    if any(value is None for value in values):
        return list(values)

    if all(isinstance(value, Mapping) for value in values):
        keys: list[str] = []
        for value in values:
            for key in value:
                if key not in keys:
                    keys.append(str(key))
        return {
            key: _collate_values(
                [cast(Mapping[str, Any], value).get(key) for value in values]
            )
            for key in keys
        }

    try:
        return default_collate(values)
    except (TypeError, RuntimeError):
        return list(values)


def _extract_tensor_label(label: Any) -> torch.Tensor | None:
    """Extract a tensor label from common LeRobot sample target structures."""
    if torch.is_tensor(label):
        return label
    if isinstance(label, Mapping):
        action = label.get("action")
        if torch.is_tensor(action):
            return action
    return None


class LeRobotBatch(dict):
    """Dictionary batch wrapper that supports `.to(device)`."""

    def to(self, device: torch.device | str):
        for key, value in list(self.items()):
            self[key] = _move_to_device(value, device)
        return self


class LeRobotCollateWrapper:
    """Collate LeRobot dict samples into `(inputs, labels)` for ComposableTrainer."""

    def __call__(
        self,
        examples: Iterable[Any],
    ) -> tuple[LeRobotBatch, torch.Tensor]:
        example_list = list(examples)
        if not example_list:
            raise ValueError("LeRobot collate received an empty batch.")

        normalized_inputs: list[dict[str, Any]] = []
        raw_labels: list[Any] = []

        for sample in example_list:
            if isinstance(sample, Mapping):
                sample_dict = dict(sample)
                label = sample_dict.get("plato_targets")
                if label is None:
                    label = sample_dict.get("action")

                payload = {
                    key: value
                    for key, value in sample_dict.items()
                    if key not in _RESERVED_KEYS
                }

                if not payload:
                    plato_inputs = sample_dict.get("plato_inputs")
                    if isinstance(plato_inputs, Mapping):
                        payload = dict(plato_inputs)

                if "action" not in payload and torch.is_tensor(label):
                    payload["action"] = label

                normalized_inputs.append(payload)
                raw_labels.append(label)
            else:
                normalized_inputs.append({"observation": sample})
                raw_labels.append(None)

        batched_inputs = _collate_values(normalized_inputs)
        if not isinstance(batched_inputs, Mapping):
            batched_inputs = {"inputs": batched_inputs}
        batch_dict = LeRobotBatch(dict(batched_inputs))

        tensor_labels = [_extract_tensor_label(label) for label in raw_labels]
        if tensor_labels and all(label is not None for label in tensor_labels):
            labels = _collate_values(cast(list[Any], tensor_labels))
            if torch.is_tensor(labels):
                return batch_dict, labels

        action_tensor = batch_dict.get("action")
        if torch.is_tensor(action_tensor):
            return batch_dict, action_tensor

        batch_size = _resolve_batch_size(batch_dict)
        if batch_size is None:
            batch_size = len(example_list)
        labels = torch.zeros(batch_size, dtype=torch.float32)
        return batch_dict, labels


def _resolve_policy_forward(
    model: nn.Module,
    batch: Mapping[str, Any],
    reduction: str,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    """Call the policy and normalize output into `(loss_tensor, loss_dict)`."""
    forward_result = model.forward(batch, reduction=reduction)

    if torch.is_tensor(forward_result):
        return forward_result, {}

    if isinstance(forward_result, tuple):
        if len(forward_result) == 0:
            raise ValueError("LeRobot policy forward returned an empty tuple.")

        first = forward_result[0]
        second = forward_result[1] if len(forward_result) > 1 else {}

        if torch.is_tensor(first):
            loss_dict = second if isinstance(second, Mapping) else {}
            return first, cast(Mapping[str, Any], loss_dict)

        if torch.is_tensor(second):
            loss_dict = first if isinstance(first, Mapping) else {}
            return second, cast(Mapping[str, Any], loss_dict)

        if isinstance(first, Mapping):
            maybe_loss = first.get("loss")
            if torch.is_tensor(maybe_loss):
                return maybe_loss, first

    if isinstance(forward_result, Mapping):
        maybe_loss = forward_result.get("loss")
        if torch.is_tensor(maybe_loss):
            return maybe_loss, forward_result

    raise TypeError(
        "LeRobot policy forward must return a tensor loss or a tuple containing "
        "a tensor loss. Received: "
        f"{type(forward_result)}."
    )


def _apply_preprocessor(
    batch: LeRobotBatch,
    context: TrainingContext,
) -> LeRobotBatch:
    """Apply the optional LeRobot preprocessor."""
    preprocessor = context.state.get("lerobot_preprocessor")
    if preprocessor is None:
        return batch

    processed = preprocessor(dict(batch))
    if not isinstance(processed, Mapping):
        raise TypeError(
            "LeRobot preprocessor must return a mapping batch. "
            f"Received {type(processed)}."
        )
    return LeRobotBatch(dict(processed))


def _summarize_loss_dict(loss_dict: Mapping[str, Any]) -> dict[str, float]:
    """Store scalar loss dictionary values for debugging/monitoring hooks."""
    summary: dict[str, float] = {}
    for key, value in loss_dict.items():
        if torch.is_tensor(value):
            try:
                summary[str(key)] = float(value.detach().cpu().item())
            except (RuntimeError, ValueError):
                continue
        elif isinstance(value, (int, float)):
            summary[str(key)] = float(value)
    return summary


def _resolve_sampler_for_loader(sampler: Any) -> Any:
    """Resolve a sampler config value to a torch DataLoader sampler object."""
    if sampler is None:
        return None
    if isinstance(sampler, torch.utils.data.Sampler):
        return sampler
    if isinstance(sampler, (list, range)):
        return torch.utils.data.SubsetRandomSampler(sampler)
    if hasattr(sampler, "get"):
        return sampler.get()
    return sampler


def _resolve_precision(precision: Any) -> str:
    """Normalize policy precision values from config."""
    if precision is None:
        return "fp32"
    if not isinstance(precision, str):
        raise TypeError("`parameters.policy.precision` must be a string.")

    normalized = precision.strip().lower()
    if normalized not in _SUPPORTED_POLICY_PRECISIONS:
        supported = ", ".join(sorted(_SUPPORTED_POLICY_PRECISIONS))
        raise ValueError(
            "Unsupported `parameters.policy.precision` value "
            f"'{precision}'. Expected one of: {supported}."
        )
    return normalized


def _resolve_runtime_device(device_value: Any, fallback_device: Any) -> torch.device:
    """
    Resolve runtime device from policy config, falling back to trainer default.

    Raises explicit errors when a requested accelerator is unavailable so users
    can detect mismatched config/environment early.
    """
    if isinstance(fallback_device, torch.device):
        fallback = fallback_device
    else:
        fallback = torch.device(str(fallback_device))

    if device_value is None:
        return fallback
    if not isinstance(device_value, str):
        raise TypeError("`parameters.policy.device` must be a string.")

    normalized = device_value.strip().lower()
    if not normalized:
        return fallback

    if normalized == "cpu":
        return torch.device("cpu")

    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "`parameters.policy.device` is set to 'cuda' but CUDA is not "
                "available on this host."
            )
        return torch.device("cuda:0")

    if normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"`parameters.policy.device` is set to '{device_value}' but CUDA "
                "is not available on this host."
            )
        try:
            gpu_index = int(normalized.split(":", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Invalid CUDA device value: '{device_value}'."
            ) from exc
        if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"`parameters.policy.device` requested CUDA device {gpu_index}, "
                f"but only {torch.cuda.device_count()} device(s) are available."
            )
        return torch.device(normalized)

    if normalized == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError(
                "`parameters.policy.device` is set to 'mps' but MPS is not "
                "available on this host."
            )
        return torch.device("mps")

    raise ValueError(
        "Unsupported `parameters.policy.device` value "
        f"'{device_value}'. Expected cpu, cuda[:index], or mps."
    )


def _autocast_context(
    context: TrainingContext,
) -> tuple[contextlib.AbstractContextManager[Any], bool]:
    """Resolve an autocast context from runtime precision and device settings."""
    precision = str(context.state.get("lerobot_precision", "fp32")).lower()
    device = context.device

    if not isinstance(device, torch.device):
        device = torch.device(str(device))

    if precision == "fp32":
        return contextlib.nullcontext(), False

    if device.type == "cuda":
        if not torch.cuda.is_available():
            if not context.state.get("lerobot_precision_warning_emitted"):
                logging.warning(
                    "LeRobot precision '%s' requested, but CUDA is unavailable. "
                    "Falling back to fp32 execution.",
                    precision,
                )
                context.state["lerobot_precision_warning_emitted"] = True
            return contextlib.nullcontext(), False

        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype), True

    if device.type == "cpu":
        if precision == "bf16":
            return torch.autocast(device_type="cpu", dtype=torch.bfloat16), True
        if not context.state.get("lerobot_precision_warning_emitted"):
            logging.warning(
                "LeRobot precision '%s' is not supported on CPU autocast. "
                "Falling back to fp32 execution.",
                precision,
            )
            context.state["lerobot_precision_warning_emitted"] = True
        return contextlib.nullcontext(), False

    if device.type == "mps":
        if precision == "fp16":
            return torch.autocast(device_type="mps", dtype=torch.float16), True
        if not context.state.get("lerobot_precision_warning_emitted"):
            logging.warning(
                "LeRobot precision '%s' is not supported on MPS autocast. "
                "Falling back to fp32 execution.",
                precision,
            )
            context.state["lerobot_precision_warning_emitted"] = True
        return contextlib.nullcontext(), False

    if not context.state.get("lerobot_precision_warning_emitted"):
        logging.warning(
            "LeRobot precision '%s' is not supported on device type '%s'. "
            "Falling back to fp32 execution.",
            precision,
            device.type,
        )
        context.state["lerobot_precision_warning_emitted"] = True
    return contextlib.nullcontext(), False


class LeRobotTrainingStepStrategy(TrainingStepStrategy):
    """Training step strategy for LeRobot policies with dict-style batches."""

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction

    def training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        examples: torch.Tensor | Mapping[str, Any],
        labels: torch.Tensor,  # pylint: disable=unused-argument
        loss_criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        context: TrainingContext,
    ) -> torch.Tensor:
        optimizer.zero_grad()
        del labels, loss_criterion

        if not isinstance(examples, Mapping):
            raise TypeError(
                "LeRobot training expects dictionary-style batches. "
                f"Received {type(examples).__name__}."
            )

        autocast_guard, autocast_enabled = _autocast_context(context)
        context.state["lerobot_autocast_enabled"] = autocast_enabled
        with autocast_guard:
            batch = _apply_preprocessor(LeRobotBatch(dict(examples)), context)
            loss, loss_dict = _resolve_policy_forward(
                model,
                batch,
                reduction=self.reduction,
            )

        if not torch.is_tensor(loss):
            raise TypeError(
                "LeRobot policy forward did not return a tensor loss."
            )

        loss.backward()
        optimizer.step()

        context.state["optimizer_step_completed"] = True
        context.state["lerobot_loss_dict"] = _summarize_loss_dict(loss_dict)
        return loss.detach()


class LeRobotTestingStrategy(TestingStrategy):
    """Compute a stable average evaluation loss for regression checks."""

    def __init__(self, collate_fn: LeRobotCollateWrapper, reduction: str = "mean"):
        self.collate_fn = collate_fn
        self.reduction = reduction

    def test_model(
        self,
        model: nn.Module,
        config: dict[str, Any],
        testset,
        sampler,
        context: TrainingContext,
    ) -> float:
        batch_size = int(config.get("batch_size", 1))
        sampler_obj = _resolve_sampler_for_loader(sampler)

        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler_obj,
            collate_fn=self.collate_fn,
        )

        model.to(context.device)
        model.eval()
        context.state["eval_loader"] = test_loader

        total_loss = 0.0
        total_weight = 0

        autocast_guard, autocast_enabled = _autocast_context(context)
        context.state["lerobot_autocast_enabled"] = autocast_enabled
        with torch.no_grad(), autocast_guard:
            for examples, labels in test_loader:
                examples = examples.to(context.device)
                labels = labels.to(context.device)

                batch = _apply_preprocessor(examples, context)
                loss, _ = _resolve_policy_forward(
                    model,
                    batch,
                    reduction=self.reduction,
                )

                batch_weight = int(labels.size(0))
                if batch_weight <= 0:
                    inferred = _resolve_batch_size(batch)
                    batch_weight = inferred if inferred is not None else 1

                total_loss += float(loss.detach().item()) * batch_weight
                total_weight += batch_weight

        model.train()
        context.state.pop("eval_loader", None)

        if total_weight == 0:
            return float("inf")

        eval_loss = total_loss / total_weight
        context.state["lerobot_eval_loss"] = eval_loss
        return eval_loss


def _resolve_dataset_stats(dataset: Any) -> Any:
    """Extract dataset statistics from LeRobot datasets/subsets when available."""
    if dataset is None:
        return None

    metadata_candidates = (
        getattr(dataset, "meta", None),
        getattr(dataset, "metadata", None),
    )
    for metadata in metadata_candidates:
        stats = getattr(metadata, "stats", None)
        if stats is not None:
            return stats

    nested_dataset = getattr(dataset, "dataset", None)
    if nested_dataset is not None and nested_dataset is not dataset:
        return _resolve_dataset_stats(nested_dataset)

    return None


class Trainer(ComposableTrainer):
    """Composable LeRobot trainer backend."""

    def __init__(self, model=None, callbacks=None):
        self._collate_wrapper = LeRobotCollateWrapper()
        self._processors_initialised = False
        self._pretrained_path = self._resolve_policy_path()
        self._runtime_precision = self._resolve_policy_precision()
        self._policy_device = self._resolve_policy_device()
        self._preprocessor_factory: Callable[..., tuple[Callable, Callable]] | None = (
            None
        )

        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=None,
            optimizer_strategy=None,
            training_step_strategy=LeRobotTrainingStepStrategy(),
            lr_scheduler_strategy=None,
            model_update_strategy=None,
            data_loader_strategy=CustomCollateFnDataLoaderStrategy(
                collate_fn=self._collate_wrapper,
                num_workers=0,
                pin_memory=True,
            ),
            testing_strategy=LeRobotTestingStrategy(self._collate_wrapper),
        )

        resolved_device = _resolve_runtime_device(self._policy_device, self.device)
        self.device = str(resolved_device)
        self.context.device = resolved_device
        self.context.state["lerobot_precision"] = self._runtime_precision
        self.context.state["lerobot_runtime_device"] = str(resolved_device)
        self.context.state["lerobot_preprocessor"] = None
        self.context.state["lerobot_postprocessor"] = None

    @staticmethod
    def _resolve_policy_path() -> str | None:
        parameters = getattr(Config(), "parameters", None)
        policy_cfg = _config_node_to_dict(getattr(parameters, "policy", None))
        candidate = policy_cfg.get("path")
        if isinstance(candidate, str):
            value = candidate.strip()
            return value if value else None
        return None

    @staticmethod
    def _resolve_policy_precision() -> str:
        parameters = getattr(Config(), "parameters", None)
        policy_cfg = _config_node_to_dict(getattr(parameters, "policy", None))
        return _resolve_precision(policy_cfg.get("precision", "fp32"))

    @staticmethod
    def _resolve_policy_device() -> str | None:
        parameters = getattr(Config(), "parameters", None)
        policy_cfg = _config_node_to_dict(getattr(parameters, "policy", None))
        candidate = policy_cfg.get("device")
        if candidate is None:
            return None
        if not isinstance(candidate, str):
            raise TypeError("`parameters.policy.device` must be a string.")
        value = candidate.strip()
        return value if value else None

    def _resolve_model_pretrained_path(self) -> str | None:
        model = self._require_model()
        model_path = getattr(model, "plato_policy_path", None)
        if isinstance(model_path, str) and model_path.strip():
            return model_path.strip()
        return self._pretrained_path

    def _ensure_pre_post_processors(self, dataset: Any) -> None:
        if self._processors_initialised:
            return

        model = self._require_model()
        policy_config = getattr(model, "config", None)
        if policy_config is None:
            raise AttributeError(
                "LeRobot trainer expects the model to expose a `config` attribute "
                "compatible with `make_pre_post_processors`."
            )

        if self._preprocessor_factory is None:
            self._preprocessor_factory = _import_make_pre_post_processors()

        dataset_stats = _resolve_dataset_stats(dataset)
        if dataset_stats is None:
            logging.warning(
                "LeRobot dataset statistics are unavailable; preprocessing will "
                "be created without explicit dataset stats."
            )

        kwargs: dict[str, Any] = {"dataset_stats": dataset_stats}
        pretrained_path = self._resolve_model_pretrained_path()
        if pretrained_path:
            kwargs["pretrained_path"] = pretrained_path

        preprocessor, postprocessor = self._preprocessor_factory(
            policy_config, **kwargs
        )
        if not callable(preprocessor) or not callable(postprocessor):
            raise TypeError(
                "LeRobot `make_pre_post_processors` must return two callables."
            )

        self.context.state["lerobot_preprocessor"] = preprocessor
        self.context.state["lerobot_postprocessor"] = postprocessor
        self._processors_initialised = True

    def train_model(self, config, trainset, sampler, **kwargs):
        self._ensure_pre_post_processors(trainset)
        self.context.state["lerobot_precision"] = self._runtime_precision
        self.context.device = torch.device(str(self.device))
        return super().train_model(config, trainset, sampler, **kwargs)

    def test_model(self, config, testset, sampler=None, **kwargs):
        self._ensure_pre_post_processors(testset)
        self.context.state["lerobot_precision"] = self._runtime_precision
        self.context.device = torch.device(str(self.device))
        return super().test_model(config, testset, sampler, **kwargs)
