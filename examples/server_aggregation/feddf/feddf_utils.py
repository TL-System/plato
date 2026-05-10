"""Shared proxy-set helpers for the FedDF server aggregation example."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from plato.config import Config


def resolve_algorithm_value(name: str, explicit_value: Any, default: Any) -> Any:
    """Resolve an example parameter from the constructor or config file."""
    if explicit_value is not None:
        return explicit_value

    algorithm_cfg = getattr(Config(), "algorithm", None)
    if algorithm_cfg is None:
        return default

    return getattr(algorithm_cfg, name, default)


def select_proxy_subset(
    dataset: Dataset,
    *,
    size: int,
    seed: int,
) -> tuple[Subset, list[int]]:
    """Build a deterministic subset of the shared proxy dataset."""
    total_examples = len(dataset)
    if total_examples == 0:
        raise ValueError("FedDF proxy dataset is empty.")

    subset_size = min(size, total_examples)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_examples, generator=generator)[:subset_size].tolist()
    indices.sort()

    return Subset(dataset, indices), indices


def unwrap_model_outputs(outputs: Any) -> torch.Tensor:
    """Normalise model outputs to a logits tensor."""
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    if not isinstance(outputs, torch.Tensor):
        raise TypeError(
            "FedDF expects the model forward pass to return a tensor or "
            f"tensor-like tuple, received {type(outputs).__name__}."
        )
    return outputs


def extract_batch_inputs(batch: Any) -> Any:
    """Return the model input tensor from a dataset batch."""
    if isinstance(batch, (tuple, list)):
        return batch[0]
    if isinstance(batch, dict):
        for key in ("input", "inputs", "image", "images", "x"):
            if key in batch:
                return batch[key]
    return batch


def collect_proxy_logits(
    model: torch.nn.Module,
    proxy_dataset: Dataset,
    *,
    batch_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Evaluate a model on the proxy set and return detached logits."""
    was_training = model.training
    device = torch.device(device)
    dataloader = DataLoader(proxy_dataset, batch_size=batch_size, shuffle=False)
    logits: list[torch.Tensor] = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            inputs = extract_batch_inputs(batch).to(device)
            batch_logits = unwrap_model_outputs(model(inputs))
            logits.append(batch_logits.detach().cpu())

    if was_training:
        model.train()

    return torch.cat(logits, dim=0)


def stack_proxy_inputs(proxy_dataset: Dataset) -> torch.Tensor:
    """Materialise the proxy inputs in their deterministic dataset order."""
    inputs = []
    for example in proxy_dataset:
        inputs.append(extract_batch_inputs(example))

    return torch.stack(inputs)
