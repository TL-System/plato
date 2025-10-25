"""
Port aggregation strategy.

Applies cosine-similarity and staleness-aware weighting for asynchronous updates.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn.functional as F

from plato.config import Config
from plato.servers.strategies.base import AggregationStrategy, ServerContext


class PortAggregationStrategy(AggregationStrategy):
    """Aggregate deltas using Port's similarity and staleness weighting."""

    async def aggregate_deltas(
        self,
        updates: list[SimpleNamespace],
        deltas_received: list[dict],
        context: ServerContext,
    ) -> dict:
        if not updates or not deltas_received:
            return {}

        trainer = getattr(context, "trainer", None)
        if trainer is None:
            raise RuntimeError("Port aggregation requires an attached trainer.")

        zeros_fn: Callable[[torch.Size], torch.Tensor] | None = getattr(
            trainer, "zeros", None
        )
        if zeros_fn is None or not callable(zeros_fn):
            raise RuntimeError(
                "Port aggregation requires the trainer to provide a zeros() helper."
            )

        total_samples = sum(update.report.num_samples for update in updates)

        aggregation_weights: list[float] = []

        for index, delta in enumerate(deltas_received):
            report = updates[index].report
            staleness = getattr(updates[index], "staleness", 0)
            num_samples = report.num_samples

            similarity = await self._cosine_similarity(delta, staleness, context)
            staleness_factor = self._staleness_factor(staleness)

            similarity_weight = (
                Config().server.similarity_weight
                if hasattr(Config().server, "similarity_weight")
                else 1
            )
            staleness_weight = (
                Config().server.staleness_weight
                if hasattr(Config().server, "staleness_weight")
                else 1
            )

            logging.info("[Client %s] similarity: %s", index, (similarity + 1) / 2)
            logging.info(
                "[Client %s] staleness: %s, staleness factor: %s",
                index,
                staleness,
                staleness_factor,
            )

            sample_ratio = num_samples / total_samples if total_samples > 0 else 0.0
            raw_weight = sample_ratio * (
                (similarity + 1) / 2 * similarity_weight
                + staleness_factor * staleness_weight
            )
            logging.info("[Client %s] raw weight = %s", index, raw_weight)
            aggregation_weights.append(raw_weight)

        weight_sum = sum(aggregation_weights)
        if weight_sum == 0:
            uniform_weight = 1.0 / len(aggregation_weights)
            aggregation_weights = [uniform_weight for _ in aggregation_weights]
        else:
            aggregation_weights = [
                weight / weight_sum for weight in aggregation_weights
            ]

        logging.info(
            "[Server #%s] normalized aggregation weights: %s",
            os.getpid(),
            aggregation_weights,
        )

        avg_update = {
            name: zeros_fn(weights.shape)
            for name, weights in deltas_received[0].items()
        }

        for index, delta in enumerate(deltas_received):
            for name, value in delta.items():
                avg_update[name] += value * aggregation_weights[index]

            await asyncio.sleep(0)

        return avg_update

    async def _cosine_similarity(
        self, update_delta: dict, staleness: int, context: ServerContext
    ) -> float:
        similarity = 1.0
        current_round = getattr(context, "current_round", 0)
        filename = f"model_{current_round - 2}.pth"
        model_path = Config().params["model_path"]
        checkpoint_path = f"{model_path}/{filename}"

        if staleness > 1 and os.path.exists(checkpoint_path):
            trainer = getattr(context, "trainer", None)
            if trainer is None:
                raise RuntimeError("Port aggregation requires an attached trainer.")

            current_model = trainer.require_model()
            previous_model = copy.deepcopy(current_model)
            previous_model.load_state_dict(torch.load(checkpoint_path))

            previous = torch.zeros(0)
            for _, weight in previous_model.cpu().state_dict().items():
                previous = torch.cat((previous, weight.view(-1)))

            current = torch.zeros(0)
            for _, weight in current_model.cpu().state_dict().items():
                current = torch.cat((current, weight.view(-1)))

            deltas = torch.zeros(0)
            for _, delta in update_delta.items():
                deltas = torch.cat((deltas, delta.view(-1)))

            similarity = F.cosine_similarity(current - previous, deltas, dim=0)

        return similarity

    @staticmethod
    def _staleness_factor(staleness: int) -> float:
        staleness_bound = (
            Config().server.staleness_bound
            if hasattr(Config().server, "staleness_bound")
            else 10
        )
        return staleness_bound / (staleness + staleness_bound)
