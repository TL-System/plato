"""
MistNet-specific client strategies.

These strategies adapt the default client pipeline for MistNet, where the
client performs only a partial forward pass and returns extracted features.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from typing import Any, Tuple

from plato.clients.strategies.defaults import DefaultTrainingStrategy
from plato.clients.strategies.base import ClientContext
from plato.config import Config


class MistNetTrainingStrategy(DefaultTrainingStrategy):
    """Training strategy that extracts features instead of training locally."""

    async def train(
        self, context: ClientContext
    ) -> Tuple[Any, Any]:
        logging.info("Training on MistNet client #%d", context.client_id)

        # MistNet delegates testing to the server
        assert not Config().clients.do_test

        tic = time.perf_counter()
        features = context.algorithm.extract_features(context.trainset, context.sampler)
        training_time = time.perf_counter() - tic

        report = SimpleNamespace(
            client_id=context.client_id,
            num_samples=context.sampler.num_samples()
            if context.sampler is not None
            else 0,
            accuracy=0,
            training_time=training_time,
            comm_time=time.time(),
            update_response=False,
            payload_length=len(features),
        )

        return report, features
