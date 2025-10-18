"""
An example for running Plato with custom clients.

To run this example:

cd examples/customized
uv run custom_client.py -c client.yml -i <client_id>
"""

import asyncio
import logging
from contextlib import contextmanager
from functools import partial
from typing import Callable

import socketio
import torch
from socketio.exceptions import ConnectionError as SocketIOConnectionError
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from plato.clients import simple
from plato.clients.composable import ComposableClientEvents
from plato.config import Config
from plato.datasources import base
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.base import (
    TestingStrategy,
    TrainingContext,
    TrainingStepStrategy,
)
from plato.trainers.strategies.loss_criterion import CrossEntropyLossStrategy
from plato.trainers.strategies.optimizer import AdamOptimizerStrategy


class DataSource(base.DataSource):
    """A custom datasource with custom training and validation datasets."""

    def __init__(self):
        super().__init__()

        self.trainset = MNIST("./data", train=True, download=True, transform=ToTensor())
        self.testset = MNIST("./data", train=False, download=True, transform=ToTensor())


class MNISTTrainingStepStrategy(TrainingStepStrategy):
    """Custom training step that flattens MNIST images and prints the loss."""

    def training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        examples: torch.Tensor,
        labels: torch.Tensor,
        loss_criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        context: TrainingContext,
    ) -> torch.Tensor:
        """Perform a single MNIST training step."""
        optimizer.zero_grad()

        flattened_examples = examples.view(examples.size(0), -1)
        outputs = model(flattened_examples)
        loss = loss_criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        print(f"train loss: {loss.item():.6f}")

        return loss


class MNISTTestingStrategy(TestingStrategy):
    """Testing strategy that flattens MNIST images before evaluation."""

    def test_model(self, model, config, testset, sampler, context):
        """Evaluate the model with flattened MNIST images."""
        batch_size = config.get("batch_size", 32)

        if sampler is not None and hasattr(sampler, "get") and callable(sampler.get):
            sampler = sampler.get()

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, sampler=sampler
        )

        model.to(context.device)
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for examples, labels in test_loader:
                examples, labels = (
                    examples.to(context.device),
                    labels.to(context.device),
                )

                flattened_examples = examples.view(examples.size(0), -1)
                outputs = model(flattened_examples)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        return accuracy


class Trainer(ComposableTrainer):
    """A custom trainer composed with MNIST-specific training and testing strategies."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=CrossEntropyLossStrategy(),
            optimizer_strategy=AdamOptimizerStrategy(lr=1e-3),
            training_step_strategy=MNISTTrainingStepStrategy(),
            testing_strategy=MNISTTestingStrategy(),
        )


class CustomClient(simple.Client):
    """An example for customizing the client."""

    def __init__(self, model=None, datasource=None, trainer=None):
        super().__init__(model=model, datasource=datasource, trainer=trainer)
        logging.info("A customized client has been initialized.")


def _ensure_client_id(client, default_id=1):
    """Ensure a client identifier is configured."""
    if client.client_id is not None:
        return

    logging.warning(
        "No client ID provided via '-i'. Defaulting to client_id=%d for this run.",
        default_id,
    )
    client.client_id = default_id
    client._sync_to_context(("client_id",))


def _run_client(client):
    """Run the client with asyncio.run and notebook fallback."""

    async def _start_client():
        await client.start_client()

    try:
        asyncio.run(_start_client())
    except RuntimeError as runtime_error:
        if "asyncio.run() cannot be called from a running event loop" not in str(
            runtime_error
        ):
            raise
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_start_client())
        finally:
            loop.close()
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
        logging.info("Client shut down gracefully (code=%s).", exc.code)


@contextmanager
def _graceful_socketio_session():
    """Temporarily disable reconnection and silence SystemExit(0) disconnects."""
    original_async_client_cls = socketio.AsyncClient
    original_on_disconnect = ComposableClientEvents.on_disconnect

    class _NoReconnectAsyncClient(original_async_client_cls):
        def __init__(self, *args, **kwargs):
            kwargs["reconnection"] = False
            super().__init__(*args, **kwargs)

    async def _on_disconnect_no_exit(self):
        try:
            await original_on_disconnect(self)
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise

    socketio.AsyncClient = _NoReconnectAsyncClient
    ComposableClientEvents.on_disconnect = _on_disconnect_no_exit
    try:
        yield
    finally:
        socketio.AsyncClient = original_async_client_cls
        ComposableClientEvents.on_disconnect = original_on_disconnect


def main():
    """
    A Plato federated learning training session using a custom client.

    To run this example:
    python examples/customized/custom_client.py -c examples/customized/client.yml -i <client_id>
    """
    model = partial(
        nn.Sequential,
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    datasource = DataSource
    trainer = Trainer

    client = CustomClient(model=model, datasource=datasource, trainer=trainer)
    client.configure()
    _ensure_client_id(client)

    try:
        with _graceful_socketio_session():
            _run_client(client)
    except SocketIOConnectionError as exc:
        server_config = getattr(Config(), "server", None)
        server_address = getattr(server_config, "address", "unknown")
        server_port = getattr(server_config, "port", "unknown")
        logging.error(
            "Unable to connect to the server at %s:%s (%s). "
            "Ensure the federated server example is running.",
            server_address,
            server_port,
            exc,
        )


if __name__ == "__main__":
    main()
