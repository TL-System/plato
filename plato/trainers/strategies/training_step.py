"""
Training step strategy implementations.

This module provides default and common training step strategies for
the composable trainer architecture.
"""

from __future__ import annotations

from collections.abc import Callable
from math import ceil
from typing import List, Optional

import torch
import torch.nn as nn

from plato.trainers.strategies.base import TrainingContext, TrainingStepStrategy

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
BackwardHook = Callable[[nn.Module, torch.Tensor, TrainingContext], None]
AggregateFn = Callable[[list[torch.Tensor]], torch.Tensor]


class DefaultTrainingStepStrategy(TrainingStepStrategy):
    """
    Default training step strategy: forward -> loss -> backward -> step.

    This implements the standard training step used in most deep learning:
    1. Zero gradients
    2. Forward pass
    3. Compute loss
    4. Backward pass
    5. Optimizer step

    Args:
        create_graph: Whether to create computation graph for higher-order derivatives
        retain_graph: Whether to retain the computation graph

    Example:
        >>> strategy = DefaultTrainingStepStrategy()
        >>> trainer = ComposableTrainer(training_step_strategy=strategy)
    """

    def __init__(self, create_graph: bool = False, retain_graph: bool = False):
        """Initialize default training step parameters."""
        self.create_graph = create_graph
        self.retain_graph = retain_graph

    def training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        examples: torch.Tensor,
        labels: torch.Tensor,
        loss_criterion: LossFn,
        context: TrainingContext,
    ) -> torch.Tensor:
        """Perform standard training step."""
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(examples)

        # Compute loss
        loss = loss_criterion(outputs, labels)

        # Backward pass
        loss.backward(create_graph=self.create_graph, retain_graph=self.retain_graph)

        # Optimizer step
        optimizer.step()

        return loss


class GradientAccumulationStepStrategy(TrainingStepStrategy):
    """
    Training step strategy with gradient accumulation.

    Gradient accumulation allows effective larger batch sizes by accumulating
    gradients over multiple batches before updating weights.

    Each window averages microbatch losses equally, retaining the original
    full-window behavior even when microbatch sizes differ. A partial window
    uses its actual microbatch count, rather than the configured window size.

    Args:
        accumulation_steps: Number of steps to accumulate gradients
        create_graph: Whether to create computation graph

    Example:
        >>> strategy = GradientAccumulationStepStrategy(accumulation_steps=4)
        >>> trainer = ComposableTrainer(training_step_strategy=strategy)
    """

    def __init__(self, accumulation_steps: int = 1, create_graph: bool = False):
        """Initialize gradient accumulation parameters."""
        if isinstance(accumulation_steps, bool) or not isinstance(
            accumulation_steps, int
        ) or accumulation_steps < 1:
            raise ValueError("accumulation_steps must be a positive integer")
        self.accumulation_steps = accumulation_steps
        self.create_graph = create_graph
        self.current_step = 0
        self._loss_total = None

    def setup(self, context: TrainingContext) -> None:
        """Reset step counter on setup."""
        self.current_step = 0
        self._loss_total = None

    def on_train_start(self, context: TrainingContext) -> None:
        self.setup(context)
        if context.model is not None:
            context.model.zero_grad(set_to_none=True)
        context.state["optimizer_step_completed"] = False

    def on_train_end(self, context: TrainingContext) -> None:
        self.on_train_start(context)

    def optimizer_steps_per_epoch(self, batches: int) -> int:
        return ceil(batches / self.accumulation_steps)

    def training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        examples: torch.Tensor,
        labels: torch.Tensor,
        loss_criterion: LossFn,
        context: TrainingContext,
    ) -> torch.Tensor:
        """Perform training step with gradient accumulation."""
        context.state["optimizer_step_completed"] = False
        if self.current_step == 0:
            optimizer.zero_grad(set_to_none=True)
        # Forward pass
        outputs = model(examples)

        # Compute loss
        loss = loss_criterion(outputs, labels)

        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps

        # Backward pass
        scaled_loss.backward(create_graph=self.create_graph)

        # Increment step counter
        self.current_step += 1
        detached_loss = loss.detach()
        self._loss_total = (
            detached_loss if self._loss_total is None
            else self._loss_total + detached_loss
        )

        # Update weights every N steps
        if self.current_step == self.accumulation_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self.current_step = 0
            self._loss_total = None
            context.state["optimizer_step_completed"] = True

        # Return unscaled loss for logging
        return loss

    def finalize(self, model, optimizer, context) -> torch.Tensor | None:
        """Flush a pending window once at the epoch boundary."""
        context.state["optimizer_step_completed"] = False
        if self.current_step == 0:
            return None
        if self._loss_total is None:
            raise RuntimeError("Pending accumulation window has no recorded loss")
        scale = self.accumulation_steps / self.current_step
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
        loss = self._loss_total / self.current_step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.current_step = 0
        self._loss_total = None
        context.state["optimizer_step_completed"] = True
        return loss


class MixedPrecisionStepStrategy(TrainingStepStrategy):
    """
    Training step strategy with automatic mixed precision (AMP).

    Mixed precision training uses FP16 for faster computation while
    maintaining FP32 for numerical stability where needed.

    Args:
        enabled: Whether to enable mixed precision (auto-detects CUDA availability)
        create_graph: Whether to create computation graph

    Example:
        >>> strategy = MixedPrecisionStepStrategy(enabled=True)
        >>> trainer = ComposableTrainer(training_step_strategy=strategy)
    """

    def __init__(self, enabled: bool | None = None, create_graph: bool = False):
        """Initialize mixed precision parameters."""
        self.enabled = enabled
        self.create_graph = create_graph
        self.scaler = None

    def setup(self, context: TrainingContext) -> None:
        """Setup gradient scaler for mixed precision."""
        if self.enabled is None:
            # Auto-detect: enable if CUDA is available
            self.enabled = torch.cuda.is_available()

        if self.enabled:
            self.scaler = torch.amp.GradScaler("cuda")

    def training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        examples: torch.Tensor,
        labels: torch.Tensor,
        loss_criterion: LossFn,
        context: TrainingContext,
    ) -> torch.Tensor:
        """Perform training step with mixed precision."""
        context.state["optimizer_step_completed"] = False
        optimizer.zero_grad()

        if self.enabled and self.scaler is not None:
            # Mixed precision training
            with torch.amp.autocast("cuda", enabled=examples.device.type == "cuda"):
                outputs = model(examples)
                loss = loss_criterion(outputs, labels)

            # Scaled backward pass
            self.scaler.scale(loss).backward(create_graph=self.create_graph)

            # Public pre-step wrappers (including clipping) must see unscaled
            # gradients even when a fused optimizer normally unscales internally.
            self.scaler.unscale_(optimizer)

            # Unscale gradients and step
            # GradScaler can suppress optimizer.step on overflow. Observe the
            # actual public step hook, rather than counting scaler attempts.
            # AMP-aware fused optimizers can skip inside step itself, so the
            # scaler's public overflow/backoff signal is checked as well.
            def completed(*_):
                context.state["optimizer_step_completed"] = True

            handle = optimizer.register_step_post_hook(completed)
            scale_before = self.scaler.get_scale()
            try:
                self.scaler.step(optimizer)
            finally:
                handle.remove()
            self.scaler.update()
            if self.scaler.get_scale() < scale_before:
                context.state["optimizer_step_completed"] = False
        else:
            # Standard precision training
            outputs = model(examples)
            loss = loss_criterion(outputs, labels)
            loss.backward(create_graph=self.create_graph)
            optimizer.step()
            context.state["optimizer_step_completed"] = True

        return loss


class GradientClippingStepStrategy(TrainingStepStrategy):
    """
    Training step strategy with gradient clipping.

    Gradient clipping prevents exploding gradients by limiting the norm
    of gradients before the optimizer step.

    Args:
        max_norm: Max norm of gradients
        norm_type: Type of norm to use (2 for L2 norm)
        create_graph: Whether to create computation graph

    Example:
        >>> strategy = GradientClippingStepStrategy(max_norm=1.0)
        >>> trainer = ComposableTrainer(training_step_strategy=strategy)
    """

    def __init__(
        self, max_norm: float = 1.0, norm_type: float = 2.0, create_graph: bool = False
    ):
        """Initialize gradient clipping parameters."""
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.create_graph = create_graph

    def training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        examples: torch.Tensor,
        labels: torch.Tensor,
        loss_criterion: LossFn,
        context: TrainingContext,
    ) -> torch.Tensor:
        """Perform training step with gradient clipping."""
        optimizer.zero_grad()

        # Forward pass
        outputs = model(examples)

        # Compute loss
        loss = loss_criterion(outputs, labels)

        # Backward pass
        loss.backward(create_graph=self.create_graph)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=self.max_norm, norm_type=self.norm_type
        )

        # Optimizer step
        optimizer.step()

        return loss


class CustomBackwardStepStrategy(TrainingStepStrategy):
    """
    Training step strategy with custom backward hook.

    This allows injecting custom logic during the backward pass,
    useful for gradient modifications or custom backpropagation.

    Args:
        backward_hook: Callable that takes (model, loss, context) and performs backward
        create_graph: Whether to create computation graph

    Example:
        >>> def my_backward(model, loss, context):
        ...     loss.backward()
        ...     # Custom gradient modifications
        ...     for param in model.parameters():
        ...         if param.grad is not None:
        ...             param.grad *= 0.9  # Example: gradient dampening
        >>>
        >>> strategy = CustomBackwardStepStrategy(backward_hook=my_backward)
        >>> trainer = ComposableTrainer(training_step_strategy=strategy)
    """

    def __init__(self, backward_hook: BackwardHook, create_graph: bool = False):
        """Initialize custom backward step parameters."""
        self.backward_hook = backward_hook
        self.create_graph = create_graph

    def training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        examples: torch.Tensor,
        labels: torch.Tensor,
        loss_criterion: LossFn,
        context: TrainingContext,
    ) -> torch.Tensor:
        """Perform training step with custom backward."""
        optimizer.zero_grad()

        # Forward pass
        outputs = model(examples)

        # Compute loss
        loss = loss_criterion(outputs, labels)

        # Custom backward pass
        self.backward_hook(model, loss, context)

        # Optimizer step
        optimizer.step()

        return loss


class MultipleForwardPassStepStrategy(TrainingStepStrategy):
    """
    Training step strategy with multiple forward passes.

    Some algorithms (like certain consistency regularization methods)
    require multiple forward passes per batch with different augmentations.

    Args:
        num_passes: Number of forward passes per batch
        aggregate_fn: Function to aggregate losses from multiple passes
        create_graph: Whether to create computation graph

    Example:
        >>> def aggregate_losses(losses):
        ...     return sum(losses) / len(losses)
        >>>
        >>> strategy = MultipleForwardPassStepStrategy(
        ...     num_passes=2,
        ...     aggregate_fn=aggregate_losses
        ... )
        >>> trainer = ComposableTrainer(training_step_strategy=strategy)
    """

    def __init__(
        self,
        num_passes: int = 2,
        aggregate_fn: AggregateFn | None = None,
        create_graph: bool = False,
    ):
        """Initialize multiple forward pass parameters."""
        self.num_passes = num_passes
        self.aggregate_fn = aggregate_fn or self._default_aggregate
        self.create_graph = create_graph

    @staticmethod
    def _default_aggregate(losses: list[torch.Tensor]) -> torch.Tensor:
        """Default aggregation: mean of losses."""
        if not losses:
            raise ValueError("At least one loss tensor is required for aggregation.")
        total = losses[0]
        for loss in losses[1:]:
            total = total + loss
        return total / len(losses)

    def training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        examples: torch.Tensor,
        labels: torch.Tensor,
        loss_criterion: LossFn,
        context: TrainingContext,
    ) -> torch.Tensor:
        """Perform training step with multiple forward passes."""
        optimizer.zero_grad()

        losses: list[torch.Tensor] = []
        for _ in range(self.num_passes):
            # Forward pass
            outputs = model(examples)

            # Compute loss
            loss = loss_criterion(outputs, labels)
            losses.append(loss)

        # Aggregate losses
        total_loss = self.aggregate_fn(losses)

        # Backward pass
        total_loss.backward(create_graph=self.create_graph)

        # Optimizer step
        optimizer.step()

        return total_loss


class ValidateBeforeStepStrategy(TrainingStepStrategy):
    """
    Training step strategy that validates inputs before training.

    This adds checks for NaN/Inf values to catch numerical issues early.

    Args:
        check_inputs: Whether to check input tensors
        check_outputs: Whether to check model outputs
        check_gradients: Whether to check gradients
        raise_on_error: Whether to raise exception or just log warning
        base_strategy: Underlying training step strategy

    Example:
        >>> strategy = ValidateBeforeStepStrategy(
        ...     check_inputs=True,
        ...     check_outputs=True,
        ...     check_gradients=True
        ... )
        >>> trainer = ComposableTrainer(training_step_strategy=strategy)
    """

    def __init__(
        self,
        check_inputs: bool = True,
        check_outputs: bool = True,
        check_gradients: bool = True,
        raise_on_error: bool = False,
        base_strategy: TrainingStepStrategy | None = None,
    ):
        """Initialize validation parameters."""
        self.check_inputs = check_inputs
        self.check_outputs = check_outputs
        self.check_gradients = check_gradients
        self.raise_on_error = raise_on_error
        self.base_strategy = base_strategy or DefaultTrainingStepStrategy()

    def setup(self, context: TrainingContext) -> None:
        """Setup base strategy."""
        self.base_strategy.setup(context)

    def on_train_start(self, context: TrainingContext) -> None:
        self.base_strategy.on_train_start(context)

    def on_train_end(self, context: TrainingContext) -> None:
        self.base_strategy.on_train_end(context)

    def on_client_id_changed(self, context: TrainingContext) -> None:
        self.base_strategy.on_client_id_changed(context)

    def optimizer_steps_per_epoch(self, batches: int) -> int:
        return self.base_strategy.optimizer_steps_per_epoch(batches)

    def _check_gradients(self, model: nn.Module) -> None:
        if self.check_gradients:
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    self._check_tensor(parameter.grad, f"gradient of {name}")

    def _check_tensor(self, tensor: torch.Tensor, name: str) -> bool:
        """Check if tensor contains NaN or Inf."""
        if torch.isnan(tensor).any():
            msg = f"NaN detected in {name}"
            if self.raise_on_error:
                raise ValueError(msg)
            print(f"Warning: {msg}")
            return False

        if torch.isinf(tensor).any():
            msg = f"Inf detected in {name}"
            if self.raise_on_error:
                raise ValueError(msg)
            print(f"Warning: {msg}")
            return False

        return True

    def training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        examples: torch.Tensor,
        labels: torch.Tensor,
        loss_criterion: LossFn,
        context: TrainingContext,
    ) -> torch.Tensor:
        """Perform validated training step."""
        # Check inputs
        if self.check_inputs:
            self._check_tensor(examples, "input examples")
            self._check_tensor(labels, "input labels")

        def validated_loss(outputs, targets):
            if self.check_outputs:
                self._check_tensor(outputs, "model outputs")
            loss = loss_criterion(outputs, targets)
            self._check_tensor(loss, "loss")
            return loss

        # Keep the selected backward/accumulation/clipping behavior. The
        # optimizer hook checks gradients immediately before actual updates.
        handle = optimizer.register_step_pre_hook(
            lambda *_: self._check_gradients(model)
        )
        try:
            return self.base_strategy.training_step(
                model, optimizer, examples, labels, validated_loss, context
            )
        finally:
            handle.remove()

    def finalize(self, model, optimizer, context) -> torch.Tensor | None:
        """Validate and forward any pending accumulated update."""
        finalize = getattr(self.base_strategy, "finalize", None)
        if not callable(finalize):
            context.state["optimizer_step_completed"] = False
            return None
        handle = optimizer.register_step_pre_hook(
            lambda *_: self._check_gradients(model)
        )
        try:
            return finalize(model, optimizer, context)
        finally:
            handle.remove()

    def teardown(self, context: TrainingContext) -> None:
        """Teardown base strategy."""
        self.base_strategy.teardown(context)
