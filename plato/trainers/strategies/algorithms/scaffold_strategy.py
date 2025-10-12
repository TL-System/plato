"""
SCAFFOLD Strategy Implementation

Reference:
Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S., Stich, S., & Suresh, A. T. (2020).
"SCAFFOLD: Stochastic Controlled Averaging for Federated Learning."
Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

Paper: https://arxiv.org/pdf/1910.06378.pdf

Description:
SCAFFOLD uses control variates to correct for "client drift" in federated learning.
It maintains two control variates:
- Server control variate (c): aggregate of all client control variates
- Client control variate (c_i): local correction for each client

The algorithm modifies the local update to:
    w_{i,t+1} = w_{i,t} - η * (g_i + c - c_i)

where g_i is the local gradient. At the end of local training, clients compute:
    c_i^new = c - (1/(η*τ)) * (x_local - x_global)
    Δc_i = c_i^new - c_i

The server then updates:
    c^new = c + (1/K) * Σ Δc_i
"""

import copy
import logging
import os
import pickle
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch

from plato.config import Config
from plato.trainers.strategies.base import ModelUpdateStrategy, TrainingContext


class SCAFFOLDUpdateStrategy(ModelUpdateStrategy):
    """
    SCAFFOLD control variate strategy for federated learning.

    SCAFFOLD corrects for client drift by maintaining control variates at both
    the server and client level. These control variates act as variance reduction
    terms that guide local training.

    The strategy implements the client-side logic:
    1. Receive server control variate at start of training
    2. Apply corrections after each training step
    3. Compute updated client control variate at end of training
    4. Return control variate delta to server

    Args:
        save_path: Optional custom path to save client control variates.
                   If None, uses Config().params["model_path"]

    Attributes:
        server_control_variate: Control variate from server (c)
        client_control_variate: Local control variate (c_i)
        global_model_weights: Snapshot of global model at start of round
        local_steps: Number of local training steps performed
        learning_rate: Learning rate used for local training
        client_control_variate_path: Path to save client control variate

    Example:
        >>> from plato.trainers.composable import ComposableTrainer
        >>> from plato.trainers.strategies.algorithms import SCAFFOLDUpdateStrategy
        >>>
        >>> # Basic usage
        >>> trainer = ComposableTrainer(
        ...     model_update_strategy=SCAFFOLDUpdateStrategy()
        ... )
        >>>
        >>> # The server should send control variates in context.state:
        >>> # context.state['server_control_variate'] = server_cv
        >>>
        >>> # After training, retrieve the delta:
        >>> # delta = trainer.get_update_payload(context)
        >>> # delta['control_variate_delta'] contains Δc_i

    Note:
        The server must implement the corresponding logic to:
        1. Send server control variate to clients
        2. Receive control variate deltas from clients
        3. Update server control variate: c += (1/K) * Σ Δc_i
    """

    def __init__(self, save_path: Optional[str] = None):
        """
        Initialize SCAFFOLD update strategy.

        Args:
            save_path: Optional custom path for saving client control variates
        """
        self.save_path = save_path
        self.server_control_variate = None
        self.client_control_variate = None
        self.global_model_weights = None
        self.local_steps = 0
        self.learning_rate = None
        self.client_control_variate_path = None

    def setup(self, context: TrainingContext) -> None:
        """
        Setup the strategy and determine save path for client control variate.

        Args:
            context: Training context with model and client_id
        """
        if self.save_path is not None:
            base_path = self.save_path
        else:
            base_path = Config().params["model_path"]

        self.client_control_variate_path = (
            f"{base_path}_scaffold_cv_{context.client_id}.pkl"
        )

        # Try to load existing client control variate from disk
        if os.path.exists(self.client_control_variate_path):
            try:
                with open(self.client_control_variate_path, "rb") as f:
                    self.client_control_variate = pickle.load(f)
                logging.info(
                    "[Client #%d] Loaded existing SCAFFOLD control variate from %s",
                    context.client_id,
                    self.client_control_variate_path,
                )
            except Exception as e:
                logging.warning(
                    "[Client #%d] Failed to load control variate: %s",
                    context.client_id,
                    str(e),
                )
                self.client_control_variate = None

    def on_train_start(self, context: TrainingContext) -> None:
        """
        Initialize control variates at the start of each training round.

        This method:
        1. Receives server control variate from context
        2. Initializes client control variate if first time
        3. Saves global model weights for later comparison
        4. Resets local step counter

        Args:
            context: Training context

        Note:
            The server should provide server_control_variate via:
            context.state['server_control_variate'] = server_cv
        """
        # Receive server control variate from context
        self.server_control_variate = context.state.get("server_control_variate")

        if self.server_control_variate is None:
            logging.warning(
                "[Client #%d] No server_control_variate found in context. "
                "Initializing with zeros.",
                context.client_id,
            )
            # Initialize server control variate with zeros
            self.server_control_variate = OrderedDict()
            for name, param in context.model.named_parameters():
                self.server_control_variate[name] = torch.zeros_like(param)

        # Initialize client control variate if first time
        if self.client_control_variate is None:
            logging.info(
                "[Client #%d] Initializing client control variate with zeros.",
                context.client_id,
            )
            self.client_control_variate = OrderedDict()
            for name, param in context.model.named_parameters():
                self.client_control_variate[name] = torch.zeros_like(param)

        # Save global model weights for Option 2 in the paper
        self.global_model_weights = copy.deepcopy(context.model.state_dict())

        # Reset local step counter
        self.local_steps = 0

        # Extract learning rate from context if available
        if "learning_rate" in context.state:
            self.learning_rate = context.state["learning_rate"]

    def after_step(self, context: TrainingContext) -> None:
        """
        Apply control variate correction after each optimizer step.

        This implements the key SCAFFOLD update:
            w_{t+1} = w_t - η * (g + c - c_i)

        The optimizer has already computed w_t - η*g, so we add:
            w += η * (c - c_i)

        Args:
            context: Training context with model and device

        Note:
            This assumes the optimizer.step() has already been called.
            The learning rate is extracted from context or config.
        """
        # Get learning rate
        if self.learning_rate is not None:
            lr = self.learning_rate
        elif "learning_rate" in context.state:
            lr = context.state["learning_rate"]
        else:
            lr = Config().trainer.lr if hasattr(Config().trainer, "lr") else 0.01

        # Apply control variate correction: w += lr * (c - c_i)
        # Only apply to weight and bias parameters (matching original implementation)
        with torch.no_grad():
            for name, param in context.model.named_parameters():
                if (
                    ("weight" in name or "bias" in name)
                    and name in self.server_control_variate
                    and name in self.client_control_variate
                ):
                    server_cv = self.server_control_variate[name].to(param.device)
                    client_cv = self.client_control_variate[name].to(param.device)

                    # Correction term
                    correction = server_cv - client_cv

                    # Debug: Check for NaN or Inf
                    if torch.isnan(correction).any() or torch.isinf(correction).any():
                        logging.error(
                            "[Client #%d] Step %d: NaN/Inf in correction for %s! "
                            "server_cv: min=%.6f max=%.6f, client_cv: min=%.6f max=%.6f",
                            context.client_id,
                            self.local_steps,
                            name,
                            server_cv.min().item(),
                            server_cv.max().item(),
                            client_cv.min().item(),
                            client_cv.max().item(),
                        )

                    # Apply correction
                    param.data.add_(correction, alpha=lr)

                    # Debug: Check param after correction
                    if self.local_steps < 3:  # Only log first few steps
                        logging.debug(
                            "[Client #%d] Step %d: Applied correction to %s: "
                            "correction range=[%.6f, %.6f], param range=[%.6f, %.6f]",
                            context.client_id,
                            self.local_steps,
                            name,
                            correction.min().item(),
                            correction.max().item(),
                            param.data.min().item(),
                            param.data.max().item(),
                        )

        # Increment local step counter
        self.local_steps += 1

    def on_train_end(self, context: TrainingContext) -> None:
        """
        Compute new client control variate and delta at end of training.

        This implements Option 2 from the paper:
            c_i^new = c - (1/(η*τ)) * (x_local - x_global)
            Δc_i = c_i^new - c_i

        where:
        - c is the server control variate
        - η is the learning rate
        - τ is the number of local steps
        - x_local is the final local model
        - x_global is the initial global model

        Args:
            context: Training context

        Note:
            The control variate delta is stored in context.state for
            retrieval by get_update_payload().
        """
        # Get learning rate
        if self.learning_rate is not None:
            eta = self.learning_rate
        elif "learning_rate" in context.state:
            eta = context.state["learning_rate"]
        else:
            eta = Config().trainer.lr if hasattr(Config().trainer, "lr") else 0.01

        # Number of local steps
        tau = max(1, self.local_steps)

        # Compute new client control variate using Option 2
        new_client_cv = OrderedDict()
        delta_cv = OrderedDict()

        for name, param in context.model.named_parameters():
            # Current local model parameter
            x_local = param.data

            # Initial global model parameter
            x_global = self.global_model_weights[name]

            # Old client control variate
            c_i_old = self.client_control_variate[name]

            # Server control variate
            c = self.server_control_variate[name].to(param.device)

            # Compute new client control variate
            # c_i^new = c - (1/(η*τ)) * (x_local - x_global)
            c_i_new = c - (x_local - x_global.to(param.device)) / (eta * tau)

            # Compute delta
            delta = c_i_new - c_i_old.to(param.device)

            # Store on CPU to save GPU memory
            new_client_cv[name] = c_i_new.detach().cpu()
            delta_cv[name] = delta.detach().cpu()

        # Update stored client control variate
        self.client_control_variate = new_client_cv

        # Save client control variate to disk
        try:
            with open(self.client_control_variate_path, "wb") as f:
                pickle.dump(self.client_control_variate, f)
            logging.info(
                "[Client #%d] Saved SCAFFOLD control variate to %s",
                context.client_id,
                self.client_control_variate_path,
            )
        except Exception as e:
            logging.error(
                "[Client #%d] Failed to save control variate: %s",
                context.client_id,
                str(e),
            )

        # Store delta in context for server
        context.state["client_control_variate_delta"] = delta_cv

    def get_update_payload(self, context: TrainingContext) -> Dict[str, Any]:
        """
        Return control variate delta to send to server.

        Args:
            context: Training context

        Returns:
            Dictionary containing 'control_variate_delta' with Δc_i

        Note:
            The server should aggregate these deltas and update its
            control variate: c += (1/K) * Σ Δc_i
        """
        return {
            "control_variate_delta": context.state.get("client_control_variate_delta")
        }

    def teardown(self, context: TrainingContext) -> None:
        """
        Cleanup resources.

        Args:
            context: Training context
        """
        # Clear large state dictionaries to free memory
        self.server_control_variate = None
        self.global_model_weights = None


class SCAFFOLDUpdateStrategyV2(SCAFFOLDUpdateStrategy):
    """
    Alternative SCAFFOLD implementation using Option 1 from the paper.

    This variant computes the client control variate using:
        c_i^new = (1/τ*η) * Σ (w_t - w_{t+1})

    This requires tracking weight updates at each step, which may use
    more memory but can be more accurate in some settings.

    Example:
        >>> trainer = ComposableTrainer(
        ...     model_update_strategy=SCAFFOLDUpdateStrategyV2()
        ... )
    """

    def __init__(self, save_path: Optional[str] = None):
        """Initialize SCAFFOLD V2 strategy."""
        super().__init__(save_path=save_path)
        self.accumulated_updates = None

    def on_train_start(self, context: TrainingContext) -> None:
        """Initialize update accumulator."""
        super().on_train_start(context)

        # Initialize accumulator for Option 1
        self.accumulated_updates = OrderedDict()
        for name, param in context.model.named_parameters():
            self.accumulated_updates[name] = torch.zeros_like(param)

    def before_step(self, context: TrainingContext) -> None:
        """Save model weights before step for Option 1 computation."""
        if not hasattr(self, "_weights_before_step"):
            self._weights_before_step = OrderedDict()

        for name, param in context.model.named_parameters():
            self._weights_before_step[name] = param.data.clone()

    def after_step(self, context: TrainingContext) -> None:
        """
        Apply control variate correction and accumulate updates for Option 1.

        Args:
            context: Training context
        """
        # First accumulate the update for Option 1
        if hasattr(self, "_weights_before_step"):
            for name, param in context.model.named_parameters():
                w_before = self._weights_before_step[name]
                w_after = param.data
                update = w_before - w_after
                self.accumulated_updates[name] += update.cpu()

        # Then apply the standard SCAFFOLD correction
        super().after_step(context)

    def on_train_end(self, context: TrainingContext) -> None:
        """
        Compute new client control variate using Option 1.

        This uses the accumulated updates:
            c_i^new = (1/(τ*η)) * Σ_t (w_t - w_{t+1})

        Args:
            context: Training context
        """
        # Get learning rate
        if self.learning_rate is not None:
            eta = self.learning_rate
        elif "learning_rate" in context.state:
            eta = context.state["learning_rate"]
        else:
            eta = Config().trainer.lr if hasattr(Config().trainer, "lr") else 0.01

        # Number of local steps
        tau = max(1, self.local_steps)

        # Compute new client control variate using Option 1
        new_client_cv = OrderedDict()
        delta_cv = OrderedDict()

        for name in self.accumulated_updates:
            # Old client control variate
            c_i_old = self.client_control_variate[name]

            # Compute new client control variate using accumulated updates
            # c_i^new = (1/(τ*η)) * Σ (w_t - w_{t+1})
            c_i_new = self.accumulated_updates[name] / (tau * eta)

            # Compute delta
            delta = c_i_new - c_i_old

            # Store
            new_client_cv[name] = c_i_new
            delta_cv[name] = delta

        # Update stored client control variate
        self.client_control_variate = new_client_cv

        # Save to disk
        try:
            with open(self.client_control_variate_path, "wb") as f:
                pickle.dump(self.client_control_variate, f)
            logging.info(
                "[Client #%d] Saved SCAFFOLD control variate to %s",
                context.client_id,
                self.client_control_variate_path,
            )
        except Exception as e:
            logging.error(
                "[Client #%d] Failed to save control variate: %s",
                context.client_id,
                str(e),
            )

        # Store delta in context
        context.state["client_control_variate_delta"] = delta_cv

        # Clear accumulator
        self.accumulated_updates = None
