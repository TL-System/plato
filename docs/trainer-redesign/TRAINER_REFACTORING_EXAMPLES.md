# Trainer Refactoring: Implementation Examples & Migration Guide

## Table of Contents

1. [Quick Start Examples](#quick-start-examples)
2. [Migration Examples](#migration-examples)
3. [Strategy Factory Patterns](#strategy-factory-patterns)
4. [Complete Algorithm Implementations](#complete-algorithm-implementations)
5. [Testing Examples](#testing-examples)
6. [Best Practices](#best-practices)

---

## Quick Start Examples

### Example 1: Using Pre-built Strategies

```python
# Before (Inheritance)
from plato.trainers import basic

class MyTrainer(basic.Trainer):
    def get_loss_criterion(self):
        return torch.nn.CrossEntropyLoss()

# After (Composition)
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies import CrossEntropyLossStrategy

trainer = ComposableTrainer(
    loss_strategy=CrossEntropyLossStrategy()
)
```

### Example 2: Combining Multiple Strategies

```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import (
    FedProxLossStrategy,
    SCAFFOLDUpdateStrategy,
    AdamOptimizerStrategy,
)

# Combine FedProx loss with SCAFFOLD control variates
trainer = ComposableTrainer(
    loss_strategy=FedProxLossStrategy(mu=0.01),
    model_update_strategy=SCAFFOLDUpdateStrategy(),
    optimizer_strategy=AdamOptimizerStrategy(lr=0.001),
)
```

### Example 3: Custom Strategy

```python
from plato.trainers.strategies.base import LossCriterionStrategy

class MyCustomLossStrategy(LossCriterionStrategy):
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.base_criterion = torch.nn.CrossEntropyLoss()

    def compute_loss(self, outputs, labels, context):
        ce_loss = self.base_criterion(outputs, labels)

        # Add custom regularization
        reg_term = self.alpha * torch.norm(
            torch.stack([p for p in context.model.parameters()])
        )

        return ce_loss + reg_term

# Use it
trainer = ComposableTrainer(
    loss_strategy=MyCustomLossStrategy(alpha=0.1)
)
```

---

## Migration Examples

### FedProx Migration

#### Before (Inheritance)

```python
# examples/customized_client_training/fedprox/fedprox_trainer.py
import torch
from plato.config import Config
from plato.trainers import basic


class FedProxLocalObjective:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.init_global_weights = self._flatten_weights(model, device)

    def _flatten_weights(self, model, device):
        weights = torch.tensor([], requires_grad=False).to(device)
        for param in model.parameters():
            weights = torch.cat((weights, torch.flatten(param)))
        return weights

    def compute_objective(self, outputs, labels):
        current_weights = self._flatten_weights(self.model, self.device)
        mu = Config().clients.proximal_term_penalty_constant or 1
        proximal_term = (mu / 2) * torch.linalg.norm(
            current_weights - self.init_global_weights, ord=2
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(outputs, labels) + proximal_term


class Trainer(basic.Trainer):
    def get_loss_criterion(self):
        local_obj = FedProxLocalObjective(self.model, self.device)
        return local_obj.compute_objective
```

#### After (Composition)

```python
# examples/customized_client_training/fedprox/fedprox_trainer_v2.py
from plato.trainers.strategies.algorithms import FedProxLossStrategy

# Option 1: Use in main script
from plato.trainers.composable import ComposableTrainer
from plato.clients import simple
from plato.servers import fedavg

def main():
    trainer = ComposableTrainer(
        loss_strategy=FedProxLossStrategy(mu=0.01)
    )
    client = simple.Client(trainer=trainer)
    server = fedavg.Server(trainer=trainer)
    server.run(client)

# Option 2: Create factory function
def create_fedprox_trainer(mu=0.01):
    return ComposableTrainer(
        loss_strategy=FedProxLossStrategy(mu=mu)
    )
```

### SCAFFOLD Migration

#### Before (Inheritance)

```python
# examples/customized_client_training/scaffold/scaffold_trainer.py
import copy
import torch
from plato.trainers import basic


class Trainer(basic.Trainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(model=model, callbacks=callbacks)
        self.server_control_variate = None
        self.client_control_variate = None
        self.global_model_weights = None
        self.local_steps = 0

    def train_run_start(self, config):
        self.server_control_variate = self.additional_data
        if self.client_control_variate is None:
            self.client_control_variate = {
                variate: torch.zeros(self.server_control_variate[variate].shape)
                for variate in self.server_control_variate
            }
        self.global_model_weights = copy.deepcopy(self.model.state_dict())
        self.local_steps = 0

    def train_step_end(self, config, batch=None, loss=None):
        for group in self.param_groups:
            lr = group["lr"]
            counter = 0
            for name in self.server_control_variate:
                if "weight" in name or "bias" in name:
                    server_cv = self.server_control_variate[name].to(self.device)
                    param = group["params"][counter]
                    param.data.add_(
                        torch.sub(server_cv, self.client_control_variate[name].to(self.device)),
                        alpha=lr,
                    )
                    counter += 1
        self.local_steps += 1

    def train_run_end(self, config):
        # Compute and save control variate delta
        # ... (complex logic)
```

#### After (Composition)

```python
# examples/customized_client_training/scaffold/scaffold_trainer_v2.py
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import SCAFFOLDUpdateStrategy

def main():
    trainer = ComposableTrainer(
        model_update_strategy=SCAFFOLDUpdateStrategy()
    )
    # ... rest of setup
```

### LG-FedAvg Migration

#### Before (Inheritance)

```python
# examples/personalized_fl/lgfedavg/lgfedavg_trainer.py
from plato.trainers import basic
from plato.utils import trainer_utils
from plato.config import Config


class Trainer(basic.Trainer):
    def perform_forward_and_backward_passes(self, config, examples, labels):
        # Train local layers first
        trainer_utils.freeze_model(self.model, Config().algorithm.global_layer_names)
        trainer_utils.activate_model(self.model, Config().algorithm.local_layer_names)
        super().perform_forward_and_backward_passes(config, examples, labels)

        # Train global layers second
        trainer_utils.activate_model(self.model, Config().algorithm.global_layer_names)
        trainer_utils.freeze_model(self.model, Config().algorithm.local_layer_names)
        loss = super().perform_forward_and_backward_passes(config, examples, labels)

        return loss
```

#### After (Composition)

```python
# examples/personalized_fl/lgfedavg/lgfedavg_trainer_v2.py
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import LGFedAvgStepStrategy
from plato.config import Config


def main():
    trainer = ComposableTrainer(
        training_step_strategy=LGFedAvgStepStrategy(
            global_layer_names=Config().algorithm.global_layer_names,
            local_layer_names=Config().algorithm.local_layer_names,
        )
    )
    # ... rest of setup
```

### FedDyn Migration

#### Before (Inheritance)

```python
# examples/customized_client_training/feddyn/feddyn_trainer.py
import copy
import torch
from plato.trainers import basic


class Trainer(basic.Trainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)
        self.global_model_param = None
        self.local_param_last_epoch = None

    def perform_forward_and_backward_passes(self, config, examples, labels):
        weight_list = labels / torch.sum(labels) * Config().clients.total_clients
        alpha_coef = Config().algorithm.alpha_coef or 0.01
        adaptive_alpha_coef = alpha_coef / torch.where(weight_list != 0, weight_list, 1.0)

        self.optimizer.zero_grad()
        outputs = self.model(examples)

        # Ordinary loss
        loss_task = self._loss_criterion(outputs, labels)

        # Linear penalty
        local_params = self.model.state_dict()
        loss_penalty = torch.zeros(adaptive_alpha_coef.shape).to(self.device)

        for parameter_name in local_params:
            loss_penalty += adaptive_alpha_coef * torch.sum(
                local_params[parameter_name] * (
                    -self.global_model_param[parameter_name].to(self.device)
                    + self.local_param_last_epoch[parameter_name].to(self.device)
                )
            )

        loss = loss_task + torch.sum(loss_penalty)
        loss.backward()
        self.optimizer.step()

        return loss
```

#### After (Composition)

```python
# examples/customized_client_training/feddyn/feddyn_trainer_v2.py
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import (
    FedDynLossStrategy,
    FedDynUpdateStrategy,
)


def main():
    trainer = ComposableTrainer(
        loss_strategy=FedDynLossStrategy(alpha_coef=0.01),
        model_update_strategy=FedDynUpdateStrategy(),
    )
    # ... rest of setup
```

---

## Strategy Factory Patterns

### Factory for Common Configurations

```python
# plato/trainers/strategies/factories.py

from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import *


class TrainerFactory:
    """Factory for creating trainers with common strategy combinations."""

    @staticmethod
    def create_fedprox_trainer(mu=0.01, **kwargs):
        """Create a FedProx trainer."""
        return ComposableTrainer(
            loss_strategy=FedProxLossStrategy(mu=mu),
            **kwargs
        )

    @staticmethod
    def create_scaffold_trainer(**kwargs):
        """Create a SCAFFOLD trainer."""
        return ComposableTrainer(
            model_update_strategy=SCAFFOLDUpdateStrategy(),
            **kwargs
        )

    @staticmethod
    def create_feddyn_trainer(alpha_coef=0.01, **kwargs):
        """Create a FedDyn trainer."""
        return ComposableTrainer(
            loss_strategy=FedDynLossStrategy(alpha_coef=alpha_coef),
            model_update_strategy=FedDynUpdateStrategy(),
            **kwargs
        )

    @staticmethod
    def create_lgfedavg_trainer(global_layers, local_layers, **kwargs):
        """Create a LG-FedAvg trainer."""
        return ComposableTrainer(
            training_step_strategy=LGFedAvgStepStrategy(
                global_layer_names=global_layers,
                local_layer_names=local_layers,
            ),
            **kwargs
        )

    @staticmethod
    def create_personalized_fl_trainer(
        personalization_layers,
        freeze_after_round=None,
        **kwargs
    ):
        """Create a trainer for personalized FL (FedPer, FedRep, etc.)."""
        return ComposableTrainer(
            model_update_strategy=PersonalizedFLStrategy(
                personalization_layers=personalization_layers,
                freeze_after_round=freeze_after_round,
            ),
            **kwargs
        )

    @staticmethod
    def create_hybrid_trainer(
        loss_strategy=None,
        optimizer_strategy=None,
        training_step_strategy=None,
        model_update_strategy=None,
        **kwargs
    ):
        """Create a trainer with custom combination of strategies."""
        return ComposableTrainer(
            loss_strategy=loss_strategy,
            optimizer_strategy=optimizer_strategy,
            training_step_strategy=training_step_strategy,
            model_update_strategy=model_update_strategy,
            **kwargs
        )


# Usage examples
def example_usage():
    # FedProx
    trainer1 = TrainerFactory.create_fedprox_trainer(mu=0.01)

    # SCAFFOLD
    trainer2 = TrainerFactory.create_scaffold_trainer()

    # FedDyn
    trainer3 = TrainerFactory.create_feddyn_trainer(alpha_coef=0.02)

    # Hybrid: FedProx + SCAFFOLD
    trainer4 = TrainerFactory.create_hybrid_trainer(
        loss_strategy=FedProxLossStrategy(mu=0.01),
        model_update_strategy=SCAFFOLDUpdateStrategy(),
    )
```

### Builder Pattern for Complex Configurations

```python
# plato/trainers/strategies/builders.py

from typing import Optional
from plato.trainers.composable import ComposableTrainer


class TrainerBuilder:
    """Builder pattern for creating trainers with fluent API."""

    def __init__(self):
        self._model = None
        self._callbacks = None
        self._loss_strategy = None
        self._optimizer_strategy = None
        self._training_step_strategy = None
        self._lr_scheduler_strategy = None
        self._model_update_strategy = None
        self._data_loader_strategy = None

    def with_model(self, model):
        """Set the model."""
        self._model = model
        return self

    def with_callbacks(self, callbacks):
        """Set callbacks."""
        self._callbacks = callbacks
        return self

    def with_loss_strategy(self, strategy):
        """Set loss computation strategy."""
        self._loss_strategy = strategy
        return self

    def with_optimizer_strategy(self, strategy):
        """Set optimizer strategy."""
        self._optimizer_strategy = strategy
        return self

    def with_training_step_strategy(self, strategy):
        """Set training step strategy."""
        self._training_step_strategy = strategy
        return self

    def with_lr_scheduler_strategy(self, strategy):
        """Set LR scheduler strategy."""
        self._lr_scheduler_strategy = strategy
        return self

    def with_model_update_strategy(self, strategy):
        """Set model update strategy."""
        self._model_update_strategy = strategy
        return self

    def with_data_loader_strategy(self, strategy):
        """Set data loader strategy."""
        self._data_loader_strategy = strategy
        return self

    def build(self) -> ComposableTrainer:
        """Build the trainer."""
        return ComposableTrainer(
            model=self._model,
            callbacks=self._callbacks,
            loss_strategy=self._loss_strategy,
            optimizer_strategy=self._optimizer_strategy,
            training_step_strategy=self._training_step_strategy,
            lr_scheduler_strategy=self._lr_scheduler_strategy,
            model_update_strategy=self._model_update_strategy,
            data_loader_strategy=self._data_loader_strategy,
        )


# Usage
from plato.trainers.strategies.algorithms import *

trainer = (
    TrainerBuilder()
    .with_loss_strategy(FedProxLossStrategy(mu=0.01))
    .with_optimizer_strategy(AdamOptimizerStrategy(lr=0.001))
    .with_model_update_strategy(SCAFFOLDUpdateStrategy())
    .with_callbacks([MyCustomCallback()])
    .build()
)
```

---

## Complete Algorithm Implementations

### FedProx Strategy (Complete)

```python
# plato/trainers/strategies/implementations/fedprox_strategy.py

import torch
import torch.nn as nn
from typing import Optional
from plato.trainers.strategies.base import LossCriterionStrategy, TrainingContext
from plato.config import Config


class FedProxLossStrategy(LossCriterionStrategy):
    """
    FedProx loss with proximal term to handle system heterogeneity.

    Reference:
    Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020.
    https://proceedings.mlsys.org/paper/2020/hash/38af86134b65d0f10fe33d30dd76442e-Abstract.html

    Loss = L(w) + (mu/2) * ||w - w_global||^2

    where:
    - L(w) is the base loss (e.g., cross-entropy)
    - w is the current model parameters
    - w_global is the global model parameters at start of training
    - mu is the proximal term penalty constant
    """

    def __init__(
        self,
        mu: float = 0.01,
        base_loss_fn: Optional[nn.Module] = None,
    ):
        """
        Initialize FedProx loss strategy.

        Args:
            mu: Proximal term penalty constant (default: 0.01)
            base_loss_fn: Base loss function (default: CrossEntropyLoss)
        """
        self.mu = mu
        self.base_loss_fn = base_loss_fn
        self._criterion = None
        self.global_weights = None

    def setup(self, context: TrainingContext):
        """Initialize base loss function."""
        if self.base_loss_fn is None:
            # Default to cross-entropy if not specified
            self._criterion = nn.CrossEntropyLoss()
        else:
            self._criterion = self.base_loss_fn

    def compute_loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        context: TrainingContext
    ) -> torch.Tensor:
        """
        Compute FedProx loss with proximal term.

        Args:
            outputs: Model outputs
            labels: Ground truth labels
            context: Training context with model and device

        Returns:
            Total loss (base loss + proximal term)
        """
        # Compute base loss
        base_loss = self._criterion(outputs, labels)

        # On first call, save global weights
        if self.global_weights is None:
            self.global_weights = {
                name: param.clone().detach()
                for name, param in context.model.named_parameters()
            }

        # Compute proximal term: (mu/2) * ||w - w_global||^2
        proximal_term = 0.0
        for name, param in context.model.named_parameters():
            if name in self.global_weights:
                proximal_term += torch.sum(
                    (param - self.global_weights[name].to(param.device)) ** 2
                )

        proximal_term = (self.mu / 2.0) * proximal_term

        return base_loss + proximal_term

    def teardown(self, context: TrainingContext):
        """Clean up at end of training."""
        self.global_weights = None
```

### SCAFFOLD Strategy (Complete)

```python
# plato/trainers/strategies/implementations/scaffold_strategy.py

import torch
import copy
import logging
import pickle
import os
from collections import OrderedDict
from typing import Dict, Any
from plato.trainers.strategies.base import ModelUpdateStrategy, TrainingContext
from plato.config import Config


class SCAFFOLDUpdateStrategy(ModelUpdateStrategy):
    """
    SCAFFOLD control variate strategy for variance reduction.

    Reference:
    Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging
    for Federated Learning", ICML 2020.
    https://arxiv.org/abs/1910.06378

    SCAFFOLD maintains control variates (c_i for client, c for server) to
    correct for client drift. The client update rule becomes:

    x_i ← x_i - η∇F_i(x_i) - η(c - c_i)

    After local training:
    c_i^new = c - (1/(ητ))(x_local - x_global)

    where:
    - η is the learning rate
    - τ is the number of local steps
    - x_global is the global model at start
    - x_local is the local model at end
    """

    def __init__(self, load_saved_cv: bool = True):
        """
        Initialize SCAFFOLD strategy.

        Args:
            load_saved_cv: Whether to load saved client control variate
        """
        self.load_saved_cv = load_saved_cv
        self.server_control_variate = None
        self.client_control_variate = None
        self.global_model_weights = None
        self.local_steps = 0
        self.learning_rate = None
        self.client_cv_path = None

    def setup(self, context: TrainingContext):
        """Setup file paths for saving/loading control variates."""
        model_path = Config().params["model_path"]
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        self.client_cv_path = os.path.join(
            model_path,
            f"scaffold_cv_client_{context.client_id}.pkl"
        )

        # Try to load existing client control variate
        if self.load_saved_cv and os.path.exists(self.client_cv_path):
            try:
                with open(self.client_cv_path, 'rb') as f:
                    self.client_control_variate = pickle.load(f)
                logging.info(
                    "[Client #%d] Loaded saved control variate from %s",
                    context.client_id,
                    self.client_cv_path
                )
            except Exception as e:
                logging.warning(
                    "[Client #%d] Failed to load control variate: %s",
                    context.client_id,
                    str(e)
                )

    def on_train_start(self, context: TrainingContext):
        """
        Initialize control variates and save global model weights.

        Called at the start of each training round.
        """
        # Get server control variate from context (sent by server)
        self.server_control_variate = context.state.get('server_control_variate')

        if self.server_control_variate is None:
            logging.warning(
                "[Client #%d] No server control variate received, "
                "initializing to zero",
                context.client_id
            )
            self.server_control_variate = {
                name: torch.zeros_like(param)
                for name, param in context.model.named_parameters()
            }

        # Initialize client control variate to zero if first participation
        if self.client_control_variate is None:
            self.client_control_variate = {
                name: torch.zeros_like(param)
                for name, param in context.model.named_parameters()
            }
            logging.info(
                "[Client #%d] Initialized client control variate to zero",
                context.client_id
            )

        # Save global model weights for computing control variate update
        self.global_model_weights = copy.deepcopy(context.model.state_dict())

        # Reset local step counter
        self.local_steps = 0

        # Get learning rate from config
        self.learning_rate = context.config.get('lr', Config().trainer.lr)

    def after_step(self, context: TrainingContext):
        """
        Apply control variate correction after each optimizer step.

        This is the key SCAFFOLD correction:
        x ← x - η(c - c_i)
        """
        if self.server_control_variate is None:
            return

        with torch.no_grad():
            for name, param in context.model.named_parameters():
                if name in self.server_control_variate:
                    # Compute correction: c - c_i
                    correction = (
                        self.server_control_variate[name].to(param.device)
                        - self.client_control_variate[name].to(param.device)
                    )

                    # Apply correction: x ← x - η(c - c_i)
                    param.data.sub_(correction, alpha=self.learning_rate)

        self.local_steps += 1

    def on_train_end(self, context: TrainingContext):
        """
        Compute new client control variate and delta to send to server.

        Formula: c_i^new = c - (1/(ητ))(x_local - x_global)
        Delta: Δc_i = c_i^new - c_i^old
        """
        eta = self.learning_rate
        tau = max(1, self.local_steps)

        new_client_cv = OrderedDict()
        delta_cv = OrderedDict()

        for name, param in context.model.named_parameters():
            # Get weights
            x_global = self.global_model_weights[name]
            x_local = param.data
            c_old = self.client_control_variate[name]
            c_server = self.server_control_variate[name]

            # Compute new client control variate
            # c_i^new = c - (x_local - x_global) / (η * τ)
            c_new = c_server.to(param.device) - (
                (x_local - x_global.to(param.device)) / (eta * tau)
            )

            # Compute delta for server
            delta = c_new - c_old.to(param.device)

            # Store (move to CPU to save memory)
            new_client_cv[name] = c_new.detach().cpu()
            delta_cv[name] = delta.detach().cpu()

        # Update stored client control variate
        self.client_control_variate = new_client_cv

        # Save to disk for next round
        try:
            with open(self.client_cv_path, 'wb') as f:
                pickle.dump(self.client_control_variate, f)
            logging.info(
                "[Client #%d] Saved control variate to %s",
                context.client_id,
                self.client_cv_path
            )
        except Exception as e:
            logging.error(
                "[Client #%d] Failed to save control variate: %s",
                context.client_id,
                str(e)
            )

        # Store delta in context for sending to server
        context.state['scaffold_delta_cv'] = delta_cv

        logging.info(
            "[Client #%d] SCAFFOLD: local_steps=%d, lr=%.6f",
            context.client_id,
            self.local_steps,
            eta
        )

    def get_update_payload(self, context: TrainingContext) -> Dict[str, Any]:
        """
        Return control variate delta to send to server.

        The server will aggregate these deltas to update global control variate:
        c^{t+1} = c^t + (1/K) Σ Δc_i
        """
        return {
            'scaffold_delta_cv': context.state.get('scaffold_delta_cv'),
            'local_steps': self.local_steps,
        }

    def teardown(self, context: TrainingContext):
        """Clean up at end of training."""
        # Control variates are preserved across rounds,
        # so we don't clear them here
        pass
```

### LG-FedAvg Strategy (Complete)

```python
# plato/trainers/strategies/implementations/lgfedavg_strategy.py

import torch
from typing import List
from plato.trainers.strategies.base import TrainingStepStrategy, TrainingContext


class LGFedAvgStepStrategy(TrainingStepStrategy):
    """
    LG-FedAvg training step with dual forward/backward passes.

    Reference:
    Liang et al., "Think Locally, Act Globally: Federated Learning with
    Local and Global Representations", NeurIPS 2020.
    https://arxiv.org/abs/2001.01523

    LG-FedAvg splits the model into:
    - Global layers: Shared across all clients (e.g., feature extractor)
    - Local layers: Personalized per client (e.g., classifier head)

    Training performs two forward/backward passes per batch:
    1. Freeze global layers, train local layers
    2. Freeze local layers, train global layers
    """

    def __init__(
        self,
        global_layer_names: List[str],
        local_layer_names: List[str]
    ):
        """
        Initialize LG-FedAvg strategy.

        Args:
            global_layer_names: List of layer name patterns for global layers
            local_layer_names: List of layer name patterns for local layers
        """
        self.global_layer_names = global_layer_names
        self.local_layer_names = local_layer_names

    def _set_requires_grad(
        self,
        model: torch.nn.Module,
        layer_names: List[str],
        requires_grad: bool
    ):
        """
        Enable/disable gradients for specific layers.

        Args:
            model: The model
            layer_names: List of layer name patterns
            requires_grad: Whether to enable gradients
        """
        for name, param in model.named_parameters():
            # Check if parameter name matches any pattern
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = requires_grad

    def training_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        examples: torch.Tensor,
        labels: torch.Tensor,
        loss_criterion: callable,
        context: TrainingContext
    ) -> torch.Tensor:
        """
        Perform LG-FedAvg training step with two passes.

        Args:
            model: The model to train
            optimizer: The optimizer
            examples: Input batch
            labels: Target labels
            loss_criterion: Loss function
            context: Training context

        Returns:
            Loss from the second pass (global layers)
        """
        # First pass: Train local layers only
        self._set_requires_grad(model, self.global_layer_names, False)
        self._set_requires_grad(model, self.local_layer_names, True)

        optimizer.zero_grad()
        outputs = model(examples)
        loss_local = loss_criterion(outputs, labels)
        loss_local.backward()
