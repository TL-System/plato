"""
A federated learning trainer for gradient leakage attacks,
where intermediate gradients can be transmitted,
and potential defense mechanisms can be applied.

This trainer has been migrated to use the new composable trainer architecture
with strategies and callbacks instead of inheritance and hooks.
"""

import math
import pickle
import random
from typing import Any, Iterable, Optional, Union, cast

import numpy as np
import torch
from defense.GradDefense.dataloader import get_root_set_loader
from defense.GradDefense.sensitivity import compute_sens
from defense.Outpost.perturb import compute_risk
from torchvision import transforms
from utils.helpers import cross_entropy_for_onehot, label_to_onehot

from plato.callbacks.trainer import TrainerCallback
from plato.config import Config
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.base import (
    DataLoaderStrategy,
    TestingStrategy,
    TrainingContext,
    TrainingStepStrategy,
)

GradientList = list[torch.Tensor]
NestedGradientList = list[GradientList]
GradientsLike = Union[GradientList, NestedGradientList]

criterion = cross_entropy_for_onehot
tt = transforms.ToPILImage()


class DLGDataLoaderStrategy(DataLoaderStrategy):
    """Custom data loader strategy that computes sensitivity for GradDefense."""

    def __init__(self):
        """Initialize the data loader strategy."""
        self.sensitivity = None

    def create_train_loader(self, trainset, sampler, batch_size, context):
        """Creates an instance of the trainloader with sensitivity computation."""
        # Calculate sensitivity with the trainset
        if hasattr(Config().algorithm, "defense"):
            if Config().algorithm.defense == "GradDefense":
                root_set_loader = get_root_set_loader(trainset)
                self.sensitivity = compute_sens(
                    model=context.model.to(context.device),
                    rootset_loader=root_set_loader,
                    device=context.device,
                )
                # Store in context for use by other components
                context.state["sensitivity"] = self.sensitivity

        # Handle different sampler types properly
        if sampler is not None:
            if isinstance(sampler, torch.utils.data.Sampler):
                # It's already a PyTorch Sampler object
                sampler_obj = sampler
            elif isinstance(sampler, (list, range)):
                # It's a list of indices, create SubsetRandomSampler
                sampler_obj = torch.utils.data.SubsetRandomSampler(sampler)
            elif hasattr(sampler, "get"):
                # It's a Plato Sampler, call get() to obtain PyTorch sampler
                sampler_obj = sampler.get()
            else:
                # Unknown type, try to use it directly
                sampler_obj = sampler
        else:
            sampler_obj = None

        return torch.utils.data.DataLoader(
            dataset=trainset, shuffle=False, batch_size=batch_size, sampler=sampler_obj
        )


class DLGTrainingStepStrategy(TrainingStepStrategy):
    """
    Custom training step strategy for DLG gradient leakage attacks.

    This strategy implements custom forward/backward passes with gradient
    computation and storage for gradient leakage analysis.
    """

    def __init__(self) -> None:
        """Initialize the training step strategy."""
        self.examples: torch.Tensor | None = None
        self.list_grad: GradientsLike | None = None
        self.feature_fc1_graph: torch.Tensor | None = None

    def training_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        examples: torch.Tensor,
        labels: torch.Tensor,
        loss_criterion: torch.nn.Module,
        context: TrainingContext,
    ) -> torch.Tensor:
        """Perform forward and backward passes for DLG attacks."""
        examples.requires_grad = True
        self.examples = examples
        model.zero_grad()

        if (
            hasattr(Config().algorithm, "target_eval")
            and Config().algorithm.target_eval
        ):
            # Set model into evaluation mode at client's training
            model.eval()
        else:
            model.train()

        # Compute gradients in the current step
        if (
            hasattr(Config().algorithm, "defense")
            and Config().algorithm.defense == "GradDefense"
            and hasattr(Config().algorithm, "clip")
            and Config().algorithm.clip is True
        ):
            per_sample_grads: NestedGradientList = []
            step_losses: list[torch.Tensor] = []
            for example, label in zip(examples, labels):
                output = model(torch.unsqueeze(example, dim=0))
                loss = loss_criterion(output, torch.unsqueeze(label, dim=0))
                step_losses.append(loss)
                grad = torch.autograd.grad(
                    loss,
                    model.parameters(),
                    retain_graph=True,
                    create_graph=True,
                    only_inputs=True,
                )
                per_sample_grads.append([g.detach().clone() for g in grad])
            loss = torch.stack(step_losses).mean()
            self.list_grad = per_sample_grads
        else:
            if (
                hasattr(Config().algorithm, "defense")
                and Config().algorithm.defense == "Soteria"
            ):
                forward_feature = getattr(model, "forward_feature", None)
                if callable(forward_feature):
                    outputs, self.feature_fc1_graph = forward_feature(examples)
                else:
                    outputs = model(examples)
            else:
                outputs = model(examples)
            # Save the ground truth and gradients
            loss = loss_criterion(outputs, labels)
            grad = torch.autograd.grad(
                loss,
                model.parameters(),
                retain_graph=True,
                create_graph=True,
                only_inputs=True,
            )
            self.list_grad = [g.detach().clone() for g in grad]

        # Store in context for use by callbacks
        context.state["examples"] = self.examples
        context.state["labels"] = labels
        context.state["list_grad"] = self.list_grad
        context.state["feature_fc1_graph"] = self.feature_fc1_graph

        return loss

    def on_train_run_start(self, trainer, config, **kwargs):
        """Initialize DLG trainer state and inject trainer into context."""
        trainer.target_grad = None
        # Inject trainer reference into context for strategy access
        trainer.context.state["trainer"] = trainer


class DLGTrainingCallbacks(TrainerCallback):
    """
    Callbacks for DLG trainer handling training lifecycle events.

    Implements the logic from train_run_start, train_step_end, and train_run_end.
    """

    def __init__(self) -> None:
        """Initialize the callback."""
        self.full_examples: torch.Tensor | None = None
        self.full_labels: torch.Tensor | None = None
        self.full_onehot_labels: torch.Tensor | None = None
        self.target_grad: GradientList | None = None

    def on_train_run_start(self, trainer, config, **kwargs) -> None:
        """Method called at the start of training run."""
        self.target_grad = None

    def on_train_epoch_start(self, trainer, config, **kwargs) -> None:
        """Store data in the first epoch."""
        if trainer.current_epoch == 1:
            # Initialize storage for full examples and labels
            self.full_examples = None
            self.full_labels = None

    def on_train_step_end(self, trainer, config, batch, loss, **kwargs) -> None:
        """Apply defense mechanisms and update model weights manually."""
        context = trainer.context

        # Retrieve stored data from context
        examples = cast(Optional[torch.Tensor], context.state.get("examples"))
        labels = cast(Optional[torch.Tensor], context.state.get("labels"))
        gradients_like = cast(Optional[GradientsLike], context.state.get("list_grad"))
        feature_fc1_graph = cast(
            Optional[torch.Tensor], context.state.get("feature_fc1_graph")
        )

        # Store data in the first epoch
        if trainer.current_epoch == 1 and examples is not None and labels is not None:
            if self.full_examples is None:
                self.full_examples = examples.detach().clone()
            else:
                self.full_examples = torch.cat((examples, self.full_examples), dim=0)

            if self.full_labels is None:
                self.full_labels = labels.detach().clone()
            else:
                self.full_labels = torch.cat((labels, self.full_labels), dim=0)

            if self.full_labels is not None:
                self.full_onehot_labels = label_to_onehot(
                    self.full_labels, num_classes=Config().parameters.model.num_classes
                )

        # Apply defense if needed
        working_gradients = gradients_like
        if hasattr(Config().algorithm, "defense") and working_gradients is not None:
            defense_name = Config().algorithm.defense
            if defense_name == "GradDefense":
                sensitivity = cast(
                    Optional[list[float]], context.state.get("sensitivity")
                )
                if sensitivity is None:
                    raise ValueError("Sensitivity must be available for GradDefense.")
                if getattr(Config().algorithm, "clip", False):
                    if isinstance(working_gradients, list) and working_gradients:
                        nested_gradients = cast(NestedGradientList, working_gradients)
                        from defense.GradDefense.perturb import (
                            noise_with_clip as graddefense_noise,
                        )

                        working_gradients = cast(
                            GradientsLike,
                            graddefense_noise(
                                dy_dx=nested_gradients,
                                sensitivity=sensitivity,
                                slices_num=Config().algorithm.slices_num,
                                perturb_slices_num=Config().algorithm.perturb_slices_num,
                                noise_intensity=Config().algorithm.scale,
                            ),
                        )
                else:
                    if isinstance(working_gradients, list) and all(
                        isinstance(item, torch.Tensor) for item in working_gradients
                    ):
                        from defense.GradDefense.perturb import (
                            noise as graddefense_noise,
                        )

                        working_gradients = cast(
                            GradientsLike,
                            graddefense_noise(
                                dy_dx=cast(GradientList, working_gradients),
                                sensitivity=sensitivity,
                                slices_num=Config().algorithm.slices_num,
                                perturb_slices_num=Config().algorithm.perturb_slices_num,
                                noise_intensity=Config().algorithm.scale,
                            ),
                        )

            elif (
                defense_name == "Soteria"
                and feature_fc1_graph is not None
                and examples is not None
                and isinstance(working_gradients, list)
                and all(isinstance(item, torch.Tensor) for item in working_gradients)
            ):
                deviation_f1_target = torch.zeros_like(feature_fc1_graph)
                deviation_f1_x_norm = torch.zeros_like(feature_fc1_graph)

                for f_index in range(deviation_f1_x_norm.size(1)):
                    deviation_f1_target[:, f_index] = 1
                    feature_fc1_graph.backward(deviation_f1_target, retain_graph=True)
                    deviation_f1_x = examples.grad
                    if deviation_f1_x is None:
                        continue
                    deviation_f1_x_norm[:, f_index] = (
                        torch.norm(
                            deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1
                        )
                        / feature_fc1_graph[:, f_index]
                    )
                    trainer.model.zero_grad()
                    deviation_f1_target[:, f_index] = 0
                    deviation_f1_x.zero_()

                deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(dim=0)
                thresh = np.percentile(
                    deviation_f1_x_norm_sum.flatten().cpu().numpy(),
                    Config().algorithm.threshold,
                )
                mask = np.where(
                    np.abs(deviation_f1_x_norm_sum.cpu().numpy()) < thresh, 0, 1
                ).astype(np.float32)
                working_gradients[6] = cast(
                    torch.Tensor, working_gradients[6]
                ) * torch.from_numpy(mask).to(trainer.device)

            elif (
                defense_name == "GC"
                and isinstance(working_gradients, list)
                and all(isinstance(item, torch.Tensor) for item in working_gradients)
            ):
                for index, grad_item in enumerate(working_gradients):
                    grad_tensor = grad_item.cpu().numpy()
                    flattened_weights = np.abs(grad_tensor.flatten())
                    thresh = np.percentile(
                        flattened_weights, Config().algorithm.prune_pct
                    )
                    pruned = np.where(np.abs(grad_tensor) < thresh, 0, grad_tensor)
                    working_gradients[index] = torch.from_numpy(pruned).to(
                        trainer.device
                    )

            elif (
                defense_name == "DP"
                and isinstance(working_gradients, list)
                and all(isinstance(item, torch.Tensor) for item in working_gradients)
            ):
                for index, grad_item in enumerate(working_gradients):
                    grad_tensor = grad_item.cpu().numpy()
                    noise = np.random.laplace(
                        0, Config().algorithm.epsilon, size=grad_tensor.shape
                    )
                    working_gradients[index] = torch.from_numpy(grad_tensor + noise).to(
                        trainer.device
                    )

            elif (
                defense_name == "Outpost"
                and isinstance(working_gradients, list)
                and all(isinstance(item, torch.Tensor) for item in working_gradients)
            ):
                iteration = trainer.current_epoch * (batch + 1)
                if random.random() < 1 / (1 + Config().algorithm.beta * iteration):
                    risk = compute_risk(trainer.model)
                    from defense.Outpost.perturb import noise as outpost_noise

                    working_gradients = cast(
                        GradientsLike, outpost_noise(dy_dx=working_gradients, risk=risk)
                    )

        gradient_updates: GradientList | None = None
        if isinstance(working_gradients, list) and all(
            isinstance(item, torch.Tensor) for item in working_gradients
        ):
            gradient_updates = [
                tensor.to(trainer.device) for tensor in working_gradients
            ]

        # Update model weights with gradients and learning rate
        if gradient_updates is not None:
            for param, grad_part in zip(trainer.model.parameters(), gradient_updates):
                param.data = (
                    param.data
                    - Config().parameters.optimizer.lr * grad_part.to(trainer.device)
                )

            # Sum up the gradients for each local update
            if self.target_grad is None:
                self.target_grad = [g.detach().clone() for g in gradient_updates]
            else:
                self.target_grad = [
                    existing + new.detach().clone()
                    for existing, new in zip(self.target_grad, gradient_updates)
                ]

    def on_train_run_end(self, trainer, config, **kwargs) -> None:
        """Method called at the end of a training run."""
        if (
            hasattr(Config().algorithm, "share_gradients")
            and Config().algorithm.share_gradients
            and self.target_grad is not None
        ):
            total_local_steps = config["epochs"] * math.ceil(
                Config().data.partition_size / config["batch_size"]
            )
            self.target_grad = [grad / total_local_steps for grad in self.target_grad]

        if self.full_examples is not None:
            self.full_examples = self.full_examples.detach()
            file_path = f"{Config().params['model_path']}/{trainer.client_id}.pickle"
            with open(file_path, "wb") as handle:
                pickle.dump(
                    [self.full_examples, self.full_onehot_labels, self.target_grad],
                    handle,
                )


class DLGTestingStrategy(TestingStrategy):
    """Custom testing strategy that processes outputs for DLG attacks."""

    def test_model(self, model, config, testset, sampler, context):
        """Test the model with custom output processing."""
        model.to(context.device)
        model.eval()

        # Create test data loader
        batch_size = config.get("batch_size", 32)

        # Handle different sampler types properly
        if sampler is not None:
            if isinstance(sampler, torch.utils.data.Sampler):
                sampler_obj = sampler
            elif isinstance(sampler, (list, range)):
                sampler_obj = torch.utils.data.SubsetRandomSampler(sampler)
            elif hasattr(sampler, "get"):
                sampler_obj = sampler.get()
            else:
                sampler_obj = sampler
        else:
            sampler_obj = None

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, sampler=sampler_obj
        )

        correct = 0
        total = 0

        with torch.no_grad():
            for examples, labels in test_loader:
                examples, labels = (
                    examples.to(context.device),
                    labels.to(context.device),
                )

                outputs = model(examples)

                # Process outputs - extract first element if tuple/list
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0
        return accuracy


class Trainer(ComposableTrainer):
    """
    The federated learning trainer for gradient leakage attacks.

    Migrated to use the new composable trainer architecture with strategies
    and callbacks instead of inheritance and hooks.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initializing the trainer with the provided model.

        Arguments:
            model: The model to train.
            callbacks: The callbacks that this trainer uses.
        """

        def weights_init(m):
            """Initializing the weights and biases in the model."""
            if hasattr(m, "weight"):
                m.weight.data.uniform_(-0.5, 0.5)
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.data.uniform_(-0.5, 0.5)

        # Create DLG-specific callbacks
        dlg_callbacks = [DLGTrainingCallbacks]
        if callbacks is not None:
            dlg_callbacks.extend(callbacks)

        # Initialize with custom strategies
        super().__init__(
            model=model,
            callbacks=dlg_callbacks,
            training_step_strategy=DLGTrainingStepStrategy(),
            data_loader_strategy=DLGDataLoaderStrategy(),
            testing_strategy=DLGTestingStrategy(),
        )

        # DLG explicit weights initialization
        if (
            hasattr(Config().algorithm, "init_params")
            and Config().algorithm.init_params
            and self.model is not None
        ):
            self.model.apply(weights_init)

        # Store reference to DLG callback for accessing stored data
        self._dlg_callback = None
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, DLGTrainingCallbacks):
                self._dlg_callback = callback
                break

    @property
    def target_grad(self):
        """Access target gradients from callback."""
        if self._dlg_callback is not None:
            return self._dlg_callback.target_grad
        return None

    @property
    def full_examples(self):
        """Access full examples from callback."""
        if self._dlg_callback is not None:
            return self._dlg_callback.full_examples
        return None

    @property
    def full_onehot_labels(self):
        """Access full onehot labels from callback."""
        if self._dlg_callback is not None:
            return self._dlg_callback.full_onehot_labels
        return None
