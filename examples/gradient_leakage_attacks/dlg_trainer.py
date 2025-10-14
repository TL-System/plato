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

    def __init__(self):
        """Initialize the training step strategy."""
        self.examples = None
        self.list_grad = None
        self.feature_fc1_graph = None

    def training_step(
        self, model, optimizer, examples, labels, loss_criterion, context
    ):
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
            self.list_grad = []
            for example, label in zip(examples, labels):
                outputs = model(torch.unsqueeze(example, dim=0))
                loss = loss_criterion(outputs, torch.unsqueeze(label, dim=0))
                grad = torch.autograd.grad(
                    loss,
                    model.parameters(),
                    retain_graph=True,
                    create_graph=True,
                    only_inputs=True,
                )
                self.list_grad.append(list((_.detach().clone() for _ in grad)))
        else:
            if (
                hasattr(Config().algorithm, "defense")
                and Config().algorithm.defense == "Soteria"
            ):
                outputs, self.feature_fc1_graph = model.forward_feature(examples)
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
            self.list_grad = list((_.detach().clone() for _ in grad))

        # Store in context for use by callbacks
        context.state["examples"] = self.examples
        context.state["labels"] = labels
        context.state["list_grad"] = self.list_grad
        context.state["feature_fc1_graph"] = self.feature_fc1_graph

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

    def __init__(self):
        """Initialize the callback."""
        self.full_examples = None
        self.full_labels = None
        self.full_onehot_labels = None
        self.target_grad = None

    def on_train_run_start(self, trainer, config, **kwargs):
        """Method called at the start of training run."""
        self.target_grad = None

    def on_train_epoch_start(self, trainer, config, **kwargs):
        """Store data in the first epoch."""
        if trainer.current_epoch == 1:
            # Initialize storage for full examples and labels
            self.full_examples = None
            self.full_labels = None

    def on_train_step_end(self, trainer, config, batch, loss, **kwargs):
        """Apply defense mechanisms and update model weights manually."""
        context = trainer.context

        # Retrieve stored data from context
        examples = context.state.get("examples")
        labels = context.state.get("labels")
        list_grad = context.state.get("list_grad")
        feature_fc1_graph = context.state.get("feature_fc1_graph")

        # Store data in the first epoch
        if trainer.current_epoch == 1 and examples is not None and labels is not None:
            try:
                self.full_examples = torch.cat((examples, self.full_examples), dim=0)
                self.full_labels = torch.cat((labels, self.full_labels), dim=0)
            except:
                self.full_examples = examples.detach().clone()
                self.full_labels = labels.detach().clone()

            self.full_onehot_labels = label_to_onehot(
                self.full_labels, num_classes=Config().parameters.model.num_classes
            )

        # Apply defense if needed
        grad = list_grad
        if hasattr(Config().algorithm, "defense") and list_grad is not None:
            if Config().algorithm.defense == "GradDefense":
                sensitivity = context.state.get("sensitivity")
                if (
                    hasattr(Config().algorithm, "clip")
                    and Config().algorithm.clip is True
                ):
                    from defense.GradDefense.perturb import noise_with_clip as noise
                else:
                    from defense.GradDefense.perturb import noise
                list_grad = noise(
                    dy_dx=list_grad,
                    sensitivity=sensitivity,
                    slices_num=Config().algorithm.slices_num,
                    perturb_slices_num=Config().algorithm.perturb_slices_num,
                    noise_intensity=Config().algorithm.scale,
                )

            elif Config().algorithm.defense == "Soteria":
                if feature_fc1_graph is not None and examples is not None:
                    deviation_f1_target = torch.zeros_like(feature_fc1_graph)
                    deviation_f1_x_norm = torch.zeros_like(feature_fc1_graph)
                    for f in range(deviation_f1_x_norm.size(1)):
                        deviation_f1_target[:, f] = 1
                        feature_fc1_graph.backward(
                            deviation_f1_target, retain_graph=True
                        )
                        deviation_f1_x = examples.grad.data
                        deviation_f1_x_norm[:, f] = (
                            torch.norm(
                                deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1
                            )
                            / (feature_fc1_graph.data[:, f])
                        )
                        trainer.model.zero_grad()
                        examples.grad.data.zero_()
                        deviation_f1_target[:, f] = 0

                    deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(axis=0)
                    thresh = np.percentile(
                        deviation_f1_x_norm_sum.flatten().cpu().numpy(),
                        Config().algorithm.threshold,
                    )
                    mask = np.where(
                        abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1
                    ).astype(np.float32)
                    list_grad[6] = list_grad[6] * torch.Tensor(mask).to(trainer.device)

            elif Config().algorithm.defense == "GC":
                for i, grad_item in enumerate(list_grad):
                    grad_tensor = grad_item.cpu().numpy()
                    flattened_weights = np.abs(grad_tensor.flatten())
                    thresh = np.percentile(
                        flattened_weights, Config().algorithm.prune_pct
                    )
                    grad_tensor = np.where(abs(grad_tensor) < thresh, 0, grad_tensor)
                    list_grad[i] = torch.Tensor(grad_tensor).to(trainer.device)

            elif Config().algorithm.defense == "DP":
                for i, grad_item in enumerate(list_grad):
                    grad_tensor = grad_item.cpu().numpy()
                    noise = np.random.laplace(
                        0, Config().algorithm.epsilon, size=grad_tensor.shape
                    )
                    grad_tensor = grad_tensor + noise
                    list_grad[i] = torch.Tensor(grad_tensor).to(trainer.device)

            elif Config().algorithm.defense == "Outpost":
                iteration = trainer.current_epoch * (batch + 1)
                # Probability decay
                if random.random() < 1 / (1 + Config().algorithm.beta * iteration):
                    # Risk evaluation
                    risk = compute_risk(trainer.model)
                    # Perturb
                    from defense.Outpost.perturb import noise

                    list_grad = noise(dy_dx=list_grad, risk=risk)

            # cast grad back to tuple type
            grad = tuple([g.to(trainer.device) for g in list_grad])

        # Update model weights with gradients and learning rate
        if grad is not None:
            for param, grad_part in zip(trainer.model.parameters(), grad):
                param.data = (
                    param.data
                    - Config().parameters.optimizer.lr * grad_part.to(trainer.device)
                )

            # Sum up the gradients for each local update
            try:
                self.target_grad = [
                    sum(x)
                    for x in zip(
                        list((_.detach().clone() for _ in grad)), self.target_grad
                    )
                ]
            except:
                self.target_grad = list((_.detach().clone() for _ in grad))

    def on_train_run_end(self, trainer, config, **kwargs):
        """Method called at the end of a training run."""
        if (
            hasattr(Config().algorithm, "share_gradients")
            and Config().algorithm.share_gradients
        ):
            try:
                total_local_steps = config["epochs"] * math.ceil(
                    Config().data.partition_size / config["batch_size"]
                )
                trainer.target_grad = [
                    x / total_local_steps for x in trainer.target_grad
                ]
            except:
                trainer.target_grad = None

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
