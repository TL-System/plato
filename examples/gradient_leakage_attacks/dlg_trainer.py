"""
A federated learning trainer for gradient leakage attacks,
where intermediate gradients can be transmitted,
and potential defense mechanisms can be applied.
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
from plato.trainers import basic
from plato.trainers.strategies.base import TrainingStepStrategy

criterion = cross_entropy_for_onehot
tt = transforms.ToPILImage()


class DLGOutputProcessingCallback(TrainerCallback):
    """Callback to process outputs for DLG trainer (extract first element from tuple)."""

    def on_test_outputs(self, trainer, outputs, **kwargs):
        """Extract first element from output tuple."""
        return outputs[0]


class DLGLifecycleCallback(TrainerCallback):
    """Callback to handle DLG-specific lifecycle events."""

    def on_train_run_start(self, trainer, config, **kwargs):
        """Initialize DLG trainer state and inject trainer into context."""
        trainer.target_grad = None
        # Inject trainer reference into context for strategy access
        trainer.context.state["trainer"] = trainer

    def on_train_step_end(self, trainer, config, batch=None, loss=None, **kwargs):
        """Apply gradient defenses and update model at the end of each training step."""
        # Apply defense if needed
        grad = trainer.list_grad

        if hasattr(Config().algorithm, "defense"):
            if Config().algorithm.defense == "GradDefense":
                if (
                    hasattr(Config().algorithm, "clip")
                    and Config().algorithm.clip is True
                ):
                    from defense.GradDefense.perturb import noise_with_clip as noise
                else:
                    from defense.GradDefense.perturb import noise
                trainer.list_grad = noise(
                    dy_dx=trainer.list_grad,
                    sensitivity=trainer.sensitivity,
                    slices_num=Config().algorithm.slices_num,
                    perturb_slices_num=Config().algorithm.perturb_slices_num,
                    noise_intensity=Config().algorithm.scale,
                )

            elif Config().algorithm.defense == "Soteria":
                deviation_f1_target = torch.zeros_like(trainer.feature_fc1_graph)
                deviation_f1_x_norm = torch.zeros_like(trainer.feature_fc1_graph)
                for f in range(deviation_f1_x_norm.size(1)):
                    deviation_f1_target[:, f] = 1
                    trainer.feature_fc1_graph.backward(
                        deviation_f1_target, retain_graph=True
                    )
                    deviation_f1_x = trainer.examples.grad.data
                    deviation_f1_x_norm[:, f] = (
                        torch.norm(
                            deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1
                        )
                        / (trainer.feature_fc1_graph.data[:, f])
                    )
                    trainer.model.zero_grad()
                    trainer.examples.grad.data.zero_()
                    deviation_f1_target[:, f] = 0

                deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(axis=0)
                thresh = np.percentile(
                    deviation_f1_x_norm_sum.flatten().cpu().numpy(),
                    Config().algorithm.threshold,
                )
                mask = np.where(
                    abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1
                ).astype(np.float32)
                trainer.list_grad[6] = trainer.list_grad[6] * torch.Tensor(mask).to(
                    trainer.device
                )

            elif Config().algorithm.defense == "GC":
                for i, grad_elem in enumerate(trainer.list_grad):
                    grad_tensor = grad_elem.cpu().numpy()
                    flattened_weights = np.abs(grad_tensor.flatten())
                    thresh = np.percentile(
                        flattened_weights, Config().algorithm.prune_pct
                    )
                    grad_tensor = np.where(abs(grad_tensor) < thresh, 0, grad_tensor)
                    trainer.list_grad[i] = torch.Tensor(grad_tensor).to(trainer.device)

            elif Config().algorithm.defense == "DP":
                for i, grad_elem in enumerate(trainer.list_grad):
                    grad_tensor = grad_elem.cpu().numpy()
                    noise_tensor = np.random.laplace(
                        0, Config().algorithm.epsilon, size=grad_tensor.shape
                    )
                    grad_tensor = grad_tensor + noise_tensor
                    trainer.list_grad[i] = torch.Tensor(grad_tensor).to(trainer.device)

            elif Config().algorithm.defense == "Outpost":
                iteration = trainer.current_epoch * (batch + 1)
                if random.random() < 1 / (1 + Config().algorithm.beta * iteration):
                    risk = compute_risk(trainer.model)
                    from defense.Outpost.perturb import noise

                    trainer.list_grad = noise(dy_dx=trainer.list_grad, risk=risk)

            grad = tuple([g.to(trainer.device) for g in trainer.list_grad])

        # Update model weights with gradients and learning rate
        for param, grad_part in zip(trainer.model.parameters(), grad):
            param.data = param.data - Config().parameters.optimizer.lr * grad_part.to(
                trainer.device
            )

        # Sum up the gradients for each local update
        try:
            trainer.target_grad = [
                sum(x)
                for x in zip(list((_.detach().clone() for _ in grad)), trainer.target_grad)
            ]
        except:
            trainer.target_grad = list((_.detach().clone() for _ in grad))

    def on_train_run_end(self, trainer, config, **kwargs):
        """Save gradients and examples at the end of training."""
        if (
            hasattr(Config().algorithm, "share_gradients")
            and Config().algorithm.share_gradients
        ):
            try:
                total_local_steps = config["epochs"] * math.ceil(
                    Config().data.partition_size / config["batch_size"]
                )
                trainer.target_grad = [x / total_local_steps for x in trainer.target_grad]
            except:
                trainer.target_grad = None

        trainer.full_examples = trainer.full_examples.detach()
        file_path = f"{Config().params['model_path']}/{trainer.client_id}.pickle"
        with open(file_path, "wb") as handle:
            pickle.dump(
                [trainer.full_examples, trainer.full_onehot_labels, trainer.target_grad], handle
            )


class DLGTrainingStepStrategy(TrainingStepStrategy):
    """Custom training step strategy for DLG gradient leakage attacks."""

    def training_step(self, model, optimizer, examples, labels, loss_criterion, context):
        """Perform one DLG training step with gradient computation."""
        trainer = context.state.get("trainer")
        if trainer is None:
            raise ValueError("Trainer must be stored in context.state['trainer']")

        # Store data in the first epoch (later epochs will still have the same partitioned data)
        if context.current_epoch == 1:
            try:
                trainer.full_examples = torch.cat((examples, trainer.full_examples), dim=0)
                trainer.full_labels = torch.cat((labels, trainer.full_labels), dim=0)
            except:
                trainer.full_examples = examples
                trainer.full_labels = labels

            trainer.full_onehot_labels = label_to_onehot(
                trainer.full_labels, num_classes=Config().parameters.model.num_classes
            )

        examples.requires_grad = True
        trainer.examples = examples
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
            trainer.list_grad = []
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
                trainer.list_grad.append(list((_.detach().clone() for _ in grad)))
        else:
            if (
                hasattr(Config().algorithm, "defense")
                and Config().algorithm.defense == "Soteria"
            ):
                outputs, trainer.feature_fc1_graph = model.forward_feature(examples)
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
            trainer.list_grad = list((_.detach().clone() for _ in grad))

        return loss


class Trainer(basic.Trainer):
    """The federated learning trainer for the gradient leakage attack."""

    def __init__(self, model=None, callbacks=None):
        """Initializing the trainer with the provided model.

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

        # Add DLG-specific callbacks
        callbacks_with_dlg = [DLGOutputProcessingCallback, DLGLifecycleCallback]
        if callbacks is not None:
            callbacks_with_dlg.extend(callbacks)

        # Create DLG training step strategy
        dlg_training_strategy = DLGTrainingStepStrategy()

        # Initialize parent with DLG strategy - bypass basic.Trainer.__init__
        # to pass custom training_step_strategy
        from plato.trainers.composable import ComposableTrainer

        ComposableTrainer.__init__(
            self,
            model=model,
            callbacks=callbacks_with_dlg,
            training_step_strategy=dlg_training_strategy,
        )

        # Legacy attributes for backward compatibility
        self._loss_criterion = None

        # DLG explicit weights initialziation
        if (
            hasattr(Config().algorithm, "init_params")
            and Config().algorithm.init_params
        ):
            self.model.apply(weights_init)

        self.examples = None
        self.trainset = None
        self.full_examples = None
        self.full_labels = None
        self.full_onehot_labels = None
        self.list_grad = None
        self.target_grad = None
        self.feature_fc1_graph = None
        self.sensitivity = None

    def get_train_loader(self, batch_size, trainset, sampler, **kwargs):
        """Creates an instance of the trainloader."""
        # Calculate sensitivity with the trainset
        if hasattr(Config().algorithm, "defense"):
            if Config().algorithm.defense == "GradDefense":
                root_set_loader = get_root_set_loader(trainset)
                self.sensitivity = compute_sens(
                    model=self.model.to(self.device),
                    rootset_loader=root_set_loader,
                    device=self.device,
                )

        return torch.utils.data.DataLoader(
            dataset=trainset, shuffle=False, batch_size=batch_size, sampler=sampler
        )

    @property
    def loss_criterion(self):
        """Legacy property for accessing loss criterion."""
        if self._loss_criterion is None:
            # Create loss criterion using the strategy
            def compute_loss_fn(outputs, labels):
                return self.loss_strategy.compute_loss(outputs, labels, self.context)

            self._loss_criterion = compute_loss_fn
        return self._loss_criterion
