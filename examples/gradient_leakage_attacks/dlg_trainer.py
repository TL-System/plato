import math
import pickle
import random

import numpy as np
import torch
from plato.config import Config
from plato.trainers import basic
from torchvision import transforms

from defense.GradDefense.dataloader import get_root_set_loader
from defense.GradDefense.sensitivity import compute_sens
from defense.Outpost.perturb import compute_risk
from utils.utils import cross_entropy_for_onehot, label_to_onehot

criterion = cross_entropy_for_onehot
tt = transforms.ToPILImage()


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

        super().__init__(model=model, callbacks=callbacks)

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
                    device=Config().device(),
                )

        return torch.utils.data.DataLoader(
            dataset=trainset, shuffle=False, batch_size=batch_size, sampler=sampler
        )

    def train_run_start(self, config):
        """Method called at the start of training run."""
        self.target_grad = None

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Perform forward and backward passes in the training loop."""
        # Store data in the first epoch (later epochs will still have the same partitioned data)
        if self.current_epoch == 1:
            try:
                self.full_examples = torch.cat((examples, self.full_examples), dim=0)
                self.full_labels = torch.cat((labels, self.full_labels), dim=0)
            except:
                self.full_examples = examples
                self.full_labels = labels

            self.full_onehot_labels = label_to_onehot(
                self.full_labels, num_classes=Config().parameters.model.num_classes
            )

        examples.requires_grad = True
        self.examples = examples
        self.model.zero_grad()

        if (
            hasattr(Config().algorithm, "target_eval")
            and Config().algorithm.target_eval
        ):
            # Set model into evaluation mode at client's training
            self.model.eval()
        else:
            self.model.train()

        # Compute gradients in the current step
        if (
            hasattr(Config().algorithm, "defense")
            and Config().algorithm.defense == "GradDefense"
            and hasattr(Config().algorithm, "clip")
            and Config().algorithm.clip is True
        ):
            self.list_grad = []
            for example, label in zip(examples, labels):
                outputs, _ = self.model(torch.unsqueeze(example, dim=0))

                loss = self._loss_criterion(outputs, torch.unsqueeze(label, dim=0))
                grad = torch.autograd.grad(
                    loss,
                    self.model.parameters(),
                    retain_graph=True,
                    create_graph=True,
                    only_inputs=True,
                )
                self.list_grad.append(list((_.detach().clone() for _ in grad)))
        else:
            try:
                outputs, self.feature_fc1_graph = self.model(examples)
            except:
                outputs = self.model(examples)
            # Save the ground truth and gradients
            loss = self._loss_criterion(outputs, labels)
            grad = torch.autograd.grad(
                loss,
                self.model.parameters(),
                retain_graph=True,
                create_graph=True,
                only_inputs=True,
            )
            self.list_grad = list((_.detach().clone() for _ in grad))

        self._loss_tracker.update(loss, labels.size(0))

        return loss

    def train_step_end(self, config, batch=None, loss=None):
        """Method called at the end of a training step."""
        # Apply defense if needed
        if hasattr(Config().algorithm, "defense"):
            if Config().algorithm.defense == "GradDefense":
                if (
                    hasattr(Config().algorithm, "clip")
                    and Config().algorithm.clip is True
                ):
                    from defense.GradDefense.clip import noise
                else:
                    from defense.GradDefense.perturb import noise
                self.list_grad = noise(
                    dy_dx=self.list_grad,
                    sensitivity=self.sensitivity,
                    slices_num=Config().algorithm.slices_num,
                    perturb_slices_num=Config().algorithm.perturb_slices_num,
                    noise_intensity=Config().algorithm.scale,
                )

            elif Config().algorithm.defense == "Soteria":
                deviation_f1_target = torch.zeros_like(self.feature_fc1_graph)
                deviation_f1_x_norm = torch.zeros_like(self.feature_fc1_graph)
                for f in range(deviation_f1_x_norm.size(1)):
                    deviation_f1_target[:, f] = 1
                    self.feature_fc1_graph.backward(
                        deviation_f1_target, retain_graph=True
                    )
                    deviation_f1_x = self.examples.grad.data
                    deviation_f1_x_norm[:, f] = torch.norm(
                        deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1
                    ) / (self.feature_fc1_graph.data[:, f])
                    self.model.zero_grad()
                    self.examples.grad.data.zero_()
                    deviation_f1_target[:, f] = 0

                deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(axis=0)
                thresh = np.percentile(
                    deviation_f1_x_norm_sum.flatten().cpu().numpy(),
                    Config().algorithm.threshold,
                )
                mask = np.where(
                    abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1
                ).astype(np.float32)
                # print(sum(mask))
                self.list_grad[6] = self.list_grad[6] * torch.Tensor(mask).to(
                    self.device
                )

            elif Config().algorithm.defense == "GC":
                for i, grad in enumerate(self.list_grad):
                    grad_tensor = grad.cpu().numpy()
                    flattened_weights = np.abs(grad_tensor.flatten())
                    # Generate the pruning threshold according to 'prune by percentage'
                    thresh = np.percentile(
                        flattened_weights, Config().algorithm.prune_pct
                    )
                    grad_tensor = np.where(abs(grad_tensor) < thresh, 0, grad_tensor)
                    self.list_grad[i] = torch.Tensor(grad_tensor).to(self.device)

            elif Config().algorithm.defense == "DP":
                for i, grad in enumerate(self.list_grad):
                    grad_tensor = grad.cpu().numpy()
                    noise = np.random.laplace(
                        0, Config().algorithm.epsilon, size=grad_tensor.shape
                    )
                    grad_tensor = grad_tensor + noise
                    self.list_grad[i] = torch.Tensor(grad_tensor).to(self.device)

            elif Config().algorithm.defense == "Outpost":
                iteration = self.current_epoch * (batch + 1)
                # Probability decay
                if random.random() < 1 / (1 + Config().algorithm.beta * iteration):
                    # Risk evaluation
                    risk = compute_risk(self.model)
                    # Perturb
                    from defense.Outpost.perturb import noise

                    self.list_grad = noise(dy_dx=self.list_grad, risk=risk)

            # cast grad back to tuple type
            grad = tuple(self.list_grad)

        # Update model weights with gradients and learning rate
        for param, grad_part in zip(self.model.parameters(), grad):
            param.data = param.data - Config().parameters.optimizer.lr * grad_part.to(
                self.device
            )

        # Sum up the gradients for each local update
        try:
            self.target_grad = [
                sum(x)
                for x in zip(list((_.detach().clone() for _ in grad)), self.target_grad)
            ]
        except:
            self.target_grad = list((_.detach().clone() for _ in grad))

    def train_run_end(self, config, **kwargs):
        """Method called at the end of a training run."""
        if (
            hasattr(Config().algorithm, "share_gradients")
            and Config().algorithm.share_gradients
        ):
            try:
                total_local_steps = config["epochs"] * math.ceil(
                    Config().data.partition_size / config["batch_size"]
                )
                self.target_grad = [x / total_local_steps for x in self.target_grad]
            except:
                self.target_grad = None

        self.full_examples = self.full_examples.detach()
        file_path = f"{Config().params['model_path']}/{self.client_id}.pickle"
        with open(file_path, "wb") as handle:
            pickle.dump(
                [self.full_examples, self.full_onehot_labels, self.target_grad], handle
            )

    @staticmethod
    def process_outputs(outputs):
        """
        Method called after the model updates have been generated.
        """
        return outputs[0]
