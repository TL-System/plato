import math
import pickle
import random
import time

# import matplotlib.pyplot as plt
import numpy as np
import torch
from plato.callbacks.handler import CallbackHandler
from plato.callbacks.trainer import PrintProgressCallback
from plato.config import Config
from plato.models import registry as models_registry
from plato.trainers import basic, tracking
from torchvision import transforms

from defense.GradDefense.dataloader import get_root_set_loader
from defense.GradDefense.sensitivity import compute_sens
from defense.Outpost.perturb import compute_risk
from utils.utils import cross_entropy_for_onehot, label_to_onehot

criterion = cross_entropy_for_onehot
tt = transforms.ToPILImage()


def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)

class Trainer(basic.Trainer):
    """ The federated learning trainer for the gradient leakage attack. """

    def __init__(self, model=None, callbacks=None):
        """Initializing the trainer with the provided model.

        Arguments:
        model: The model to train.
        client_id: The ID of the client using this trainer (optional).
        """
        super().__init__(model=model, callbacks=callbacks)

        # DLG explicit weights initialziation
        if hasattr(Config().algorithm,
                   'init_params') and Config().algorithm.init_params:
            self.model.apply(weights_init)

    def train_model(self, config, trainset, sampler, **kwargs):
        """ The default training loop when a custom training loop is not supplied. """
        partition_size = Config().data.partition_size
        batch_size = config['batch_size']
        tic = time.perf_counter()

        self.run_history.reset()

        self.train_run_start(config)

        self.train_loader = Trainer.get_train_loader(batch_size, trainset,
                                                     sampler)

        # Initializing the loss criterion
        _loss_criterion = self.get_loss_criterion()

        self.model.to(self.device)
        self.model.train()

        total_epochs = config["epochs"]

        if hasattr(Config().algorithm, 'defense'):
            if Config().algorithm.defense == 'GradDefense':
                root_set_loader = get_root_set_loader(trainset)
                sensitivity = compute_sens(model=self.model,
                                           rootset_loader=root_set_loader,
                                           device=Config().device())

        target_grad = None
        total_local_steps = total_epochs * math.ceil(
            partition_size / batch_size)

        for self.current_epoch in range(1, total_epochs + 1):
            self._loss_tracker.reset()
            self.train_epoch_start(config)

            for batch_id, (examples, labels) in enumerate(self.train_loader):
                examples, labels = examples.to(self.device), labels.to(
                    self.device)
                examples.requires_grad = True
                self.model.zero_grad()

                # Store data in the first epoch (later epochs will still have the same partitioned data)
                if self.current_epoch == 1:
                    try:
                        full_examples = torch.cat((examples, full_examples),
                                                  dim=0)
                        full_labels = torch.cat((labels, full_labels), dim=0)
                    except:
                        full_examples = examples
                        full_labels = labels

                # plt.imshow(tt(examples[0].cpu()))
                # plt.title("Ground truth image")

                # Compute gradients in the current step
                if hasattr(Config().algorithm, 'defense') and Config().algorithm.defense == 'GradDefense' and \
                        hasattr(Config().algorithm, 'clip') and Config().algorithm.clip is True:
                    list_grad = []
                    for index in range(len(examples)):
                        outputs, _ = self.model(
                            torch.unsqueeze(examples[index], dim=0))

                        loss = _loss_criterion(
                            outputs, torch.unsqueeze(labels[index], dim=0))
                        grad = torch.autograd.grad(loss,
                                                   self.model.parameters(),
                                                   retain_graph=True,
                                                   create_graph=True,
                                                   only_inputs=True)
                        list_grad.append(
                            list((_.detach().clone() for _ in grad)))
                else:
                    outputs, feature_fc1_graph = self.model(examples)

                    # Save the ground truth and gradients
                    loss = _loss_criterion(outputs, labels)
                    grad = torch.autograd.grad(loss,
                                               self.model.parameters(),
                                               retain_graph=True,
                                               create_graph=True,
                                               only_inputs=True)
                    list_grad = list((_.detach().clone() for _ in grad))

                self._loss_tracker.update(loss, labels.size(0))

                # Apply defense if needed
                if hasattr(Config().algorithm, 'defense'):
                    if Config().algorithm.defense == 'GradDefense':
                        if hasattr(Config().algorithm,
                                   'clip') and Config().algorithm.clip is True:
                            from defense.GradDefense.clip import noise
                        else:
                            from defense.GradDefense.perturb import noise
                        list_grad = noise(
                            dy_dx=list_grad,
                            sensitivity=sensitivity,
                            slices_num=Config().algorithm.slices_num,
                            perturb_slices_num=Config(
                            ).algorithm.perturb_slices_num,
                            noise_intensity=Config().algorithm.scale)

                    elif Config().algorithm.defense == 'Soteria':
                        deviation_f1_target = torch.zeros_like(
                            feature_fc1_graph)
                        deviation_f1_x_norm = torch.zeros_like(
                            feature_fc1_graph)
                        for f in range(deviation_f1_x_norm.size(1)):
                            deviation_f1_target[:, f] = 1
                            feature_fc1_graph.backward(deviation_f1_target,
                                                       retain_graph=True)
                            deviation_f1_x = examples.grad.data
                            deviation_f1_x_norm[:, f] = torch.norm(
                                deviation_f1_x.view(deviation_f1_x.size(0),
                                                    -1),
                                dim=1) / (feature_fc1_graph.data[:, f])
                            self.model.zero_grad()
                            examples.grad.data.zero_()
                            deviation_f1_target[:, f] = 0

                        deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(
                            axis=0)
                        thresh = np.percentile(
                            deviation_f1_x_norm_sum.flatten().cpu().numpy(),
                            Config().algorithm.threshold)
                        mask = np.where(
                            abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0,
                            1).astype(np.float32)
                        # print(sum(mask))
                        list_grad[6] = list_grad[6] * torch.Tensor(mask).to(
                            self.device)

                    elif Config().algorithm.defense == 'GC':
                        for i in range(len(list_grad)):
                            grad_tensor = list_grad[i].cpu().numpy()
                            flattened_weights = np.abs(grad_tensor.flatten())
                            # Generate the pruning threshold according to 'prune by percentage'
                            thresh = np.percentile(
                                flattened_weights,
                                Config().algorithm.prune_pct)
                            grad_tensor = np.where(
                                abs(grad_tensor) < thresh, 0, grad_tensor)
                            list_grad[i] = torch.Tensor(grad_tensor).to(
                                self.device)

                    elif Config().algorithm.defense == 'DP':
                        for i in range(len(list_grad)):
                            grad_tensor = list_grad[i].cpu().numpy()
                            noise = np.random.laplace(0,
                                                      Config().algorithm.epsilon,
                                                      size=grad_tensor.shape)
                            grad_tensor = grad_tensor + noise
                            list_grad[i] = torch.Tensor(grad_tensor).to(
                                self.device)

                    elif Config().algorithm.defense == 'Outpost':
                        iteration = self.current_epoch * (batch_id + 1)
                        # Probability decay
                        if random.random() < 1 / (
                                1 + Config().algorithm.beta * iteration):
                            # Risk evaluation
                            risk = compute_risk(self.model)
                            # Perturb
                            from defense.Outpost.perturb import noise
                            list_grad = noise(dy_dx=list_grad, risk=risk)

                    # cast grad back to tuple type
                    grad = tuple(list_grad)

                # Update model weights with gradients and learning rate
                for (param, grad_part) in zip(self.model.parameters(), grad):
                    param.data = param.data - Config(
                    ).parameters.optimizer.lr * grad_part

                # Sum up the gradients for each local update
                try:
                    target_grad = [
                        sum(x)
                        for x in zip(list((_.detach().clone()
                                           for _ in grad)), target_grad)
                    ]
                except:
                    target_grad = list((_.detach().clone() for _ in grad))

                self.train_step_end(config, batch=batch_id, loss=loss)

            full_onehot_labels = label_to_onehot(
                full_labels, num_classes=Config().trainer.num_classes)

            # Simulate client's speed
            if (self.client_id != 0
                    and hasattr(Config().clients, "speed_simulation")
                    and Config().clients.speed_simulation):
                self.simulate_sleep_time()

            # Saving the model at the end of this epoch to a file so that
            # it can later be retrieved to respond to server requests
            # in asynchronous mode when the wall clock time is simulated
            if (hasattr(Config().server, "request_update")
                    and Config().server.request_update):
                self.model.cpu()
                training_time = time.perf_counter() - tic
                filename = f"{self.client_id}_{self.current_epoch}_{training_time}.pth"
                self.save_model(filename)
                self.model.to(self.device)

            self.run_history.update_metric("train_loss",
                                           self._loss_tracker.average)
            self.train_epoch_end(config)

        if hasattr(Config().algorithm,
                   'share_gradients') and Config().algorithm.share_gradients:
            try:
                target_grad = [x / total_local_steps for x in target_grad]
            except:
                target_grad = None

        full_examples = full_examples.detach()
        file_path = f"{Config().params['model_path']}/{self.client_id}.pickle"
        with open(file_path, 'wb') as handle:
            pickle.dump([full_examples, full_onehot_labels, target_grad],
                        handle)

        self.train_run_end(config)
        self.callback_handler.call_event("on_train_run_end", self, config)

    def test_model(self, config, testset, sampler, **kwargs):
        """
        Evaluates the model with the provided test dataset and test sampler.

        :param testset: the test dataset.
        :param sampler: the test sampler.
        :param kwargs (optional): Additional keyword arguments.

        """
        batch_size = config["batch_size"]

        if sampler is None:
            test_loader = torch.utils.data.DataLoader(testset,
                                                      batch_size=batch_size,
                                                      shuffle=False)
        else:
            # Use a testing set following the same distribution as the training set
            test_loader = torch.utils.data.DataLoader(testset,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      sampler=sampler.get())

        correct = 0
        total = 0

        self.model.to(self.device)
        with torch.no_grad():
            for examples, labels in test_loader:
                examples, labels = examples.to(self.device), labels.to(
                    self.device)

                outputs, _ = self.model(examples)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total
