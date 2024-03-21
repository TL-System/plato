import torch
import logging
import random
import time
import os

from torch.utils.data import Dataset
from plato.config import Config
from plato.trainers import basic
import torch.nn as nn
import torch.nn.functional as F

class ELRLoss(nn.Module):
    """ Early learning regularization loss from https://proceedings.neurips.cc/paper/2020/file/ea89621bee7c88b2c5be6681c8ef4906-Paper.pdf"""
    def __init__(self, num_examp=60000, num_classes=10, beta=0.7):
        super(ELRLoss, self).__init__()
        self.device = Config().device()
        self.target = torch.zeros(num_examp, num_classes).to(self.device) 
        self.beta = beta
        self.lamda = 0.1 # should read from configuration file.

    def forward(self, outputs, labels, indices):
        y_pred = F.softmax(outputs,dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()

        self.target[indices] = self.beta * self.target[indices] + (1-self.beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
        ce_loss = F.cross_entropy(outputs, labels)
        elr_loss = ((1-(self.target[indices] * y_pred).sum(dim=1)).log()).mean()
        loss = ce_loss +  self.lamda * elr_loss
        
        return  loss
class IndexedDataSet(Dataset):
    """A toy trainer to test noisy data source."""
    def __init__(self, dataset) -> None:
        super().__init__()
        self._wrapped_dataset = dataset

    def __len__(self):
        return len(self._wrapped_dataset)

    def __getitem__(self, index):
        return (index, self._wrapped_dataset.__getitem__(index))

class Trainer(basic.Trainer):
    """A toy trainer to test noisy data source."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)
        self.cache_root =  os.path.expanduser("~/.cache")
        self.server_id = os.getppid()

    def train_model(self, config, trainset, sampler, **kwargs):
        """The default training loop when a custom training loop is not supplied."""
        batch_size = config["batch_size"]
        self.sampler = sampler
        tic = time.perf_counter()

        self.run_history.reset()

        self.train_run_start(config)
        self.callback_handler.call_event("on_train_run_start", self, config)

        # self.train_loader = self.get_train_loader(batch_size, trainset, sampler)
        self.train_loader = self.get_indexed_train_loader(batch_size, trainset, sampler)

        # Initializing the loss criterion
        self._loss_criterion = self.get_loss_criterion()

        # Initializing the optimizer
        self.optimizer = self.get_optimizer(self.model)
        self.lr_scheduler = self.get_lr_scheduler(config, self.optimizer)
        self.optimizer = self._adjust_lr(config, self.lr_scheduler, self.optimizer)

        self.model.to(self.device)
        self.model.train()

        total_epochs = config["epochs"]

        # Corrected label and indices
        # [[indices_1, labels_1], [indices_2, labels_2], ...]
        corrections = []
        corrected = False

        for self.current_epoch in range(1, total_epochs + 1):
            self._loss_tracker.reset()
            self.train_epoch_start(config)
            self.callback_handler.call_event("on_train_epoch_start", self, config)

            for batch_id, (indices, (examples, labels)) in enumerate(self.train_loader):
                self.train_step_start(config, batch=batch_id)
                self.callback_handler.call_event(
                    "on_train_step_start", self, config, batch=batch_id
                )

                examples, labels = examples.to(self.device), labels.to(self.device)

                loss = self.perform_forward_and_backward_passes(
                    config, examples, labels, indices
                )
                if not corrected:
                    corrections.append(self.magic_label_correction(indices))

                self.train_step_end(config, batch=batch_id, loss=loss)
                self.callback_handler.call_event(
                    "on_train_step_end", self, config, batch=batch_id, loss=loss
                )

            corrected = True

            self.lr_scheduler_step()

            if hasattr(self.optimizer, "params_state_update"):
                self.optimizer.params_state_update()

            # Simulate client's speed
            if (
                self.client_id != 0
                and hasattr(Config().clients, "speed_simulation")
                and Config().clients.speed_simulation
            ):
                self.simulate_sleep_time()

            # Saving the model at the end of this epoch to a file so that
            # it can later be retrieved to respond to server requests
            # in asynchronous mode when the wall clock time is simulated
            if (
                hasattr(Config().server, "request_update")
                and Config().server.request_update
            ):
                self.model.cpu()
                training_time = time.perf_counter() - tic
                filename = f"{self.client_id}_{self.current_epoch}_{training_time}.pth"
                self.save_model(filename)
                self.model.to(self.device)

            self.run_history.update_metric("train_loss", self._loss_tracker.average)
            self.train_epoch_end(config)
            self.callback_handler.call_event("on_train_epoch_end", self, config)


        self.save_pseudo_labels(corrections)

        self.train_run_end(config)
        self.callback_handler.call_event("on_train_run_end", self, config)

    def magic_label_correction(self, indices):
        '''Update the labels at target positions'''
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices)

        pseudo_labels = torch.randint(0, 10, indices.size())
        return [indices, pseudo_labels]

    def save_pseudo_labels(self, corrections):
        # Organize corrected labels, corrections should be formatted as
        # [[indices_1, labels_1], [indices_2, labels_2], ...]
        if len(corrections) > 0: 
            indices = torch.cat([x[0] for x in corrections]) 
            pseudo_labels = torch.cat([x[1] for x in corrections])
            
            # Dump pseudo labels to file
            label_file = f"{self.server_id}-client-{self.client_id}-labels.pt"
            label_file = os.path.join(self.cache_root, label_file)
            torch.save([indices, pseudo_labels], label_file)

            logging.info(f" [Client #{self.client_id}] Replaced labels at {indices} to {pseudo_labels}")
        else: 
            logging.info(f"[Client #{self.client_id}] Keeps the label untouched.")


    def get_indexed_train_loader(self, batch_size, trainset, sampler):
        return torch.utils.data.DataLoader(
            dataset=IndexedDataSet(trainset), shuffle=False, batch_size=batch_size, sampler=sampler
        )
    def get_loss_criterion(self):
        #Returns the regularization loss criterion.
        return ELRLoss()

    def perform_forward_and_backward_passes(self, config, examples, labels,indices):
        """Perform forward and backward passes in the training loop.

        Arguments:
        config: the configuration.
        examples: data samples in the current batch.
        labels: labels in the current batch.

        Returns: loss values after the current batch has been processed."""
        
        self.optimizer.zero_grad()

        outputs = self.model(examples)
        
        loss = self._loss_criterion(outputs, labels, indices)
        self._loss_tracker.update(loss, labels.size(0))

        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()

        self.optimizer.step()

        return loss
    
