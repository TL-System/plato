import torch
import logging
import random
import time
import copy
import os
import numpy as np

from torch.utils.data import Dataset
from plato.config import Config
from plato.trainers import basic
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler

from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist


def get_output(loader, net, device, criterion=None):
    net.eval()
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    net = net.to(device)
    with torch.no_grad():
        for i, (indices, (examples, labels)) in enumerate(loader):
            examples = examples.to(device)
            labels = labels.to(device)
            labels = labels.long()

            logits = net(examples)
            predictions = F.softmax(logits, dim=1)
            loss = criterion(predictions, labels)

            if i == 0:
                all_predictions = np.array(predictions.cpu())
                all_loss = np.array(loss.cpu())
                all_logits = np.array(logits.cpu())
                all_indices = np.array(indices)
            else:
                all_predictions = np.concatenate(
                    (all_predictions, predictions.cpu()), axis=0
                )
                all_loss = np.concatenate((all_loss, loss.cpu()), axis=0)
                all_logits = np.concatenate((all_logits, logits.cpu()), axis=0)
                all_indices = np.concatenate((all_indices, indices.cpu()), axis=0)

    return all_predictions, all_loss, all_logits, all_indices


def lid_term(X, batch, k=20):
    eps = 1e-6
    X = np.asarray(X, dtype=np.float32)

    batch = np.asarray(batch, dtype=np.float32)
    f = lambda v: -k / (np.sum(np.log(v / (v[-1] + eps))) + eps)
    distances = cdist(X, batch)

    # get the closest k neighbours
    sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=distances)[:, 1 : k + 1]
    m, n = sort_indices.shape
    idx = np.ogrid[:m, :n]
    idx[1] = sort_indices
    # sorted matrix
    distances_ = distances[tuple(idx)]
    lids = np.apply_along_axis(f, axis=1, arr=distances_)
    return lids


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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
        self.cache_root = os.path.expanduser("~/.cache")
        self.server_id = None
        self.stage = None
        self.random_seed = 1
        self.relabel_ratio = 0.5 # Need to be in config
        self.confidence_thres = 0.5 # Need to be in config
        self.relabel_only = False

        # Local Proximal Reg
        self.init_model = None # Model before training
        self.mu = 0.0 # Estimated noisy level

    def set_server_id(self, server_id):
        self.server_id = server_id
    
    def set_stage(self, stage):
        self.stage = stage

    def set_relabel_only(self):
        self.relabel_only = True
    
    def set_noisy_level(self, mu):
        self.mu = mu

    def train_model(self, config, trainset, sampler, **kwargs):
        # Normal training in warm up phase
        # if self.warm_up:
        #     return super().train_model(config, trainset, sampler, **kwargs)
        self.init_model = copy.deepcopy(self.model)
        """Stage 1: Preprocessing"""
        if self.stage == 1:
            self.stage_1(config, trainset, sampler, **kwargs)
        elif self.stage == 2:
            self.stage_2(config, trainset, sampler, **kwargs)
        elif self.stage == 3:
            self.stage_3(config, trainset, sampler, **kwargs)
            
    def stage_1(self, config, trainset, sampler, **kwargs):
        super().train_model(config, trainset, sampler, **kwargs)

        batch_size = config["batch_size"]
        self.sampler = sampler
        indexed_data_loader = self.get_indexed_train_loader(
            batch_size, trainset, sampler
        )
        no_reduction_criterion = nn.CrossEntropyLoss(reduction="none")
        local_output, loss, _, local_indices = get_output(
            indexed_data_loader, self.model, self.device, no_reduction_criterion
        )


        LID_local = list(lid_term(local_output, local_output))
        LID_client = np.mean(LID_local)

        # Save the LID value and cumulative loss values for current client
        LID_file = f"{self.server_id}_LID_client_{self.client_id}.pt"
        LID_file = os.path.join(self.cache_root, LID_file)
        torch.save(LID_client, LID_file)

        cumulative_loss_file = f"{self.server_id}_loss_{self.client_id}.pt"
        cumulative_loss_file = os.path.join(self.cache_root, cumulative_loss_file)
        if os.path.exists(cumulative_loss_file):
            cumulative_loss = torch.load(cumulative_loss_file)
        else:
            cumulative_loss = {}

        loss_dict = {}
        loss_for_relabel = []
        for index, loss_val in zip(local_indices, loss):
            if index in cumulative_loss:
                loss_dict[index] = loss_val + cumulative_loss[index]
            else:
                loss_dict[index] = loss_val
            
            loss_for_relabel.append(loss_dict[index])
        
        torch.save(loss_dict, cumulative_loss_file)

        # Relabel anyway, may not be applied if server determines this client is clean
        loss = np.array(loss_for_relabel)

        gmm_loss = GaussianMixture(n_components=2, random_state=self.random_seed).fit(np.array(loss).reshape(-1, 1))
        labels_loss = gmm_loss.predict(np.array(loss).reshape(-1, 1))
        gmm_clean_label_loss = np.argsort(gmm_loss.means_[:, 0])[0]

        pred_n = np.where(labels_loss.flatten() != gmm_clean_label_loss)[0]
        estimated_noisy_level = len(pred_n) / len(local_indices)

        relabel_idx = (-loss).argsort()[:int(len(local_indices) * estimated_noisy_level * self.relabel_ratio)]
        relabel_idx = list(set(np.where(np.max(local_output, axis=1) > self.confidence_thres)[0]) & set(relabel_idx))
        
        pseudo_labels = np.argmax(local_output, axis=1)[relabel_idx]
        real_indices = local_indices[relabel_idx]
        if len(relabel_idx):
            self.save_pseudo_labels([[list(real_indices), list(pseudo_labels)]])

        mu_file = f"{self.server_id}_mu_{self.client_id}.pt"
        mu_file = os.path.join(self.cache_root, mu_file)
        torch.save(estimated_noisy_level, mu_file)

    def stage_2(self, config, trainset, sampler, **kwargs):
        if not self.relabel_only:
            # Clean clients perform normal FedAvg
            super().train_model(config, trainset, sampler, **kwargs)
        else:
            # Noisy clients relabel local samples
            batch_size = config["batch_size"]
            self.sampler = sampler
            indexed_data_loader = self.get_indexed_train_loader(
                batch_size, trainset, sampler
            )
            no_reduction_criterion = nn.CrossEntropyLoss(reduction="none")
            local_output, _, _, local_indices = get_output(
                indexed_data_loader, self.model, self.device, no_reduction_criterion
            )

            relabel_idx = list(np.where(np.max(local_output, axis=1) > self.confidence_thres)[0])
            pseudo_labels = np.argmax(local_output, axis=1)
            real_indices = local_indices[relabel_idx]
            if len(relabel_idx):
                self.save_pseudo_labels([[list(real_indices), list(pseudo_labels)]])

    def stage_3(self, config, trainset, sampler, **kwargs):
        super().train_model(config, trainset, sampler, **kwargs)
        
    def save_pseudo_labels(self, corrections):
        # Organize corrected labels, corrections should be formatted as
        # [[indices_1, labels_1], [indices_2, labels_2], ...]
        if len(corrections) > 0:
            indices = torch.cat([torch.tensor(x[0]) for x in corrections])
            pseudo_labels = torch.cat([torch.tensor(x[1]) for x in corrections])

            # Dump pseudo labels to file
            label_file = f"{self.server_id}-client-{self.client_id}-label-updates.pt"
            label_file = os.path.join(self.cache_root, label_file)
            torch.save([indices, pseudo_labels], label_file)

            logging.info(
                f" [Client #{self.client_id}] Replaced labels at {indices} to {pseudo_labels}"
            )
        else:
            logging.info(f"[Client #{self.client_id}] Keeps the label untouched.")

    def get_indexed_train_loader(self, batch_size, trainset, sampler):
        return torch.utils.data.DataLoader(
            dataset=IndexedDataSet(trainset),
            shuffle=False,
            batch_size=batch_size,
            sampler=sampler,
        )

    def perform_forward_and_backward_passes(self, config, examples, labels):
        self.optimizer.zero_grad()

        # Mixup augmentation
        examples, targets_a, targets_b, lam = mixup_data(examples, labels)

        outputs = self.model(examples)

        # loss = self._loss_criterion(outputs, labels)
        loss = mixup_criterion(self._loss_criterion, outputs, targets_a, targets_b, lam)

        # Local Proximal Reg
        if self.stage == 1:
            w_diff = torch.tensor(0.).to(self.device)
            for w, w_t in zip(self.init_model.parameters(), self.model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2) 
            w_diff = torch.sqrt(w_diff)
            loss += 5.0 * self.mu * w_diff

        self._loss_tracker.update(loss, labels.size(0))

        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()

        self.optimizer.step()

        return loss