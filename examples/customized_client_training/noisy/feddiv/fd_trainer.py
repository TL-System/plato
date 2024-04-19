import torch
import logging
import random
import time
import os
import numpy as np

from torch.utils.data import Dataset
from plato.config import Config
from plato.trainers import basic
from feddiv.gmm_filter import GaussianMixture
from feddiv.fd_utils import *
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler


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
        self.warm_up = False
        self.local_filter = None
        self.de_bias = None
        self.train_counter = None

    def set_server_id(self, server_id):
        self.server_id = server_id

    def get_de_bias(self, sampler):
        de_bias_file = f"{self.server_id}_de_bias_{self.client_id}.pt"
        de_bias_file = os.path.join(self.cache_root, de_bias_file)
        if os.path.exists(de_bias_file):
            print("Debias file exsits, reading from file")
            de_bias = torch.load(de_bias_file)
            de_bias["counter"] += 1
            return de_bias
        else:
            print("Debias file not exsits, Creating")
            de_bias = {}
            de_bias["de_biased_labels"] = {x: 0 for x in sampler.subset_indices}
            de_bias["de_biased_probs"] = {x: 0 for x in sampler.subset_indices}
            num_classes = Config().parameters.model.num_classes
            de_bias["phat"] = np.ones([1, num_classes], dtype=np.float32) / num_classes
            de_bias["counter"] = 0
            return de_bias
        
    def set_local_filter(self, global_filter_stat, sampler):
        self.local_filter = GaussianMixture(**global_filter_stat)
        self.de_bias = self.get_de_bias(sampler)

    def train_model(self, config, trainset, sampler, **kwargs):
        # Normal training in warm up phase
        if self.warm_up:
            return super().train_model(config, trainset, sampler, **kwargs)

        # Training
        batch_size = config["batch_size"]
        self.sampler = sampler

        self.run_history.reset()

        # Revise data samples
        indexed_data_loader = self.get_indexed_train_loader(batch_size, trainset, sampler)
        no_reduction_criterion = nn.CrossEntropyLoss(reduction='none')
        local_output, loss, _, local_indices = get_output(indexed_data_loader, self.model, self.device, no_reduction_criterion)

        local_split = local_data_splitting(loss, self.local_filter)

        revised_sample_idx, relabeled_trainset, self.de_bias, relabel_indices = relabeling_and_reselection(
                local_split, local_indices, local_output, trainset, self.de_bias
            )
        
        revised_sampler = SubsetRandomSampler(list(revised_sample_idx), generator=sampler.generator)
        
        if len(revised_sample_idx) == 0:
            logging.warn("Nothing left in the revised sample, using original sampler.")
            super().train_model(config, relabeled_trainset, sampler, **kwargs)
        else:
            # Perform the normal training with relabeled trainset and revised sampler
            super().train_model(config, relabeled_trainset, revised_sampler, **kwargs)

        # Save pseudo labels
        if len(relabel_indices):
            pseudo_labels = relabeled_trainset.targets[relabel_indices]
            self.save_pseudo_labels([[list(relabel_indices), list(pseudo_labels)]])
        
        # Evaluate the local model after local training
        _, loss, local_logits, local_indices = get_output(indexed_data_loader, self.model, self.device, no_reduction_criterion)

        self.de_bias = client_cached_phat_update(self.de_bias, local_logits, local_indices)
        filter_update_history = self.update_local_filter(loss)

         # Save the debias status for current client
        de_bias_file = f"{self.server_id}_de_bias_{self.client_id}.pt"
        de_bias_file = os.path.join(self.cache_root, de_bias_file)
        torch.save(self.de_bias, de_bias_file)

        
        filter_update_file = f"{self.server_id}_filter_{self.client_id}.pt"
        filter_update_file = os.path.join(self.cache_root, filter_update_file)
        torch.save(filter_update_history, filter_update_file)

    def update_local_filter(self, loss, filter_update_epoch = 3):
        normalized_loss = calculate_normalized_loss(loss)
        training_data = normalized_loss.reshape(-1, 1)
        self.local_filter.fit(training_data, filter_update_epoch)
        return self.local_filter.history_

    def save_pseudo_labels(self, corrections):
        # Organize corrected labels, corrections should be formatted as
        # [[indices_1, labels_1], [indices_2, labels_2], ...]
        if len(corrections) > 0:
            indices = torch.cat([torch.tensor(x[0]) for x in corrections])
            pseudo_labels = torch.cat([torch.tensor(x[1]) for x in corrections])

            # Dump pseudo labels to file
            label_file = f"{self.server_id}-client-{self.client_id}-labels.pt"
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