"""
The LIVE Netflix Video QoE datasets.

For more information about the datasets, refer to
https://live.ece.utexas.edu/research/LIVE_NFLXStudy/nflx_index.html.
"""

import copy
import logging
import os
import re

import numpy as np
import scipy.io as sio
import torch

from plato.config import Config
from plato.datasources import base

FEATURE_NAMES = ["VQA", "R$_1$", "R$_2$", "M", "I"]


class QoENFLXDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        VQA = torch.from_numpy(self.dataset[idx, [0]].astype(np.float)).float()
        R1 = torch.from_numpy(self.dataset[idx, [1]].astype(np.float)).float()
        R2 = self.dataset[idx, [2]].astype(np.int)
        M = torch.from_numpy(self.dataset[idx, [3]].astype(np.float)).float()
        I = torch.from_numpy(self.dataset[idx, [4]].astype(np.float)).float()
        label = self.dataset[idx, [5]]
        sample = {"VQA": VQA, "R1": R1, "R2": R2, "Mem": M, "Impair": I, "label": label}

        return sample


class DataSource(base.DataSource):
    """A data source for QoENFLX datasets."""

    def __init__(self, **kwargs):
        super().__init__()

        logging.info("Dataset: QoENFLX")
        dataset_path = Config().params["data_path"] + "/QoENFLX/VideoATLAS/"
        db_files = os.listdir(dataset_path)
        db_files.sort(
            key=lambda var: [
                int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
            ]
        )
        Nvideos = len(db_files)

        pre_load_train_test_data_LIVE_Netflix = sio.loadmat(
            Config().params["data_path"]
            + "/QoENFLX/TrainingMatrix_LIVENetflix_1000_trials.mat"
        )["TrainingMatrix_LIVENetflix_1000_trials"]

        # randomly pick a trial out of the 1000
        nt_rand = np.random.choice(
            np.shape(pre_load_train_test_data_LIVE_Netflix)[1], 1
        )
        n_train = [
            ind
            for ind in range(0, Nvideos)
            if pre_load_train_test_data_LIVE_Netflix[ind, nt_rand] == 1
        ]
        n_test = [
            ind
            for ind in range(0, Nvideos)
            if pre_load_train_test_data_LIVE_Netflix[ind, nt_rand] == 0
        ]

        X = np.zeros((len(db_files), len(FEATURE_NAMES)))
        y = np.zeros((len(db_files), 1))

        feature_labels = list()
        for typ in FEATURE_NAMES:
            if typ == "VQA":
                feature_labels.append("STRRED" + "_" + "mean")
            elif typ == "R$_1$":
                feature_labels.append("ds_norm")
            elif typ == "R$_2$":
                feature_labels.append("ns")
            elif typ == "M":
                feature_labels.append("tsl_norm")
            else:
                feature_labels.append("lt_norm")

        for i, f in enumerate(db_files):
            data = sio.loadmat(dataset_path + f)
            for feat_cnt, feat in enumerate(feature_labels):
                X[i, feat_cnt] = data[feat]
            y[i] = data["final_subj_score"]

        X_train_before_scaling = X[n_train, :]
        X_test_before_scaling = X[n_test, :]
        y_train = y[n_train]
        y_test = y[n_test]

        self.trainset = copy.deepcopy(
            np.concatenate((X_train_before_scaling, y_train), axis=1)
        )
        self.testset = copy.deepcopy(
            np.concatenate((X_test_before_scaling, y_test), axis=1)
        )

    @staticmethod
    def get_train_loader(batch_size, trainset, sampler, shuffle=False):
        """The custom train loader for QoENFLX."""
        return torch.utils.data.DataLoader(
            QoENFLXDataset(trainset),
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
        )

    @staticmethod
    def get_test_loader(batch_size, testset):
        """The custom test loader for QoENFLX."""
        return torch.utils.data.DataLoader(
            QoENFLXDataset(testset), batch_size=batch_size, shuffle=False
        )
