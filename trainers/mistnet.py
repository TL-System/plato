"""
The federated learning trainer for MistNet, used by both the client and the
server.

Reference:

P. Wang, et al. "MistNet: Towards Private Neural Network Training with Local
Differential Privacy," found in docs/papers.
"""

import torch
import numpy as np

from config import Config
from utils import unary_encoding
from trainers import trainer


class FeatureDataset(torch.utils.data.Dataset):
    """Used to prepare a feature dataset for a DataLoader in PyTorch."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label


class Trainer(trainer.Trainer):
    """A federated learning trainer for MistNet, used by both the client and the
    server.
    """
    def extract_features(self, dataset, cut_layer: str, epsilon=None):
        """Extracting features using layers before the cut_layer.

        dataset: The training or testing dataset.
        cut_layer: Layers before this one will be used for extracting features.
        epsilon: If epsilon is not None, local differential privacy should be
                applied to the features extracted.
        """
        self.model.eval()

        batch_size = Config().trainer.batch_size
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True)

        feature_dataset = []
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with torch.no_grad():
                logits = self.model.forward_to(inputs, cut_layer)
                if epsilon is not None:
                    logits = logits.detach().cpu().numpy()
                    logits = unary_encoding.encode(logits)
                    logits = unary_encoding.randomize(logits, epsilon)
                    logits = torch.from_numpy(logits.astype('float32'))

            for i in np.arange(logits.shape[0]):  # each sample in the batch
                feature_dataset.append((logits[i], targets[i]))

        return feature_dataset

    def train(self, trainset, cut_layer=None):
        super().train(FeatureDataset(trainset), cut_layer)

    def test(self, testset, batch_size, cut_layer=None):
        super().test(FeatureDataset(testset), batch_size, cut_layer)
