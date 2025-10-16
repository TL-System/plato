"""
The feature dataset server received from clients.
"""

import torch

from plato.datasources import base


def _flatten_feature_iterables(items):
    """Flatten nested lists while keeping feature tuples intact."""
    for item in items:
        if isinstance(item, list):
            yield from _flatten_feature_iterables(item)
        else:
            yield item


def _expand_feature_batch(feature_pair):
    """
    Expand a batched feature tuple into individual samples when possible.

    Args:
        feature_pair: Tuple of (features, targets) possibly batched.

    Returns:
        Iterable of (feature, target) pairs.
    """
    if not isinstance(feature_pair, tuple) or len(feature_pair) != 2:
        return [feature_pair]

    features, targets = feature_pair

    if torch.is_tensor(features) and torch.is_tensor(targets):
        if features.dim() >= 1 and targets.dim() >= 1 and features.size(0) == targets.size(0):
            return list(zip(features, targets))

    return [feature_pair]


class DataSource(base.DataSource):
    """The feature dataset."""

    def __init__(self, features, **kwargs):
        super().__init__()

        self.feature_dataset = []

        for feature_pair in _flatten_feature_iterables(features):
            expanded = _expand_feature_batch(feature_pair)
            for sample in expanded:
                if isinstance(sample, torch.Tensor):
                    # Fall back to zero label if only feature tensor provided.
                    self.feature_dataset.append((sample, torch.zeros(1)))
                elif isinstance(sample, tuple):
                    if len(sample) == 1:
                        self.feature_dataset.append((sample[0], torch.zeros(1)))
                    elif len(sample) >= 2:
                        self.feature_dataset.append((sample[0], sample[1]))
                else:
                    self.feature_dataset.append((sample, torch.zeros(1)))

        self.trainset = self.feature_dataset
        self.testset = []

    def __len__(self):
        return len(self.trainset)

    def __getitem__(self, item):
        return self.trainset[item]
