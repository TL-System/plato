"""
The feature dataset server received from clients.
"""

from typing import Any, Iterable

import torch

from plato.datasources import base


class DataSource(base.DataSource):
    """The feature dataset."""

    def __init__(self, features, **kwargs):
        super().__init__()

        self.feature_dataset = []

        for item in self._yield_items(features):
            self._append_feature(item)

        self.trainset = self.feature_dataset
        self.testset = []

    def __len__(self):
        return len(self.trainset)

    def __getitem__(self, item):
        return self.trainset[item]

    def _append_feature(self, item: Any) -> None:
        """
        Append flattened feature items, expanding batched tensors into per-sample entries.
        """
        if isinstance(item, tuple):
            if len(item) >= 2:
                data, target = item[0], item[1]
            elif len(item) == 1:
                data = item[0]
                target = torch.tensor(0, dtype=torch.long)
            else:
                return

            if torch.is_tensor(data) and torch.is_tensor(target):
                if (
                    data.dim() >= 1
                    and target.dim() >= 1
                    and data.size(0) == target.size(0)
                ):
                    for i in range(data.size(0)):
                        feature = data[i]
                        label = target[i]
                        if torch.is_tensor(label):
                            label = label.squeeze()
                        self.feature_dataset.append((feature, label))
                    return

            self.feature_dataset.append((data, target))
        else:
            # Non-tuple entries are ignored as they don't conform to (feature, label)
            pass

    def _yield_items(self, items: Iterable[Any]):
        """Recursively yield non-list items from nested iterables."""
        for item in items:
            if isinstance(item, list):
                yield from self._yield_items(item)
            else:
                yield item
