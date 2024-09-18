"""
Dataloader of GradDefense

Reference:
Wang et al., "Protect Privacy from Gradient Leakage Attack in Federated Learning," INFOCOM 2022.
https://github.com/wangjunxiao/GradDefense
"""

import numpy as np
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

DEFAULT_NUM_WORKERS = 8
ROOTSET_PER_CLASS = 5
ROOTSET_SIZE = 50


def extract_root_set(
    dataset: Dataset,
    sample_per_class: int = ROOTSET_PER_CLASS,
    seed: int = None,
):
    """Extract root dataset."""
    num_classes = len(dataset.classes)
    class2sample = {i: [] for i in range(num_classes)}
    select_indices = []
    if seed is None:
        index_pool = range(len(dataset))
    else:
        index_pool = np.random.RandomState(seed=seed).permutation(len(dataset))
    for i in index_pool:
        current_class = dataset[i][1]
        if len(class2sample[current_class]) < sample_per_class:
            class2sample[current_class].append(i)
            select_indices.append(i)
        elif len(select_indices) == sample_per_class * num_classes:
            break
    return select_indices, class2sample


def get_root_set_loader(trainset):
    """Obtain root dataset loader."""
    rootset_indices, __ = extract_root_set(trainset)
    root_set = Subset(trainset, rootset_indices)
    root_dataloader = DataLoader(
        root_set, batch_size=len(root_set), num_workers=DEFAULT_NUM_WORKERS
    )

    return root_dataloader
