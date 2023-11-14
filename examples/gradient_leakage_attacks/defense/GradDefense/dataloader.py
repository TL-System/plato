import numpy as np
import torch
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

DEFAULT_NUM_WORKERS = 8
rootset_per_class = 5
rootset_size = 50

# TODO: total_num_samples is not used


def extract_root_set(
    dataset: Dataset,
    sample_per_class: int = rootset_per_class,
    total_num_samples: int = rootset_size,
    seed: int = None,
):
    num_classes = len(dataset.classes)
    class2sample = {i: [] for i in range(num_classes)}
    select_indices = []
    if seed == None:
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
    rootset_indices, __ = extract_root_set(trainset)
    root_set = Subset(trainset, rootset_indices)
    root_dataloader = DataLoader(
        root_set, batch_size=len(root_set), num_workers=DEFAULT_NUM_WORKERS
    )

    return root_dataloader
