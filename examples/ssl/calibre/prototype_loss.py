"""
Implementation of prototype loss for Calibre algorithm.
"""

import torch
from torch import nn
from torch.nn import functional as F


def cosine_distance(prototypes: torch.Tensor, queries: torch.Tensor):
    """Compute the consine similarity between prototypes and queries.

    :params prototypes: A `torch.Tensor` with shape [n_proto, D]
    :params queries: A `torch.Tensor` with shape [n_samples, D]

    :return A `torch.Tensor` with shape [n_samples, n_proto]

    """
    # Assumes both prototypes and queries are already normalized
    return -torch.mm(queries, prototypes.t())


def euclidean_distance(prototypes: torch.Tensor, queries: torch.Tensor):
    """Compute the euclidean distance between prototypes and queries.

    :params prototypes: A `torch.Tensor` with shape [n_proto, D]
    :params queries: A `torch.Tensor` with shape [n_samples, D]

    :return A `torch.Tensor` with shape [n_samples, n_proto]
    """
    distance = torch.cdist(prototypes, queries, p=2)
    return torch.square(distance)


distance_fns = {"euclidean": euclidean_distance, "cosine": cosine_distance}


def get_prototype_loss(
    prototypes: torch.Tensor,
    queries: torch.Tensor,
    query_labels: torch.Tensor,
    distance_type: str = "euclidean",
):
    """Get the prototype loss."""
    # 1. get the distance function
    dis_fn = distance_fns[distance_type]

    # 2. normalize the inputs with epsilon to prevent NaN from zero-norm vectors
    prototypes = nn.functional.normalize(prototypes, dim=1, eps=1e-8)
    queries = nn.functional.normalize(queries, dim=1, eps=1e-8)

    # 3. compute the distances
    # with shape, [n_proto, n_samples]
    distances = dis_fn(prototypes, queries)
    # convert to [n_samples, n_proto]
    distances = distances.T

    logits = -distances

    # Calculate logits with softmax
    return F.cross_entropy(logits, query_labels)
