"""
Clustering based on the encodinds.
"""

import torch
from torch import nn
import kmeans_pytorch


def compute_kmeans(matrix, num_clusters, tol=1e-4, device=torch.device("cpu")):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """

    # convert to float
    matrix = matrix.float()

    # transfer to device
    matrix = matrix.to(device)

    # initialize
    initial_state = kmeans_pytorch.initialize(matrix, num_clusters)

    iteration = 0
    while True:
        dis = kmeans_pytorch.pairwise_distance(matrix, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_previous = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(matrix, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(torch.sum((initial_state - initial_state_previous) ** 2, dim=1))
        )

        # increment iteration
        iteration = iteration + 1

        if center_shift**2 < tol:
            break

    return choice_cluster.cpu(), initial_state.cpu()


def kmeans_clustering(encodings, n_clusters=10, device=None):
    """Cluster the samples based on the features."""
    # kmeans
    if device is None:
        device = (
            encodings.get_device()
            if encodings.is_cuda
            else torch.device("mps")
            if encodings.is_mps
            else torch.device("cpu")
        )
    encodings = nn.functional.normalize(encodings, dim=1)

    # # the output tensors will be placed under the cpu
    cluster_ids, cluster_centers = compute_kmeans(
        matrix=encodings,
        num_clusters=n_clusters,
        device=device,
    )
    # move them to the device
    cluster_ids = cluster_ids.to(device)
    cluster_centers = cluster_centers.to(device)

    return cluster_ids, cluster_centers
