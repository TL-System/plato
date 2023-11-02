"""
Clustering based on the encodinds.
"""

import torch


def kmeans_clustering(features, n_clusters, max_iters=100):
    """Computing the keams"""
    # Initialize centroids randomly
    centroids = features[torch.randperm(features.size(0))[:n_clusters]]

    for _ in range(max_iters):
        # Assign each data point to the nearest centroid
        distances = torch.cdist(features, centroids)  # Compute distances
        cluster_ids = torch.argmin(distances, dim=1)  # Assign labels

        # Update centroids as the mean of the assigned data points
        new_centroids = torch.stack(
            [features[cluster_ids == i].mean(0) for i in range(n_clusters)]
        )

        # Check for convergence
        if torch.all(new_centroids == centroids):
            break

        centroids = new_centroids

    return cluster_ids, centroids
