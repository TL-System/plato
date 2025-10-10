"""
Clustering based on encodings.
"""

import torch
from sklearn.cluster import KMeans


def kmeans_clustering(features, n_clusters, max_iter=200):
    """Cluster features using the K-means algorithm."""
    device = features.device

    features = features.detach().cpu().numpy()

    # Check for NaN values and replace them with zeros
    # This can happen due to numerical instability in normalization
    if torch.isnan(torch.from_numpy(features)).any():
        features = torch.nan_to_num(torch.from_numpy(features), nan=0.0).numpy()

    kmeans = KMeans(n_init="auto", n_clusters=n_clusters, max_iter=max_iter).fit(
        features
    )
    cluster_ids = torch.from_numpy(kmeans.labels_).long()
    centroids = torch.from_numpy(kmeans.cluster_centers_).float()
    centroids = torch.nn.functional.normalize(centroids, dim=1, eps=1e-8)

    cluster_ids = cluster_ids.to(device)
    centroids = centroids.to(device)
    return cluster_ids, centroids
