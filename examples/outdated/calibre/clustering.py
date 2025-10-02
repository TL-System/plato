"""
Clustering based on encodings.
"""

import torch

from sklearn.cluster import KMeans


def kmeans_clustering(features, n_clusters, max_iter=200):
    """Cluster features using the K-means algorithm."""
    device = features.device

    features = features.detach().cpu().numpy()
    kmeans = KMeans(n_init="auto", n_clusters=n_clusters, max_iter=max_iter).fit(
        features
    )
    cluster_ids = torch.from_numpy(kmeans.labels_).int()
    centroids = torch.from_numpy(kmeans.cluster_centers_).float()
    centroids = torch.nn.functional.normalize(centroids, dim=1)

    cluster_ids = cluster_ids.to(device)
    centroids = centroids.to(device)
    return cluster_ids, centroids
