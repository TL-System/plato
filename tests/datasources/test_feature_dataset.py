"""Unit tests for FeatureDataset."""

import torch

from plato.datasources.feature_dataset import FeatureDataset


def test_feature_dataset_returns_feature_and_label():
    """Dataset should return the feature and label unmodified."""
    feature = torch.randn(3, 32, 32)
    label = torch.tensor(5)

    dataset = FeatureDataset([(feature, label)])

    loaded_feature, loaded_label = dataset[0]

    assert torch.equal(loaded_feature, feature)
    assert loaded_label.item() == label.item()


def test_feature_dataset_handles_multiple_samples():
    """Dataset should iterate over all provided samples."""
    features = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    samples = [(features[i], labels[i]) for i in range(len(labels))]

    dataset = FeatureDataset(samples)

    assert len(dataset) == len(samples)
    all_labels = [dataset[i][1].item() for i in range(len(dataset))]
    assert all_labels == labels.tolist()


def test_feature_dataset_ignores_extra_fields():
    """Dataset should ignore additional fields beyond feature and label."""
    feature = torch.randn(3, 32, 32)
    label = torch.tensor(2)
    extra = {"meta": 123}

    dataset = FeatureDataset([(feature, label, extra)])

    loaded_feature, loaded_label = dataset[0]

    assert torch.equal(loaded_feature, feature)
    assert loaded_label.item() == label.item()
