"""Tests for FeatureDataset that prepares split learning features."""

import torch

from plato.datasources.feature_dataset import FeatureDataset


def test_feature_dataset_extracts_first_two_elements():
    """Ensure extra elements in sample tuples are ignored."""
    features = torch.randn(8, 4, 4)
    target = torch.tensor(3)
    extra = torch.tensor([1, 2, 3])

    dataset = FeatureDataset([(features, target, extra)])

    loaded_feature, loaded_target = dataset[0]

    assert torch.equal(loaded_feature, features)
    assert torch.equal(loaded_target, target)


def test_feature_dataset_handles_plain_tensor():
    """Fallback when sample is just a tensor (no label)."""
    features = torch.randn(8, 4, 4)

    dataset = FeatureDataset([features])

    loaded_feature, loaded_target = dataset[0]

    assert torch.equal(loaded_feature, features)
    assert loaded_target.shape == (1,)
    assert torch.allclose(loaded_target, torch.zeros_like(loaded_target))
