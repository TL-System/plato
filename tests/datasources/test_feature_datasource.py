"""Tests for feature datasource used in split learning."""

import torch

from plato.datasources.feature import DataSource


def test_feature_datasource_expands_batched_features():
    """Dataset should expand batched tensors into per-sample entries."""
    batch_size = 4
    features = torch.randn(batch_size, 8, 4, 4)
    targets = torch.randint(0, 10, (batch_size,))

    datasource = DataSource([[[(features, targets)]]])

    assert len(datasource) == batch_size

    sample_feature, sample_target = datasource[0]
    assert sample_feature.shape == (8, 4, 4)
    assert sample_target.ndim == 0


def test_feature_datasource_handles_unbatched_entries():
    """Existing single-sample entries should be preserved."""
    feature = torch.randn(8, 4, 4)
    target = torch.tensor(1)

    datasource = DataSource([[[(feature, target)]]])

    assert len(datasource) == 1
    sample_feature, sample_target = datasource[0]
    assert torch.equal(sample_feature, feature)
    assert torch.equal(sample_target, target)


def test_feature_datasource_defaults_label_when_missing():
    """Features without labels should receive a zero label."""
    feature = torch.randn(8, 4, 4)

    datasource = DataSource([[[(feature,)]]])

    assert len(datasource) == 1
    _, sample_target = datasource[0]
    assert sample_target.item() == 0


def test_feature_datasource_ignores_non_tuple_entries():
    """Non-tuple items should be skipped from the resulting dataset."""
    feature = torch.randn(8, 4, 4)
    target = torch.tensor(3)

    datasource = DataSource([["noise", (feature, target)]])

    assert len(datasource) == 1
    _, sample_target = datasource[0]
    assert sample_target.item() == 3
