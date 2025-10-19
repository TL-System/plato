"""Behavioural tests for key model implementations."""

import pytest
import torch

from plato.models import lenet5, resnet


def test_lenet5_forward_produces_log_probs():
    """The full LeNet-5 forward pass should emit log-probabilities per class."""
    model = lenet5.Model(num_classes=10)
    model.eval()
    batch = torch.randn(2, 1, 28, 28)

    outputs = model(batch)

    assert outputs.shape == (2, 10)
    probs = outputs.exp().sum(dim=1)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-5)


def test_lenet5_split_forward_matches_full():
    """Server-side forward after the cut layer should agree with the full model."""
    torch.manual_seed(0)
    full_model = lenet5.Model()

    client_model = lenet5.Model()
    client_model.load_state_dict(full_model.state_dict())
    client_model.cut_layer = "relu1"

    server_model = lenet5.Model(cut_layer="relu1")
    server_model.load_state_dict(full_model.state_dict())
    server_model.train()

    batch = torch.randn(4, 1, 28, 28)
    cut_activations = client_model.forward_to(batch)

    split_outputs = server_model.forward(cut_activations)
    full_outputs = full_model(batch)

    assert torch.allclose(split_outputs, full_outputs, atol=1e-6)


def test_resnet_rejects_unknown_variant():
    """The ResNet factory should validate the requested variant name."""
    with pytest.raises(ValueError):
        resnet.Model.get(model_name="resnet_13")


def test_resnet_split_roundtrip_matches_full():
    """Forward-to/from roundtrip should reproduce the full forward output."""
    torch.manual_seed(0)
    model = resnet.Model.get(model_name="resnet_18", num_classes=7, cut_layer="layer2")
    model.eval()

    batch = torch.randn(3, 3, 32, 32)
    cut_activations = model.forward_to(batch)
    split_outputs = model.forward_from(cut_activations)
    full_outputs = model.forward(batch)

    assert split_outputs.shape == (3, 7)
    assert torch.allclose(split_outputs, full_outputs, atol=1e-6)
