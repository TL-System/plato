#!/usr/bin/env python3
"""
Test to verify that old and new LG-FedAvg implementations produce identical results.

This test creates a simple model and dataset, then trains using both implementations
to verify they update parameters identically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class SimpleNet(nn.Module):
    """Simple network for testing with clearly separated layers."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, 1)
        self.conv2 = nn.Conv2d(10, 20, 3, 1)
        self.fc1 = nn.Linear(20 * 5 * 5, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def freeze_model_old(model, layer_names):
    """Old implementation's freeze function."""
    if layer_names is not None:
        for name, param in model.named_parameters():
            if any(param_name in name for param_name in layer_names):
                param.requires_grad = False


def activate_model_old(model, layer_names):
    """Old implementation's activate function."""
    if layer_names is not None:
        for name, param in model.named_parameters():
            if any(param_name in name for param_name in layer_names):
                param.requires_grad = True


def training_step_old(model, optimizer, examples, labels, loss_criterion,
                      global_layer_names, local_layer_names):
    """
    Old implementation's training step.
    Mimics the behavior of lgfedavg_trainer.Trainer.perform_forward_and_backward_passes
    """
    # First pass: Train local layers only
    freeze_model_old(model, global_layer_names)
    activate_model_old(model, local_layer_names)

    optimizer.zero_grad()
    outputs = model(examples)
    loss = loss_criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # Second pass: Train global layers only
    activate_model_old(model, global_layer_names)
    freeze_model_old(model, local_layer_names)

    optimizer.zero_grad()
    outputs = model(examples)
    loss = loss_criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    return loss


def set_requires_grad_new(model, layer_names, requires_grad):
    """New implementation's set_requires_grad function."""
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = requires_grad


def training_step_new(model, optimizer, examples, labels, loss_criterion,
                      global_layer_names, local_layer_names):
    """
    New implementation's training step.
    Mimics the behavior of LGFedAvgStepStrategy.training_step
    """
    # First pass: Train local layers
    set_requires_grad_new(model, global_layer_names, False)  # Freeze global
    set_requires_grad_new(model, local_layer_names, True)     # Activate local

    optimizer.zero_grad()
    outputs = model(examples)
    loss_first = loss_criterion(outputs, labels)
    loss_first.backward()
    optimizer.step()

    # Second pass: Train global layers
    set_requires_grad_new(model, local_layer_names, False)   # Freeze local
    set_requires_grad_new(model, global_layer_names, True)    # Activate global

    optimizer.zero_grad()
    outputs = model(examples)
    loss_second = loss_criterion(outputs, labels)
    loss_second.backward()
    optimizer.step()

    # Re-enable all gradients
    set_requires_grad_new(model, global_layer_names, True)
    set_requires_grad_new(model, local_layer_names, True)

    return loss_second


def test_implementations_match():
    """Test that old and new implementations produce identical parameter updates."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create synthetic dataset
    num_samples = 16
    x_data = torch.randn(num_samples, 1, 28, 28)
    y_data = torch.randint(0, 10, (num_samples,))
    
    # Define layer configuration (matching config file)
    local_layer_names = ['conv1', 'conv2', 'fc1']
    global_layer_names = ['fc2']
    
    print("=" * 70)
    print("Testing LG-FedAvg Implementation Equivalence")
    print("=" * 70)
    print(f"\nLocal layers: {local_layer_names}")
    print(f"Global layers: {global_layer_names}")
    print(f"Dataset: {num_samples} samples\n")
    
    # Test 1: Initialize two identical models
    torch.manual_seed(42)
    model_old = SimpleNet()
    
    torch.manual_seed(42)
    model_new = SimpleNet()
    
    # Verify models start identical
    params_match = True
    for (name_old, param_old), (name_new, param_new) in zip(
        model_old.named_parameters(), model_new.named_parameters()
    ):
        if not torch.allclose(param_old, param_new):
            params_match = False
            print(f"❌ Initial parameters don't match for {name_old}")
    
    if params_match:
        print("✓ Models initialized identically")
    
    # Create optimizers with same seed
    optimizer_old = torch.optim.SGD(model_old.parameters(), lr=0.01)
    optimizer_new = torch.optim.SGD(model_new.parameters(), lr=0.01)
    
    # Loss criterion
    loss_criterion = nn.CrossEntropyLoss()
    
    # Run one training step with OLD implementation
    model_old.train()
    loss_old = training_step_old(
        model_old, optimizer_old, x_data, y_data, loss_criterion,
        global_layer_names, local_layer_names
    )
    
    # Run one training step with NEW implementation
    model_new.train()
    loss_new = training_step_new(
        model_new, optimizer_new, x_data, y_data, loss_criterion,
        global_layer_names, local_layer_names
    )
    
    print(f"\nLoss after one step:")
    print(f"  Old implementation: {loss_old.item():.6f}")
    print(f"  New implementation: {loss_new.item():.6f}")
    print(f"  Difference: {abs(loss_old.item() - loss_new.item()):.10f}")
    
    # Compare final parameters
    print("\nParameter comparison after one training step:")
    all_match = True
    for (name_old, param_old), (name_new, param_new) in zip(
        model_old.named_parameters(), model_new.named_parameters()
    ):
        match = torch.allclose(param_old, param_new, rtol=1e-5, atol=1e-7)
        if match:
            print(f"  ✓ {name_old:20s} - MATCH")
        else:
            print(f"  ✗ {name_old:20s} - DIFFER")
            diff = (param_old - param_new).abs().max().item()
            print(f"    Max difference: {diff:.10f}")
            all_match = False
    
    # Check gradient states
    print("\nGradient states after training:")
    print("  OLD implementation:")
    for name, param in model_old.named_parameters():
        print(f"    {name:20s} requires_grad={param.requires_grad}")
    
    print("\n  NEW implementation:")
    for name, param in model_new.named_parameters():
        print(f"    {name:20s} requires_grad={param.requires_grad}")
    
    # Final verdict
    print("\n" + "=" * 70)
    if all_match and torch.allclose(loss_old, loss_new):
        print("✓✓✓ SUCCESS: Both implementations are IDENTICAL ✓✓✓")
        print("=" * 70)
        return True
    else:
        print("✗✗✗ FAILURE: Implementations DIFFER ✗✗✗")
        print("=" * 70)
        return False


def test_multiple_steps():
    """Test multiple training steps to ensure consistent behavior over time."""
    
    torch.manual_seed(42)
    
    # Create dataset
    num_samples = 32
    x_data = torch.randn(num_samples, 1, 28, 28)
    y_data = torch.randint(0, 10, (num_samples,))
    
    local_layer_names = ['conv1', 'conv2', 'fc1']
    global_layer_names = ['fc2']
    
    print("\n" + "=" * 70)
    print("Testing Multiple Training Steps")
    print("=" * 70)
    
    # Initialize models
    torch.manual_seed(42)
    model_old = SimpleNet()
    
    torch.manual_seed(42)
    model_new = SimpleNet()
    
    optimizer_old = torch.optim.SGD(model_old.parameters(), lr=0.01)
    optimizer_new = torch.optim.SGD(model_new.parameters(), lr=0.01)
    
    loss_criterion = nn.CrossEntropyLoss()
    
    num_steps = 5
    print(f"\nRunning {num_steps} training steps...\n")
    
    all_match = True
    for step in range(num_steps):
        # Train with old implementation
        model_old.train()
        loss_old = training_step_old(
            model_old, optimizer_old, x_data, y_data, loss_criterion,
            global_layer_names, local_layer_names
        )
        
        # Train with new implementation  
        model_new.train()
        loss_new = training_step_new(
            model_new, optimizer_new, x_data, y_data, loss_criterion,
            global_layer_names, local_layer_names
        )
        
        # Compare
        params_match = True
        for (_, param_old), (_, param_new) in zip(
            model_old.named_parameters(), model_new.named_parameters()
        ):
            if not torch.allclose(param_old, param_new, rtol=1e-5, atol=1e-7):
                params_match = False
                break
        
        status = "✓" if params_match else "✗"
        print(f"  Step {step + 1}: Loss_old={loss_old.item():.6f}, "
              f"Loss_new={loss_new.item():.6f}, "
              f"Params match: {status}")
        
        if not params_match:
            all_match = False
    
    print("\n" + "=" * 70)
    if all_match:
        print("✓✓✓ SUCCESS: Implementations remain identical over multiple steps ✓✓✓")
    else:
        print("✗✗✗ FAILURE: Implementations diverge over multiple steps ✗✗✗")
    print("=" * 70)
    
    return all_match


if __name__ == "__main__":
    # Run tests
    test1_passed = test_implementations_match()
    test2_passed = test_multiple_steps()
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Single step test: {'PASS ✓' if test1_passed else 'FAIL ✗'}")
    print(f"Multiple steps test: {'PASS ✓' if test2_passed else 'FAIL ✗'}")
    print("=" * 70)
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Implementations are algorithmically identical. 🎉\n")
        exit(0)
    else:
        print("\n⚠️  Some tests failed. Implementations may differ. ⚠️\n")
        exit(1)
