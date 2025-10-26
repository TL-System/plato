"""
Sensitivity computation of GradDefense.

Reference:
Wang et al., "Protect Privacy from Gradient Leakage Attack in Federated Learning," INFOCOM 2022.
https://github.com/wangjunxiao/GradDefense
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader


def compute_sens(
    model: nn.Module,
    rootset_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    loss_fn: nn.Module | None = None,
) -> list[float]:
    """Compute sensitivity."""
    x, y = next(iter(rootset_loader))

    x = x.to(device).requires_grad_()
    y = y.to(device)
    model = model.to(device)

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    # Compute prediction and loss
    pred = model(x)
    if isinstance(pred, tuple):
        pred = pred[0]

    loss = loss_fn(pred, y)
    # Backward propagation
    dy_dx = torch.autograd.grad(
        outputs=loss, inputs=model.parameters(), create_graph=True
    )

    vector_jacobian_products = []
    for layer in dy_dx:
        # Input-gradient Jacobian
        d2y_dx2 = torch.autograd.grad(
            outputs=layer,
            inputs=x,
            grad_outputs=torch.ones_like(layer),
            retain_graph=True,
        )[0]
        vector_jacobian_products.append(d2y_dx2.detach().clone())

    sensitivity = []
    for layer_vjp in vector_jacobian_products:
        sum_norms = torch.zeros((), device=layer_vjp.device)
        for sample_vjp in layer_vjp:
            # Sample-wise Frobenius norm
            sum_norms = sum_norms + torch.norm(sample_vjp)
        f_norm = sum_norms / layer_vjp.shape[0]
        sensitivity.append(float(f_norm.detach().cpu()))

    return sensitivity
