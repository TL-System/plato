"""
Sensitivity computation of GradDefense

Reference:
Wang et al., "Protect Privacy from Gradient Leakage Attack in Federated Learning," INFOCOM 2022.
https://github.com/wangjunxiao/GradDefense
"""

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader


def compute_sens(
    model: nn.Module,
    rootset_loader: DataLoader,
    device: torch.device,
    loss_fn=nn.CrossEntropyLoss(),
):
    """Compute sensitivity."""
    x, y = next(iter(rootset_loader))

    x = x.to(device).requires_grad_()
    y = y.to(device)
    model = model.to(device)

    # Compute prediction and loss
    try:
        pred, _ = model(x)
    except:
        pred = model(x)

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
        f_norm_sum = 0
        for sample_vjp in layer_vjp:
            # Sample-wise Frobenius norm
            f_norm_sum += torch.norm(sample_vjp)
        f_norm = f_norm_sum / len(layer_vjp)
        sensitivity.append(f_norm.cpu().numpy())

    return sensitivity
