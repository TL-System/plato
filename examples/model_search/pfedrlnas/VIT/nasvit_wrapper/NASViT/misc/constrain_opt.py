# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Constraint(torch.optim.Optimizer):
    """
    first_step: gradient of objective 1, and log the grad,
    second_step: gradient of objective 2, and do something based on the logged gradient at step one
    closure: the objective 2 for second step
    """

    def __init__(
        self, params, base_optimizer, g_star=0.05, alpha=1.0, beta=1.0, **kwargs
    ):
        defaults = dict(g_star=g_star, **kwargs)
        super(Constraint, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.g_star = g_star
        self.alpha = alpha
        self.beta = beta
        self.g_constraint = 0.0
        self.grad_inner = 0.0
        self.proj_student = True

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # p.add_( - p.grad * 1e-3) # SGD learning rate
                constraint_grad = (
                    torch.ones_like(p.grad) * p.grad
                )  # deepcopy, otherwise the c_grad would be a pointer
                self.state[p]["constraint_grad"] = constraint_grad

                if "constraint_grad_norm" in self.state[p].keys():
                    self.state[p]["constraint_grad_norm"] = (
                        1.0 * constraint_grad.norm()
                        + 0.0 * self.state[p]["constraint_grad_norm"]
                    )
                else:
                    self.state[p]["constraint_grad_norm"] = constraint_grad.norm()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False, step=True):
        """
        calculate the projection here
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    if "constraint_grad" in self.state[p].keys():
                        p.grad = self.state[p]["constraint_grad"]
                    else:
                        continue

                if "constraint_grad" not in self.state[p].keys():
                    continue

                if "grad_inner" not in self.state[p].keys():
                    self.state[p]["grad_inner"] = (
                        p.grad * self.state[p]["constraint_grad"]
                    ).sum()
                else:
                    self.state[p]["grad_inner"] = (
                        0.0 * self.state[p]["grad_inner"]
                        + 1.0 * (p.grad * self.state[p]["constraint_grad"]).sum()
                    )

                if self.proj_student:
                    adaptive_step_x = 0.0

                    p.grad.add_(
                        self.state[p]["constraint_grad"] * adaptive_step_x
                        + self.state[p]["constraint_grad"]
                    )
                else:
                    adaptive_step_x = self.state[p]["grad_inner"] / (
                        1e-6 + p.grad.norm().pow(2)
                    )
                    adaptive_step_x = torch.clamp(-adaptive_step_x, min=0.0, max=2.0)

                    p.grad.add_(
                        p.grad * adaptive_step_x + self.state[p]["constraint_grad"]
                    )
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None, g_value=None, g_constraint=None):
        assert closure is not None, (
            "Requires closure, but it was not provided, raise an error"
        )
        assert g_value is not None, "Requires g value"
        assert g_constraint is not None, "Requires g constraint"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.g_value = g_value
        self.g_constraint = g_constraint
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
