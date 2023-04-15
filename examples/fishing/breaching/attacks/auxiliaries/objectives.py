"""Various objective functions that can be re-used for multiple attacks."""

import torch
from typing import List

from .make_functional import make_functional_with_buffers


class GradientLoss(torch.nn.Module):
    """Super-class to simplify gradient-based objectives."""

    def __init__(self):
        super().__init__()
        self.task_regularization = 0

    def initialize(self, loss_fn, cfg_impl, local_hyperparams=None):
        self.loss_fn = loss_fn
        self.local_hyperparams = local_hyperparams
        if self.local_hyperparams is None:
            self._grad_fn = self._grad_fn_single_step
        else:
            self._grad_fn = self._grad_fn_multi_step

        self.cfg_impl = cfg_impl

    def forward(self, model, gradient_data, candidate, labels):
        gradient, task_loss = self._grad_fn(model, candidate, labels)
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            objective = self.gradient_based_loss(gradient, gradient_data)
        if self.task_regularization != 0:
            objective += self.task_regularization * task_loss
        return objective, task_loss.detach()

    def gradient_based_loss(self, gradient_rec, gradient_data):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()

    def _grad_fn_single_step(self, model, candidate, labels):
        """Compute a single gradient."""
        model.zero_grad()
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            task_loss = self.loss_fn(model(candidate), labels)
        gradient = torch.autograd.grad(task_loss, model.parameters(), create_graph=True)
        return gradient, task_loss

    def _grad_fn_multi_step(self, model, candidate, labels):
        """Compute the full graph for multiple local update steps."""
        model.zero_grad()
        func_model, params, buffers = make_functional_with_buffers(model)
        initial_params = [p.clone() for p in params]

        seen_data_idx = 0
        for i in range(self.local_hyperparams["steps"]):
            data = candidate[seen_data_idx : seen_data_idx + self.local_hyperparams["data_per_step"]]
            seen_data_idx += self.local_hyperparams["data_per_step"]
            seen_data_idx = seen_data_idx % candidate.shape[0]
            labels = self.local_hyperparams["labels"][i]
            with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
                task_loss = self.loss_fn(func_model(params, buffers, data), labels)

            step_gradient = torch.autograd.grad(task_loss, params, create_graph=True)

            # Update parameters in graph:
            params = [param - self.local_hyperparams["lr"] * grad for param, grad in zip(params, step_gradient)]

        # Finally return differentiable difference in state:
        gradient = [p_local - p_server for p_local, p_server in zip(params, initial_params)]

        # Return last loss as the "best" task loss
        return gradient, task_loss


class Euclidean(GradientLoss):
    """Gradient matching based on the euclidean distance of two gradient vectors."""

    def __init__(self, scale=1.0, task_regularization=0.0, **kwargs):
        super().__init__()
        self.scale = scale
        self.task_regularization = task_regularization

    def gradient_based_loss(self, gradient_rec, gradient_data):
        return self._euclidean(gradient_rec, gradient_data) * self.scale

    def __repr__(self):
        return f"Euclidean loss with scale={self.scale} and task reg={self.task_regularization}"

    @staticmethod
    @torch.jit.script
    def _euclidean(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor]):
        objective = gradient_rec[0].new_zeros(1,)
        for rec, data in zip(gradient_rec, gradient_data):
            objective += (rec - data).pow(2).sum()
        return 0.5 * objective


class EuclideanTag(GradientLoss):
    """Gradient matching based on the euclidean distance of two gradient vectors plus TAG regularizer

    from Deng et al., "TAG: Gradient Attack on Transformer-based Language Models"
    How to scale each layer is unclear to me based on the paper, so I am recycling decay schemes from
    the InvertingGradients repo.
    """

    def __init__(self, scale=1.0, task_regularization=0.0, tag_scale=0.1, scale_scheme="linear", **kwargs):
        super().__init__()
        self.scale = scale
        self.task_regularization = task_regularization
        self.tag_scale = tag_scale
        self.scale_scheme = scale_scheme
        self.weights = None

    def gradient_based_loss(self, gradient_rec, gradient_data):
        if self.weights is None:
            setup = dict(dtype=gradient_rec[0].dtype, device=gradient_rec[0].device)
            if self.scale_scheme == "linear":
                weights = torch.arange(len(gradient_rec), 0, -1, **setup) / len(gradient_rec)
            elif self.scale_scheme == "exp":
                weights = torch.arange(len(gradient_rec), 0, -1, **setup)
                weights = weights.softmax(dim=0)
                weights = weights / weights[0]
            else:
                weights = gradient_rec[0].new_ones(len(gradient_rec))
        return self._weighted_euclidean_l1(gradient_rec, gradient_data, weights, self.tag_scale) * self.scale

    def __repr__(self):
        return (
            f"Tag loss with scale={self.scale}, weight scheme {self.scale_scheme}, L1 scale {self.tag_scale} "
            f"and task reg={self.task_regularization}"
        )

    @staticmethod
    @torch.jit.script
    def _weighted_euclidean_l1(
        gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor], weights: torch.Tensor, tag_scale: float,
    ):
        objective = gradient_rec[0].new_zeros(1,)
        for rec, data, weight in zip(gradient_rec, gradient_data, weights):
            objective += (rec - data).pow(2).sum() + tag_scale * weight * (rec - data).abs().sum()
        return 0.5 * objective


class L1Loss(GradientLoss):
    """Gradient matching based on the L1 distance of two gradient vectors."""

    def __init__(self, scale=1.0, task_regularization=0.0, **kwargs):
        super().__init__()
        self.scale = scale
        self.task_regularization = task_regularization

    def gradient_based_loss(self, gradient_rec, gradient_data):
        return self._l1loss(gradient_rec, gradient_data) * self.scale

    def __repr__(self):
        return f"L1 loss with scale={self.scale} and task reg={self.task_regularization}"

    @staticmethod
    @torch.jit.script
    def _l1loss(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor]):
        objective = gradient_rec[0].new_zeros(1,)
        for rec, data in zip(gradient_rec, gradient_data):
            objective += (rec - data).abs().sum()
        return 0.5 * objective

        return objective


class CosineSimilarity(GradientLoss):
    """Gradient matching based on cosine similarity of two gradient vectors."""

    def __init__(self, scale=1.0, task_regularization=0.0, **kwargs):
        super().__init__()
        self.scale = scale
        self.task_regularization = task_regularization

    def gradient_based_loss(self, gradient_rec, gradient_data):
        return self._cosine_sim(gradient_rec, gradient_data) * self.scale

    def __repr__(self):
        return f"Cosine Similarity with scale={self.scale} and task reg={self.task_regularization}"

    @staticmethod
    @torch.jit.script
    def _cosine_sim(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor]):
        scalar_product = gradient_rec[0].new_zeros(1,)
        rec_norm = gradient_rec[0].new_zeros(1,)
        data_norm = gradient_rec[0].new_zeros(1,)

        for rec, data in zip(gradient_rec, gradient_data):
            scalar_product += (rec * data).sum()
            rec_norm += rec.pow(2).sum()
            data_norm += data.pow(2).sum()

        objective = 1 - scalar_product / (rec_norm.sqrt() * data_norm.sqrt())
        return objective


class AngularSimilarity(CosineSimilarity):
    """Gradient matching based on angular similarity of two gradient vectors.

    This is basically a more linear cosine similarity."""

    def __init__(self, scale=1.0, task_regularization=0.0, fudge_factor=1e-7, **kwargs):
        super().__init__()
        self.scale = scale
        self.task_regularization = task_regularization
        self.fudge_factor = 1e-7

    def gradient_based_loss(self, gradient_rec, gradient_data):
        cosine = 1 - self._cosine_sim(gradient_rec, gradient_data)
        angle = torch.acos(cosine.clamp(min=-1 + self.fudge_factor, max=1 - self.fudge_factor))

        return angle / torch.pi * self.scale

    def __repr__(self):
        return f"Angular Similarity with scale={self.scale} and task reg={self.task_regularization}"


class MaskedCosineSimilarity(GradientLoss):
    """Gradient matching based on cosine similarity of two gradient vectors.
    All positions that are zero in the data gradient are masked.
    """

    def __init__(self, scale=1.0, mask_value=1e-6, task_regularization=0.0, **kwargs):
        super().__init__()
        self.scale = scale
        self.mask_value = 1e-6
        self.task_regularization = task_regularization

    def __repr__(self):
        return f"Masked Cosine Similarity with scale={self.scale} and task reg={self.task_regularization}. Mask val={self.mask_value}"

    def gradient_based_loss(self, gradient_rec, gradient_data):
        scalar_product, rec_norm, data_norm = 0.0, 0.0, 0.0
        for rec, data in zip(gradient_rec, gradient_data):
            mask = data.abs() > self.mask_value
            scalar_product += (rec * data * mask).sum()
            rec_norm += (rec * mask).pow(2).sum()
            data_norm += (data * mask).pow(2).sum()

        objective = 1 - scalar_product / rec_norm.sqrt() / data_norm.sqrt()

        return objective * self.scale


class FastCosineSimilarity(GradientLoss):
    """Gradient matching based on cosine similarity of two gradient vectors.
    No gradient flows through the normalization."""

    def __init__(self, scale=1.0, task_regularization=0.0, **kwargs):
        super().__init__()
        self.scale = scale
        self.task_regularization = task_regularization

    def gradient_based_loss(self, gradient_rec, gradient_data):
        return self._cosine_sim(gradient_rec, gradient_data) * self.scale

    @staticmethod
    @torch.jit.script
    def _cosine_sim(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor]):
        scalar_product = gradient_rec[0].new_zeros(1,)
        rec_norm = gradient_rec[0].new_zeros(1,)
        data_norm = gradient_rec[0].new_zeros(1,)

        for rec, data in zip(gradient_rec, gradient_data):
            scalar_product += (rec * data).sum()
            rec_norm += rec.detach().pow(2).sum()
            data_norm += data.detach().pow(2).sum()

        objective = 1 - scalar_product / rec_norm.sqrt() / data_norm.sqrt()

        return objective

    def __repr__(self):
        return f"Fast Cosine Similarity with scale={self.scale} and task reg={self.task_regularization}"


class PearlmutterEuclidean(torch.nn.Module):
    """Use a first-order approximation of \nabla_x \nabla_g instead of the correct autograd value."""

    def __init__(
        self,
        scale=1.0,
        eps=1e-3,
        level_gradients=False,
        fudge_factor=1e-6,
        task_regularization=0.0,
        implementation="forward",
        **kwargs,
    ):
        super().__init__()
        self.scale = scale
        self.task_regularization = task_regularization

        self.eps = eps
        self.level_gradients = level_gradients
        self.fudge_factor = fudge_factor
        self.implementation = implementation

    def initialize(self, loss_fn, cfg_impl, local_hyperparams=None):
        self.loss_fn = loss_fn
        self.local_hyperparams = local_hyperparams
        if self.local_hyperparams is not None:
            raise ValueError("This loss is only implemented for local gradients so far.")

        self.cfg_impl = cfg_impl
        if self.implementation == "forward":
            self._forward_impl = self._forward_differences
        elif self.implementation == "backward":
            self._forward_impl = self._backward_differences
        elif self.implementation == "central":
            self._forward_impl = self._central_differences
        elif self.implementation == "upwind":
            self._forward_impl = self._upwind_differences
        else:
            raise ValueError(f"Invalid finite difference implementation {self.implementation} given.")

    def __repr__(self):
        return (
            f"Pearlmutter-type Finite Differences Loss with scale={self.scale} and task reg={self.task_regularization}."
            f"Finite Difference Eps: {self.eps}. Level gradients: {self.level_gradients}. "
            f"{f'Fudge-factor: {self.fudge_factor}' if self.level_gradients else ''}"
        )

    def forward(self, model, gradient_data, candidate, labels):
        """Run through model twice to approximate 2nd-order derivative on residual."""
        model.zero_grad()
        # Keep original parameters
        original_parameters = [p.detach().clone() for p in model.parameters()]
        # Compute derivative and accumulate into candidate.grad
        objective_value, task_loss = self._forward_impl(model, gradient_data, candidate, labels)

        # Unpatch model:# Want to do this faster, somehow
        for param, original_param in zip(model.parameters(), original_parameters):
            param.data.copy_(original_param)
        return objective_value, task_loss

    def _forward_differences(self, model, gradient_data, candidate, labels):
        """Run through model twice to approximate 2nd-order derivative on residual."""
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            task_loss = self.loss_fn(model(candidate), labels)
        # Compute both model gradients and candidate gradients
        *gradients, dLdx = torch.autograd.grad(task_loss, (*model.parameters(), candidate), create_graph=False)
        if self.level_gradients:
            # Only a good idea if normalize_gradients=True
            grad_norm = torch.stack([g.pow(2).sum() for g in gradients]).sum().sqrt()
            torch._foreach_div_(gradients, max(grad_norm, self.fudge_factor))

        # Compute first-order gradients of objective
        objective_value, first_order_grad = self._compute_objective_and_first_order(candidate, gradients, gradient_data)
        # Adapts eps to different block strengths
        eps_n = self.eps / torch.stack([g.pow(2).sum() for g in gradients]).sum().sqrt()
        # Patch model and compute loss at offset vector:
        torch._foreach_add_(list(model.parameters()), first_order_grad, alpha=eps_n)
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            offset_task_loss = self.loss_fn(model(candidate), labels)
        (dLv_dx,) = torch.autograd.grad(offset_task_loss, (candidate,), create_graph=False)

        # Compute finite difference approximation
        candidate.grad += (dLv_dx - dLdx) / eps_n * self.scale
        # Add task loss
        candidate.grad += self.task_regularization * dLdx

        return objective_value, task_loss

    def _backward_differences(self, model, gradient_data, candidate, labels):
        """Run through model twice to approximate 2nd-order derivative on residual."""
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            task_loss = self.loss_fn(model(candidate), labels)
        # Compute both model gradients and candidate gradients
        *gradients, dLdx = torch.autograd.grad(task_loss, (*model.parameters(), candidate), create_graph=False)
        if self.level_gradients:
            # Only a good idea if normalize_gradients=True
            grad_norm = torch.stack([g.pow(2).sum() for g in gradients]).sum().sqrt()
            torch._foreach_div_(gradients, max(grad_norm, self.fudge_factor))

        # Compute first-order gradients of objective
        objective_value, first_order_grad = self._compute_objective_and_first_order(candidate, gradients, gradient_data)
        # Adapts eps to different block strengths
        eps_n = self.eps / torch.stack([g.pow(2).sum() for g in gradients]).sum().sqrt()
        # Patch model and compute loss at offset vector:
        torch._foreach_sub_(list(model.parameters()), first_order_grad, alpha=eps_n)
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            offset_task_loss = self.loss_fn(model(candidate), labels)
        (dLv_dx,) = torch.autograd.grad(offset_task_loss, (candidate,), create_graph=False)

        # Compute finite difference approximation
        candidate.grad += (dLdx - dLv_dx) / eps_n * self.scale
        # Add task loss
        candidate.grad += self.task_regularization * dLdx

        return objective_value, task_loss

    def _central_differences(self, model, gradient_data, candidate, labels):
        """Run through model twice to approximate 2nd-order derivative on residual."""
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            task_loss = self.loss_fn(model(candidate), labels)
        # Compute both model gradients and candidate gradients
        *gradients, dLdx = torch.autograd.grad(task_loss, (*model.parameters(), candidate), create_graph=False)

        # Compute first-order gradients of objective
        objective_value, first_order_grad = self._compute_objective_and_first_order(candidate, gradients, gradient_data)
        # Adapts eps to different block strengths
        eps_n = self.eps / torch.stack([g.pow(2).sum() for g in gradients]).sum().sqrt()

        # Patch model and compute loss at offset vectors:
        torch._foreach_add_(list(model.parameters()), first_order_grad, alpha=0.5 * eps_n)
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            offset_plus = self.loss_fn(model(candidate), labels)
        (dLvp_dx,) = torch.autograd.grad(offset_plus, (candidate,), create_graph=False)

        torch._foreach_sub_(list(model.parameters()), first_order_grad, alpha=eps_n)
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            offset_minus = self.loss_fn(model(candidate), labels)
        (dLvm_dx,) = torch.autograd.grad(offset_minus, (candidate,), create_graph=False)

        # Compute finite difference approximation
        candidate.grad += (dLvp_dx - dLvm_dx) / eps_n * self.scale
        # Add task loss
        candidate.grad += self.task_regularization * dLdx

        return objective_value, task_loss

    def _upwind_differences(self, model, gradient_data, candidate, labels):
        """Run through model twice to approximate 2nd-order derivative on residual."""
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            task_loss = self.loss_fn(model(candidate), labels)
        # Compute both model gradients and candidate gradients
        *gradients, dLdx = torch.autograd.grad(task_loss, (*model.parameters(), candidate), create_graph=False)

        # Compute first-order gradients of objective
        objective_value, first_order_grad = self._compute_objective_and_first_order(candidate, gradients, gradient_data)
        # Adapts eps to different block strengths
        eps_n = self.eps / torch.stack([g.pow(2).sum() for g in gradients]).sum().sqrt()

        # Patch model and compute loss at offset vectors:
        torch._foreach_add_(list(model.parameters()), first_order_grad, alpha=0.5 * eps_n)
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            offset_plus = self.loss_fn(model(candidate), labels)
        (dLvp_dx,) = torch.autograd.grad(offset_plus, (candidate,), create_graph=False)

        torch._foreach_sub_(list(model.parameters()), first_order_grad, alpha=eps_n)
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            offset_minus = self.loss_fn(model(candidate), labels)
        (dLvm_dx,) = torch.autograd.grad(offset_minus, (candidate,), create_graph=False)

        # Compute both finite differences
        Dp = (dLvp_dx - dLdx) / eps_n
        Dm = (dLdx - dLvm_dx) / eps_n
        # Upwind based on dLdx
        # Compute finite difference approximation
        candidate.grad += (torch.max(dLdx, 0)[0] * Dm + torch.min(dLdx, 0)[0] * Dp) * self.scale
        # Add task loss
        candidate.grad += self.task_regularization * dLdx

        return objective_value, task_loss

    def _compute_objective_and_first_order(self, candidate, gradients, gradient_data):
        residuals = torch._foreach_sub(gradients, gradient_data)  # Save one copy of the gradient list here ?
        # Gradients have already been populated. Make sure not to kill them later on.
        with torch.no_grad():
            with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
                objective_value = 0.5 * self.scale * torch.stack([r.detach().pow(2).sum() for r in residuals]).sum()
        return objective_value, residuals


class PearlmutterCosine(PearlmutterEuclidean):
    """Use a first-order approximation of \nabla_x \nabla_g instead of the correct autograd value."""

    def _compute_objective_and_first_order(self, candidate, gradients, gradient_data):
        with torch.no_grad():
            scalar_product, grad_norm, data_norm = self._cosine_sim_components(gradients, gradient_data)

        first_order_cosine = torch._foreach_div(gradient_data, -grad_norm * data_norm)
        torch._foreach_sub_(first_order_cosine, gradients, alpha=-scalar_product / (grad_norm.pow(3) * data_norm))

        objective_value = self.scale * (1 - scalar_product / (grad_norm * data_norm))
        return objective_value, first_order_cosine

    @staticmethod
    @torch.jit.script
    def _cosine_sim_components(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor]):
        scalar_product = torch.tensor(0, device=gradient_rec[0].device, dtype=gradient_rec[0].dtype)
        rec_norm = torch.tensor(0, device=gradient_rec[0].device, dtype=gradient_rec[0].dtype)
        data_norm = torch.tensor(0, device=gradient_rec[0].device, dtype=gradient_rec[0].dtype)

        for rec, data in zip(gradient_rec, gradient_data):
            scalar_product += (rec * data).sum()
            rec_norm += rec.detach().pow(2).sum()
            data_norm += data.detach().pow(2).sum()

        return scalar_product, rec_norm.sqrt(), data_norm.sqrt()


objective_lookup = {
    "euclidean": Euclidean,
    "cosine-similarity": CosineSimilarity,
    "masked-cosine-similarity": MaskedCosineSimilarity,
    "fast-cosine-similarity": FastCosineSimilarity,
    "angular": AngularSimilarity,
    "l1": L1Loss,
    "pearlmutter-loss": PearlmutterEuclidean,
    "pearlmutter-cosine": PearlmutterCosine,
    "tag-euclidean": EuclideanTag,
}
