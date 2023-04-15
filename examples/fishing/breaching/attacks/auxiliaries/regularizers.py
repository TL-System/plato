"""Various regularizers that can be re-used for multiple attacks."""

import torch

from .deepinversion import DeepInversionFeatureHook


class _LinearFeatureHook:
    """Hook to retrieve input to given module."""

    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        input_features = input[0]
        self.features = input_features

    def close(self):
        self.hook.remove()


class FeatureRegularization(torch.nn.Module):
    """Feature regularization implemented for the last linear layer at the end."""

    def __init__(self, setup, scale=0.1):
        super().__init__()
        self.setup = setup
        self.scale = scale

    def initialize(self, models, shared_data, labels, *args, **kwargs):
        self.measured_features = []
        for user_data in shared_data:
            # Assume last two gradient vector entries are weight and bias:
            weights = user_data["gradients"][-2]
            bias = user_data["gradients"][-1]
            grads_fc_debiased = weights / bias[:, None]
            features_per_label = []
            for label in labels:
                if bias[label] != 0:
                    features_per_label.append(grads_fc_debiased[label])
                else:
                    features_per_label.append(torch.zeros_like(grads_fc_debiased[0]))
            self.measured_features.append(torch.stack(features_per_label))

        self.refs = [None for model in models]
        for idx, model in enumerate(models):
            for module in model.modules():
                # Keep only the last linear layer here:
                if isinstance(module, torch.nn.Linear):
                    self.refs[idx] = _LinearFeatureHook(module)

    def forward(self, tensor, *args, **kwargs):
        regularization_value = 0
        for ref, measured_val in zip(self.refs, self.measured_features):
            regularization_value += (ref.features - measured_val).pow(2).mean()
        return regularization_value * self.scale

    def __repr__(self):
        return f"Feature space regularization, scale={self.scale}"


class LinearLayerRegularization(torch.nn.Module):
    """Linear layer regularization implemented for arbitrary linear layers. WIP Example."""

    def __init__(self, setup, scale=0.1):
        super().__init__()
        self.setup = setup
        self.scale = scale

    def initialize(self, models, gradient_data, *args, **kwargs):
        self.measured_features = []
        self.refs = [list() for model in models]

        for idx, (model, user_data) in enumerate(zip(models, shared_data)):
            # 1) Find linear layers:
            linear_layers = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    linear_layers.append(name)
                    self.refs[idx].append(_LinearFeatureHook(module))
            named_grads = {name: g for (g, (name, param)) in zip(user_data["gradients"], model.named_parameters())}

            # 2) Check features
            features = []
            for linear_layer in linear_layers:
                weights = named_grads[linear_layer + ".weight"]
                bias = named_grads[linear_layer + ".bias"]
                grads_fc_debiased = (weights / bias[:, None]).mean(dim=0)  # At some point todo: Make this smarter
                features.append(grads_fc_debiased)
            self.measured_features.append(features)

    def forward(self, tensor, *args, **kwargs):
        regularization_value = 0
        for model_ref, data_ref in zip(self.refs, self.measured_features):
            for linear_layer, data in zip(model_ref, data_ref):
                regularization_value += (linear_layer.features.mean(dim=0) - data).pow(2).sum()

    def __repr__(self):
        return f"Feature space regularization, scale={self.scale}"


class TotalVariation(torch.nn.Module):
    """Computes the total variation value of an (image) tensor, based on its last two dimensions.
    Optionally also Color TV based on its last three dimensions.

    The value of this regularization is scaled by 1/sqrt(M*N) times the given scale."""

    def __init__(self, setup, scale=0.1, inner_exp=1, outer_exp=1, double_opponents=False, eps=1e-8):
        """scale is the overall scaling. inner_exp and outer_exp control isotropy vs anisotropy.
        Optionally also includes proper color TV via double opponents."""
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.inner_exp = inner_exp
        self.outer_exp = outer_exp
        self.eps = eps
        self.double_opponents = double_opponents

        grad_weight = torch.tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]], **setup).unsqueeze(0).unsqueeze(1)
        grad_weight = torch.cat((torch.transpose(grad_weight, 2, 3), grad_weight), 0)
        self.groups = 6 if self.double_opponents else 3
        grad_weight = torch.cat([grad_weight] * self.groups, 0)

        self.register_buffer("weight", grad_weight)

    def initialize(self, models, *args, **kwargs):
        pass

    def forward(self, tensor, *args, **kwargs):
        """Use a convolution-based approach."""
        if self.double_opponents:
            tensor = torch.cat(
                [
                    tensor,
                    tensor[:, 0:1, :, :] - tensor[:, 1:2, :, :],
                    tensor[:, 0:1, :, :] - tensor[:, 2:3, :, :],
                    tensor[:, 1:2, :, :] - tensor[:, 2:3, :, :],
                ],
                dim=1,
            )
        diffs = torch.nn.functional.conv2d(
            tensor, self.weight, None, stride=1, padding=1, dilation=1, groups=self.groups
        )
        squares = (diffs.abs() + self.eps).pow(self.inner_exp)
        squared_sums = (squares[:, 0::2] + squares[:, 1::2]).pow(self.outer_exp)
        return squared_sums.mean() * self.scale

    def __repr__(self):
        return (
            f"Total Variation, scale={self.scale}. p={self.inner_exp} q={self.outer_exp}. "
            f"{'Color TV: double oppponents' if self.double_opponents else ''}"
        )


class OrthogonalityRegularization(torch.nn.Module):
    """This is the orthogonality regularizer described Qian et al.,

    "MINIMAL CONDITIONS ANALYSIS OF GRADIENT-BASED RECONSTRUCTION IN FEDERATED LEARNING"
    """

    def __init__(self, setup, scale=0.1):
        super().__init__()
        self.setup = setup
        self.scale = scale

    def initialize(self, models, *args, **kwargs):
        pass

    def forward(self, tensor, *args, **kwargs):
        if tensor.shape[0] == 1:
            return 0
        else:
            B = tensor.shape[0]
            full_products = (tensor.unsqueeze(0) * tensor.unsqueeze(1)).pow(2).view(B, B, -1).mean(dim=2)
            idx = torch.arange(0, B, out=torch.LongTensor())
            full_products[idx, idx] = 0
            return full_products.sum()

    def __repr__(self):
        return f"Input Orthogonality, scale={self.scale}"


class NormRegularization(torch.nn.Module):
    """Implement basic norm-based regularization, e.g. an L2 penalty."""

    def __init__(self, setup, scale=0.1, pnorm=2.0):
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.pnorm = pnorm

    def initialize(self, models, *args, **kwargs):
        pass

    def forward(self, tensor, *args, **kwargs):
        return 1 / self.pnorm * tensor.pow(self.pnorm).mean() * self.scale

    def __repr__(self):
        return f"Input L^p norm regularization, scale={self.scale}, p={self.pnorm}"


class DeepInversion(torch.nn.Module):
    """Implement a DeepInversion based regularization as proposed in DeepInversion as used for reconstruction in
    Yin et al, "See through Gradients: Image Batch Recovery via GradInversion".
    """

    def __init__(self, setup, scale=0.1, first_bn_multiplier=10):
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.first_bn_multiplier = first_bn_multiplier

    def initialize(self, models, *args, **kwargs):
        """Initialize forward hooks."""
        self.losses = [list() for model in models]
        for idx, model in enumerate(models):
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    self.losses[idx].append(DeepInversionFeatureHook(module))

    def forward(self, tensor, *args, **kwargs):
        rescale = [self.first_bn_multiplier] + [1.0 for _ in range(len(self.losses[0]) - 1)]
        feature_reg = 0
        for loss in self.losses:
            feature_reg += sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss)])
        return self.scale * feature_reg

    def __repr__(self):
        return f"Deep Inversion Regularization (matching batch norms), scale={self.scale}, first-bn-mult={self.first_bn_multiplier}"


regularizer_lookup = dict(
    total_variation=TotalVariation,
    orthogonality=OrthogonalityRegularization,
    norm=NormRegularization,
    deep_inversion=DeepInversion,
    features=FeatureRegularization,
)
