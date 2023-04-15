"""Implements a malicious block that can be inserted at the front on normal models to break them."""
import torch

import math
from statistics import NormalDist
from scipy.stats import laplace


class ImprintBlock(torch.nn.Module):
    structure = "cumulative"

    def __init__(self, data_shape, num_bins, connection="linear", gain=1e-3, linfunc="fourier", mode=0):
        """
        data_shape is the shape of the input data
        num_bins is how many "paths" to include in the model
        connection is how this block should coonect back to the input shape (optional)

        linfunc is the choice of linear query function ('avg', 'fourier', 'randn', 'rand').
        If linfunc is fourier, then the mode parameter determines the mode of the DCT-2 that is used as linear query.
        """
        super().__init__()
        self.data_shape = data_shape
        self.data_size = torch.prod(torch.as_tensor(data_shape))
        self.num_bins = num_bins
        self.linear0 = torch.nn.Linear(self.data_size, num_bins)

        self.bins = self._get_bins(linfunc)
        with torch.no_grad():
            self.linear0.weight.data = self._init_linear_function(linfunc, mode) * gain
            self.linear0.bias.data = self._make_biases() * gain

        self.connection = connection
        if connection == "linear":
            self.linear2 = torch.nn.Linear(num_bins, self.data_size)
            with torch.no_grad():
                self.linear2.weight.data = torch.ones_like(self.linear2.weight.data) / gain
                self.linear2.bias.data -= torch.as_tensor(self.bins).mean()

        self.nonlin = torch.nn.ReLU()

    @torch.no_grad()
    def _init_linear_function(self, linfunc="fourier", mode=0):
        K, N = self.num_bins, self.data_size
        if linfunc == "avg":
            weights = torch.ones_like(self.linear0.weight.data) / N
        elif linfunc == "fourier":
            weights = torch.cos(math.pi / N * (torch.arange(0, N) + 0.5) * mode).repeat(K, 1) / N * max(mode, 0.33) * 4
            # dont ask about the 4, this is WIP
            # nonstandard normalization
        elif linfunc == "randn":
            weights = torch.randn(N).repeat(K, 1)
            std, mu = torch.std_mean(weights[0])  # Enforce mean=0, std=1 with higher precision
            weights = (weights - mu) / std / math.sqrt(N)  # Move to std=1 in output dist
        elif linfunc == "rand":
            weights = torch.rand(N).repeat(K, 1)  # This might be a terrible idea haven't done the math
            std, mu = torch.std_mean(weights[0])  # Enforce mean=0, std=1
            weights = (weights - mu) / std / math.sqrt(N)  # Move to std=1 in output dist
        else:
            raise ValueError(f"Invalid linear function choice {linfunc}.")

        return weights

    def _get_bins(self, linfunc="avg"):
        bins = []
        mass_per_bin = 1 / (self.num_bins)
        bins.append(-10)  # -Inf is not great here, but NormalDist(mu=0, sigma=1).cdf(10) approx 1
        for i in range(1, self.num_bins):
            if "fourier" in linfunc:
                bins.append(laplace(loc=0.0, scale=1 / math.sqrt(2)).ppf(i * mass_per_bin))
            else:
                bins.append(NormalDist().inv_cdf(i * mass_per_bin))
        return bins

    def _make_biases(self):
        new_biases = torch.zeros_like(self.linear0.bias.data)
        for i in range(new_biases.shape[0]):
            new_biases[i] = -self.bins[i]
        return new_biases

    def forward(self, x, *args, **kwargs):
        x_in = x
        x = self.linear0(x.flatten(start_dim=1))
        x = self.nonlin(x)
        if self.connection == "linear":
            output = self.linear2(x)
        elif self.connection == "cat":
            output = torch.cat([x, x_in[:, self.num_bins :]], dim=1)
        elif self.connection == "softmax":
            s = torch.softmax(x, dim=1)[:, :, None]
            output = (x_in[:, None, :] * s).sum(dim=1)
        else:
            output = x_in.flatten(start_dim=1) + x.mean(dim=1, keepdim=True)
        return output.unflatten(dim=1, sizes=self.data_shape)


class SparseImprintBlock(ImprintBlock):
    structure = "sparse"

    """This block is sparse instead of cumulative which is more efficient in noise/param tradeoffs but requires
    two ReLUs that construct the hard-tanh nonlinearity."""

    def __init__(self, data_shape, num_bins, connection="linear", gain=1, linfunc="fourier", mode=0):
        super().__init__(data_shape, num_bins, connection, gain, linfunc, mode)
        self.nonlin = torch.nn.Hardtanh(min_val=0, max_val=gain)

    def _get_bins(self, linfunc="avg"):
        bins = []
        mass = 0
        for path in range(self.num_bins + 1):
            mass += 1 / (self.num_bins + 2)
            if "fourier" in linfunc:
                bins.append(laplace(loc=0, scale=1 / math.sqrt(2)).ppf(mass))
            else:
                bins += [NormalDist(mu=0, sigma=1).inv_cdf(mass)]
        bin_sizes = [bins[i + 1] - bins[i] for i in range(len(bins) - 1)]
        self.bin_sizes = bin_sizes
        return bins[1:]

    @torch.no_grad()
    def _init_linear_function(self, linfunc="fourier", mode=0):
        weights = super()._init_linear_function(linfunc, mode)
        for i, row in enumerate(weights):
            row /= torch.as_tensor(self.bin_sizes[i], device=weights.device)
        return weights

    def _make_biases(self):
        new_biases = torch.zeros_like(self.linear0.bias.data)
        for i, (bin_val, bin_width) in enumerate(zip(self.bins, self.bin_sizes)):
            new_biases[i] = -bin_val / bin_width
        return new_biases


class OneShotBlock(ImprintBlock):
    structure = "cumulative"

    """One-shot attack with minimal additional parameters. Can target a specific data point if its target_val is known."""

    def __init__(self, data_shape, num_bins, connection="linear", gain=1e-3, linfunc="fourier", mode=0, target_val=0):
        self.virtual_bins = num_bins
        self.target_val = target_val
        num_bins = 2
        super().__init__(data_shape, num_bins, connection, gain, linfunc, mode)

    def _get_bins(self, linfunc="avg"):
        bins = []
        mass_per_bin = 1 / (self.virtual_bins)
        bins.append(-10)  # -Inf is not great here, but NormalDist(mu=0, sigma=1).cdf(10) approx 1
        for i in range(1, self.virtual_bins):
            if "fourier" in linfunc:
                bins.append(laplace(loc=0.0, scale=1 / math.sqrt(2)).ppf(i * mass_per_bin))
            else:
                bins.append(NormalDist().inv_cdf(i * mass_per_bin))
            if self.target_val < bins[-1]:
                break
        return bins[-2:]


class OneShotBlockSparse(SparseImprintBlock):
    structure = "sparse"

    def __init__(self, data_shape, num_bins, connection="linear", gain=1e-3, linfunc="fourier", mode=0):
        """
        data_shape is the data_shape of the input data
        num_bins is how many "paths" to include in the model
        target_val=0 in this variant.
        """
        super().__init__(data_shape, 1, connection, gain, linfunc, mode)
        self.num_bins = num_bins

    def _get_bins(self):
        # Here we just build bins of uniform mass
        left_bins = []
        bins = []
        mass_per_bin = 1 / self.num_bins
        bins = [-NormalDist().inv_cdf(0.5), -NormalDist().inv_cdf(0.5 + mass_per_bin)]
        self.bin_sizes = [bins[i + 1] - bins[i] for i in range(len(bins) - 1)]
        bins = bins[:-1]  # here we need to throw away one on the right
        return bins


class CuriousAbandonHonesty(ImprintBlock):
    """Replicates the attack of Boenisch et al, "When the Curious Abandon Honesty: Federated Learning Is Not Private"
    This is a sparse ReLU block.
    """

    structure = "sparse"

    def __init__(self, data_shape, num_bins, mu=0, sigma=0.5, scale_factor=0.95, connection="linear"):
        """
        data_shape is the shape of the input data, num_bins is the number of inserted rows.
        mu, sigma and scale_factor control the attack as described in the paper
        connection is how this block should coonect back to the input shape (optional)
        gain can scale this layer.
        """
        torch.nn.Module.__init__(self)
        self.data_shape = data_shape
        self.data_size = torch.prod(torch.as_tensor(data_shape))
        self.num_bins = num_bins

        self.linear0 = torch.nn.Linear(self.data_size, num_bins)

        with torch.no_grad():
            self.linear0.weight.data = self._init_trap_weights(sigma, scale_factor)
            self.linear0.bias.data = self._make_biases(mu)

        self.connection = connection
        if connection == "linear":
            self.linear2 = torch.nn.Linear(num_bins, self.data_size)
            with torch.no_grad():
                self.linear2.weight.data = torch.ones_like(self.linear2.weight.data)
                self.linear2.bias.data.zero_()

        self.nonlin = torch.nn.ReLU()

    @torch.no_grad()
    def _init_trap_weights(self, sigma, scale_factor):
        N, K = self.data_size, self.num_bins

        # indices = torch.argsort(torch.rand(K, N), dim=1) # This has insane memory requirements in pytorch
        indices = torch.zeros((K, N), dtype=torch.long)
        for row in range(K):
            indices[row] = torch.randperm(N)
        negative_weight_indices = indices[:, : int(N / 2)]
        positive_weight_indices = indices[:, int(N / 2) :]

        sampled_weights = torch.randn(K, int(N / 2)) * sigma

        negative_samples = sampled_weights
        positive_samples = -scale_factor * sampled_weights

        final_weights = torch.empty(K, N)
        final_weights.scatter_(1, negative_weight_indices, negative_samples)
        final_weights.scatter_(1, positive_weight_indices, positive_samples)
        return final_weights

    def _make_biases(self, mu):
        new_biases = torch.ones_like(self.linear0.bias.data) * mu
        return new_biases
