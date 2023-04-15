"""Utility functions for class/feature fishing attacks."""

import numbers
import torch

import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment

import torchvision.transforms as transforms


default_setup = dict(device=torch.device("cpu"), dtype=torch.float)


def wrap_indices(indices):
    if isinstance(indices, numbers.Number):
        return [indices]
    else:
        return list(indices)


def check_with_tolerance(value, list, threshold=1e-3):
    for i in list:
        if abs(value - i) < threshold:
            return True

    return False


def order_gradients(self, recovered_single_gradients, gt_single_gradients, setup=default_setup):
    single_gradients = []
    num_data = len(gt_single_gradients)

    for grad_i in recovered_single_gradients:
        single_gradients.append(torch.cat([torch.flatten(i) for i in grad_i]))

    similarity_matrix = torch.zeros(num_data, num_data, **setup)
    for idx, x in enumerate(single_gradients):
        for idy, y in enumerate(gt_single_gradients):
            similarity_matrix[idy, idx] = torch.nn.CosineSimilarity(dim=0)(y, x).detach()

    try:
        _, rec_assignment = linear_sum_assignment(similarity_matrix.cpu().numpy(), maximize=True)
    except ValueError:
        log.info(f"ValueError from similarity matrix {similarity_matrix.cpu().numpy()}")
        log.info("Returning trivial order...")
        rec_assignment = list(range(num_data))

    return [recovered_single_gradients[i] for i in rec_assignment]


def reconstruct_feature(shared_data, cls_to_obtain):
    if type(shared_data) is not list:
        shared_grad = shared_data["gradients"]
    else:
        shared_grad = shared_data

    weights = shared_grad[-2]
    bias = shared_grad[-1]
    grads_fc_debiased = weights / bias[:, None]

    if bias[cls_to_obtain] != 0:
        return grads_fc_debiased[cls_to_obtain]
    else:
        return torch.zeros_like(grads_fc_debiased[0])


def cal_single_gradients(model, loss_fn, true_user_data, setup=default_setup):
    true_data = true_user_data["data"]
    num_data = len(true_data)
    labels = true_user_data["labels"]
    model = model.to(**setup)

    single_gradients = []
    single_losses = []

    for ii in range(num_data):
        cand_ii = true_data[ii : (ii + 1)]
        label_ii = labels[ii : (ii + 1)]
        model.zero_grad()
        spoofed_loss_ii = loss_fn(model(cand_ii), label_ii)
        gradient_ii = torch.autograd.grad(spoofed_loss_ii, model.parameters())
        gradient_ii = [g_ii.reshape(-1) for g_ii in gradient_ii]
        gradient_ii = torch.cat(gradient_ii)
        single_gradients.append(gradient_ii)
        single_losses.append(spoofed_loss_ii)

    return single_gradients, single_losses


def print_gradients_norm(single_gradients, single_losses, which_to_recover=-1, return_results=False):
    grad_norm = []
    losses = []

    if not return_results:
        print("grad norm   |   loss")

    for i, gradient_ii in enumerate(single_gradients):
        if not return_results:
            if i == which_to_recover:
                print(f"{float(torch.norm(gradient_ii)):2.4f} | {float(single_losses[i]):4.2f} - target")
            else:
                print(f"{float(torch.norm(gradient_ii)):2.4f} | {float(single_losses[i]):4.2f}")

        grad_norm.append(float(torch.norm(gradient_ii)))
        losses.append(float(single_losses[i]))

    if return_results:
        return torch.stack(grad_norm), torch.stack(losses)


def random_transformation(img):
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img.shape[-2:], scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=1),
            # transforms.RandomVerticalFlip(p=1),
            transforms.GaussianBlur(3),
        ]
    )

    return transform(img)


def estimate_gt_stats(est_features, sample_sizes, indx=0):
    aggreg_data = []
    est_feature = est_features[indx]

    for i in range(len(est_feature)):
        feat_i = est_feature[i]
        size_i = sample_sizes[i]
        aggreg_data.append(feat_i * (size_i ** (1 / 2)))

    return np.mean(est_feature), np.std(aggreg_data)


def find_best_feat(est_features, sample_sizes, method="kstest"):
    if "kstest" in method:
        statistics = []
        for i in range(len(est_features)):
            tmp_series = est_features[i]
            tmp_series = (tmp_series - np.mean(tmp_series)) / np.std(tmp_series)
            statistics.append(stats.kstest(tmp_series, "norm")[0])

        return np.argmin(statistics)
    elif "most-spread" in method or "most-high-mean" in method:
        means = []
        stds = []
        for i in range(len(est_features)):
            mu, sigma = estimate_gt_stats(est_features, sample_sizes, indx=1)
            means.append(mu)
            stds.append(sigma)

        if "most-spread" in method:
            return np.argmax(stds)
        else:
            return np.argmax(means)
    else:
        raise ValueError(f"Method {method} not implemented.")

    return np.argmax(p_values)
