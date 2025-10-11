"""Metrics computation for gradient leakage attack evaluation"""

import math
from statistics import mean

import lpips
import torch
from torchmetrics import StructuralSimilarityIndexMeasure

from plato.config import Config

LOSS_FN = lpips.LPIPS(net="vgg").to(Config().device())


def compute_ssim(dummy_data, ground_truth):
    """Compute SSIM."""
    # Assuming for now that the matching dummy_data and gt are given
    dummy_data = torch.flatten(dummy_data)
    ground_truth = torch.flatten(ground_truth)

    dummy_data_mean = torch.mean(dummy_data.double()).item()
    ground_truth_mean = torch.mean(ground_truth.double()).item()
    dummy_mean_square = dummy_data_mean**2
    gt_mean_square = ground_truth_mean**2

    dummy_data_var = torch.var(dummy_data, unbiased=False).item()
    ground_truth_var = torch.var(ground_truth, unbiased=False).item()
    cov = covar(dummy_data, ground_truth)

    c1 = 0.0001
    c2 = 0.0009

    term1 = (2 * dummy_data_mean * ground_truth_mean) + c1
    term2 = (2 * cov) + c2
    term3 = dummy_mean_square + gt_mean_square + c1
    term4 = dummy_data_var + ground_truth_var + c2

    return (term1 * term2) / (term3 * term4)


def compute_ssim_library(dummy_data, ground_truth):
    """Compute SSIM from library."""
    ssim = StructuralSimilarityIndexMeasure().to(Config().device())
    return ssim(dummy_data, ground_truth).item()


def get_evaluation_dict(dummy_data, ground_truth, num_images, model, ds):
    """Compute all evaluation metrics."""
    eval_dict = {}
    (
        eval_dict["data_mses"],
        eval_dict["lpipss"],
        eval_dict["ssims"],
    ) = ([], [], [])
    for i in range(num_images):
        # Initialize image data MSE and LPIPS values to be infinite
        eval_dict["data_mses"].append(math.inf)
        eval_dict["lpipss"].append(math.inf)
        # Initialize SSIM to be -1 (minimum value possible)
        eval_dict["ssims"].append(-1.0)
        # Find the closest ground truth data after the misordering
        for j in range(num_images):
            # Image data MSEs, LPIPSs, PSNRs and SSIMs are stored in lists
            # where the ith entry is for the ith dummy data
            eval_dict["data_mses"][i] = min(
                eval_dict["data_mses"][i],
                torch.mean((dummy_data[i] - ground_truth[j]) ** 2).item(),
            )
            eval_dict["lpipss"][i] = min(
                eval_dict["lpipss"][i],
                LOSS_FN.forward(dummy_data[i], ground_truth[j]).item(),
            )
            eval_dict["ssims"][i] = max(
                eval_dict["ssims"][i],
                compute_ssim_library(
                    torch.unsqueeze(dummy_data[i], dim=0),
                    torch.unsqueeze(ground_truth[j], dim=0),
                ),
            )
        # Find the mean for the MSE, LPIPS and PSNR
        eval_dict["avg_data_mses"] = mean(eval_dict["data_mses"])
        eval_dict["avg_lpips"] = mean(eval_dict["lpipss"])
        eval_dict["avg_psnr"] = compute_psnr(dummy_data, ground_truth, factor=1 / ds)
        eval_dict["avg_ssim"] = mean(eval_dict["ssims"])
        with torch.no_grad():
            eval_dict["avg_feat_mses"] = torch.mean(
                (model(dummy_data)[0] - model(ground_truth)[0]) ** 2
            ).item()

    return eval_dict


def compute_psnr(img_batch, ref_batch, batched=False, factor=1.0):
    """
    Compute PSNR.

    This code based on https://github.com/JonasGeiping/invertinggradients.
    Standard PSNR.
    """

    def get_psnr(img_in, img_ref):
        mse = ((img_in - img_ref) ** 2).mean()
        if mse > 0 and torch.isfinite(mse):
            return 10 * torch.log10(factor**2 / mse)
        elif not torch.isfinite(mse):
            return img_batch.new_tensor(float("nan"))
        else:
            return img_batch.new_tensor(float("inf"))

    if batched:
        psnr = get_psnr(img_batch.detach(), ref_batch)
    else:
        [B, C, m, n] = img_batch.shape
        psnrs = []
        for sample in range(B):
            psnrs.append(
                get_psnr(
                    img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]
                )
            )
        psnr = torch.stack(psnrs, dim=0).mean()

    return psnr.item()


def covar(a, b):
    """Compute covariance."""
    a_bar = torch.mean(a).item()
    b_bar = torch.mean(b).item()
    cov = 0
    for i in enumerate(a):
        cov += (a[i].item() - a_bar) * (b[i].item() - b_bar)
    return cov / (len(a) - 1)
