import math
from statistics import mean

import lpips
import torch
from torchmetrics import StructuralSimilarityIndexMeasure

loss_fn = lpips.LPIPS(net="vgg")


def find_ssim(dummy_data, ground_truth):
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


def find_ssim_library(dummy_data, ground_truth):
    ssim = StructuralSimilarityIndexMeasure()
    return ssim(dummy_data, ground_truth).item()


def get_evaluation_dict(dummy_data, ground_truth, num_images):
    eval_dict = {}
    (
        eval_dict["mses"],
        eval_dict["lpipss"],
        eval_dict["psnrs"],
        eval_dict["ssims"],
        eval_dict["library_ssims"],
    ) = ([], [], [], [], [])
    for i in range(num_images):
        # Initialize MSE and LPIPS values to be infinite
        eval_dict["mses"].append(math.inf)
        eval_dict["lpipss"].append(math.inf)
        # Initialize SSIM to be -1 (minimum value possible)
        eval_dict["ssims"].append(-1.0)
        eval_dict["library_ssims"].append(-1.0)
        # Find the closest ground truth data after the misordering
        for j in range(num_images):
            # MSEs, LPIPSs, PSNRs and SSIMs are stored in lists
            # where the ith entry is for the ith dummy data
            eval_dict["mses"][i] = min(
                eval_dict["mses"][i],
                torch.mean((dummy_data[i] - ground_truth[j]) ** 2).item(),
            )
            # VERY EXPENSIVE TO COMPUTE
            # eval_dict["lpipss"][i] = min(
            #     eval_dict["lpipss"][i],
            #     loss_fn.forward(dummy_data[i], ground_truth[j]).item(),
            # )
        #     eval_dict["ssims"][i] = max(
        #         eval_dict["ssims"][i], find_ssim(dummy_data[i], ground_truth[j])
        #     )
        #     eval_dict["library_ssims"][i] = max(
        #         eval_dict["library_ssims"][i],
        #         find_ssim_library(
        #             torch.unsqueeze(dummy_data[i], dim=0),
        #             torch.unsqueeze(ground_truth[j], dim=0),
        #         ),
        #     )
        # eval_dict["psnrs"].append(-10 * math.log10(eval_dict["mses"][i]))
        # Find the mean for the MSE, LPIPS and PSNR
        eval_dict["avg_mses"] = mean(eval_dict["mses"])
        # eval_dict["avg_lpips"] = mean(eval_dict["lpipss"])
        # eval_dict["avg_psnr"] = mean(eval_dict["psnrs"])
        # eval_dict["avg_ssim"] = mean(eval_dict["ssims"])
        # eval_dict["avg_library_ssim"] = mean(eval_dict["library_ssims"])

    return eval_dict


def covar(a, b):
    a_bar = torch.mean(a).item()
    b_bar = torch.mean(b).item()
    cov = 0
    for i in range(len(a)):
        cov += (a[i].item() - a_bar) * (b[i].item() - b_bar)
    return cov / (len(a) - 1)
