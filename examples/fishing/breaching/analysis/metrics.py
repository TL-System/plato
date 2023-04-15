"""Various metrics."""
import torch
from functools import partial
import warnings

warnings.filterwarnings(
    "ignore", message=".*torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.*",
)  # this is kornia's problem


def cw_ssim(img_batch, ref_batch, scales=5, skip_scales=None, K=1e-6, reduction="mean"):
    """Batched complex wavelet structural similarity.

    As in Zhou Wang and Eero P. Simoncelli, "TRANSLATION INSENSITIVE IMAGE SIMILARITY IN COMPLEX WAVELET DOMAIN"
    Ok, not quite, this implementation computes no local SSIM and neither averaging over local patches and uses only
    the existing wavelet structure to provide a similar scale-invariant decomposition.

    skip_scales can be a list like [True, False, False, False] marking levels to be skipped.
    K is a small fudge factor.
    """
    try:
        from pytorch_wavelets import DTCWTForward
    except ModuleNotFoundError:
        warnings.warn(
            "To utilize wavelet SSIM, install pytorch wavelets from https://github.com/fbcotter/pytorch_wavelets."
        )
        return torch.as_tensor(float("NaN")), torch.as_tensor(float("NaN"))

    # 1) Compute wavelets:
    setup = dict(device=img_batch.device, dtype=img_batch.dtype)
    if skip_scales is not None:
        include_scale = [~s for s in skip_scales]
        total_scales = scales - sum(skip_scales)
    else:
        include_scale = True
        total_scales = scales
    xfm = DTCWTForward(J=scales, biort="near_sym_b", qshift="qshift_b", include_scale=include_scale).to(**setup)
    img_coefficients = xfm(img_batch)
    ref_coefficients = xfm(ref_batch)

    # 2) Multiscale complex SSIM:
    ssim = 0
    for xs, ys in zip(img_coefficients[1], ref_coefficients[1]):
        if len(xs) > 0:
            xc = torch.view_as_complex(xs)
            yc = torch.view_as_complex(ys)

            conj_product = (xc * yc.conj()).sum(dim=2).abs()
            square_img = (xc * xc.conj()).abs().sum(dim=2)
            square_ref = (yc * yc.conj()).abs().sum(dim=2)

            ssim_val = (2 * conj_product + K) / (square_img + square_ref + K)
            ssim += ssim_val.mean(dim=[1, 2, 3])
    ssim = ssim / total_scales
    return ssim.mean().item(), ssim.max().item()


def gradient_uniqueness(model, loss_fn, user_data, server_payload, setup, query=0, fudge=1e-7):
    """Count the number of gradient entries that are only affected by a single data point."""

    r"""Formatting suggestion:
      print(f'Unique entries (hitting 1 or all): {unique_entries:.2%}, average hits: {average_hits_per_entry:.2%} \n'
      f'Stats (as N hits:val): {dict(zip(uniques[0].tolist(), uniques[1].tolist()))}\n'
      f'Unique nonzero (hitting 1 or all): {nonzero_uniques:.2%} Average nonzero: {nonzero_hits_per_entry:.2%}. \n'
      f'nonzero-Stats (as N hits:val): {dict(zip(uniques_nonzero[0].tolist(), uniques_nonzero[1].tolist()))}')
    """
    payload = server_payload["queries"][query]
    parameters = payload["parameters"]
    buffers = payload["buffers"]

    with torch.no_grad():
        for param, server_state in zip(model.parameters(), parameters):
            param.copy_(server_state.to(**setup))
        for buffer, server_state in zip(model.buffers(), buffers):
            buffer.copy_(server_state.to(**setup))

    # Compute the forward pass
    gradients = []
    for data_point, label in zip(user_data["data"], user_data["labels"]):
        model.zero_grad()
        loss = loss_fn(model(data_point[None, :]), label[None])
        data_grads = torch.autograd.grad(loss, model.parameters())
        gradients += [torch.cat([g.reshape(-1) for g in data_grads])]

    average_gradient = torch.stack(gradients, dim=0).mean(dim=0, keepdim=True)

    gradient_per_example = torch.stack(gradients, dim=0)

    val = (gradient_per_example - average_gradient).abs() < fudge
    nonzero_val = val[:, average_gradient[0].abs() > fudge]
    unique_entries = (val.sum(dim=0) == 1).float().mean() + (val.sum(dim=0) == len(gradients)).float().mean()
    # hitting a single entry or all entries is equally good for rec
    average_hits_per_entry = val.sum(dim=0).float().mean()
    nonzero_hits_per_entry = (nonzero_val).sum(dim=0).float().mean()
    unique_nonzero_hits = (nonzero_val.sum(dim=0) == 1).float().mean() + (
        nonzero_val.sum(dim=0) == len(gradients)
    ).float().mean()
    return (
        unique_entries,
        average_hits_per_entry,
        unique_nonzero_hits,
        nonzero_hits_per_entry,
        val.sum(dim=0).unique(return_counts=True),
        nonzero_val.sum(dim=0).unique(return_counts=True),
    )


def psnr_compute(img_batch, ref_batch, batched=False, factor=1.0, clip=False):
    """Standard PSNR."""
    if clip:
        img_batch = torch.clamp(img_batch, 0, 1)

    if batched:
        mse = ((img_batch.detach() - ref_batch) ** 2).mean()
        if mse > 0 and torch.isfinite(mse):
            return 10 * torch.log10(factor ** 2 / mse)
        elif not torch.isfinite(mse):
            return [torch.tensor(float("nan"), device=img_batch.device)] * 2
        else:
            return [torch.tensor(float("inf"), device=img_batch.device)] * 2
    else:
        B = img_batch.shape[0]
        mse_per_example = ((img_batch.detach() - ref_batch) ** 2).view(B, -1).mean(dim=1)
        if any(mse_per_example == 0):
            return [torch.tensor(float("inf"), device=img_batch.device)] * 2
        elif not all(torch.isfinite(mse_per_example)):
            return [torch.tensor(float("nan"), device=img_batch.device)] * 2
        else:
            psnr_per_example = 10 * torch.log10(factor ** 2 / mse_per_example)
            return psnr_per_example.mean().item(), psnr_per_example.max().item()


def registered_psnr_compute(img_batch, ref_batch, factor=1.0):
    """Use kornia for now."""
    return _registered_psnr_compute_kornia(img_batch, ref_batch, factor)


def _registered_psnr_compute_kornia(img_batch, ref_batch, factor=1.0):
    """Kornia version. Todo: Use a smarter/deeper matching tool."""
    try:
        from kornia.geometry import ImageRegistrator, HomographyWarper  # lazy import here as well
    except ModuleNotFoundError:
        warnings.warn("To utilize registered PSNR, install kornia.")
        return torch.as_tensor(float("NaN")), torch.as_tensor(float("NaN"))

    B = img_batch.shape[0]
    default_psnrs = []
    registered_psnrs = []
    # If only this was parallelized, todo ...
    for img, ref in zip(img_batch.detach(), ref_batch.detach()):
        img, ref = img[None, ...], ref[None, ...]
        mse = ((img - ref) ** 2).mean()
        default_psnrs += [10 * torch.log10(factor ** 2 / mse)]
        # Align by homography:
        registrator = ImageRegistrator("similarity", num_iterations=2500)
        registrator.warper = partial(HomographyWarper, padding_mode="reflection")
        registrator.to(ref.device)
        homography = registrator.register(img, ref)
        warped_img = registrator.warp_src_into_dst(img)
        # Compute new PSNR:
        mse = ((warped_img.detach() - ref_batch) ** 2).mean()
        registered_psnrs += [10 * torch.log10(factor ** 2 / mse)]

    # Return best of default and warped PSNR:
    result = torch.stack([torch.stack(default_psnrs), torch.stack(registered_psnrs)]).max(dim=0)[0]
    return result.mean().item(), result.max().item()


def _registered_psnr_compute_kornia_loftr(img_batch, ref_batch, factor=1.0):
    """Kornia version. WIP."""
    try:
        from kornia.feature import LoFTR
        from kornia.geometry.homography import find_homography_dlt
    except ModuleNotFoundError:
        warnings.warn("To utilize this registered PSNR implementation, install kornia.")
        return torch.as_tensor(float("NaN")), torch.as_tensor(float("NaN"))

    B = img_batch.shape[0]
    mse_per_example = ((img_batch.detach() - ref_batch) ** 2).view(B, -1).mean(dim=1)
    default_psnrs = 10 * torch.log10(factor ** 2 / mse_per_example)
    # Align by homography:
    matcher = LoFTR(pretrained="indoor")
    with torch.no_grad():
        correspondences_dict = matcher(
            dict(image0=img_batch.mean(dim=1, keepdim=True), image1=ref_batch.mean(dim=1, keepdim=True))
        )
        homography = find_homography_dlt(correspondences_dict["keypoints0"], correspondences_dict["keypoints1"])
        warped_imgs = homography_warp(img_batch, homography, ref_batch.shape[-2:])
    # Compute new PSNR:
    mse_per_example = ((warped_imgs.detach() - ref_batch) ** 2).view(B, -1).mean(dim=1)
    registered_psnrs = 10 * torch.log10(factor ** 2 / mse_per_example)

    # Return best of default and warped PSNR:
    return torch.stack([default_psnrs, registered_psnrs]).max(dim=0)[0].mean()


def _registered_psnr_compute_skimage(img_batch, ref_batch, factor=1.0):
    """Use ORB features to register images onto reference before computing PSNR scores."""
    try:
        import skimage.feature  # Lazy metric stuff import
        import skimage.measure
        import skimage.transform
    except ModuleNotFoundError:
        warnings.warn("To utilize this registered PSNR implementation, install scikit-image.")
        return torch.as_tensor(float("NaN")), torch.as_tensor(float("NaN"))

    descriptor_extractor = skimage.feature.ORB(n_keypoints=800)

    psnr_vals = torch.zeros(img_batch.shape[0])
    for idx, (img, ref) in enumerate(zip(img_batch, ref_batch)):
        default_psnr = psnr_compute(img, ref, factor=1.0, batched=True)
        try:
            img_np, ref_np = img.numpy(), ref.numpy()  # move to numpy
            descriptor_extractor.detect_and_extract(ref_np.mean(axis=0))  # and grayscale for ORB
            keypoints_src, descriptors_src = descriptor_extractor.keypoints, descriptor_extractor.descriptors
            descriptor_extractor.detect_and_extract(img_np.mean(axis=0))
            keypoints_tgt, descriptors_tgt = descriptor_extractor.keypoints, descriptor_extractor.descriptors

            matches = skimage.feature.match_descriptors(descriptors_src, descriptors_tgt, cross_check=True)
            # Look for an affine transform and search with RANSAC over matches:
            model_robust, inliers = skimage.measure.ransac(
                (keypoints_tgt[matches[:, 1]], keypoints_src[matches[:, 0]]),
                skimage.transform.EuclideanTransform,
                min_samples=len(matches) - 1,
                residual_threshold=4,
                max_trials=2500,
            )  # :>
            warped_img = skimage.transform.warp(img_np.transpose(1, 2, 0), model_robust, mode="wrap", order=1)
            # Compute normal PSNR from here:
            registered_psnr = psnr_compute(torch.as_tensor(warped_img), ref.permute(1, 2, 0), factor=1.0, batched=True)
            if registered_psnr.isfinite():
                psnr_vals[idx] = max(registered_psnr, default_psnr)
            else:
                psnr_vals[idx] = default_psnr
        except (TypeError, IndexError, RuntimeError, ValueError):
            # TypeError if RANSAC fails
            # IndexError if not enough matches are found
            # RunTimeError if ORB does not find enough features
            # ValueError if empty match sequence
            # This matching implementation fills me with joy
            psnr_vals[idx] = default_psnr
    return psnr_vals.mean()


def image_identifiability_precision(
    reconstructed_user_data,
    true_user_data,
    dataloader,
    scores=["pixel", "lpips", "self"],
    lpips_scorer=None,
    model=None,
    fudge=1e-3,
):
    """Nearest-neighbor metric as described in Yin et al., "See through Gradients: Image Batch Recovery via GradInversion"
    This version prints separate metrics for different choices of score functions.
    It's a bit messier to do it all in one go, but otherwise the data has to be loaded three separate times.

    For a self score, the model has to be provided.
    For an LPIPS score, the lpips scorer has to be provided.
    """
    # Compare the reconstructed images to each image in the dataloader with the appropriate label
    # This could be batched and partially cached to make it faster in the future ...
    identified_images = dict(zip(scores, [0 for entry in scores]))
    for batch_idx, reconstruction in enumerate(reconstructed_user_data["data"]):
        batch_label = true_user_data["labels"][batch_idx]
        label_subset = [idx for (idx, label) in dataloader.dataset.lookup.items() if label == batch_label]
        # Find the closest match to the reconstruction in all of the validation data:
        distances = dict(zip(scores, [[] for entry in scores]))
        for idx in label_subset:
            comparable_data = dataloader.dataset[idx][0].to(device=reconstruction.device)

            for score in scores:
                if score == "lpips":
                    with torch.inference_mode():
                        distances[score] += [lpips_scorer(reconstruction, comparable_data, normalize=False).mean()]
                elif score == "self" and model is not None:
                    features_rec = _return_model_features(model, reconstruction)
                    features_comp = _return_model_features(model, comparable_data)
                    distances[score] += [
                        1 - torch.nn.functional.cosine_similarity(features_rec.view(-1), features_comp.view(-1), dim=0)
                    ]
                else:
                    distances[score] += [torch.norm(comparable_data.view(-1) - reconstruction.view(-1))]

        # Verify that this match is actually close to the true user data:
        for score in scores:
            minimal_distance_data_idx = label_subset[torch.stack(distances[score]).argmin()]
            candidate_solution = dataloader.dataset[minimal_distance_data_idx][0].to(device=reconstruction.device)
            true_solution = true_user_data["data"][batch_idx]
            distance_to_true = torch.norm(candidate_solution.view(-1) - true_solution.view(-1))

            if distance_to_true < fudge:  # This should be tiny by all accounts
                identified_images[score] += 1

    return {k: v / len(reconstructed_user_data["data"]) for k, v in identified_images.items()}


@torch.inference_mode()
def _return_model_features(model, inputs):
    features = dict()  # The named-hook + dict construction should be a bit more robust
    if inputs.ndim == 3:
        inputs = inputs.unsqueeze(0)

    def named_hook(name):
        def hook_fn(module, input, output):
            features[name] = input[0]

        return hook_fn

    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, (torch.nn.Linear)):
            hook = module.register_forward_hook(named_hook(name))
            feature_layer_name = name
            break
    model(inputs)
    hook.remove()
    return features[feature_layer_name]
