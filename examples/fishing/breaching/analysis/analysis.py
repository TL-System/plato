"""Simple report function based on PSNR and maybe SSIM and maybe better ideas..."""
import torch

import re
from .metrics import psnr_compute, registered_psnr_compute, image_identifiability_precision, cw_ssim
from ..cases import construct_dataloader

import copy
import logging

log = logging.getLogger(__name__)


def report(
    reconstructed_user_data,
    true_user_data,
    server_payload,
    model_template,
    order_batch=True,
    compute_full_iip=False,
    compute_rpsnr=True,
    compute_ssim=True,
    cfg_case=None,
    setup=dict(device=torch.device("cpu"), dtype=torch.float),
):
    log.info("Starting evaluations for attack effectiveness report...")
    model = copy.deepcopy(model_template)  # Copy just in case and discard later
    model.to(**setup)
    metadata = server_payload[0]["metadata"]
    if metadata["modality"] == "text":
        modality_metrics = _run_text_metrics(
            reconstructed_user_data, true_user_data, server_payload, cfg_case, order_batch
        )
    else:
        modality_metrics = _run_vision_metrics(
            reconstructed_user_data,
            true_user_data,
            server_payload,
            model,
            order_batch,
            compute_full_iip,
            compute_rpsnr,
            compute_ssim,
            cfg_case,
            setup,
        )
    if reconstructed_user_data["labels"] is not None:
        test_label_acc = count_integer_overlap(
            reconstructed_user_data["labels"].view(-1),
            true_user_data["labels"].view(-1),
            # maxlength=cfg_case.data.vocab_size,
        ).item()
    else:
        test_label_acc = 0

    feat_mse = 0.0
    for idx, payload in enumerate(server_payload):
        parameters = payload["parameters"]
        buffers = payload["buffers"]

        with torch.no_grad():
            for param, server_state in zip(model.parameters(), parameters):
                param.copy_(server_state.to(**setup))
            if buffers is not None:
                for buffer, server_state in zip(model.buffers(), buffers):
                    buffer.copy_(server_state.to(**setup))
            else:
                if len(true_user_data["buffers"]) > 0:
                    for buffer, user_state in zip(model.buffers(), true_user_data["buffers"]):
                        buffer.copy_(user_state.to(**setup))

            # Compute the forward passes
            feats_rec = model(reconstructed_user_data["data"].to(device=setup["device"]))
            feats_true = model(true_user_data["data"].to(device=setup["device"]))
            relevant_features = true_user_data["labels"]
            feat_mse += (feats_rec - feats_true)[..., relevant_features.view(-1)].pow(2).mean().item()

    # Record model parameters:
    parameters = sum([p.numel() for p in model.parameters()])

    if metadata["modality"] == "text":
        m = modality_metrics
        log.info(
            f"METRICS: | Accuracy: {m['accuracy']:2.4f} | S-BLEU: {m['sacrebleu']:4.2f} | FMSE: {feat_mse:2.4e} | "
            + "\n"
            f" G-BLEU: {m['google_bleu']:4.2f} | ROUGE1: {m['rouge1']:4.2f}| ROUGE2: {m['rouge2']:4.2f} | ROUGE-L: {m['rougeL']:4.2f}"
            f"| Token Acc T:{m['token_acc']:2.2%}/A:{m['token_avg_accuracy']:2.2%} "
            f"| Label Acc: {test_label_acc:2.2%}"
        )

    else:
        m = modality_metrics
        iip_scoring = " | ".join([f"{k}: {v:5.2%}" for k, v in m.items() if "IIP" in k])
        log.info(
            f"METRICS: | MSE: {m['mse']:2.4f} | PSNR: {m['psnr']:4.2f} | FMSE: {feat_mse:2.4e} | LPIPS: {m['lpips']:4.2f}|"
            + "\n"
            f" R-PSNR: {m['rpsnr']:4.2f} | {iip_scoring} | SSIM: {m['ssim']:2.4f} | "
            f"max R-PSNR: {m['max_rpsnr']:4.2f} | max SSIM: {m['max_ssim']:2.4f} | Label Acc: {test_label_acc:2.2%}"
        )

    metrics = dict(
        **modality_metrics,
        feat_mse=feat_mse,
        parameters=parameters,
        label_acc=test_label_acc,
    )
    return metrics


def _run_text_metrics(reconstructed_user_data, true_user_data, server_payload, cfg_case, order_batch=True):
    import datasets
    from ..cases.data.datasets_text import _get_tokenizer

    text_metrics = dict()

    candidate_metrics = ["accuracy", "bleu", "rouge", "google_bleu", "sacrebleu"]
    metrics = {name: datasets.load_metric(name, cache_dir=cfg_case.data.path) for name in candidate_metrics}

    tokenizer = _get_tokenizer(
        server_payload[0]["metadata"]["tokenizer"],
        server_payload[0]["metadata"]["vocab_size"],
        cache_dir=cfg_case.data.path,
    )

    if order_batch:
        order = compute_text_order(reconstructed_user_data, true_user_data, vocab_size=cfg_case.data.vocab_size)
        reconstructed_user_data["data"] = reconstructed_user_data["data"][order]
        if reconstructed_user_data["labels"] is not None:
            reconstructed_user_data["labels"] = reconstructed_user_data["labels"][order]
        if "confidence" in reconstructed_user_data:
            reconstructed_user_data["confidence"] = reconstructed_user_data["confidence"][order]
    else:
        order = None
    text_metrics["order"] = order

    # Accuracy:
    for rec_example, ref_example in zip(reconstructed_user_data["data"], true_user_data["data"]):
        metrics["accuracy"].add_batch(predictions=rec_example, references=ref_example)
    text_metrics["accuracy"] = metrics["accuracy"].compute()["accuracy"]

    # Per sentence Accuracy:
    B = reconstructed_user_data["data"].shape[0]
    accuracies = []
    for rec_sentence, ref_sentence in zip(reconstructed_user_data["data"], true_user_data["data"]):
        accuracies.append((rec_sentence == ref_sentence).float().mean().item())
    text_metrics["intra-sentence_accuracy"] = accuracies
    text_metrics["max-sentence_accuracy"] = max(accuracies)

    for name in ["bleu", "google_bleu"]:
        # Metrics that operate on lists of words [re-encoded into word-level to reduce tokenizer impact]
        RE_split = r"[\w']+"
        rec_sent_words = [
            re.findall(RE_split, sentence) for sentence in tokenizer.batch_decode(reconstructed_user_data["data"])
        ]
        ref_sent_words = [re.findall(RE_split, sentence) for sentence in tokenizer.batch_decode(true_user_data["data"])]
        num_sentences = len(rec_sent_words)
        try:
            score = metrics[name].compute(predictions=rec_sent_words, references=[ref_sent_words] * num_sentences)
            text_metrics[name] = score[name]
        except ZeroDivisionError:  # huggingface BLEU breaks for a totally wrong sentence
            text_metrics[name] = 0.0

    for name in ["sacrebleu", "rouge"]:
        # Metrics that operate on full sentences
        rec_sentence = tokenizer.batch_decode(reconstructed_user_data["data"])
        ref_sentence = tokenizer.batch_decode(true_user_data["data"])

        num_sentences = len(rec_sentence)
        if name == "rouge":
            score = metrics[name].compute(predictions=rec_sentence, references=ref_sentence)
        else:
            score = metrics[name].compute(predictions=rec_sentence, references=[ref_sentence] * num_sentences)
        if name == "sacrebleu":
            text_metrics[name] = score["score"] / 100
        else:
            text_metrics["rouge1"] = score["rouge1"].mid.fmeasure
            text_metrics["rouge2"] = score["rouge2"].mid.fmeasure
            text_metrics["rougeL"] = score["rougeL"].mid.fmeasure

    # Token measurements:
    test_word_acc = count_integer_overlap(
        reconstructed_user_data["data"].view(-1),
        true_user_data["data"].view(-1),
        maxlength=cfg_case.data.vocab_size,
    )
    text_metrics["token_acc"] = test_word_acc.item()
    # Per sentence token overlap:
    B = reconstructed_user_data["data"].shape[0]
    overlaps = []
    for rec_sentence, ref_sentence in zip(reconstructed_user_data["data"], true_user_data["data"]):
        overlaps.append(count_integer_overlap(rec_sentence, ref_sentence, maxlength=cfg_case.data.vocab_size).item())
    text_metrics["intra-sentence_token_acc"] = overlaps

    # Frequency-corrected token acc:
    avg_token_val = average_per_token_accuracy(
        reconstructed_user_data["data"].view(-1),
        true_user_data["data"].view(-1),
        maxlength=cfg_case.data.vocab_size,
    )
    text_metrics["token_avg_accuracy"] = avg_token_val.item()

    return text_metrics


def _run_vision_metrics(
    reconstructed_user_data,
    true_user_data,
    server_payload,
    model,
    order_batch=True,
    compute_full_iip=False,
    compute_rpsnr=True,
    compute_ssim=True,
    cfg_case=None,
    setup=dict(device=torch.device("cpu"), dtype=torch.float),
):
    import lpips  # lazily import this only if vision reporting is used.

    lpips_scorer = lpips.LPIPS(net="alex", verbose=False).to(**setup)

    metadata = server_payload[0]["metadata"]
    if hasattr(metadata, "mean"):
        dm = torch.as_tensor(metadata.mean, **setup)[None, :, None, None]
        ds = torch.as_tensor(metadata.std, **setup)[None, :, None, None]
    else:
        dm, ds = torch.tensor(0, **setup), torch.tensor(1, **setup)

    rec_denormalized = torch.clamp(reconstructed_user_data["data"].to(**setup) * ds + dm, 0, 1)
    ground_truth_denormalized = torch.clamp(true_user_data["data"].to(**setup) * ds + dm, 0, 1)

    if order_batch:
        order = compute_batch_order(lpips_scorer, rec_denormalized, ground_truth_denormalized, setup)
        reconstructed_user_data["data"] = reconstructed_user_data["data"][order]
        if reconstructed_user_data["labels"] is not None:
            reconstructed_user_data["labels"] = reconstructed_user_data["labels"][order]
        rec_denormalized = rec_denormalized[order]
    else:
        order = None

    mse_score = (rec_denormalized - ground_truth_denormalized).pow(2).mean(dim=[1, 2, 3])
    avg_mse, max_mse = mse_score.mean().item(), mse_score.max().item()
    avg_psnr, max_psnr = psnr_compute(rec_denormalized, ground_truth_denormalized, factor=1)
    avg_ssim, max_ssim = cw_ssim(rec_denormalized, ground_truth_denormalized, scales=5)

    # Hint: This part switches to the lpips [-1, 1] normalization:
    lpips_score = lpips_scorer(rec_denormalized, ground_truth_denormalized, normalize=True)
    avg_lpips, max_lpips = lpips_score.mean().item(), lpips_score.max().item()

    # Compute registered psnr. This is a bit computationally intensive:
    if compute_rpsnr:
        avg_rpsnr, max_rpsnr = registered_psnr_compute(rec_denormalized, ground_truth_denormalized, factor=1)
    else:
        avg_rpsnr, max_rpsnr = float("nan"), float("nan")

    # Compute IIP score if data config is passed:
    if cfg_case is not None:
        dataloader = construct_dataloader(cfg_case.data, cfg_case.impl, user_idx=None, return_full_dataset=True)
        if compute_full_iip:
            scores = ["pixel", "lpips", "self"]
        else:
            scores = ["pixel"]
        iip_scores = image_identifiability_precision(
            reconstructed_user_data, true_user_data, dataloader, lpips_scorer=lpips_scorer, model=model, scores=scores
        )
    else:
        iip_scores = dict(none=float("NaN"))

    vision_metrics = dict(
        mse=avg_mse,
        psnr=avg_psnr,
        lpips=avg_lpips,
        rpsnr=avg_rpsnr,
        ssim=avg_ssim,
        max_ssim=max_ssim,
        max_rpsnr=max_rpsnr,
        order=order,
        **{f"IIP-{k}": v for k, v in iip_scores.items()},
    )
    return vision_metrics


def count_integer_overlap(rec_labels, true_labels, maxlength=50527):
    # if rec_labels is not None:
    #     if any(rec_labels.sort()[0] != true_labels):
    #         found_labels = 0
    #         label_pool = true_labels.clone().tolist()
    #         for label in rec_labels:
    #             if label in label_pool:
    #                 found_labels += 1
    #                 label_pool.remove(label)
    #         test_label_acc = found_labels / len(true_labels)
    #     else:
    #         test_label_acc = 1
    # else:
    #     test_label_acc = 0

    # much faster (measured with timeit:)
    if rec_labels is not None:
        test_label_acc = (
            1
            - (
                torch.bincount(rec_labels.view(-1), minlength=maxlength)
                - torch.bincount(true_labels[true_labels != -100].view(-1), minlength=maxlength)
            )
            .abs()
            .sum()
            / 2
            / rec_labels.numel()
        )
    else:
        test_label_acc = 0
    return test_label_acc


def average_per_token_accuracy(rec_labels, true_labels, maxlength=50527):
    if rec_labels is not None:
        binsrec = torch.bincount(rec_labels.view(-1), minlength=maxlength)
        binstrue = torch.bincount(true_labels[true_labels != -100].view(-1), minlength=maxlength)

        true_tokens = binstrue > 0
        per_token_accuracy = torch.clamp(
            binsrec[true_tokens] / binstrue[true_tokens], 0.0, 1.0
        )  # discount overcounting

        avg_token_val = per_token_accuracy.mean()
        # avg_freq_adjusted_val = (per_token_accuracy * binstrue[true_tokens] / true_labels.numel()).sum() # total acc ;>
    else:
        avg_token_val = 0
    return avg_token_val


def compute_batch_order(lpips_scorer, rec_denormalized, ground_truth_denormalized, setup):
    """Re-order a batch of images according to LPIPS statistics of source batch, trying to match similar images.

    This implementation basically follows the LPIPS.forward method, but for an entire batch."""
    from scipy.optimize import linear_sum_assignment  # Again a lazy import

    B = rec_denormalized.shape[0]
    L = lpips_scorer.L
    assert ground_truth_denormalized.shape[0] == B

    with torch.inference_mode():
        # Compute all features [assume sufficient memory is a given]
        features_rec = []
        for input in rec_denormalized:
            input_scaled = lpips_scorer.scaling_layer(input)
            output = lpips_scorer.net.forward(input_scaled)
            layer_features = {}
            for kk in range(L):
                layer_features[kk] = normalize_tensor(output[kk])
            features_rec.append(layer_features)

        features_gt = []
        for input in ground_truth_denormalized:
            input_scaled = lpips_scorer.scaling_layer(input)
            output = lpips_scorer.net.forward(input_scaled)
            layer_features = {}
            for kk in range(L):
                layer_features[kk] = normalize_tensor(output[kk])
            features_gt.append(layer_features)

        # Compute overall similarities:
        similarity_matrix = torch.zeros(B, B, **setup)
        for idx, x in enumerate(features_gt):
            for idy, y in enumerate(features_rec):
                for kk in range(L):
                    diff = (x[kk] - y[kk]) ** 2
                    similarity_matrix[idx, idy] += spatial_average(lpips_scorer.lins[kk](diff)).squeeze()
    try:
        _, rec_assignment = linear_sum_assignment(similarity_matrix.cpu().numpy(), maximize=False)
    except ValueError:
        print(f"ValueError from similarity matrix {similarity_matrix.cpu().numpy()}")
        print("Returning trivial order...")
        rec_assignment = list(range(B))
    return torch.as_tensor(rec_assignment, device=setup["device"], dtype=torch.long)


def compute_text_order(reconstructed_user_data, true_user_data, vocab_size):
    from scipy.optimize import linear_sum_assignment  # Again a lazy import

    """Simple text ordering based just on token overlap."""
    B = reconstructed_user_data["data"].shape[0]
    overlaps = torch.zeros(B, B, device=true_user_data["data"].device)
    for (idx, rec_sentence) in enumerate(reconstructed_user_data["data"]):
        for (idy, ref_sentence) in enumerate(true_user_data["data"]):
            overlap = count_integer_overlap(rec_sentence, ref_sentence, maxlength=vocab_size)
            overlaps[idx, idy] = overlap
    try:
        _, rec_assignment = linear_sum_assignment(overlaps.T.cpu().numpy(), maximize=True)
    except ValueError:
        print(f"ValueError from overlap matrix {overlaps.cpu().numpy()}")
        print("Returning trivial order...")
        rec_assignment = list(range(B))
    return torch.as_tensor(rec_assignment, device=true_user_data["data"].device, dtype=torch.long)


def normalize_tensor(in_feat, eps=1e-10):
    """From https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/__init__.py."""
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


def spatial_average(in_tens, keepdim=True):
    """https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py ."""
    return in_tens.mean([2, 3], keepdim=keepdim)


def find_oneshot(rec_denormalized, ground_truth_denormalized):
    one_shot = (rec_denormalized - ground_truth_denormalized).pow(2)
    one_shot_idx = one_shot.view(one_shot.shape[0], -1).mean(dim=-1).argmin()
    return one_shot_idx
