import torch
from plato.config import Config
from utils import load_gpt2_from_dict
from consts import BERT_CLS_TOKEN, BERT_SEP_TOKEN, BERT_PAD_TOKEN
import numpy as np


def get_aux_lm(device):
    lm = load_gpt2_from_dict(
        "transformer_wikitext-103.pth", device, output_hidden_states=True
    ).to(device)
    lm.eval()

    return lm


def fix_special_tokens(x_embeds, bert_embeddings_weight, pads):
    x_embeds.data[:, 0] = bert_embeddings_weight[BERT_CLS_TOKEN]
    if pads is not None:
        for sen_id in range(x_embeds.shape[0]):
            x_embeds.data[sen_id, pads[sen_id] :] = bert_embeddings_weight[
                BERT_PAD_TOKEN
            ]
            x_embeds.data[sen_id, pads[sen_id] - 1] = bert_embeddings_weight[
                BERT_SEP_TOKEN
            ]
    elif x_embeds.shape[0] == 1:
        x_embeds.data[:, -1] = bert_embeddings_weight[BERT_SEP_TOKEN]
    return x_embeds


def get_closest_tokens(inputs_embeds, unused_tokens, embeddings_weight, metric="cos"):
    embeddings_weight = embeddings_weight.repeat(inputs_embeds.shape[0], 1, 1)
    if metric == "l2":
        d = torch.cdist(inputs_embeds, embeddings_weight, p=2)
    elif metric == "cos":
        dp = torch.bmm(inputs_embeds, embeddings_weight.transpose(1, 2))
        norm1 = inputs_embeds.norm(p=2, dim=2).unsqueeze(2)
        norm2 = embeddings_weight.norm(p=2, dim=2).unsqueeze(1)
        d = -dp / (norm1 * norm2)
    else:
        assert False

    d[:, :, unused_tokens] = 1e9
    return d, d.min(dim=2)[1]


def get_reconstruction_loss(model, x_embeds, y_labels, true_grads, create_graph=False):
    grads = compute_grads(model, x_embeds, y_labels, create_graph=create_graph)
    return grad_dist(true_grads, grads)


def compute_grads(model, x_embeds, y_labels, create_graph=False):
    outs = model(inputs_embeds=x_embeds, labels=y_labels)
    return torch.autograd.grad(
        outs.loss, model.parameters(), create_graph=create_graph, allow_unused=True
    )


def grad_dist(grads1, grads2):
    ret = 0.0
    n_g = 0
    for g1, g2 in zip(grads1, grads2):
        if (g1 is not None) and (g2 is not None):
            if Config().algorithm.loss == "cos":
                ret += 1.0 - (g1 * g2).sum() / (
                    g1.view(-1).norm(p=2) * g2.view(-1).norm(p=2)
                )
            elif Config().algorithm.loss == "dlg":
                ret += (g1 - g2).square().sum()
            elif Config().algorithm.loss == "tag":
                ret += (
                    g1 - g2
                ).square().sum() + Config().algorithm.tag_factor * torch.abs(
                    g1 - g2
                ).sum()
            else:
                assert False
            n_g += 1
    if Config().algorithm.loss == "cos":
        ret /= n_g
    return ret


def get_loss(lm, model, ids, x_embeds, true_labels, true_grads, create_graph=False):
    perplexity = lm(input_ids=ids, labels=ids).loss
    rec_loss = get_reconstruction_loss(
        model, x_embeds, true_labels, true_grads, create_graph=create_graph
    )
    return (
        perplexity,
        rec_loss,
        rec_loss + Config().algorithm.coeff_perplexity * perplexity,
    )


def swap_tokens(x_embeds, max_len, cos_ids, lm, model, true_labels, true_grads):
    print("Attempt swap", flush=True)
    best_x_embeds, best_tot_loss = None, None
    changed = None
    for sen_id in range(x_embeds.data.shape[0]):
        for sample_idx in range(200):
            perm_ids = np.arange(x_embeds.shape[1])

            if sample_idx != 0:
                if sample_idx % 4 == 0:  # swap two tokens
                    i, j = 1 + np.random.randint(
                        max_len[sen_id] - 2
                    ), 1 + np.random.randint(max_len[sen_id] - 2)
                    perm_ids[i], perm_ids[j] = perm_ids[j], perm_ids[i]
                elif sample_idx % 4 == 1:  # move a token to another place
                    i = 1 + np.random.randint(max_len[sen_id] - 2)
                    j = 1 + np.random.randint(max_len[sen_id] - 1)
                    if i < j:
                        perm_ids = np.concatenate(
                            [
                                perm_ids[:i],
                                perm_ids[i + 1 : j],
                                perm_ids[i : i + 1],
                                perm_ids[j:],
                            ]
                        )
                    else:
                        perm_ids = np.concatenate(
                            [
                                perm_ids[:j],
                                perm_ids[i : i + 1],
                                perm_ids[j:i],
                                perm_ids[i + 1 :],
                            ]
                        )
                elif sample_idx % 4 == 2:  # move a sequence to another place
                    b = 1 + np.random.randint(max_len[sen_id] - 1)
                    e = 1 + np.random.randint(max_len[sen_id] - 1)
                    if b > e:
                        b, e = e, b
                    p = 1 + np.random.randint(max_len[sen_id] - 1 - (e - b))
                    if p >= b:
                        p += e - b
                    if p < b:
                        perm_ids = np.concatenate(
                            [perm_ids[:p], perm_ids[b:e], perm_ids[p:b], perm_ids[e:]]
                        )
                    elif p >= e:
                        perm_ids = np.concatenate(
                            [perm_ids[:b], perm_ids[e:p], perm_ids[b:e], perm_ids[p:]]
                        )
                    else:
                        assert False
                elif sample_idx % 4 == 3:  # take some prefix and put it at the end
                    i = 1 + np.random.randint(max_len[sen_id] - 2)
                    perm_ids = np.concatenate(
                        [perm_ids[:1], perm_ids[i:-1], perm_ids[1:i], perm_ids[-1:]]
                    )

            new_ids = cos_ids.clone()
            new_ids[sen_id] = cos_ids[sen_id, perm_ids]
            new_x_embeds = x_embeds.clone()
            new_x_embeds[sen_id] = x_embeds[sen_id, perm_ids, :]

            _, _, new_tot_loss = get_loss(
                lm, model, new_ids, new_x_embeds, true_labels, true_grads
            )

            if (best_tot_loss is None) or (new_tot_loss < best_tot_loss):
                best_x_embeds = new_x_embeds
                best_tot_loss = new_tot_loss
                if sample_idx != 0:
                    changed = sample_idx % 4
        if not (changed is None):
            change = [
                "Swapped tokens",
                "Moved token",
                "Moved sequence",
                "Put prefix at the end",
            ][changed]
            print(change, flush=True)
        x_embeds.data = best_x_embeds
