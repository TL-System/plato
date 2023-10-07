import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config


def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))


def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


# https://github.com/facebookresearch/text-adversarial-attack/blob/main/src/utils.py
def load_gpt2_from_dict(dict_path, device, output_hidden_states=False):
    state_dict = torch.load(dict_path, map_location=device)["model"]

    config = GPT2Config(
        vocab_size=30522,
        n_embd=1024,
        n_head=8,
        activation_function="relu",
        n_layer=24,
        output_hidden_states=output_hidden_states,
    )
    model = GPT2LMHeadModel(config)
    model.load_state_dict(state_dict)
    # The input embedding is not loaded automatically
    model.set_input_embeddings(
        embedding_from_weights(state_dict["transformer.wte.weight"].cpu())
    )

    return model


def embedding_from_weights(w):
    layer = nn.Embedding(w.size(0), w.size(1))
    layer.weight.data = w
    return layer
