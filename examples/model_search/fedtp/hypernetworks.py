"""
Hypernetworks used in FedTP.

Modified based on https://github.com/zhyczy/FedTP/blob/main/models/Hypernetworks.py.
"""

from collections import OrderedDict

from torch import nn
from torch.nn.utils import spectral_norm

from plato.config import Config


class ViTHyper(nn.Module):
    """
    The HyerNetwork for generating Vision Transformer (ViT)'s attention maps.
    """

    def __init__(
        self,
        n_nodes,
        embedding_dim,
        hidden_dim,
        dim,
        client_sample,
        heads=8,
        dim_head=64,
        n_hidden=1,
        depth=6,
        spec_norm=False,
    ):
        # pylint:disable=too-many-arguments
        super().__init__()
        self.dim = dim
        self.inner_dim = dim_head * heads
        self.depth = depth
        self.client_sample = client_sample
        # embedding layer
        self.embeddings = nn.Embedding(
            num_embeddings=n_nodes, embedding_dim=embedding_dim
        )

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim))
            if spec_norm
            else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim))
                if spec_norm
                else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.to_qkv_value_list = nn.ModuleList([])
        for _ in range(self.depth):
            if len(Config().parameters.hypernet.attention.split(",")) > 1:
                to_qkv_value = nn.ModuleList(
                    [nn.Linear(hidden_dim, self.dim * self.inner_dim) for _ in range(3)]
                )
            else:
                to_qkv_value = nn.Linear(hidden_dim, self.dim * self.inner_dim * 3)
            self.to_qkv_value_list.append(to_qkv_value)

    def forward(self, idx):
        "The forward pass of hypernetwork."
        weights = 0
        emd = self.embeddings(idx)
        features = self.mlp(emd)
        weights = OrderedDict()
        for dep in range(self.depth):
            layer_d_qkv_value_hyper = self.to_qkv_value_list[dep]
            attention_map = Config().parameters.hypernet.attention.split(",")
            if len(attention_map) == 1:
                layer_d_qkv_value = layer_d_qkv_value_hyper(features).view(
                    self.inner_dim * 3, self.dim
                )
                name = Config().parameters.hypernet.attention % (dep)
                weights[name] = layer_d_qkv_value.cpu()
            else:
                layer_d_qkv_value = [
                    layer(features).view(self.inner_dim, self.dim)
                    for layer in layer_d_qkv_value_hyper
                ]
                name = Config().parameters.hypernet.attention % (dep, dep, dep)
                names = name.split(",")
                key = names[0]
                query = names[1]
                value = names[2]
                weights[key] = layer_d_qkv_value[0].cpu()
                weights[query] = layer_d_qkv_value[1].cpu()
                weights[value] = layer_d_qkv_value[2].cpu()
        return weights


class ShakesHyper(nn.Module):
    """
    HyperNetwork for transformer.
    """

    # pylint:disable=too-many-instance-attributes
    def __init__(
        self,
        n_nodes,
        embedding_dim,
        hidden_dim,
        dim,
        client_sample,
        heads=8,
        dim_head=64,
        n_hidden=1,
        depth=6,
        spec_norm=False,
    ):
        # pylint:disable=too-many-arguments
        # pylint:disable=too-many-locals

        super().__init__()
        self.dim = dim
        self.inner_dim = dim_head * heads
        self.depth = depth
        self.client_sample = client_sample
        # embedding layer
        self.embeddings = nn.Embedding(
            num_embeddings=n_nodes, embedding_dim=embedding_dim
        )

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim))
            if spec_norm
            else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim))
                if spec_norm
                else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.wqs_value_list = nn.ModuleList([])
        self.wks_value_list = nn.ModuleList([])
        self.wvs_value_list = nn.ModuleList([])

        for _ in range(self.depth):
            wq_value = nn.Linear(hidden_dim, dim * heads * dim_head)
            wk_value = nn.Linear(hidden_dim, dim * heads * dim_head)
            wv_value = nn.Linear(hidden_dim, dim * heads * dim_head)
            self.wqs_value_list.append(wq_value)
            self.wks_value_list.append(wk_value)
            self.wvs_value_list.append(wv_value)

    def forward(self, idx, test):
        "The forward pass of hypernetwork."
        weights = 0
        emd = self.embeddings(idx)
        features = self.mlp(emd)
        if test is False:
            weights = [OrderedDict() for x in range(self.client_sample)]
            for dep in range(self.depth):
                layer_d_q_value_hyper = self.wqs_value_list[dep]
                layer_d_q_value = layer_d_q_value_hyper(features).view(
                    -1, self.inner_dim, self.dim
                )
                layer_d_k_value_hyper = self.wks_value_list[dep]
                layer_d_k_value = layer_d_k_value_hyper(features).view(
                    -1, self.inner_dim, self.dim
                )
                layer_d_v_value_hyper = self.wvs_value_list[dep]
                layer_d_v_value = layer_d_v_value_hyper(features).view(
                    -1, self.inner_dim, self.dim
                )
                for index in range(self.client_sample):
                    weights[index][
                        "encoder.layer_stack." + str(dep) + ".slf_attn.w_qs.weight"
                    ] = layer_d_q_value[index]
                    weights[index][
                        "encoder.layer_stack." + str(dep) + ".slf_attn.w_ks.weight"
                    ] = layer_d_k_value[index]
                    weights[index][
                        "encoder.layer_stack." + str(dep) + ".slf_attn.w_vs.weight"
                    ] = layer_d_v_value[index]
        else:
            weights = OrderedDict()
            for dep in range(self.depth):
                layer_d_q_value_hyper = self.wqs_value_list[dep]
                layer_d_q_value = layer_d_q_value_hyper(features).view(
                    self.inner_dim, self.dim
                )
                layer_d_k_value_hyper = self.wks_value_list[dep]
                layer_d_k_value = layer_d_k_value_hyper(features).view(
                    self.inner_dim, self.dim
                )
                layer_d_v_value_hyper = self.wvs_value_list[dep]
                layer_d_v_value = layer_d_v_value_hyper(features).view(
                    self.inner_dim, self.dim
                )
                weights["encoder.layer_stack." + str(dep) + ".slf_attn.w_qs.weight"] = (
                    layer_d_q_value
                )
                weights["encoder.layer_stack." + str(dep) + ".slf_attn.w_ks.weight"] = (
                    layer_d_k_value
                )
                weights["encoder.layer_stack." + str(dep) + ".slf_attn.w_vs.weight"] = (
                    layer_d_v_value
                )
        return weights
