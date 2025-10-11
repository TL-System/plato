"""
Modified ViT.
"""

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


class Scaler(nn.Module):
    "The scaler module for different rates of the models."

    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, feature):
        "Forward function."
        output = feature / self.rate if self.training else feature
        return output


# pylint:disable=invalid-name
def pair(t):
    """
    Helper function.
    """
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    """
    Perpare norm layers.
    """

    def __init__(self, dim, fn, model_rate=1.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.scaler = Scaler(model_rate)

    def forward(self, x, **kwargs):
        """
        Forward function.
        """
        return self.fn(self.scaler(self.norm(x)), **kwargs)


class FeedForward(nn.Module):
    """
    Feed forward network.
    """

    def __init__(self, dim, hidden_dim, dropout=0.0, model_rate=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            Scaler(model_rate),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Forward function.
        """
        return self.net(x)


class Attention(nn.Module):
    """
    MHS module.
    """

    # pylint:disable=too-many-arguments
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, model_rate=1.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        self.scaler = Scaler(model_rate)

    def forward(self, x):
        """
        Forward function.
        """
        qkv = self.scaler(self.to_qkv(x)).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.scaler(out)
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Transformer blocks
    """

    # pylint:disable=too-many-arguments
    def __init__(
        self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, model_rate=1.0
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                model_rate=model_rate,
                            ),
                            model_rate,
                        ),
                        PreNorm(
                            dim,
                            FeedForward(
                                dim, mlp_dim, dropout=dropout, model_rate=model_rate
                            ),
                            model_rate,
                        ),
                    ]
                )
            )

    def forward(self, x):
        "Forward function"
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# pylint:disable=too-many-instance-attributes
class ViT(nn.Module):
    """
    Vision image transformer.
    """

    # pylint:disable=too-many-locals
    def __init__(
        self,
        *,
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        pool="cls",
        channels=1,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.1,
        model_rate=1.0,
        configs=None,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        if configs is not None:
            depth = configs[0]
            model_rate = configs[1]
        self.channel = channels
        dim = int(model_rate * dim)
        mlp_dim = int(mlp_dim * model_rate)
        dim_head = int(model_rate * dim_head)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, (
            "Image dimensions must be divisible by the patch size."
        )

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, model_rate
        )

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        """
        Forward function.
        """
        if img.size(-1) != 32:
            img = torch.nn.functional.interpolate(img, size=32, mode="bicubic")
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

    def get_net(self, func=max):
        """
        Get a random net, or max net or min net.
        """
        return [func([1, 2, 3, 4, 5, 6]), func([0.5, 1])]
