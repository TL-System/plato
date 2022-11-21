""" 
Code for DeepViT. The implementation has heavy reference to timm.
"""
import torch
import torch.nn as nn
from functools import partial
import pickle
from torch.nn.parameter import Parameter

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model

from .layers import *

from torch.nn import functional as F

import numpy as np


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'Deepvit_base_patch16_224_16B': _cfg(
        url='',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'Deepvit_base_patch16_224_24B': _cfg(
        url='',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'Deepvit_base_patch16_224_32B': _cfg(
        url='',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'Deepvit_L_384': _cfg(
        url='',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
}



class DeepVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, group = False, re_atten=True, cos_reg = False,
                 use_cnn_embed=False, apply_transform=None, transform_scale=False, scale_adjustment=1.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # use cosine similarity as a regularization term
        self.cos_reg = cos_reg

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            if use_cnn_embed:
                self.patch_embed = PatchEmbed_CNN(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            else:
                self.patch_embed = PatchEmbed(
                    img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        d = depth if isinstance(depth, int) else len(depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, d)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, share=depth[i], num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, group = group, 
                re_atten=re_atten, apply_transform=apply_transform[i], transform_scale=transform_scale, scale_adjustment=scale_adjustment)
            for i in range(len(depth))])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        if self.cos_reg:
            atten_list = []
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        attn = None
        for blk in self.blocks:
            x, attn = blk(x, attn)
            if self.cos_reg:
                atten_list.append(attn)

        x = self.norm(x)
        if self.cos_reg and self.training:
            return x[:, 0], atten_list
        else:
            return x[:, 0]

    def forward(self, x):
        if self.cos_reg and self.training:
            x, atten = self.forward_features(x)
            x = self.head(x)
            return x, atten
        else:
            x = self.forward_features(x)
            x = self.head(x)
            return x


@register_model
def deepvit_patch16_224_re_attn_16b(pretrained=False, **kwargs):
    apply_transform = [False] * 0 + [True] * 16
    model = DeepVisionTransformer(
        patch_size=16, embed_dim=384, depth=[False] * 16, apply_transform=apply_transform, num_heads=12, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  **kwargs)
    # We following the same settings for original ViT
    model.default_cfg = default_cfgs['Deepvit_base_patch16_224_16B']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model

@register_model
def deepvit_patch16_224_re_attn_24b(pretrained=False, **kwargs):
    apply_transform = [False] * 0 + [True] * 24
    model = DeepVisionTransformer(
        patch_size=16, embed_dim=384, depth=[False] * 24, apply_transform=apply_transform, num_heads=12, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  **kwargs)
    # We following the same settings for original ViT
    model.default_cfg = default_cfgs['Deepvit_base_patch16_224_24B']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model
 
@register_model
def deepvit_patch16_224_re_attn_32b(pretrained=False, **kwargs):
    apply_transform = [False] * 0 + [True] * 32
    model = DeepVisionTransformer(
        patch_size=16, embed_dim=384, depth=[False] * 32, apply_transform=apply_transform, num_heads=12, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  **kwargs)
    # We following the same settings for original ViT
    model.default_cfg = default_cfgs['Deepvit_base_patch16_224_32B']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model
@register_model
def deepvit_S(pretrained=False, **kwargs):
    apply_transform = [False] * 11 + [True] * 5
    model = DeepVisionTransformer(
        patch_size=16, embed_dim=396, depth=[False] * 16, apply_transform=apply_transform, num_heads=12, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  transform_scale=True, use_cnn_embed = True, scale_adjustment=0.5, **kwargs)
    # We following the same settings for original ViT
    model.default_cfg = default_cfgs['Deepvit_base_patch16_224_32B']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model
@register_model
def deepvit_L(pretrained=False, **kwargs):
    apply_transform = [False] * 20 + [True] * 12
    model = DeepVisionTransformer(
        patch_size=16, embed_dim=420, depth=[False] * 32, apply_transform=apply_transform, num_heads=12, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), use_cnn_embed = True, scale_adjustment=0.5, **kwargs)
    # We following the same settings for original ViT
    model.default_cfg = default_cfgs['Deepvit_base_patch16_224_32B']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model

@register_model
def deepvit_L_384(pretrained=False, **kwargs):
    apply_transform = [False] * 20 + [True] * 12
    model = DeepVisionTransformer(
        img_size=384, patch_size=16, embed_dim=420, depth=[False] * 32, apply_transform=apply_transform, num_heads=12, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), use_cnn_embed = True, scale_adjustment=0.5, **kwargs)
    # We following the same settings for original ViT
    model.default_cfg = default_cfgs['Deepvit_L_384']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model
