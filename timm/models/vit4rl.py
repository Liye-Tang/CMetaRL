import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from .helpers import build_model_with_cfg, named_apply, adapt_input_conv
from .layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from .registry import register_model

_logger = logging.getLogger(__name__)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Identity()
        self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        # print("patch_size{}".format(patch_size))
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 4 if distilled else 3
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token3 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        # self.pos_drop = nn.Dropout(p=drop_rate)

        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.low_blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer,
                act_layer=act_layer)
            for i in range(int(depth / 3))])

        self.mid_blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer,
                act_layer=act_layer)
            for i in range(int(depth / 3))])

        self.top_blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer,
                act_layer=act_layer)
            for i in range(int(depth / 3))])

        self.norm = norm_layer(embed_dim)
        self.pooling_x = nn.MaxPool2d((3, 3), stride=(2, 1), padding=1)
        self.pooling_y = nn.MaxPool2d((3, 3), stride=(1, 2), padding=1)
        self.reduction = nn.Sequential(
            nn.Linear(192, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
        )

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token1, std=.02)
            trunc_normal_(self.cls_token2, std=.02)
            trunc_normal_(self.cls_token3, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def copy_token(self):
        self.cls_token3 = torch.nn.Parameter(self.cls_token1.clone())
        self.cls_token2 = torch.nn.Parameter(self.cls_token1.clone())

    def forward_features1(self, x):
        x = self.patch_embed(x)
        # print("patch_embed.shape{}".format(x.shape))
        cls_token1 = self.cls_token1.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        cls_token2 = self.cls_token2.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        cls_token3 = self.cls_token3.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token1, cls_token2, cls_token3, x), dim=1)
        else:
            x = torch.cat((cls_token1, cls_token2, cls_token3, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        x = self.low_blocks(x)
        x = self.update_pool_low(x)
        x = self.mid_blocks(x)
        x = self.update_pool_top(x)
        x = self.top_blocks(x)
        x = self.norm(x)
        return self.pre_logits(x[:, 0])

    def forward_features2(self, x):
        x = self.patch_embed(x)
        # print("patch_embed.shape{}".format(x.shape))
        cls_token1 = self.cls_token1.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        cls_token2 = self.cls_token2.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        cls_token3 = self.cls_token3.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token1, cls_token2, cls_token3, x), dim=1)
        else:
            x = torch.cat((cls_token1, cls_token2, cls_token3, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        x = self.low_blocks(x)
        x = self.update_pool_low(x)
        x = self.mid_blocks(x)
        x = self.update_pool_top(x)
        x = self.top_blocks(x)
        x = self.norm(x)
        return self.pre_logits(x[:, 1])

    def forward_features3(self, x):
        x = self.patch_embed(x)
        # print("patch_embed.shape{}".format(x.shape))
        cls_token1 = self.cls_token1.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        cls_token2 = self.cls_token2.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        cls_token3 = self.cls_token3.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token1, cls_token2, cls_token3, x), dim=1)
        else:
            x = torch.cat((cls_token1, cls_token2, cls_token3, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        x = self.low_blocks(x)
        x = self.update_pool_low(x)
        x = self.mid_blocks(x)
        x = self.update_pool_top(x)
        x = self.top_blocks(x)
        x = self.norm(x)
        return self.pre_logits(x[:, 2])

    # def revise(self,x):
    #     B,C,L=x.shape
    #     W = int(math.sqrt(int(L/2)))
    #     H = 2*W
    #     return x.view(B,C,H,W).transpose(-1,-2).reshape(B,C,-1)

    # def pool_permute(self,x):
    #     B,C,L=x.shape
    #     return x.view(B,C,int(math.sqrt(L)),int(math.sqrt(L))).transpose(-1,-2).reshape(B,C,-1)

    def revise_2D_low(self, x):
        B, C, L = x.shape
        W, H = int(math.sqrt(int(L))), int(math.sqrt(int(L)))
        return x.view(B, C, H, W)

    # def revise_2D_top(self,x):
    #     B,C,L=x.shape
    #     W = int((math.sqrt(int(L*8+1))+1)/4)
    #     H = 2*W-1
    #     return x.view(B,C,H,W)

    def revise_2D_top(self, x):
        B, C, L = x.shape
        W = int(math.sqrt(int(L / 2)))
        H = 2 * W
        return x.view(B, C, H, W)

    def revise_1D(self, x):
        B, C, H, W = x.shape
        return x.reshape(B, C, H * W)

    def update_pool_top(self, x):
        x_policy1 = torch.unsqueeze(x[:, 0], 1)
        x_policy2 = torch.unsqueeze(x[:, 1], 1)
        x_policy3 = torch.unsqueeze(x[:, 2], 1)
        x_rec = torch.unsqueeze(x[:, 3], 1)
        x_pool = self.revise_2D_top(x[:, 4:].transpose(2, 1))
        x_pool = self.pooling_x(x_pool)
        x_pool = self.revise_1D(x_pool)
        x_pool = x_pool.transpose(2, 1)
        x_pool = x_pool + self.reduction(x_pool)
        return torch.cat([x_policy1, x_policy2, x_policy3, x_rec, x_pool], dim=1)

    def update_pool_low(self, x):
        x_policy1 = torch.unsqueeze(x[:, 0], 1)
        x_policy2 = torch.unsqueeze(x[:, 1], 1)
        x_policy3 = torch.unsqueeze(x[:, 2], 1)
        x_rec = torch.unsqueeze(x[:, 3], 1)
        x_pool = self.revise_2D_low(x[:, 4:].transpose(2, 1))
        x_pool = self.pooling_y(x_pool)
        x_pool = self.revise_1D(x_pool)
        x_pool = x_pool.transpose(2, 1)
        x_pool = x_pool + self.reduction(x_pool)
        return torch.cat([x_policy1, x_policy2, x_policy3, x_rec, x_pool], dim=1)

    def forward_reconstruction(self, x):
        x = self.patch_embed(x)
        cls_token1 = self.cls_token1.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        cls_token2 = self.cls_token2.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        cls_token3 = self.cls_token3.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token1, cls_token2, cls_token3, x), dim=1)
        else:
            x = torch.cat((cls_token1, cls_token2, cls_token3, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        x = self.low_blocks(x)
        x = self.update_pool_low(x)
        x = self.mid_blocks(x)
        x = self.update_pool_top(x)
        x = self.top_blocks(x)
        x = self.norm(x)
        return x[:, 3]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)