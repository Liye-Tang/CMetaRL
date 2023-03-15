import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
# from .helpers import build_model_with_cfg, named_apply, adapt_input_conv
from models.layers import Mlp, trunc_normal_, lecun_normal_
# from .registry import register_model

_logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        # B is the batch_size, N is the number of tokens, and C embedding dim
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

    def __init__(self, input_dim, num_input_tokens=2, num_policy_tokens=10, embed_dim=6*64, depth=3,
                 mlp_ratio=4., qkv_bias=True, norm_layer=None, out_feature_dim=None, num_heads=6,
                 act_layer=None):
        super().__init__()
        
        # the number of the input tokens and policy tokens
        self.num_input_tokens = num_input_tokens
        self.num_policy_tokens = num_policy_tokens
        self.num_features = self.embed_dim = embed_dim
        self.out_feature_dim = out_feature_dim
        self.input_dim = input_dim
        self.num_heads = num_heads
        
        # init the policy and the cls token
        self.policy_tokens = nn.Parameter(torch.zeros(1, self.num_policy_tokens, embed_dim))
        # self.policy_tokens = [nn.Parameter(torch.zeros(1, 1, embed_dim)).to(device) for i in range(self.num_policy_tokens)]
        
        # action
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_input_tokens + self.num_policy_tokens, embed_dim))
        
        self.low_blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer,
                act_layer=act_layer)
            for i in range(int(depth))])

        self.norm = norm_layer(embed_dim)

        self.input_layer = nn.Linear(self.input_dim, self.num_features)
        self.output_layer = nn.Linear(self.num_features, self.out_feature_dim)

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=.02)
        # for policy_token in self.policy_tokens:
        #     trunc_normal_(policy_token, std=.02)
        trunc_normal_(self.policy_tokens, std=.02)
        self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    def forward(self, x, token_num):
        x = self.input_layer(x)
        # policy_tokens = [policy_token.expand(x.shape[0], -1, -1) for policy_token in self.policy_tokens]
        policy_tokens = self.policy_tokens.expand(x.shape[0], -1, -1)
        # print(self.policy_tokens)
        x = torch.cat((policy_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.low_blocks(x)
        x = self.norm(x)
        x = x[torch.arange(token_num.shape[0]).to(device), token_num.squeeze(1)]
        return self.output_layer(x)


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