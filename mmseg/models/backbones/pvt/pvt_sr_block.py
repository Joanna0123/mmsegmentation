import math
import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from detectron2.layers import DeformUnfold


class Mlp_sr(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention_sr(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class SpatialReduction_Conv33(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, dim=256, depth=0):
        super().__init__()
        self.depth = depth
        if depth == 0:
            self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
            self.norm = nn.BatchNorm2d(dim)
        else:
            self.convs = nn.ModuleList([
                nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, groups=1, bias=False, dilation=1)
                for i in range(depth)])
            self.norms = nn.ModuleList([
                    nn.BatchNorm2d(dim)
                    for i in range(depth)])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.depth == 0:
            x = self.relu(self.norm(self.conv(x)))
        else:
            for i in range(self.depth):
                x = self.relu(self.norms[i](self.convs[i](x)))

        return x

class Block_sr(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.sr_ratio = sr_ratio
        self.sr = SpatialReduction_Conv33(dim, int(math.log(sr_ratio, 2)))

        self.norm1 = norm_layer(dim)
        self.attn = Attention_sr(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_sr(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.sr(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).reshape(B, C, -1).permute(0, 2, 1)

        H_sr, W_sr = H // self.sr_ratio, W // self.sr_ratio
        x = x + self.drop_path(self.attn(self.norm1(x), H_sr, W_sr))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = F.interpolate(
                x.reshape(B, H_sr, W_sr, C).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear", align_corners=True).reshape(B, C, H * W).permute(0, 2, 1)

        return x
