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


class Mlp_sample(nn.Module):
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


class Attention_sample(nn.Module):
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

        # sample
        kernel = 3
        self.K = kernel * kernel
        self.conv_offset = nn.Linear(head_dim, 18, bias=qkv_bias)
        self.unfold = DeformUnfold(kernel_size=(3, 3), padding=1, dilation=1)

    def forward(self, x, H, W):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        offset = self.conv_offset(x.view(B, N, self.num_heads, C // self.num_heads)).reshape(B, N, self.num_heads, 18).permute(0, 2, 3, 1).reshape(B * self.num_heads, 18, H, W)

        k = k.permute(0, 1, 3, 2).reshape(B * self.num_heads, C // self.num_heads, H, W) # (B * h, C // h, H, W)
        v = v.permute(0, 1, 3, 2).reshape(B * self.num_heads, C // self.num_heads, H, W)
        k = self.unfold(k, offset).transpose(2, 1).contiguous().view(B * self.num_heads * N, C // self.num_heads, self.K)
        v = self.unfold(v, offset).transpose(2, 1).contiguous().view(B * self.num_heads * N, C // self.num_heads, self.K)

        attn = torch.bmm(q.reshape(B * self.num_heads * N, 1, C // self.num_heads), k) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(v, attn.view(B * self.num_heads * N, self.K, 1))

        x = x.view(B, self.num_heads, N, C // self.num_heads).permute(0, 2, 1, 3).contiguous().view(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Attention_sample_absolute(nn.Module):
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

        # sample
        kernel = 3
        self.K = kernel * kernel
        self.conv_offset = nn.Linear(head_dim, 18, bias=qkv_bias)
        self.unfold = DeformUnfold(kernel_size=(3, 3), padding=1, dilation=1)

        # absolute position
        self.pos_emb = nn.Parameter(torch.randn(self.K, head_dim))

    def forward(self, x, H, W):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        offset = self.conv_offset(x.view(B, N, self.num_heads, C // self.num_heads)).reshape(B, N, self.num_heads, 18).permute(0, 2, 3, 1).reshape(B * self.num_heads, 18, H, W)

        k = k.permute(0, 1, 3, 2).reshape(B * self.num_heads, C // self.num_heads, H, W) # (B * h, C // h, H, W)
        v = v.permute(0, 1, 3, 2).reshape(B * self.num_heads, C // self.num_heads, H, W)
        k = self.unfold(k, offset).transpose(2, 1).contiguous().view(B * self.num_heads * N, C // self.num_heads, self.K)
        v = self.unfold(v, offset).transpose(2, 1).contiguous().view(B * self.num_heads * N, C // self.num_heads, self.K)

        attn = torch.bmm(q.reshape(B * self.num_heads * N, 1, C // self.num_heads), k) * self.scale

        attn_pos = torch.mm(q.reshape(B * self.num_heads * N, C // self.num_heads), self.pos_emb.transpose(0, 1).contiguous()).squeeze(1)
        attn = attn.squeeze(1) + attn_pos

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(v, attn.view(B * self.num_heads * N, self.K, 1))

        x = x.view(B, self.num_heads, N, C // self.num_heads).permute(0, 2, 1, 3).contiguous().view(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x, s):
    b, h, l, _, device, dtype = *x.shape, x.device, x.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((x, col_pad), dim = 3)
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    # flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_pad = torch.zeros((b, h, s - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 2)
    # final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = flat_x_padded.reshape(b, h, l + 1, s + l - 1)
    final_x = final_x[:, :, :l, (l-1):]
    return final_x

def relative_logits_1d(q, rel_k):
    b, heads, h, w, dim = q.shape
    logits = einsum('b h x y d, r d -> b h x y r', q, rel_k)
    logits = rearrange(logits, 'b h x y r -> b (h x) y r')
    logits = rel_to_abs(logits, 9)
    # logits = logits.reshape(b, heads, h, w, w)
    # logits = expand_dim(logits, dim = 3, k = h)
    return logits

class RelPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.fmap_size = fmap_size
        # self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        # self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)
        self.rel_height = nn.Parameter(torch.randn(height + 9 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width + 9 - 1, dim_head) * scale)

    def forward(self, q):
        # h, w = self.fmap_size

        # q = rearrange(q, 'b h (x y) d -> b h x y d', x = h, y = w)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        # rel_logits_w = rearrange(rel_logits_w, 'b h x i y j-> b h (x y) (i j)')

        q = rearrange(q, 'b h x y d -> b h y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        # rel_logits_h = rearrange(rel_logits_h, 'b h x i y j -> b h (y x) (j i)')
        return rel_logits_w + rel_logits_h

class Attention_sample_relative(nn.Module):
    def __init__(self, dim, num_heads=8, fea_size=(224, 224), qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
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

        # sample
        kernel = 3
        self.K = kernel * kernel
        self.conv_offset = nn.Linear(head_dim, 18, bias=qkv_bias)
        self.unfold = DeformUnfold(kernel_size=(3, 3), padding=1, dilation=1)

        # relative position
        self.pos_emb = RelPosEmb(fea_size, head_dim)

    def forward(self, x, H, W):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        offset = self.conv_offset(x.view(B, N, self.num_heads, C // self.num_heads)).reshape(B, N, self.num_heads, 18).permute(0, 2, 3, 1).reshape(B * self.num_heads, 18, H, W)

        k = k.permute(0, 1, 3, 2).reshape(B * self.num_heads, C // self.num_heads, H, W) # (B * h, C // h, H, W)
        v = v.permute(0, 1, 3, 2).reshape(B * self.num_heads, C // self.num_heads, H, W)
        k = self.unfold(k, offset).transpose(2, 1).contiguous().view(B * self.num_heads * N, C // self.num_heads, self.K)
        v = self.unfold(v, offset).transpose(2, 1).contiguous().view(B * self.num_heads * N, C // self.num_heads, self.K)

        attn = torch.bmm(q.reshape(B * self.num_heads * N, 1, C // self.num_heads), k) * self.scale

        attn_pos = self.pos_emb(q.reshape(B * self.num_heads, H, W, C // self.num_heads, 1).permute(0, 4, 1, 2, 3)).view(B * self.num_heads * N, self.K)
        attn = attn.squeeze(1) + attn_pos

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(v, attn.view(B * self.num_heads * N, self.K, 1))

        x = x.view(B, self.num_heads, N, C // self.num_heads).permute(0, 2, 1, 3).contiguous().view(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block_sample(nn.Module):

    def __init__(self, dim, num_heads, pos_type='none', fea_size=(769, 769), mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if pos_type == 'none':
            self.attn = Attention_sample(
                dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        elif pos_type == 'abs':
            self.attn = Attention_sample_absolute(
                dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        elif pos_type == 'rel':
            self.attn = Attention_sample_relative(
                dim,
                num_heads=num_heads, fea_size=fea_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_sample(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

def build_attention_sample_share(pos_type, dim, num_heads=8, fea_size=(224, 224), qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
    if pos_type == 'none':
        return Attention_sample(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
    elif pos_type == 'abs':
        return Attention_sample_absolute(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
    elif pos_type == 'rel':
        return Attention_sample_relative(
            dim,
            num_heads=num_heads, fea_size=fea_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)

class Block_sample_share(nn.Module):

    def __init__(self, dim, num_heads, fea_size=(224, 224), mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_sample(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, attn, x, H, W):
        x = x + self.drop_path(attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
