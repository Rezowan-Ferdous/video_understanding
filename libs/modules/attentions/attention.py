import torch.nn as nn
from einops import rearrange
import math


def custom_round(x, divisor):
    result = x / divisor
    fraction = result - int(result)  # Get the fractional part
    if fraction >= 0.5:
        return math.ceil(result)
    else:
        return math.floor(result)


class Attention_Temporal(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=2,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            with_qkv=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv_4 = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.qkv_8 = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.qkv_16 = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj_4 = nn.Linear(dim, dim)
            self.proj_8 = nn.Linear(dim, dim)
            self.proj_16 = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)


    def forward(self, x, B):
        x= x.permute(0,2,1)
        BK,T,C = x.shape
        t1 = T // 4
        t2 = T // 2


        x_4 = x[:, T - t1:, ]
        x_8 = x[:, t2:, ]
        x_16 = x
        K = BK // B


        qkv_4 = self.qkv_4(x_4)


        qkv_4 = rearrange(
            qkv_4,
            "(b k) t (qkv num_heads c) -> qkv (b k) num_heads t c",
            k=K,
            qkv=3,
            num_heads=self.num_heads,
        )
        q_4, k_4, v_4 = (qkv_4[0], qkv_4[1], qkv_4[2])


        qkv_8 = self.qkv_8(x_8)


        qkv_8 = rearrange(
            qkv_8,
            "(b k) t (qkv num_heads c) -> qkv (b k) num_heads t c",
            k=K,
            qkv=3,
            num_heads=self.num_heads,
        )
        q_8, k_8, v_8 = (qkv_8[0], qkv_8[1], qkv_8[2])


        qkv_16 = self.qkv_16(x_16)


        qkv_16 = rearrange(
            qkv_16,
            "(b k) t (qkv num_heads c) -> qkv (b k) num_heads t c",
            k=K,
            qkv=3,
            num_heads=self.num_heads,
        )
        q_16, k_16, v_16 = (qkv_16[0], qkv_16[1], qkv_16[2])


        attn_4 = (q_4 @ k_4.transpose(-2, -1)) * self.scale
        attn_4 = attn_4.softmax(dim=-1)
        attn_4 = self.attn_drop(attn_4)
        x_4 = attn_4 @ v_4
        x_4 = rearrange(x_4, "(b k) num_heads t c -> (b k) t (num_heads c)", b=B)


        attn_8 = (q_8 @ k_8.transpose(-2, -1)) * self.scale
        attn_8 = attn_8.softmax(dim=-1)
        attn_8 = self.attn_drop(attn_8)
        x_8 = attn_8 @ v_8
        x_8 = rearrange(x_8, "(b k) num_heads t c -> (b k) t (num_heads c)", b=B)


        attn_16 = (q_16 @ k_16.transpose(-2, -1)) * self.scale
        attn_16 = attn_16.softmax(dim=-1)
        attn_16 = self.attn_drop(attn_16)
        x_16 = attn_16 @ v_16
        x_16 = rearrange(x_16, "(b k) num_heads t c -> (b k) t (num_heads c)", b=B)


        x_4 = self.proj_4(x_4)
        # x_8[:, t1:, :] = 0.5 * x_8[:, t1:, :] + 0.5 * x_4


        ##
        x_8_slice = x_8[:, t1:, :]  # Slice from t1 onwards
        x_4_slice = x_4[:, :x_8_slice.shape[1], :]  # Slice x_4 to match x_8_slice
        # Ensure both slices have the same size by trimming the larger tensor
        min_size = min(x_8_slice.shape[1], x_4_slice.shape[1])


        x_8_slice = x_8_slice[:, :min_size, :]
        x_4_slice = x_4_slice[:, :min_size, :]


        # Now perform the addition
        x_8[:, t1:t1 + min_size, :] = 0.5 * x_8_slice + 0.5 * x_4_slice
        ##




        x_8 = self.proj_8(x_8)
        x_16[:, t2:, :] = 0.5 * x_16[:, t2:, :] + 0.5 * x_8
        x_16 = self.proj_drop(self.proj_16(x_16))


        return x_16




class Attention_Spatial(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        with_qkv=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)


    def forward(self, x, B):
        BT, K, C = x.shape
        T = BT // B
        qkv = self.qkv(x)
        # For Intra-Spatial: (BT, heads, K, C)
        # Atten: K*K, Values: K*C
        qkv = rearrange(
            qkv,
            "(b t) k (qkv num_heads c) -> qkv (b t) num_heads k c",
            t=T,
            qkv=3,
            num_heads=self.num_heads,
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)


        x = attn @ v
        x = rearrange(
            x,
            "(b t) num_heads k c -> (b t) k (num_heads c)",
            b=B,
        )
        x = self.proj(x)
        return self.proj_drop(x)
