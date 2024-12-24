import csv
import math
from functools import partial
from einops.layers.torch import Rearrange
from einops import rearrange, reduce
import torch.nn.functional as F
from torch import nn, einsum
import torch
import sys
import copy
import numpy as np

sys.path.append('/home/zzb/NKDM')
from Code.Contrastive.Contrastive_Model import ContrastiveSemantic
from Code.Diffusion.Diffusion_Func import exists, default


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        # Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        Rearrange('b c (l p1) -> b (c p1) l', p1=2),
        nn.Conv1d(dim * 2, default(dim_out, dim), 1)
    )


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class WeightStandardizedConv2d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 ', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 ',
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)

        # print(x.shape)
        # print(mean.shape)
        # print(var.shape)
        # print(self.g.shape)

        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, *args):
        x = self.norm(x)
        return self.fn(x, *args)


# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        # emb = emb.to(device)
        # emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = emb.to(device)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(
            half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            # time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            # for EEG task -> time_emb reshape for (b, c, _)
            time_emb = rearrange(time_emb, 'b c -> b c 1 ')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


# Latent Diffusion Model cross attention for EEG
class CrossAttentionEEG(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=125, dropout=0):
        super().__init__()
        query_dim = int(query_dim)
        inner_dim = dim_head * heads
        # print(context_dim)
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        # print(context_dim)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = int(default(dim_out, dim))
        project_in = nn.Sequential(
            nn.Linear(int(dim), inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(int(dim), inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=8, d_head=125, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        dim = int(dim)
        # context_dim = int(dim)
        self.attn1 = CrossAttentionEEG(query_dim=dim, heads=n_heads, dim_head=d_head,
                                       dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttentionEEG(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head,
                                       dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class BasicTransformerBlock_multi(nn.Module):
    def __init__(self, dim, n_heads=8, d_head=125, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        dim = int(dim)
        context_dim = int(dim)
        self.attn1 = CrossAttentionEEG(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttentionEEG(query_dim=dim, context_dim=context_dim, heads=n_heads,
                                       dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        # x= self.attn1(x) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


# for EGG 1d attention block
class LinearAttention1d(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        # b, c, h, w = x.shape
        b, c, l = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) l -> b h c l', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / l

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c l -> b (h c) l', h=self.heads, l=l)
        return self.to_out(out)


class Attention1d(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        # b, c, h, w = x.shape
        b, c, l = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) l -> b h c l', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h l c -> b (h c) l', h=self.heads, l=l)
        return self.to_out(out)


class MultiScale_Encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.t500 = nn.Sequential(
            nn.Conv1d(64, 64, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.t250 = nn.Sequential(
            nn.Conv1d(64, 128, 1, 1),
            nn.LeakyReLU(),
            nn.Conv1d(128, 128, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(128, 128, 3, 2, padding=1)
        )

        self.t125 = nn.Sequential(
            nn.Conv1d(64, 256, 1, 1),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x1000 = x.clone()
        x500 = self.t500(x).clone()
        x250 = self.t250(x).clone()
        x125 = self.t125(x).clone()
        return x1000, x500, x250, x125
        # return x500, x250, x125





class Semantic_CA(nn.Module):
    def __init__(self, channel_list=None) -> None:
        super().__init__()
        # Unet backbone channels -> 32, 32, 32, 32, 64, 64, 128, 128
        # Input semantic embedding channl -> 1
        if channel_list is None:
            channel_list = [32, 32, 32, 32, 64, 64, 128, 128]

        self.channel_list = channel_list

        self.c32 = nn.ModuleList([])
        self.c64 = nn.ModuleList([])
        self.c128 = nn.ModuleList([])

        for i in range(4):
            self.c32.append(
                nn.Sequential(
                    nn.Linear(1000, 500),
                    # nn.LeakyReLU(),
                    nn.ReLU(),
                    nn.Linear(500, 32),
                    nn.Sigmoid()
                )
            )

        for i in range(2):
            self.c64.append(
                nn.Sequential(
                    nn.Linear(1000, 500),
                    # nn.LeakyReLU(),
                    nn.ReLU(),
                    nn.Linear(500, 64),
                    nn.Sigmoid()
                )
            )

        for i in range(2):
            self.c128.append(
                nn.Sequential(
                    nn.Linear(1000, 500),
                    # nn.LeakyReLU(),
                    nn.ReLU(),
                    nn.Linear(500, 128),
                    nn.Sigmoid()
                )
            )

    def forward(self, x):
        ca_list = []
        for l in self.c32:
            ca_list.append(l(x))
        for l in self.c64:
            ca_list.append(l(x))
        for l in self.c128:
            ca_list.append(l(x))

        return ca_list



"""
================================================================NKDM Model================================================================
"""

# latest!!!
class Unet_NKDM_SEM(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4),
            channels=3,
            self_condition=False,
            resnet_block_groups=8,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            guided_sr3=False,
            latent_diffusion=False,
    ):
        super().__init__()

        # determine dimensions
        self.guided_sr3 = guided_sr3
        self.latent_diffusion = latent_diffusion

        self.channels = 4 * channels if self.guided_sr3 else channels

        self.self_condition = self_condition
        input_channels = self.channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                BasicTransformerBlock(dim=1000 / (2 ** ind), context_dim=1000) if self.latent_diffusion else Residual(
                    PreNorm(dim_in, LinearAttention1d(dim_in))), # TODO: change self-atten 
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention1d(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                BasicTransformerBlock(dim=125 * (2 ** ind), context_dim=1000) if self.latent_diffusion else Residual(
                    PreNorm(dim_in, LinearAttention1d(dim_in))),# TODO: change self-atten 
                Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

        self.semantic_ca = Semantic_CA()
        
        

    def forward(self, x, time, guided=None, semantic=None, x_self_cond=None, test=False):
        # extend channel dim for convolution
        x = torch.unsqueeze(x, 1)
        guided_embedding = guided.unsqueeze(1)
        semantic_embedding = self.semantic_ca(semantic)

        if self.guided_sr3:
            assert x.shape[-1] == guided.shape[-1], 'x & guided dim dismatch'
            x = torch.cat([x, guided], dim=1)

        x = self.init_conv(x)
        
        r = x.clone()
        t = self.time_mlp(time)

        h = []
        for i, (block1, block2, attn, downsample) in enumerate(self.downs):
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x, guided_embedding) if self.latent_diffusion else attn(x)
            h.append(x)
            x = downsample(x)


        for i in range(len(h)):
            h[i] = h[i] * semantic_embedding[i].unsqueeze(-1)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for i, (block1, block2, attn, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x, guided_embedding) if self.latent_diffusion else attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        x = torch.squeeze(x, 1)
        return x



if __name__ == '__main__':
    model = Unet_NKDM_SEM(
        dim=32,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        guided_sr3=False,
        latent_diffusion=True,
    )

    i = torch.ones(1, 1000)
    g = torch.ones(1, 1000)
    s = torch.ones(1, 1000)
    t = torch.randint(0, 10, (1,)).long()

    o = model(i, t, g, s)




