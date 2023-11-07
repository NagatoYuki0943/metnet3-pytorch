from contextlib import contextmanager
from functools import partial
from collections import namedtuple

import torch
from torch import nn, Tensor, einsum
import torch.distributed as dist
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Sequential

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

from beartype import beartype
from beartype.typing import Tuple, Union, List, Optional

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pack_one(x, pattern):
    return pack([x], pattern)

def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

def safe_div(num, den, eps = 1e-10):
    return num / den.clamp(min = eps)

# prepare batch norm in maxvit for distributed training

def MaybeSyncBatchnorm2d(is_distributed = None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm2d

@contextmanager
def freeze_batchnorm(bn):
    assert not exists(next(bn.parameters(), None))

    was_training = bn.training
    was_tracking_stats = bn.track_running_stats # in some versions of pytorch, running mean and variance still gets updated even in eval mode it seems..

    bn.eval()
    bn.track_running_stats = False

    yield bn

    bn.train(was_training)
    bn.track_running_stats = was_tracking_stats

# loss scaling in section 4.3.2

class LossScaleFunction(Function):
    @staticmethod
    def forward(ctx, x, eps):
        ctx.eps = eps
        assert x.ndim == 4
        return x

    @staticmethod
    def backward(ctx, grads):
        num_channels = grads.shape[1]

        safe_div_ = partial(safe_div, eps = ctx.eps)

        weight = safe_div_(1., grads.norm(p = 2, keepdim = True, dim = (-1, -2)))
        l1_normed_weight = safe_div_(weight, weight.sum(keepdim = True, dim = 1))

        scaled_grads = num_channels * l1_normed_weight * grads

        return scaled_grads, None

class LossScaler(Module):
    def __init__(self, eps = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return LossScaleFunction.apply(x, self.eps)

# center crop

class CenterPad(Module):
    def __init__(self, target_dim):
        super().__init__()
        self.target_dim = target_dim

    def forward(self, x):
        target_dim = self.target_dim
        *_, height, width = x.shape
        assert target_dim >= height and target_dim >= width

        height_pad = target_dim - height
        width_pad = target_dim - width
        left_height_pad = height_pad // 2
        left_width_pad = width_pad // 2

        return F.pad(x, (left_height_pad, height_pad - left_height_pad, left_width_pad, width_pad - left_width_pad), value = 0.)

class CenterCrop(Module):
    def __init__(self, crop_dim):
        super().__init__()
        self.crop_dim = crop_dim

    def forward(self, x):
        crop_dim = self.crop_dim
        *_, height, width = x.shape
        assert (height >= crop_dim) and (width >= crop_dim)

        cropped_height_start_idx = (height - crop_dim) // 2
        cropped_width_start_idx = (width - crop_dim) // 2

        height_slice = slice(cropped_height_start_idx, cropped_height_start_idx + crop_dim)
        width_slice = slice(cropped_width_start_idx, cropped_width_start_idx + crop_dim)
        return x[..., height_slice, width_slice]

# down and upsample

# they use maxpool for downsample, and convtranspose2d for upsample
# todo: figure out the 4x upsample from 4km to 1km

Downsample2x = partial(nn.MaxPool2d, kernel_size = 2, stride = 2)

def Upsample2x(dim, dim_out = None):
    dim_out = default(dim_out, dim)
    return nn.ConvTranspose2d(dim, dim_out, kernel_size = 2, stride = 2)

# conditionable resnet block

class Block(Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = ChanLayerNorm(dim_out)
        self.act = nn.ReLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        cond_dim = None
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.mlp = None

        if exists(cond_dim):
            self.mlp = Sequential(
                nn.ReLU(),
                nn.Linear(cond_dim, dim_out * 2)
            )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, cond = None):

        scale_shift = None

        assert not (exists(self.mlp) ^ exists(cond))

        if exists(self.mlp) and exists(cond):
            cond = self.mlp(cond)
            cond = rearrange(cond, 'b c -> b c 1 1')
            scale_shift = cond.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class ResnetBlocks(Module):
    def __init__(
        self,
        dim,
        *,
        dim_in = None,
        depth = 1,
        cond_dim = None
    ):
        super().__init__()
        curr_dim = default(dim_in, dim)

        blocks = []
        for _ in range(depth):
            blocks.append(ResnetBlock(dim = curr_dim, dim_out = dim, cond_dim = cond_dim))
            curr_dim = dim

        self.blocks = ModuleList(blocks)

    def forward(self, x, cond = None):

        for block in self.blocks:
            x = block(x, cond = cond)

        return x

# multi-headed rms normalization, for query / key normalized attention

class RMSNorm(Module):
    def __init__(
        self,
        dim,
        *,
        heads
    ):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# they use layernorms after the conv in the resnet blocks for some reason

class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * var.clamp(min = self.eps).rsqrt() * self.g + self.b

# MBConv

class SqueezeExcitation(Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)

class MBConvResidual(Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    batchnorm_klass = MaybeSyncBatchnorm2d()

    net = Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        batchnorm_klass(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
        batchnorm_klass(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        batchnorm_klass(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

# attention related classes

class Attention(Module):
    def __init__(
        self,
        dim,
        cond_dim = None,
        heads = 32,
        dim_head = 32,
        dropout = 0.,
        window_size = 8,
        num_registers = 1
    ):
        super().__init__()
        assert num_registers > 0
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        dim_inner = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.has_cond = exists(cond_dim)

        self.film = None

        if self.has_cond:
            self.film = Sequential(
                nn.Linear(cond_dim, dim * 2),
                nn.SiLU(),
                nn.Linear(dim * 2, dim * 2),
                Rearrange('b (r d) -> r b 1 d', r = 2)
            )

        self.norm = nn.LayerNorm(dim, elementwise_affine = not self.has_cond)

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)

        self.q_norm = RMSNorm(dim_head, heads = heads)
        self.k_norm = RMSNorm(dim_head, heads = heads)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

        # relative positional bias

        num_rel_pos_bias = (2 * window_size - 1) ** 2

        self.rel_pos_bias = nn.Embedding(num_rel_pos_bias + 1, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        rel_pos_indices = F.pad(rel_pos_indices, (num_registers, 0, num_registers, 0), value = num_rel_pos_bias)
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(
        self,
        x: Tensor,
        cond: Optional[Tensor] = None
    ):
        device, h, bias_indices = x.device, self.heads, self.rel_pos_indices

        x = self.norm(x)

        # conditioning

        if exists(self.film):
            assert exists(cond)

            gamma, beta = self.film(cond)
            x = x * gamma + beta

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale

        q, k = self.q_norm(q), self.k_norm(k)

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias

        bias = self.rel_pos_bias(bias_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # combine heads out

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class MaxViT(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        cond_dim = 32,   # for conditioniong on lead time embedding
        heads = 32,
        dim_head = 32,
        window_size = 8,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        num_register_tokens = 4
    ):
        super().__init__()
        depth = (depth,) if isinstance(depth, int) else depth
        assert num_register_tokens > 0

        self.cond_dim = cond_dim

        # variables

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])

        # window size

        self.window_size = window_size

        self.register_tokens = nn.ParameterList([])

        # iterate through stages

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                conv = MBConv(
                    stage_dim_in,
                    layer_dim,
                    downsample = is_first,
                    expansion_rate = mbconv_expansion_rate,
                    shrinkage_rate = mbconv_shrinkage_rate
                )

                block_attn = Attention(dim = layer_dim, cond_dim = cond_dim, heads = heads, dim_head = dim_head, dropout = dropout, window_size = window_size, num_registers = num_register_tokens)

                grid_attn = Attention(dim = layer_dim, cond_dim = cond_dim, heads = heads, dim_head = dim_head, dropout = dropout, window_size = window_size, num_registers = num_register_tokens)

                register_tokens = nn.Parameter(torch.randn(num_register_tokens, layer_dim))

                self.layers.append(ModuleList([
                    conv,
                    block_attn,
                    grid_attn
                ]))

                self.register_tokens.append(register_tokens)

    def forward(
        self,
        x: Tensor,
        cond: Tensor
    ):
        assert cond.shape == (x.shape[0], self.cond_dim)

        b, w = x.shape[0], self.window_size

        for (conv, block_attn, grid_attn), register_tokens in zip(self.layers, self.register_tokens):
            x = conv(x)

            # block-like attention

            x = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w)

            # prepare register tokens

            r = repeat(register_tokens, 'n d -> b x y n d', b = b, x = x.shape[1],y = x.shape[2])
            r, register_batch_ps = pack_one(r, '* n d')

            x, window_ps = pack_one(x, 'b x y * d')
            x, batch_ps  = pack_one(x, '* n d')
            x, register_ps = pack([r, x], 'b * d')

            x = block_attn(x, cond = cond) + x

            r, x = unpack(x, register_ps, 'b * d')

            x = unpack_one(x, batch_ps, '* n d')
            x = unpack_one(x, window_ps, 'b x y * d')
            x = rearrange(x, 'b x y w1 w2 d -> b d (x w1) (y w2)')

            r = unpack_one(r, register_batch_ps, '* n d')

            # grid-like attention

            x = rearrange(x, 'b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w)

            # prepare register tokens

            r = reduce(r, 'b x y n d -> b n d', 'mean')
            r = repeat(r, 'b n d -> b x y n d', x = x.shape[1], y = x.shape[2])
            r, register_batch_ps = pack_one(r, '* n d')

            x, window_ps = pack_one(x, 'b x y * d')
            x, batch_ps  = pack_one(x, '* n d')
            x, register_ps = pack([r, x], 'b * d')

            x = grid_attn(x, cond = cond) + x

            r, x = unpack(x, register_ps, 'b * d')

            x = unpack_one(x, batch_ps, '* n d')
            x = unpack_one(x, window_ps, 'b x y * d')
            x = rearrange(x, 'b x y w1 w2 d -> b d (w1 x) (w2 y)')

        return x

# main MetNet3 module

Predictions = namedtuple('Predictions', [
    'surface',
    'hrrr',
    'precipitation'
])

LossBreakdown = namedtuple('LossBreakdown', [
    'surface',
    'hrrr',
    'precipitation'
])

class MetNet3(Module):
    def __init__(
        self,
        *,
        dim = 512,
        num_lead_times = 722,
        lead_time_embed_dim = 32,
        input_spatial_size = 624,
        attn_depth = 12,
        attn_dim_head = 64,
        attn_heads = 32,
        attn_dropout = 0.1,
        vit_window_size = 8,
        vit_mbconv_expansion_rate = 4,
        vit_mbconv_shrinkage_rate = 0.25,
        sparse_input_2496_channels = 8,
        dense_input_2496_channels = 8,
        dense_input_4996_channels = 8,
        surface_and_hrrr_target_spatial_size = 128,
        surface_target_channels = 6,
        hrrr_target_channels = 617,
        precipitation_target_channels = 2,
        crop_size_post_16km = 48,
        resnet_block_depth = 2,
    ):
        super().__init__()
        self.sparse_input_2496_shape = (sparse_input_2496_channels, input_spatial_size, input_spatial_size)
        self.dense_input_2496_shape = (dense_input_2496_channels, input_spatial_size, input_spatial_size)
        self.dense_input_4996_shape = (dense_input_4996_channels, input_spatial_size, input_spatial_size)

        self.surface_and_hrrr_target_spatial_size = surface_and_hrrr_target_spatial_size

        self.surface_target_shape = ((self.surface_and_hrrr_target_spatial_size,) * 2)
        self.hrrr_target_shape = (hrrr_target_channels, *self.surface_target_shape)
        self.precipitation_target_shape = (surface_and_hrrr_target_spatial_size * 4,) * 2

        self.lead_time_embedding = nn.Embedding(num_lead_times, lead_time_embed_dim)

        dim_in_4km = sparse_input_2496_channels + dense_input_2496_channels

        self.to_skip_connect_4km = CenterCrop(crop_size_post_16km * 4)

        self.resnet_blocks_down_4km = ResnetBlocks(
            dim = dim,
            dim_in = dim_in_4km,
            cond_dim = lead_time_embed_dim,
            depth = resnet_block_depth
        )

        self.downsample_and_pad_to_8km = Sequential(
            Downsample2x(),
            CenterPad(input_spatial_size)
        )

        dim_in_8km = dense_input_4996_channels + dim

        self.resnet_blocks_down_8km = ResnetBlocks(
            dim = dim,
            dim_in = dim_in_8km,
            cond_dim = lead_time_embed_dim,
            depth = resnet_block_depth
        )

        self.downsample_to_16km = Downsample2x()

        self.vit = MaxViT(
            dim = dim,
            depth = attn_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            dropout = attn_dropout,
            cond_dim = lead_time_embed_dim,
            window_size = vit_window_size,
            mbconv_expansion_rate = vit_mbconv_expansion_rate,
            mbconv_shrinkage_rate = vit_mbconv_shrinkage_rate,
        )

        self.crop_post_16km = CenterCrop(crop_size_post_16km)

        self.upsample_16km_to_8km = Upsample2x(dim)

        self.to_skip_connect_8km = CenterCrop(crop_size_post_16km * 2)

        self.resnet_blocks_up_8km = ResnetBlocks(
            dim = dim,
            dim_in = dim + dim_in_8km,
            cond_dim = lead_time_embed_dim,
            depth = resnet_block_depth
        )

        self.upsample_8km_to_4km = Upsample2x(dim)

        self.crop_post_4km = CenterCrop(surface_and_hrrr_target_spatial_size)

        self.resnet_blocks_up_4km = ResnetBlocks(
            dim = dim,
            dim_in = dim + dim_in_4km,
            cond_dim = lead_time_embed_dim,
            depth = resnet_block_depth
        )

        self.upsample_4x_to_1km = nn.ConvTranspose2d(dim, dim, kernel_size = 4, stride = 4)

        self.resnet_blocks_up_1km = ResnetBlocks(
            dim = dim,
            depth = resnet_block_depth,
            cond_dim = lead_time_embed_dim
        )

        self.to_surface_pred = Sequential(
            ChanLayerNorm(dim),
            nn.Conv2d(dim, surface_target_channels, 1)
        )

        self.to_hrrr_pred = Sequential(
            ChanLayerNorm(dim),
            nn.Conv2d(dim, hrrr_target_channels, 1)
        )

        self.to_precipitation_pred = Sequential(
            ChanLayerNorm(dim),
            nn.Conv2d(dim, precipitation_target_channels, 1)
        )

        self.batchnorm_hrrr = MaybeSyncBatchnorm2d()(hrrr_target_channels, affine = False)

        self.mse_loss_scaler = LossScaler()

    def forward(
        self,
        lead_times,
        sparse_input_2496,
        dense_input_2496,
        dense_input_4996,
        surface_target = None,
        hrrr_target = None,
        precipitation_target = None
    ):
        batch = lead_times.shape[0]

        assert batch == sparse_input_2496.shape[0] == dense_input_2496.shape[0] == dense_input_4996.shape[0], 'batch size across all inputs must be the same'
        
        assert sparse_input_2496.shape[1:] == self.sparse_input_2496_shape
        assert dense_input_2496.shape[1:] == self.dense_input_2496_shape
        assert dense_input_4996.shape[1:] == self.dense_input_4996_shape

        cond = self.lead_time_embedding(lead_times)

        x = torch.cat((sparse_input_2496, dense_input_2496), dim = 1)

        skip_connect_4km = self.to_skip_connect_4km(x)

        x = self.resnet_blocks_down_4km(x, cond = cond)

        x = self.downsample_and_pad_to_8km(x)

        x = torch.cat((dense_input_4996, x), dim = 1)

        skip_connect_8km = self.to_skip_connect_8km(x)

        x = self.resnet_blocks_down_8km(x, cond = cond)

        x = self.downsample_to_16km(x)

        x = self.vit(x, cond = cond)

        x = self.crop_post_16km(x)

        x = self.upsample_16km_to_8km(x)

        x = torch.cat((skip_connect_8km, x), dim = 1)

        x = self.resnet_blocks_up_8km(x, cond = cond)

        x = self.upsample_8km_to_4km(x)

        x = torch.cat((skip_connect_4km, x), dim = 1)

        x = self.resnet_blocks_up_4km(x, cond = cond)

        x = self.crop_post_4km(x)

        surface_pred = self.to_surface_pred(x)
        hrrr_pred = self.to_hrrr_pred(x)

        x = self.upsample_4x_to_1km(x)

        x = self.resnet_blocks_up_1km(x, cond = cond)

        precipitation_pred = self.to_precipitation_pred(x)

        exist_targets = [exists(target) for target in (surface_target, hrrr_target, precipitation_target)]

        pred =  Predictions(surface_pred, hrrr_pred, precipitation_pred)

        if not any(exist_targets):
            return pred

        assert all(exist_targets), 'all targets must be passed in for loss calculation'

        assert batch == surface_target.shape[0] == hrrr_target.shape[0] == precipitation_target.shape[0]

        assert surface_target.shape[1:] == self.surface_target_shape
        assert hrrr_target.shape[1:] == self.hrrr_target_shape
        assert precipitation_target.shape[1:] == self.precipitation_target_shape

        # calculate categorical losses

        surface_pred = rearrange(surface_pred, '... h w -> ... (h w)')
        precipitation_pred = rearrange(precipitation_pred, '... h w -> ... (h w)')

        surface_target = rearrange(surface_target, '... h w -> ... (h w)')
        precipitation_target = rearrange(precipitation_target, '... h w -> ... (h w)')

        surface_loss = F.cross_entropy(surface_pred, surface_target)
        precipition_loss = F.cross_entropy(precipitation_pred, precipitation_target)

        # calculate HRRR mse loss

        # use a batchnorm to normalize each channel to mean zero and unit variance

        normed_hrrr_target = self.batchnorm_hrrr(hrrr_target)

        with freeze_batchnorm(self.batchnorm_hrrr) as frozen_batchnorm:
            normed_hrrr_pred = frozen_batchnorm(hrrr_pred)

        # proposed loss gradient rescaler from section 4.3.2

        normed_hrrr_pred = self.mse_loss_scaler(normed_hrrr_pred)

        hrrr_loss = F.mse_loss(normed_hrrr_pred, normed_hrrr_target)

        # total loss

        total_loss = hrrr_loss + precipition_loss + surface_loss

        loss_breakdown = LossBreakdown(surface_loss, hrrr_loss, precipition_loss)

        return total_loss, loss_breakdown
