""" Sin-cos, fourier, rotary position embedding modules and functions

Hacked together by / Copyright 2022 Ross Wightman
"""
import math
from typing import List, Tuple, Optional, Union

import torch
from torch import nn as nn

def ndgrid(*tensors) -> Tuple[torch.Tensor, ...]:
    """generate N-D grid in dimension order.

    The ndgrid function is like meshgrid except that the order of the first two input arguments are switched.

    That is, the statement
    [X1,X2,X3] = ndgrid(x1,x2,x3)

    produces the same result as

    [X2,X1,X3] = meshgrid(x2,x1,x3)

    This naming is based on MATLAB, the purpose is to avoid confusion due to torch's change to make
    torch.meshgrid behaviour move from matching ndgrid ('ij') indexing to numpy meshgrid defaults of ('xy').

    """
    try:
        return torch.meshgrid(*tensors, indexing='ij')
    except TypeError:
        # old PyTorch < 1.10 will follow this path as it does not have indexing arg,
        # the old behaviour of meshgrid was 'ij'
        return torch.meshgrid(*tensors)
    

def pixel_freq_bands(
        num_bands: int,
        max_freq: float = 224.,
        linear_bands: bool = True,
        device: Optional[torch.device] = None,
):
    if linear_bands:
        bands = torch.linspace(1.0, max_freq / 2, num_bands, dtype=torch.float32, device=device)
    else:
        bands = 2 ** torch.linspace(0, math.log(max_freq, 2) - 1, num_bands, dtype=torch.float32, device=device)
    return bands * torch.pi


def freq_bands(
        num_bands: int,
        temperature: float = 10000.,
        step: int = 2,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    exp = torch.arange(0, num_bands, step, dtype=torch.int64, device=device).to(torch.float32) / num_bands
    bands = 1. / (temperature ** exp)
    return bands


def build_fourier_pos_embed(
        feat_shape: List[int],
        bands: Optional[torch.Tensor] = None,
        num_bands: int = 64,
        max_res: int = 224,
        temperature: float = 10000.,
        linear_bands: bool = False,
        include_grid: bool = False,
        in_pixels: bool = True,
        ref_feat_shape: Optional[List[int]] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        clip_length: Optional[int] = None,
        original_clip_length: Optional[int] = None,
) -> List[torch.Tensor]:
    """

    Args:
        feat_shape: Feature shape for embedding.
        bands: Pre-calculated frequency bands.
        num_bands: Number of frequency bands (determines output dim).
        max_res: Maximum resolution for pixel based freq.
        temperature: Temperature for non-pixel freq.
        linear_bands: Linear band spacing for pixel based freq.
        include_grid: Include the spatial grid in output.
        in_pixels: Output in pixel freq.
        ref_feat_shape: Reference feature shape for resize / fine-tune.
        dtype: Output dtype.
        device: Output device.

    Returns:

    """
    if bands is None:
        if in_pixels:
            bands = pixel_freq_bands(
                num_bands,
                float(max_res),
                linear_bands=linear_bands,
                device=device,
            )
        else:
            bands = freq_bands(
                num_bands,
                temperature=temperature,
                step=1 if clip_length is None else 2,
                device=device,
            )
    else:
        if device is None:
            device = bands.device
        if dtype is None:
            dtype = bands.dtype

    if in_pixels:
        t = [torch.linspace(-1., 1., steps=s, device=device, dtype=torch.float32) for s in feat_shape]
    elif clip_length:
        t = [torch.arange(clip_length, device=device, dtype=torch.int64).to(torch.float32)]
    else:
        t = [torch.arange(s, device=device, dtype=torch.int64).to(torch.float32) for s in feat_shape]

    if ref_feat_shape is not None:
        if clip_length is None:
            # eva's scheme for resizing rope embeddings (ref shape = pretrain)
            t = [x / f * r for x, f, r in zip(t, feat_shape, ref_feat_shape)]
        else:
            t = [x / f * r for x, f, r in zip(t, [clip_length], [original_clip_length])]
    grid = torch.stack(ndgrid(t), dim=-1)
    grid = grid.unsqueeze(-1)
    pos = grid * bands

    pos_sin, pos_cos = pos.sin().to(dtype=dtype), pos.cos().to(dtype)
    #angle = pos.to(dtype = dtype)
    out = [grid, pos_sin, pos_cos] if include_grid else [pos_sin, pos_cos]
    #out = [grid, angle, angle] if include_grid else [angle, angle]
    return out


class FourierEmbed(nn.Module):

    def __init__(
            self,
            max_res: int = 224,
            num_bands: int = 64,
            concat_grid=True,
            keep_spatial=False,
    ):
        super().__init__()
        self.max_res = max_res
        self.num_bands = num_bands
        self.concat_grid = concat_grid
        self.keep_spatial = keep_spatial
        self.register_buffer(
            'bands',
            pixel_freq_bands(max_res, num_bands),
            persistent=False,
        ) 

    def forward(self, x):
        B, C = x.shape[:2]
        feat_shape = x.shape[2:]
        emb = build_fourier_pos_embed(
            feat_shape,
            self.bands,
            include_grid=self.concat_grid,
            dtype=x.dtype,
            device=x.device,
        )
        emb = torch.cat(emb, dim=-1)
        emb = emb.transpose(-1, -2).flatten(len(feat_shape))
        batch_expand = (B,) + (-1,) * (x.ndim - 1)

        # FIXME support nD
        if self.keep_spatial:
            x = torch.cat([x, emb.unsqueeze(0).expand(batch_expand).permute(0, 3, 1, 2)], dim=1)
        else:
            x = torch.cat([x.permute(0, 2, 3, 1), emb.unsqueeze(0).expand(batch_expand)], dim=-1)
            x = x.reshape(B, feat_shape.numel(), -1)

        return x


def rot(x):
    return torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)


def apply_rot_embed(x: torch.Tensor, sin_emb, cos_emb):
    if sin_emb.ndim == 3:
        return x * cos_emb.unsqueeze(1).expand_as(x) + rot(x) * sin_emb.unsqueeze(1).expand_as(x)
    return x * cos_emb + rot(x) * sin_emb


def apply_rot_embed_list(x: List[torch.Tensor], sin_emb, cos_emb):
    if isinstance(x, torch.Tensor):
        x = [x]
    return [t * cos_emb + rot(t) * sin_emb for t in x]


def apply_rot_embed_cat(x: torch.Tensor, emb):
    # if emb.ndim == 3:
    #     pos_emb, tem_emb = emb.tensor_split(2, 0)
    #     pos_sin, pos_cos = pos_emb.tensor_split(2, -1)
    #     tem_sin, tem_cos = tem_emb.tensor_split(2, -1)
    #     return x * (pos_cos * tem_cos - pos_sin * tem_sin) + rot(x) * (pos_cos * tem_sin + pos_sin * tem_cos) 
    sin_emb, cos_emb = emb.tensor_split(2, -1)
    if sin_emb.ndim == 3:
        return x * cos_emb.unsqueeze(1).expand_as(x) + rot(x) * sin_emb.unsqueeze(1).expand_as(x)
    return x * cos_emb + rot(x) * sin_emb


def apply_keep_indices_nlc(x, pos_embed, keep_indices):
    pos_embed = pos_embed.unsqueeze(0).expand(x.shape[0], -1, -1)
    pos_embed = pos_embed.gather(1, keep_indices.unsqueeze(-1).expand(-1, -1, pos_embed.shape[-1]))
    return pos_embed


def build_rotary_pos_embed(
        feat_shape: List[int],
        bands: Optional[torch.Tensor] = None,
        dim: int = 64,
        max_res: int = 224,
        temperature: float = 10000.,
        linear_bands: bool = False,
        in_pixels: bool = True,
        ref_feat_shape: Optional[List[int]] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        clip_length: Optional[List[int]] = None,
        original_clip_length: Optional[List[int]] = None,
):
    """

    Args:
        feat_shape: Spatial shape of the target tensor for embedding.
        bands: Optional pre-generated frequency bands
        dim: Output dimension of embedding tensor.
        max_res: Maximum resolution for pixel mode.
        temperature: Temperature (inv freq) for non-pixel mode
        linear_bands: Linearly (instead of log) spaced bands for pixel mode
        in_pixels: Pixel vs language (inv freq) mode.
        dtype: Output dtype.
        device: Output device.

    Returns:

    """
    if clip_length is None:
        sin_emb, cos_emb = build_fourier_pos_embed(
            feat_shape,
            bands=bands,
            num_bands=dim // 4,
            max_res=max_res,
            temperature=temperature,
            linear_bands=linear_bands,
            in_pixels=in_pixels,
            ref_feat_shape=ref_feat_shape,
            device=device,
            dtype=dtype,
        )
    else:
        sin_emb, cos_emb = build_fourier_pos_embed(
            feat_shape,
            bands=bands,
            num_bands=dim,
            max_res=max_res,
            temperature=temperature,
            linear_bands=linear_bands,
            in_pixels=in_pixels,
            ref_feat_shape=ref_feat_shape,
            device=device,
            dtype=dtype,
            clip_length=clip_length,
            original_clip_length=original_clip_length
        )
    num_spatial_dim = 1
    # this would be much nicer as a .numel() call to torch.Size(), but torchscript sucks
    if clip_length is None:
        for x in feat_shape:
            num_spatial_dim *= x
    else:
        num_spatial_dim *= clip_length
    sin_emb = sin_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)
    cos_emb = cos_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)
    return sin_emb, cos_emb


class RotaryEmbedding(nn.Module):
    """ Rotary position embedding

    NOTE: This is my initial attempt at impl rotary embedding for spatial use, it has not
    been well tested, and will likely change. It will be moved to its own file.

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(
            self,
            dim,
            max_res=224,
            temperature=10000,
            in_pixels=True,
            linear_bands: bool = False,
            feat_shape: Optional[List[int]] = None,
            ref_feat_shape: Optional[List[int]] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_res = max_res
        self.temperature = temperature
        self.in_pixels = in_pixels
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape

        if feat_shape is None:
            # only cache bands
            if in_pixels:
                bands = pixel_freq_bands(
                    dim // 4,
                    float(max_res),
                    linear_bands=linear_bands,
                )
            else:
                bands = freq_bands(
                    dim // 4,
                    temperature=temperature,
                    step=1,
                )
                print(bands)
            self.register_buffer(
                'bands',
                bands,
                persistent=False,
            )
            self.pos_embed_sin = None
            self.pos_embed_cos = None
        else:
            # cache full sin/cos embeddings if shape provided up front
            emb_sin, emb_cos = build_rotary_pos_embed(
                feat_shape=feat_shape,
                dim=dim,
                max_res=max_res,
                linear_bands=linear_bands,
                in_pixels=in_pixels,
                ref_feat_shape=self.ref_feat_shape,
            )
            self.bands = None
            self.register_buffer(
                'pos_embed_sin',
                emb_sin,
                persistent=False,
            )
            self.register_buffer(
                'pos_embed_cos',
                emb_cos,
                persistent=False,
            )

    def get_embed(self, shape: Optional[List[int]] = None):
        if self.bands is not None:
            # rebuild embeddings every call, use if target shape changes
            assert shape is not None
            return build_rotary_pos_embed(
                shape,
                self.bands,
                in_pixels=self.in_pixels,
            )
        else:
            return self.pos_embed_sin, self.pos_embed_cos

    def forward(self, x):
        # assuming channel-first tensor where spatial dim are >= 2
        sin_emb, cos_emb = self.get_embed(x.shape[2:])
        return apply_rot_embed(x, sin_emb, cos_emb)


class RotaryEmbeddingCat(nn.Module):
    """ Rotary position embedding w/ concatenatd sin & cos

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(
            self,
            dim,
            max_res=224,
            temperature=10000,
            in_pixels=True,
            linear_bands: bool = False,
            feat_shape: Optional[List[int]] = None,
            ref_feat_shape: Optional[List[int]] = None,
            clip_length: Optional[int] = None,
            original_clip_length: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_res = max_res
        self.temperature = temperature
        self.in_pixels = in_pixels
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape
        self.clip_length = clip_length

        if self.clip_length is None:
            self.temporal_embed = None

        if feat_shape is None:
            # only cache bands
            if in_pixels:
                bands = pixel_freq_bands(
                    dim // 4,
                    float(max_res),
                    linear_bands=linear_bands,
                )
            else:
                bands = freq_bands(
                    dim // 4,
                    temperature=temperature,
                    step=1,
                )
            self.register_buffer(
                'bands',
                bands,
                persistent=False,
            )
            self.pos_embed = None
        else:
            # cache full sin/cos embeddings if shape provided up front
            embeds = build_rotary_pos_embed(
                feat_shape=feat_shape,
                dim=dim,
                max_res=max_res,
                linear_bands=linear_bands,
                in_pixels=in_pixels,
                ref_feat_shape=self.ref_feat_shape,
            )
            self.bands = None
            self.register_buffer(
                'pos_embed',
                torch.cat(embeds, -1),
                persistent=False,
            )
        if self.clip_length is not None:
            #apply 1D ROPE on temporal dimension
            temporal_embeds = build_rotary_pos_embed(
                feat_shape=feat_shape,
                dim=dim,
                max_res=max_res,
                linear_bands=linear_bands,
                in_pixels=in_pixels,
                ref_feat_shape=self.ref_feat_shape,
                clip_length=self.clip_length,
                original_clip_length = original_clip_length,
            )
            self.temporal_bands = None
            self.register_buffer(
                'temporal_embed',
                torch.cat(temporal_embeds, -1),
                persistent=False,
            )

    def get_embed(self, shape: Optional[List[int]] = None):
        if self.bands is not None and shape is not None:
            # rebuild embeddings every call, use if target shape changes
            embeds = build_rotary_pos_embed(
                shape,
                self.bands,
                in_pixels=self.in_pixels,
                ref_feat_shape=self.ref_feat_shape,
                clip_length = self.clip_length,
            )
            return torch.cat(embeds, -1)
        elif self.clip_length is None:
            return self.pos_embed
        elif self.pos_embed is not None:
            return self.pos_embed, self.temporal_embed
        else:
            assert False, "get_embed() requires pre-computed pos_embed or valid shape w/ pre-computed bands"

    def forward(self, x):
        # assuming channel-first tensor where spatial dim are >= 2
        pos_embed = self.get_embed(x.shape[2:])
        return apply_rot_embed_cat(x, pos_embed)
    
if __name__ == '__main__':
    clip_length = 16
    rope = RotaryEmbeddingCat(
        768 // 12, #embedding // head_nums
        in_pixels=False,
        feat_shape=(14,14),
        ref_feat_shape=(16,16), #patch size
        clip_length = 16, # num of frames
        original_clip_length= 16, # num of frames during pretraining
    )
    feat_shape = (14,14)
    num_patches = 196
    rot_pos_embed, rot_temporal_embed = rope.get_embed()
    rot_pos_embed = rot_pos_embed.repeat(clip_length, 1)
    rot_temporal_embed = rot_temporal_embed.repeat_interleave(num_patches, 0)
    rope = torch.stack((rot_pos_embed, rot_temporal_embed), dim = 0)


    print(rope.shape)