""" EVA

EVA from https://github.com/baaivision/EVA , paper: https://arxiv.org/abs/2211.07636

@article{EVA,
  title={EVA: Exploring the Limits of Masked Visual Representation Learning at Scale},
  author={Fang, Yuxin and Wang, Wen and Xie, Binhui and Sun, Quan and Wu, Ledell and Wang, Xinggang and Huang,
  Tiejun and Wang, Xinlong and Cao, Yue},
  journal={arXiv preprint arXiv:2211.07636},
  year={2022}
}

EVA-02: A Visual Representation for Neon Genesis - https://arxiv.org/abs/2303.11331
@article{EVA02,
  title={EVA-02: A Visual Representation for Neon Genesis},
  author={Fang, Yuxin and Sun, Quan and Wang, Xinggang and Huang, Tiejun and Wang, Xinlong and Cao, Yue},
  journal={arXiv preprint arXiv:2303.11331},
  year={2023}
}

This file contains EVA & EVA02 model implementations evolved from BEiT, additional models in vision_transformer.py.

Modifications by / Copyright 2023 Ross Wightman, original copyrights below
"""
# EVA models Copyright (c) 2022 BAAI-Vision
# EVA02 models Copyright (c) 2023 BAAI-Vision

import math
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint

from timm.layers import PatchEmbed, Mlp, GluMlp, SwiGLU, LayerNorm, DropPath, PatchDropout, \
    trunc_normal_, resample_patch_embed, resample_abs_pos_embed, \
    to_2tuple, use_fused_attn

from resources.models.rope import RotaryEmbeddingCat, apply_rot_embed_cat, apply_keep_indices_nlc
#from rope import RotaryEmbeddingCat, apply_rot_embed_cat, apply_keep_indices_nlc
from flash_attn import flash_attn_func


import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

#from avion.models.utils import remap_keys_from_open_clip_to_vit, remap_state_dict_keys
from einops import rearrange

class EvaAttention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qkv_fused: bool = True, #Note this is always false
            fused_attn: bool = True,
            flash_attn: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            attn_head_dim: Optional[int] = None,
            norm_layer: Optional[Callable] = None,
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            attn_drop:
            proj_drop:
            attn_head_dim:
            norm_layer:
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = fused_attn
        #self.flash_attn = flash_attn
        self.attn_drop_rate = attn_drop

        if qkv_fused:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
            self.q_proj = self.k_proj = self.v_proj = None
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
                self.register_buffer('k_bias', torch.zeros(all_head_dim), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
            else:
                self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(all_head_dim) if norm_layer is not None else nn.Identity()
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if flash_attn == True:
            #self.attn = FlashMHA(self.dim, self.num_heads, cross_attn=False, dropout=self.attn_drop_rate, use_flash_attn=True)
            self.attn = flash_attn
    def forward(
            self,
            x,
            rope: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        B, N, C = x.shape

        if self.qkv is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            #print(q.shape)

        if rope is not None:
            q = torch.cat([q[:, :, :1, :], apply_rot_embed_cat(q[:, :, 1:, :], rope)], 2).type_as(v)
            k = torch.cat([k[:, :, :1, :], apply_rot_embed_cat(k[:, :, 1:, :], rope)], 2).type_as(v)

        if self.fused_attn:
            if self.attn is not None:
                #flash_attn_layer = FlashMHA(self.dim, self.num_heads, cross_attn=False, bias=True, dropout=self.attn_drop_rate, use_flash_attn=True)
                #print(q.shape) #[8,12,197,64]
                #print(f'q_shape is{q.shape}')
                #x = self.attn(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), need_weights=False, attn_mask=attn_mask, dropout_p=self.attn_drop.p if self.training else 0.)
                x = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=self.attn_drop.p if self.training else 0., softmax_scale=None, causal=False)
            else:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            if attn_mask is not None:
                attn_mask = attn_mask.to(torch.bool)
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            attn = self.attn_drop(attn)
            x = attn @ v

        if self.attn is not None:
            x = x.reshape(B, N, C)
        else:
            x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EvaBlock(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            attn_head_dim: Optional[int] = None,
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            proj_drop:
            attn_drop:
            drop_path:
            init_values:
            act_layer:
            norm_layer:
            attn_head_dim:
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EvaAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None,
        )
        #self. attn = FlashMHA(dim, num_heads=num_heads,dropout=attn_drop,use_flash_attn=True) #use flash attn
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp  = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed weights is used, matches existing GluMLP
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x



class Eva(nn.Module):
    """ Eva Vision Transformer w/ Abs & Rotary Pos Embed

    This class implements the EVA and EVA02 models that were based on the BEiT ViT variant
      * EVA - abs pos embed, global avg pool
      * EVA02 - abs + rope pos embed, global avg pool, SwiGLU, scale Norm in MLP (ala normformer)
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: Callable = LayerNorm,
            init_values: Optional[float] = None,
            class_token: bool = True,
            use_abs_pos_emb: bool = True,
            use_rot_pos_emb: bool = False,
            use_post_norm: bool = False,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            ref_feat_shape: Optional[Union[Tuple[int, int], int]] = None,
            head_init_scale: float = 0.001,
            use_fast_conv1: bool = False,
            use_flash_attn: bool = False,
            clip_length: int = 16,
            original_clip_length: Optional[int] = 4,
    ):
        """

        Args:
            img_size:
            patch_size:
            in_chans:
            num_classes:
            global_pool:
            embed_dim:
            depth:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            drop_rate:
            pos_drop_rate:
            proj_drop_rate:
            attn_drop_rate:
            drop_path_rate:
            norm_layer:
            init_values:
            class_token:
            use_abs_pos_emb:
            use_rot_pos_emb:
            use_post_norm:
            ref_feat_shape:
            head_init_scale:
        """
        super().__init__()
        self.use_fast_conv1 = use_fast_conv1
        self.use_flash_attn = use_flash_attn
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False
        self.clip_length = clip_length
        self.original_clip_length = original_clip_length

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        # self.patch_embed = PatchEmbed(
        #     img_size=img_size,
        #     patch_size=patch_size,
        #     in_chans=in_chans,
        #     embed_dim=embed_dim,
        #     dynamic_img_pad=dynamic_img_pad,
        #     **embed_args,
        # )
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        num_patches = self.grid_size[0] * self.grid_size[1]
        self.num_patches = num_patches
        #ref_feat_shape = (clip_length, ref_feat_shape[0], ref_feat_shape[1])
        # num_patches = self.patch_embed.num_patches
        # print(f'This is test, {self.patch_embed.grid_size}')
        # assert grid_size == self.patch_embed.grid_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None

        self.pos_embed = nn.Parameter(
            torch.zeros(num_patches + self.num_prefix_tokens, embed_dim)) if use_abs_pos_emb else None
        #self.pos_embed = self.pos_embed.unbind(0)
        #print(f"Test:{num_patches}, {self.num_prefix_tokens}, {embed_dim}")
        #self.pos_embed = self.pos_embed.squeeze()
        #print(self.pos_embed.shape)
        
        self.temporal_embed = nn.Parameter(
            torch.zeros(clip_length, embed_dim)) if use_abs_pos_emb else None
        
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
                return_indices=True,
            )
        else:
            self.patch_drop = None

        if use_rot_pos_emb:
            ref_feat_shape = to_2tuple(ref_feat_shape) if ref_feat_shape is not None else None
            self.rope = RotaryEmbeddingCat(
                embed_dim // num_heads,
                in_pixels=False,
                feat_shape=None if dynamic_img_size else self.grid_size,
                ref_feat_shape=ref_feat_shape,
                clip_length=self.clip_length,
                original_clip_length=self.original_clip_length,
            )
        else:
            self.rope = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        #block_fn = EvaBlockPostNorm if use_post_norm else EvaBlock
        block_fn = EvaBlock

        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                mlp_ratio=mlp_ratio,
                swiglu_mlp=swiglu_mlp,
                scale_mlp=scale_mlp,
                scale_attn_inner=scale_attn_inner,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
            )
            for i in range(depth)])

        use_fc_norm = self.global_pool == 'avg'
        self.norm = nn.Identity() if use_fc_norm else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        #self.image_projection = nn.Parameter(torch.zeros(embed_dim, output_dim))
        #self.conv1 = nn.Conv2d(3, embed_dim, bias=True, kernel_size=self.patch_size, stride=self.patch_size)
        if use_fast_conv1:
            self.conv1 = nn.Linear(in_features=3 * patch_size ** 2, out_features=embed_dim, bias=True)
        else:
            self.conv1 = nn.Conv2d(3, embed_dim, bias=True, kernel_size=self.patch_size, stride=self.patch_size)

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.temporal_embed is not None:
            trunc_normal_(self.temporal_embed, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)

        self.fix_init_weight()
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            #rescale(layer.attn.proj.weight.data, layer_id + 1) flashattn do not need to rescale
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        # if self.image_projection is not None:
        #     nn.init.normal_(self.image_projection, std=self.embed_dim ** -0.5)
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'pos_embed', 'temporal_embed', 'cls_token'}
        return nwd

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )
        return matcher

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    #def _temporal_embed(self, x) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    def video_patch_embed(self, x: torch.Tensor) ->  Tuple[torch.Tensor]:
        #B, C, T, H, W = x.shape
        if self.use_fast_conv1:
            if self.clip_length == 1:
                x = rearrange(x, "b c (hh sh) (ww sw) -> b (hh ww) (c sh sw)", sh=self.patch_size, sw=self.patch_size)
            else:
                x = rearrange(x, "b c t (hh sh) (ww sw) -> b (t hh ww) (c sh sw)", sh=self.patch_size, sw=self.patch_size)
            x = self.conv1(x)
        return x
        
    def _compute_rope(self, tem_emb, pos_emb):
        pos_sin, pos_cos = pos_emb.tensor_split(2, -1)
        tem_sin, tem_cos = tem_emb.tensor_split(2, -1)
        cos_emb = pos_cos * tem_cos - pos_sin * tem_sin
        sin_emb = pos_cos * tem_sin + pos_sin * tem_cos
        rot_pos_embed = torch.cat((sin_emb, cos_emb), dim = -1)
        return rot_pos_embed

    def _pos_embed(self, x) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            if self.pos_embed is not None:
                pos_embed = resample_abs_pos_embed(
                    self.pos_embed,
                    (H, W),
                    num_prefix_tokens=self.num_prefix_tokens,
                )
            else:
                pos_embed = None
            x = x.view(B, -1, C)
            rot_pos_embed = self.rope.get_embed(shape=(H, W)) if self.rope is not None else None
        else:
            pos_embed = self.pos_embed
            rot_pos_embed, rot_temporal_embed = self.rope.get_embed() if self.rope is not None else None
            rot_pos_embed = rot_pos_embed.repeat(self.clip_length, 1)
            rot_temporal_embed = rot_temporal_embed.repeat_interleave(self.num_patches, 0)
            rot_pos_embed = self._compute_rope(rot_temporal_embed, rot_pos_embed)
            #rot_pos_embed = torch.stack((rot_pos_embed, rot_temporal_embed), dim = 0)
            #print(rot_pos_embed.shape)

        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if pos_embed is not None:
            cls_embed = self.pos_embed[0, :].unsqueeze(0)
            pos_embed = self.pos_embed[1:, :].repeat(self.clip_length, 1)
            temporal_embed = self.temporal_embed.repeat_interleave(self.num_patches, 0).to(dtype=x.dtype, device=x.device)
            pos_embed = pos_embed + temporal_embed
            pos_embed = torch.cat([cls_embed, pos_embed], dim=0)
            x = x + pos_embed
        x = self.pos_drop(x)

        # obtain shared rotary position embedding and apply patch dropout
        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)
        return x, rot_pos_embed

    def forward_features(self, x):
        x = self.video_patch_embed(x)
        x, rope = self._pos_embed(x)
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope)
            else:
                x = blk(x, rope=rope)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
    

def eva02_base_patch16_clip_224(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4 * 2 / 3, num_frames = 16, original_num_frames = 4, use_fast_conv1=True, use_flash_attn=True, output_dim=256, drop_rate=0., drop_path_rate=0., pretrained=False, **kwargs) -> Eva:
    """ A EVA-CLIP specific variant that adds additional attn scale layernorm to eva02_base """
    model = Eva(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_classes=output_dim,
        num_heads=num_heads,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        qkv_fused=False,
        mlp_ratio=mlp_ratio,
        swiglu_mlp=True,
        scale_mlp=True,
        scale_attn_inner=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(14,14),  # 224/16
        global_pool=kwargs.pop('global_pool', 'token'),
        use_fast_conv1=True,
        use_flash_attn=use_flash_attn,
        clip_length=num_frames,
        original_clip_length=original_num_frames
    )
    #model = Eva(, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def eva02_large_patch14_clip_224(img_size=224, patch_size=14, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4 * 2 / 3, num_frames = 16, original_num_frames = 4, use_fast_conv1=True, use_flash_attn=True, output_dim=256, drop_rate=0., drop_path_rate=0., pretrained=False, **kwargs) -> Eva:
    """ A EVA-CLIP specific variant that adds additional attn scale layernorm to eva02_base """
    model = Eva(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_classes=output_dim,
        num_heads=num_heads,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        qkv_fused=False,
        mlp_ratio=mlp_ratio,
        swiglu_mlp=True,
        scale_mlp=True,
        scale_attn_inner=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16,16),  # 224/14
        global_pool=kwargs.pop('global_pool', 'token'),
        use_fast_conv1=True,
        use_flash_attn=use_flash_attn,
        clip_length=num_frames,
        original_clip_length=original_num_frames
    )
    #model = Eva(, pretrained=pretrained, **dict(model_args, **kwargs))
    return model

if __name__ == '__main__':
    import torch.cuda.amp as amp
    scaler = amp.GradScaler(enabled=True)


    model = eva02_base_patch16_clip_224(num_frames=4)
    # checkpoint_path = '/home/wangxiaoqi/AVION/pretrain/eva02_base_patch16_clip_224.merged2b_s8b_b131k/open_clip_pytorch_model.bin'
    # checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # remapped_state_dict = remap_state_dict_keys(checkpoint)
    # # remapped_state_dict = remap_keys_from_open_clip_to_vit(
    # #         remapped_state_dict,
    # #         use_fast_conv1=True,
    # #         use_flash_attn=True,
    # #     )
    # missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
    # print("missing_keys: ", missing_keys)
    # print("unexpected_keys: ", unexpected_keys)
    # print("All keys", checkpoint.keys())

    input = Variable(torch.FloatTensor(2, 3, 4, 224, 224)).cuda()
    model = model.cuda()
    with amp.autocast(enabled=True):
        output = model(input)
    print('net output size:')
    print(output.shape)