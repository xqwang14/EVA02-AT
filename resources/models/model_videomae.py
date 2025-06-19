from functools import partial
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
import torch.utils.checkpoint as checkpoint
from typing import Callable, Optional, Tuple, Union
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))
from flash_attn.modules.mha import MHA as FlashMHA
from flash_attn.modules.mlp import Mlp as FlashMlp


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }
    
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            use_flash_attn: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.use_flash_attn = use_flash_attn
        if not use_flash_attn:
            self.attn = nn.MultiheadAttention(d_model, n_head, dropout=attn_drop)
        else:
            self.attn = FlashMHA(d_model, n_head, cross_attn=False, dropout=attn_drop, use_flash_attn=True)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        if not use_flash_attn:
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("drop1", nn.Dropout(drop)),
                ("c_proj", nn.Linear(mlp_width, d_model)),
                ("drop2", nn.Dropout(drop)),
            ]))
        else:
            self.mlp = FlashMlp(d_model, hidden_features=mlp_width, activation=act_layer())
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        attn_mask = attn_mask.to(x.dtype) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def cross_attention(self, x: torch.Tensor, context: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, context, context, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if not self.use_flash_attn:
            x = x + self.drop_path(self.ls_1(self.attention(self.ln_1(x), attn_mask=attn_mask)))
        else:
            x = x + self.drop_path(self.ls_1(self.attn(self.ln_1(x))))
        x = x + self.drop_path(self.ls_2(self.mlp(self.ln_2(x))))
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            use_flash_attn: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                act_layer=act_layer, norm_layer=norm_layer,
                use_flash_attn=use_flash_attn)
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


class VisionTransformer(nn.Module):
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            num_frames: int = 1,
            ls_init_value: float = None,
            global_average_pool: bool = False,
            output_dim: int = None, #default is 256
            patch_dropout: float = 0.,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            tubelet_size: int = 1,
            ln_pre: bool = True,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
            mim_pretrain: bool = False,
            use_checkpoint: bool = True,
            use_fast_conv1: bool = False,
            use_flash_attn: bool = False,
    ):
        super().__init__()
        self.use_fast_conv1 = use_fast_conv1
        self.use_flash_attn = use_flash_attn
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.mim_pretrain = mim_pretrain
        self.width = width
        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.patches_per_frame = self.grid_size[0] * self.grid_size[1]
        self.output_dim = output_dim
        if use_fast_conv1:
            self.conv1 = nn.Linear(in_features=3 * patch_size ** 2, out_features=width, bias=not ln_pre)
        else:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=not ln_pre)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        assert num_frames >= 1
        self.num_frames = num_frames
        if num_frames > 1:
            self.temporal_embedding = nn.Parameter(torch.zeros(num_frames, width))

        assert not (patch_dropout > 0. and drop_rate > 0.)
        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0. else nn.Identity()

        if ln_pre:
            self.ln_pre = norm_layer(width)
        else:
            self.ln_pre = nn.Identity()

        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate,
            act_layer=act_layer,
            norm_layer=norm_layer,
            use_flash_attn=use_flash_attn,
        )

        self.global_average_pool = global_average_pool
        self.ln_post = norm_layer(width)
        if output_dim is None:
            self.image_projection = None
        else:
            self.image_projection = nn.Parameter(scale * torch.randn(width, output_dim))

        self.init_parameters()

    def init_parameters(self):
        # TODO: compare the two styles
        # Mimicking timm's initialization
        nn.init.normal_(self.class_embedding, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.02)

        for block in self.transformer.resblocks:
            for n, p in block.named_parameters():
                if 'weight' in n:
                    trunc_normal_(p, std=0.02)
                elif 'bias' in n:
                    nn.init.zeros_(p)
                else:
                    raise NotImplementedError('Unknown parameters named {}'.format(n)) 
        if self.image_projection is not None:
            nn.init.normal_(self.image_projection, std=self.width ** -0.5)

        # Same init as TextTransformer
        # nn.init.normal_(self.token_embedding.weight, std=0.02)
        # nn.init.normal_(self.positional_embedding, std=0.01)

        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # if self.image_projection is not None:
        #     nn.init.normal_(self.image_projection, std=self.output_dim ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def get_num_layers(self):
        return len(self.transformer.resblocks)

    def forward_mask(self, x, mask):
        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible
        return x_vis

    def forward(self, x: torch.Tensor, mask = None):
        if self.use_fast_conv1:
            if self.num_frames == 1:
                x = rearrange(x, "b c (hh sh) (ww sw) -> b (hh ww) (c sh sw)", sh=self.patch_size[0], sw=self.patch_size[1])
                x = self.conv1(x)
                x = torch.cat(
                    [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                x = x + self.positional_embedding.to(x.dtype)
            else:
                #print(f'x shapx.shape)
                x = rearrange(x, "b c t (hh sh) (ww sw) -> b (t hh ww) (c sh sw)", sh=self.patch_size[0], sw=self.patch_size[1])
                x = self.conv1(x)
                x = torch.cat(
                    [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                cls_embed = self.positional_embedding[0, :].unsqueeze(0)
                tile_pos_embed = self.positional_embedding[1:, :].repeat(self.num_frames, 1)
                tile_temporal_embed = self.temporal_embedding.repeat_interleave(self.patches_per_frame, 0)
                total_pos_embed = tile_pos_embed + tile_temporal_embed
                total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=0)
                x = x + total_pos_embed.to(x.dtype).unsqueeze(0)
        else:
            if self.num_frames == 1:
                x = self.conv1(x)  # shape = [*, width, grid, grid]
                x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
                x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
                x = torch.cat(
                    [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                x = x + self.positional_embedding.to(x.dtype)
            else:
                x = x.permute(0, 2, 1, 3, 4).contiguous()  # B, C, T, H, W =>  B, T, C, H, W
                B, F, C, H, W = x.shape
                x = x.view(-1, C, H, W)
                x = self.conv1(x)
                x = x.flatten(2).transpose(2, 1)    # BT, C', H, W => BT, HW, C'
                x = x.reshape(B, -1, self.width)
                x = torch.cat(
                    [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                cls_embed = self.positional_embedding[0, :].unsqueeze(0)
                tile_pos_embed = self.positional_embedding[1:, :].repeat(self.num_frames, 1)
                tile_temporal_embed = self.temporal_embedding.repeat_interleave(self.patches_per_frame, 0)
                total_pos_embed = tile_pos_embed + tile_temporal_embed
                total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=0)
                x = x + total_pos_embed.to(x.dtype).unsqueeze(0)
                if self.mim_pretrain == True:
                    pos_embed = total_pos_embed.to(x.dtype).unsqueeze(0)

        # apply mask on x
        if mask is not None:
            x = x[:, 1:, :]
            pos_embed = pos_embed[:, 1:, :]
            x = self.forward_mask(x, mask)
        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = self.pos_drop(x)

        if not self.use_flash_attn:
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
        else:
            x = self.transformer(x)

        if not self.mim_pretrain:
            if self.global_average_pool:
                x = x.mean(dim=1)
            else:
                x = x[:, 0]

        x = self.ln_post(x)

        if self.image_projection is not None:
            x = x @ self.image_projection

        if self.mim_pretrain == True:
            return x, pos_embed
        else:
            return x

# class PretrainVisionTransformerEncoder(nn.Module):
#     """ Vision Transformer with support for patch or hybrid CNN input stage
#     """
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
#                  num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
#                  drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2, use_checkpoint=False,
#                  use_flash_attn=False,
#                  channel_last=False,
#                  use_learnable_pos_emb=False):
#         super().__init__()


#     def forward_features(self, x, mask):
#         _, _, T, _, _ = x.shape
#         x = self.patch_embed(x)
        
#         x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

#         B, _, C = x.shape
#         x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible

#         if self.use_checkpoint:
#             for blk in self.blocks:
#                 x_vis = checkpoint.checkpoint(blk, x_vis)
#         else:   
#             for blk in self.blocks:
#                 x_vis = blk(x_vis)

#         x_vis = self.norm(x_vis)
#         return x_vis

#     def forward(self, x, mask):
#         x = self.forward_features(x, mask)
#         x = self.head(x)
#         return x

class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=768, width=768, layers=12, heads=12, mlp_ratio=4.,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, tubelet_size=1, use_checkpoint=False, use_flash_attn=False,
                 ):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 3 * tubelet_size * patch_size ** 2 
        self.num_features = self.embed_dim = width  # num_features for consistency with other models
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint
        self.use_flash_attn = use_flash_attn


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate,
            act_layer=nn.GELU,
            norm_layer=norm_layer,
            use_flash_attn=use_flash_attn,
        )
        self.norm =  norm_layer(width)
        self.head = nn.Linear(width, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        if not self.use_flash_attn:
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
        else:
            x = self.transformer(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x

class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 num_frames=16,
                 encoder_in_chans=3, 
                 encoder_num_classes=None, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=768, #  decoder_num_classes=768, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 use_checkpoint=False,
                 use_flash_attn_at_encoder=True,
                 use_flash_attn_at_decoder=True,
                 tubelet_size=1,
                 use_fast_conv1=False,
                 ):
        super().__init__()
        self.encoder = VisionTransformer(
            image_size=img_size, 
            patch_size=patch_size, 
            num_frames=num_frames,
            output_dim=encoder_num_classes, 
            width=encoder_embed_dim, 
            layers=encoder_depth,
            heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            mim_pretrain = True,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_flash_attn=use_flash_attn_at_encoder,
            use_fast_conv1 = use_fast_conv1,
        )

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, 
            num_classes=decoder_num_classes, 
            width=decoder_embed_dim, 
            layers=decoder_depth,
            heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_flash_attn=use_flash_attn_at_decoder)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        #self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask):
        _, _, T, _, _ = x.shape
        x_vis, pos_embed = self.encoder(x, mask) # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]
        x = self.decoder(x_full, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]

        return x


def VIDEOMAE_VITB16(pretrained=False, num_frames=16, use_grad_checkpointing=False, decoder_depth=8, use_flash_attn_at_encoder=False, use_flash_attn_at_decoder=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        num_frames=num_frames,
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=None,
        decoder_num_classes=768,
        decoder_embed_dim=768,
        decoder_num_heads=6,
        mlp_ratio=4, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        use_checkpoint = use_grad_checkpointing,
        decoder_depth = decoder_depth,
        use_flash_attn_at_encoder=use_flash_attn_at_encoder,
        use_flash_attn_at_decoder=use_flash_attn_at_decoder,
        )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


def vit_base_patch16_224(pretrained=False, num_classes=400, clip_length=16, fc_drop_rate=0., drop_rate=0., drop_path_rate=0., attn_drop_rate=0., use_flash_attn=False, use_checkpoint=False, **kwargs):
    model = VisionTransformer(
        image_size=224, width=768, patch_size=16, num_frames=clip_length, output_dim=num_classes, drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate, layers=12, heads=12, mlp_ratio=4,
        use_flash_attn=use_flash_attn, norm_layer=partial(nn.LayerNorm, eps=1e-6), mim_pretrain=False,
    )
    return model

if __name__ == '__main__':
    import torch.cuda.amp as amp
    from torch.autograd import Variable
    from avion.data.transforms import GroupMultiScaleCrop, Permute, TubeMaskingGeneratorGPU
    scaler = amp.GradScaler(enabled=True)

    model=VIDEOMAE_VITB16(pretrained=False, num_frames=16)
    clip_length = 16
    input_size = 224
    patch_size = (16,16)
    window_size = (clip_length, input_size // patch_size[0], input_size // patch_size[1])
    patch_size = patch_size
    input = Variable(torch.FloatTensor(2, 3, 16, 224, 224)).cuda()
    bool_masked_pos = TubeMaskingGeneratorGPU(input.shape[0], window_size, 0.9, device='cuda')().flatten(1).to(torch.bool)
    print(bool_masked_pos, bool_masked_pos.shape)

    model = model.cuda()
    with amp.autocast(enabled=True):
        output = model(input,bool_masked_pos)
    print('net output size:')
    print(output.shape)

