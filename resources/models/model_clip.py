import clip
from collections import OrderedDict
# #Modify the root to the parent folder to aviod errors when importing avion
# import sys
# from pathlib import Path
# current_dir = Path(__file__).resolve().parent
# parent_dir = current_dir.parent.parent
# sys.path.append(str(parent_dir))
# #sys.path.append('/home/wangxiaoqi/avion/third_party/decord/python')
# #-------------------end of modification----------------------------------

import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

import open_clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
#from avion.models.eva_woRope import eva02_base_patch16_clip_224, eva02_large_patch14_clip_224, eva02_enormous_patch14_clip_224
from resources.models.eva import eva02_base_patch16_clip_224, eva02_large_patch14_clip_224
from resources.models.transformer import TextTransformer, VisionTransformer
from resources.models.utils import enable_grad_checkpointing, remap_keys_from_open_clip_to_vit, remap_state_dict_keys
from resources.models.utils import inflate_positional_embeds

class VideoClassifier(nn.Module):
    def __init__(self,
                 vision_model: nn.Module,
                 dropout: float,
                 num_classes: int,
                 **kwargs,
                 ):
        super().__init__()
        self.visual = vision_model
        self.dropout = nn.Dropout(dropout)
        if hasattr(self.visual, "image_projection"):
            self.visual.image_projection = None
        self.fc_cls = nn.Linear(vision_model.width, num_classes, bias=True)
        self.fc_cls.weight.data.normal_(mean=0.0, std=0.01)
        self.fc_cls.bias.data.zero_()

    def forward(self, image):
        image_embed = self.visual(image)
        if isinstance(image_embed, list):
            assert len(image_embed) == 1
            image_embed = image_embed[0]
        logit = self.fc_cls(self.dropout(image_embed))
        return logit
    

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 vision_model: nn.Module,
                 text_model: nn.Module,
                 vision_width: int = None,
                 text_width: int = None,
                 freeze_temperature=False,
                 **kwargs
    ):
        super().__init__()

        self.visual = vision_model
        self.textual = text_model

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if freeze_temperature:
            self.logit_scale.requires_grad_(False)
        if vision_width is not None:
            self.vision_width = vision_width
            self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        else:
            self.image_projection = None
        if text_width is not None:
            self.text_width = text_width
            self.text_projection = nn.Parameter(torch.empty(text_width, embed_dim))
        else:
            self.text_projection = None

        self.init_parameters()

    def init_parameters(self):
        if self.image_projection is not None:
            trunc_normal_(self.image_projection, std=self.vision_width ** -0.5)
        if self.text_projection is not None:
            trunc_normal_(self.text_projection, std=self.text_width ** -0.5)

    def encode_image(self, image):
        x = self.visual(image)
        if self.image_projection is not None:
            x = x @ self.image_projection.to(x.dtype)
        return x

    def encode_text(self, text, cast_dtype=None):
        x = self.textual(text, cast_dtype=cast_dtype)
        if self.text_projection is not None:
            x = x @ self.text_projection.to(x.dtype)
        return x

    def forward(self, image, text):
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text, cast_dtype=image_embed.dtype)

        return F.normalize(image_embed, dim=-1), F.normalize(text_embed, dim=-1), self.logit_scale.exp()


def CLIP_VITB16(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    pretrain_textual_path=None,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = VisionTransformer(
        224, 16, 768, 12, 12, 4,
        #use_rot_pos_emb = True, #2024/5/22 Add RoPE
        output_dim=project_embed_dim, patch_dropout=patch_dropout,
        drop_path_rate=drop_path_rate,
        num_frames=num_frames,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=49408, width=512, heads=8, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)

    if pretrain_zoo == "openai":
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-B/16', device='cpu')
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            clip_model.state_dict(),
            use_fast_conv1=use_fast_conv1,
            use_flash_attn=use_flash_attn,
            #rope = True,
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    elif pretrain_zoo == "open_clip":
        assert pretrain_path is not None
        state_dict = torch.load(pretrain_path)
        print("=> loading open_clip model")
        remapped_state_dict = remap_keys_from_open_clip_to_vit(state_dict, use_fast_conv1=use_fast_conv1, use_flash_attn=use_flash_attn)
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    elif pretrain_zoo == 'mim':
        state_dict = torch.load(pretrain_path, map_location='cpu')
        textual_state_dict = torch.load(pretrain_textual_path, map_location='cpu')
        print("=> Load checkpoint from mim model %s" % pretrain_path)
        for model_key in ['model','module','state_dict']:
            if model_key in state_dict:
                checkpoint_model = state_dict[model_key]
                if list(checkpoint_model.keys())[0].startswith('module.'):
                    renamed_ckpt = {k[7:]: v for k, v in checkpoint_model.items()}
                    checkpoint_model = renamed_ckpt
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = state_dict
        for key in ['head.weight', 'head.bias']:
            if key in checkpoint_model and checkpoint_model[key].shape != model.state_dict()[key].shape:
                print("Removing key %s from pretrained checkpoint" % key)
                checkpoint_model.pop(key)

        new_dict = OrderedDict()
        for key in checkpoint_model.keys():
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_key = key.replace('encoder.', 'visual.')
                if use_flash_attn and 'attn.qkv' in key:
                    new_dict[new_key.replace('attn.qkv', 'attn.Wqkv')] = checkpoint_model[key]
                elif use_flash_attn and 'attn.q_bias' in key:
                    q_bias = checkpoint_model[key]
                    v_bias = checkpoint_model[key.replace('attn.q_bias', 'attn.v_bias')]
                    new_dict[new_key.replace('attn.q_bias', 'attn.Wqkv.bias')] = torch.cat(
                        (q_bias, torch.zeros_like(v_bias), v_bias))
                elif use_flash_attn and 'attn.v_bias' in key:
                    continue
                elif use_flash_attn and 'attn.proj' in key:
                    new_dict[new_key.replace('attn.proj', 'attn.out_proj')] = checkpoint_model[key]
                else:
                    new_dict[new_key] = checkpoint_model[key]
            else:
                if use_flash_attn and 'attn.qkv' in key:
                    new_dict[key.replace('attn.qkv', 'attn.Wqkv')] = checkpoint_model[key]
                elif use_flash_attn and 'attn.q_bias' in key:
                    q_bias = checkpoint_model[key]
                    v_bias = checkpoint_model[key.replace('attn.q_bias', 'attn.v_bias')]
                    new_dict[key.replace('attn.q_bias', 'attn.Wqkv.bias')] = torch.cat(
                        (q_bias, torch.zeros_like(v_bias), v_bias))
                elif use_flash_attn and 'attn.v_bias' in key:
                    continue
                elif use_flash_attn and 'attn.proj' in key:
                    new_dict[key.replace('attn.proj', 'attn.out_proj')] = checkpoint_model[key]
                elif not key.startswith('encoder.'):
                    new_dict[key] = checkpoint_model[key]
        
        #for keys in pretrain_textual_path:

        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            new_dict,
            use_fast_conv1=use_fast_conv1,
            use_flash_attn=use_flash_attn,
            #rope = True,
        )
        print('=> inflating PE in models due to different frame numbers')
        remapped_state_dict = inflate_positional_embeds(
        model.state_dict(), remapped_state_dict,
        num_frames=num_frames,
        load_temporal_fix='bilinear',
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=True)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
        return model
    else:
        raise NotImplementedError
    return model


def CLIP_VITL14(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    vocab_size=49408,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = VisionTransformer(
        224, 14, 1024, 24, 16, 4,
        output_dim=project_embed_dim, patch_dropout=patch_dropout,
        drop_path_rate=drop_path_rate,
        num_frames=num_frames,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=vocab_size, width=768, heads=12, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)

    if pretrain_zoo == "openai":
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-L/14', device='cpu')
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            clip_model.state_dict(), 24,
            context_length=context_length,
            vocab_size=vocab_size,
            use_fast_conv1=use_fast_conv1,
            use_flash_attn=use_flash_attn,
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    elif pretrain_zoo == "open_clip":
        assert pretrain_path is not None
        state_dict = torch.load(pretrain_path)
        print("=> loading open_clip model")
        remapped_state_dict = remap_keys_from_open_clip_to_vit(state_dict, use_flash_attn=use_flash_attn)
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    else:
        raise NotImplementedError
    return model


def CLIP_VITL14_336PX(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    vocab_size=49408,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = VisionTransformer(
        336, 14, 1024, 24, 16, 4,
        output_dim=project_embed_dim, patch_dropout=patch_dropout,
        drop_path_rate=drop_path_rate,
        num_frames=num_frames,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=vocab_size, width=768, heads=12, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)

    if pretrain_zoo == "openai":
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-L/14@336px', device='cpu')
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            clip_model.state_dict(), 24,
            context_length=context_length,
            vocab_size=vocab_size,
            use_fast_conv1=use_fast_conv1,
            use_flash_attn=use_flash_attn,
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    elif pretrain_zoo == "open_clip":
        assert pretrain_path is not None
        state_dict = torch.load(pretrain_path)
        print("=> loading open_clip model")
        remapped_state_dict = remap_keys_from_open_clip_to_vit(state_dict, use_flash_attn=use_flash_attn)
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    else:
        raise NotImplementedError
    return model


def EVA_B_16(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    drop_rate=0.,
    drop_path_rate=0.,
    vocab_size=49408,
    num_frames=1,
    original_num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path='./eva02_base_patch16_clip_224.merged2b_s8b_b131k/open_clip_pytorch_model.bin',
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = eva02_base_patch16_clip_224(
        224, 16, 768, 12, 12, 4*2/3,
        output_dim=project_embed_dim,
        num_frames=num_frames,
        original_num_frames = original_num_frames,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
        drop_rate=drop_rate, 
        drop_path_rate= drop_path_rate,
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=vocab_size, width=512, heads=8, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)

    #checkpoint_path = './eva02_base_patch16_clip_224.merged2b_s8b_b131k/open_clip_pytorch_model.bin'
    if pretrain_zoo == 'none':
        return model
    if pretrain_zoo == 'full':
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
        return model
    else:
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        remapped_state_dict = remap_state_dict_keys(use_fast_conv1=use_fast_conv1, state_dict=checkpoint)
        # remapped_state_dict = remap_keys_from_open_clip_to_vit(
        #         remapped_state_dict,
        #         context_length=context_length,
        #         vocab_size=vocab_size,
        #         use_fast_conv1=use_fast_conv1,
        #         use_flash_attn=use_flash_attn,
        #     )
        
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    #print("All keys: ", remapped_state_dict.keys())

    return model


def EVA_L_14(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    drop_rate=0.,
    drop_path_rate=0.,
    vocab_size=49408,
    num_frames=1,
    original_num_frames = 1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=768,
    pretrain_zoo='openai',
    pretrain_path='./eva02_large_patch14_clip_224.merged2b_s4b_b131k/open_clip_pytorch_model.bin',
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = eva02_large_patch14_clip_224(
        224, 14, 1024, 24, 16, 4*2/3,
        output_dim=project_embed_dim,
        num_frames=num_frames,
        original_num_frames = original_num_frames,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
        drop_rate=drop_rate, 
        drop_path_rate= drop_path_rate,
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=vocab_size, width=768, heads=12, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)

    #checkpoint_path = './eva02_base_patch16_clip_224.merged2b_s8b_b131k/open_clip_pytorch_model.bin'
    if pretrain_zoo == 'none':
        return model
    if pretrain_zoo == 'full':
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
        return model
    else:
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        remapped_state_dict = remap_state_dict_keys(use_fast_conv1=use_fast_conv1, state_dict=checkpoint)
        # remapped_state_dict = remap_keys_from_open_clip_to_vit(
        #         remapped_state_dict,
        #         context_length=context_length,
        #         vocab_size=vocab_size,
        #         use_fast_conv1=use_fast_conv1,
        #         use_flash_attn=use_flash_attn,
        #     )
        
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    #print("All keys: ", remapped_state_dict.keys())

    return model

def EVA_E_14(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    drop_rate=0.,
    drop_path_rate=0.,
    vocab_size=49408,
    num_frames=1,
    original_num_frames = 1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=768,
    pretrain_zoo='openai',
    pretrain_path='./eva02_large_patch14_clip_224.merged2b_s4b_b131k/open_clip_pytorch_model.bin',
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = eva02_enormous_patch14_clip_224(
        224, 14, 1792, 64, 16, 15360 / 1792,
        output_dim=project_embed_dim,
        num_frames=num_frames,
        original_num_frames = original_num_frames,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
        drop_rate=drop_rate, 
        drop_path_rate= drop_path_rate,
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=vocab_size, width=1280, heads=20, layers=32, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)

    #checkpoint_path = './eva02_base_patch16_clip_224.merged2b_s8b_b131k/open_clip_pytorch_model.bin'
    if pretrain_zoo == 'none':
        return model
    if pretrain_zoo == 'full':
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
        return model
    else:
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        remapped_state_dict = remap_state_dict_keys(use_fast_conv1=use_fast_conv1, state_dict=checkpoint)
        # remapped_state_dict = remap_keys_from_open_clip_to_vit(
        #         remapped_state_dict,
        #         context_length=context_length,
        #         vocab_size=vocab_size,
        #         use_fast_conv1=use_fast_conv1,
        #         use_flash_attn=use_flash_attn,
        #     )
        
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    #print("All keys: ", remapped_state_dict.keys())

    return model

if __name__ == '__main__':

    model = EVA_E_14(
            freeze_temperature=False,
            use_grad_checkpointing=False,
            use_bidirectional_lm=False,
            context_length=77,
            patch_dropout=0.,
            drop_path_rate=0.,
            vocab_size=49408,
            num_frames=1,
            use_fast_conv1=True,
            use_flash_attn=True,
            project_embed_dim=512,
            pretrain_zoo='openai',
            pretrain_path='/home/wangxiaoqi/AVION/pretrain/eva02_base_patch16_clip_224.merged2b_s8b_b131k/open_clip_pytorch_model.bin',
            )

    from torch.autograd import Variable
    import torch.cuda.amp as amp
    scaler = amp.GradScaler(enabled=True)

    input_video = Variable(torch.FloatTensor(8, 3, 16, 224, 224)).cuda()
    model = model.cuda()
    with amp.autocast(enabled=True):
        image_features, text_features, logit_scale = model()
    print('net output size:')
    print(output.shape)