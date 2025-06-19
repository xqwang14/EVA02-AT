import argparse
from collections import OrderedDict
from functools import partial
import json
import os

#Modify the root to the parent folder to aviod errors when importing avion
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
#sys.path.append('/home/wangxiaoqi/avion/third_party/decord/python')
#-------------------end of modification----------------------------------
import time
import numpy as np
import pandas as pd
import glob
import os.path as osp

import kornia as K
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.distributed.optim import ZeroRedundancyOptimizer
import torchvision
import torchvision.transforms._transforms_video as transforms_video
from timm.data.loader import MultiEpochsDataLoader

from resources.data.clip_dataset import VideoCaptionDatasetCLIP, VideoCaptionDatasetMCQ
from resources.data.tokenizer import tokenize
from resources.data.transforms import Permute

from resources.losses.losses import MaxMarginRankingLoss
import resources.models.model_clip as model_clip
from resources.models.utils import inflate_positional_embeds, inflate_positional_embeds_eva
from resources.optim.schedulers import cosine_scheduler
import resources.utils.distributed as dist_utils
from resources.utils.evaluation_ek100mir import get_mAP, get_nDCG
from resources.utils.evaluation_charades import get_charades_map
from resources.utils.evaluation_egomcq import egomcq_accuracy_metrics
from resources.utils.meters import AverageMeter, ProgressMeter
from resources.utils.misc import check_loss_nan

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def softmax_numpy(sim, dim=0):
    sim = torch.Tensor(sim)
    sim = F.softmax(sim, dim=dim)
    return sim.numpy()

def create_and_save_dict(sim_mat, txt_ids, vis_ids, filename='test4.pkl'):
    import pickle
    """
    Create a dictionary with specified version and numpy arrays, and save it to a pickle file.

    Parameters:
    sim_mat (np.ndarray): A 2D numpy array of similarity scores.
    txt_ids (np.ndarray): A 1D numpy array of text identifiers.
    vis_ids (np.ndarray): A 1D numpy array of visual identifiers.
    filename (str): The name of the file to save the dictionary.

    Returns:
    None
    """
    # Create the dictionary
    data_dict = {
        'version': '0.1',
        "challenge": "multi_instance_retrieval",
        "sls_pt": 4,
        "sls_tl": 3,
        "sls_td": 3,
        'sim_mat': sim_mat,
        'txt_ids': txt_ids,
        'vis_ids': vis_ids
    }
    print(txt_ids)
    print(vis_ids)

    # Save the dictionary to a pickle file
    with open(filename, 'wb') as file:
        pickle.dump(data_dict, file)

    print(f"Dictionary saved to {filename}")

def get_args_parser():
    parser = argparse.ArgumentParser(description='AVION finetune ek100 mir', add_help=False)
    parser.add_argument('--dataset', default='ek100_mir', type=str, choices=['ek100_mir', 'charades_ego', 'ego4d_mcq'])
    parser.add_argument('--root', default='/home/wangxiaoqi/EK100_320p_15sec/', type=str, help='path to train dataset root')
    parser.add_argument('--train-metadata', type=str,
                        default='datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_train.csv')
    parser.add_argument('--val-metadata', type=str,
                        default='datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv')
    parser.add_argument('--relevancy-train', type=str,
                        default='datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_train.pkl')
    parser.add_argument('--relevancy-test', type=str,
                        default='datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--video-chunk-length', default=-1, type=int)
    parser.add_argument('--clip-length', default=16, type=int, help='clip length')
    parser.add_argument('--clip-stride', default=4, type=int, help='clip stride')
    parser.add_argument('--norm-style', default='openai', type=str, choices=['openai', 'timm'])
    parser.add_argument('--fused-decode-crop', action='store_true', dest='fused_decode_crop')
    parser.add_argument('--no-fused-decode-crop', action='store_false', dest='fused_decode_crop')
    parser.set_defaults(fused_decode_crop=False)
    parser.add_argument('--decode-threads', default=1, type=int)
    parser.add_argument('--use-multi-epochs-loader', action='store_true')
    # model
    parser.add_argument('--model', default='CLIP_VITB16', type=str)
    parser.add_argument('--grad-checkpointing', action='store_true', dest='use_grad_checkpointing')
    parser.add_argument('--no-grad-checkpointing', action='store_false', dest='use_grad_checkpointing')
    parser.set_defaults(use_grad_checkpointing=False)
    parser.add_argument('--use-fast-conv1', action='store_true', dest='use_fast_conv1')
    parser.add_argument('--disable-fast-conv1', action='store_false', dest='use_fast_conv1')
    parser.set_defaults(use_fast_conv1=False)
    parser.add_argument('--use-flash-attn', action='store_true', dest='use_flash_attn')
    parser.add_argument('--disable-flash-attn', action='store_false', dest='use_flash_attn')
    parser.set_defaults(use_flash_attn=False)
    parser.add_argument('--patch-dropout', default=0., type=float)
    parser.add_argument('--drop-path-rate', default=0., type=float)
    parser.add_argument('--pretrain-model', default='', type=str, help='path of pretrained model')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    # clip loss
    parser.add_argument('--local-loss', action='store_true')
    parser.add_argument('--gather-with-grad', action='store_true', dest='gather_with_grad')
    parser.add_argument('--no-gather-with-grad', action='store_false', dest='gather_with_grad')
    parser.set_defaults(gather_with_grad=True)
    # training
    parser.add_argument('--use-zero', action='store_true', dest='use_zero', help='use ZeRO optimizer')
    parser.add_argument('--no-use-zero', action='store_false', dest='use_zero', help='use ZeRO optimizer')
    parser.set_defaults(use_zero=False)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=64, type=int, help='number of samples per-device/per-gpu')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float, help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float, help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int, help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.01, type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval-freq', default=5, type=int)
    parser.add_argument('--disable-amp', action='store_true', help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--grad-clip-norm', default=None, type=float)
    # system
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate', action='store_true', help='eval only')
    parser.add_argument('--flip', action='store_true', help='apply image(video) filp')
    parser.add_argument('--submit-file', action='store_true', help='create submit file')
    parser.set_defaults(submit_file=False)
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--multigpu', action='store_true',
                        help='Use multiple GPUs')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=list, help='GPU id to use.')
    return parser


def main(args):
    dist_utils.init_distributed_mode(args)
    dist_utils.random_seed(args.seed, dist_utils.get_rank())

    if args.pretrain_model:
        ckpt_path = args.pretrain_model
    else:
        raise Exception('no checkpoint found, add it by `--pretrain-model ${CHECKPOINT_PATH}`')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    old_args = ckpt['args']
    print(f'old_args:{old_args}')
    print("=> creating model: {}".format(old_args.model))

    #context length is 77 and vocab size is 49408
    model = getattr(model_clip, old_args.model)(
        freeze_temperature=True,
        use_grad_checkpointing=args.use_grad_checkpointing,
        context_length=77,
        vocab_size=49408,
        patch_dropout=args.patch_dropout,
        num_frames=args.clip_length,
        original_num_frames = old_args.clip_length,
        drop_path_rate=args.drop_path_rate,
        use_fast_conv1=args.use_fast_conv1,
        use_flash_attn=args.use_flash_attn,
        use_quick_gelu=True,
        project_embed_dim=old_args.project_embed_dim,
        pretrain_zoo=old_args.norm_style,
        pretrain_path=args.pretrain_model,
    )
    model.logit_scale.requires_grad = False
    print('=> inflating PE in models due to different frame numbers')
    if 'EVA' in old_args.model:
        state_dict = inflate_positional_embeds_eva(
            model.state_dict(), state_dict,
            num_frames=args.clip_length,
            load_temporal_fix='bilinear',
        )
    else:
        state_dict = inflate_positional_embeds(
            model.state_dict(), state_dict,
            num_frames=args.clip_length,
            load_temporal_fix='bilinear',
        )
    model.load_state_dict(state_dict, strict=True)
    print("=> loaded resume checkpoint '{}' (epoch {})".format(ckpt_path, ckpt['epoch']))
    #print(f'Distributed Status is:{args.distributed}')
    model.cuda(args.gpu)
    #print(model)

    criterion = MaxMarginRankingLoss(
        margin=0.2,
        fix_norm=True,
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        rank=args.rank,
        world_size=args.world_size,
    ).cuda(args.gpu)

    n_wd, n_non_wd = [], []
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        #print(n,p.shape)
        if not p.requires_grad:
            continue  # frozen weights
        #Set all the parameters except: bias LN BN and positional embedding a weight decay to limit the size of weight
        #why not set wd to these layers? bias would not affect the overfitting or the number of parameters.
        #BN LN do normalization already. And for these layers with single dimension of parameters, it is meaningless to do weight decay on it.
        if (p.ndim < 2 or 'bias' in n or
            'ln' in n or 'bn' in n or
            'pos_embed' in n or 'positional_embedding' in n
        ):
            n_non_wd.append(n)
            p_non_wd.append(p)
        else:
            n_wd.append(n)
            p_wd.append(p)
    
    crop_size = 336 if old_args.model.endswith("_336PX") else 224 #There is a CLIP_VITL_336PX model available; else it would resize the video to 224px

    # idk how the tokenizer works
    tokenizer = partial(tokenize, context_length=77)
    #openai norm values
    mean, std = [108.3272985, 116.7460125, 104.09373615000001], [68.5005327, 66.6321579, 70.32316305]
    #video preprocessing. Still wonder why there is a permute.
    base_val_transform_ls = [
        Permute([3, 0, 1, 2]),
        torchvision.transforms.Resize(crop_size),
        torchvision.transforms.CenterCrop(crop_size),
        transforms_video.NormalizeVideo(mean=mean, std=std),
    ]
    val_transform = torchvision.transforms.Compose(base_val_transform_ls)

    if args.dataset == 'ego4d_mcq':
        val_dataset = VideoCaptionDatasetMCQ(
            args=args, metadata= args.val_metadata,
            transform=val_transform, is_training=False, tokenizer=tokenizer,
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            chunk_len=args.video_chunk_length,
            fast_rcc=args.fused_decode_crop, rcc_params=(crop_size,),
        )
    else:
        val_dataset = VideoCaptionDatasetCLIP(
            args=args, metadata= args.val_metadata,
            transform=val_transform, is_training=False, tokenizer=tokenizer,
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            chunk_len=args.video_chunk_length,
            fast_rcc=args.fused_decode_crop, rcc_params=(crop_size,),
        )

    #set up the samplers for dataloaders:
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

    #set up the dataloaders:
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=val_sampler, drop_last=False
    )
    print('len(val_loader) = {}'.format(len(val_loader)))


    # #inputs[0] is the video, size B,3,16,224,224; inputs[1] is the tokenized text, size 64,77; inputs[2] are action labels, size 64. In MIR tasks, inputs[2] is not needed.
    # for data_iter, inputs in enumerate(train_loader):
    #     #print(data_iter)
    #     inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]
    #     #print(inputs[0].shape, inputs[1].shape, inputs[1], inputs[2])
    #     #relevancies = inputs.pop()
    #     break
    if args.dataset == 'ego4d_mcq':
        validate_mcq(val_loader, [], model)
    else:
        validate_mir(val_loader, [], model, criterion, args)

def validate_mcq(val_loader, transform_gpu, model, use_half=False):
    model.eval()
    with amp.autocast(enabled=not args.disable_amp):
        with torch.no_grad():
            print('=> start forwarding')
            all_preds = []
            all_gts = []
            all_types = []
            end_time = time.time()
            for i, inputs in enumerate(val_loader):
                if i % args.print_freq == 0:
                    print('finish batch {}/{} in {} sec'.format(i, len(val_loader), time.time() - end_time))
                    end_time = time.time()
                texts_query = inputs[0].cuda(non_blocking=True)
                frames_options = inputs[1].cuda(non_blocking=True)
                answer = inputs[3]
                q_type = inputs[4]
                if len(inputs) == 7:
                    masks_query = inputs[5].cuda(non_blocking=True)
                else:
                    masks_query = None

                batch_size = frames_options.shape[0]
                frames_options = frames_options.view(-1, *frames_options.shape[2:])
                texts_query = texts_query.view(-1, *texts_query.shape[2:])
                #print(frames_options.shape, texts_query.shape)
                if args.fused_decode_crop and len(transform_gpu) > 0:
                    frames_options = frames_options.permute(0, 4, 1, 2, 3)
                #print(frames_options.shape, texts_query.shape)
                image_features, query_features, _ = model(frames_options, texts_query)
                # image_features = dist_utils.get_model(model).encode_image(frames_options)
                image_features = image_features.view(batch_size, -1, *image_features.shape[1:])

                # if masks_query is not None:
                #     query_features = dist_utils.get_model(model).encode_text(texts_query, attention_mask=masks_query)
                # else:
                #     query_features = dist_utils.get_model(model).encode_text(texts_query)

                all_gts.append(answer)
                all_types.append(q_type)
                for j in range(batch_size):
                    similarity_matrix = torch.matmul(query_features[j], image_features[j].T)
                    similarity_matrix = similarity_matrix.cpu().detach()
                    all_preds.append(similarity_matrix)
            all_preds = torch.stack(all_preds)
            all_gts = torch.cat(all_gts)
            all_types = torch.cat(all_types)
            metrics = egomcq_accuracy_metrics(all_preds, all_gts, all_types)
            print(metrics)

def validate_mir(val_loader, transform_gpu, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = ['loss', 'max_margin_loss']
    iters_per_epoch = len(val_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Test: "
    )

    # switch to eval mode
    model.eval()

    all_video_embed = [[] for _ in range(args.world_size)]
    all_text_embed = [[] for _ in range(args.world_size)]
    #all_logit_scale = [[] for _ in range(args.world_size)]
    total_num = 0
    with amp.autocast(enabled=not args.disable_amp):
        with torch.no_grad():
            end = time.time()
            for i, inputs in enumerate(val_loader):
                # measure data loading time
                data_time.update(time.time() - end)

                inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]
                #remove row and col
                inputs.pop()
                inputs.pop()
                relevancies = inputs.pop()
                

                # compute output
                if args.fused_decode_crop and len(transform_gpu) > 0:
                    inputs[0] = inputs[0].permute(0, 4, 1, 2, 3)
                    inputs[0] = transform_gpu(inputs[0])

                # print(inputs[1].dtype)
                # print(inputs[1].shape)
                # if inputs[1].isinstance(list) and len(inputs[1]) > 1:
                #     image_features = model.encode_image(inputs[0])
                #     text_features = []
                #     for text in inputs[1]:
                #         text_feature = model.encode_text(text)
                #         text_features.append(text_feature)
                #     text_features = text_features.mean(dim=0)
                # else:
                image_features, text_features, logit_scale = model(*inputs)
                if args.flip == True:
                    inputs_flip = inputs
                    inputs_flip[0] = torch.flip(inputs[0], dims=[-1])
                    image_features_flip = model.encode_image(inputs_flip[0])
                    image_features = (image_features + image_features_flip)/2

                gathered_image_features = [torch.zeros_like(image_features) for _ in range(args.world_size)]
                gathered_text_features = [torch.zeros_like(text_features) for _ in range(args.world_size)]
                #gathered_logit_scale = [torch.zeros_like(logit_scale) for _ in range(args.world_size)]
                torch.distributed.all_gather(gathered_image_features, image_features)
                torch.distributed.all_gather(gathered_text_features, text_features)
                #torch.distributed.all_gather(gathered_logit_scale, logit_scale)
                for j in range(args.world_size):
                    all_video_embed[j].append(gathered_image_features[j].detach().cpu())
                    all_text_embed[j].append(gathered_text_features[j].detach().cpu())
                    #all_logit_scale[j].append(gathered_logit_scale[j].detach().cpu())
                loss_dict = criterion(image_features, text_features, weight=relevancies)

                for k in loss_dict:
                    metrics[k].update(loss_dict[k].item(), args.batch_size)

                total_num += image_features.shape[0] * args.world_size

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                mem.update(torch.cuda.max_memory_allocated() // 1e9)

                if i % args.print_freq == 0:
                    progress.display(i)
    progress.synchronize()
    for j in range(args.world_size):
        all_video_embed[j] = torch.cat(all_video_embed[j], dim=0).numpy()
        all_text_embed[j] = torch.cat(all_text_embed[j], dim=0).numpy()
        #all_logit_scale[j] = torch.cat(all_logit_scale[j], dim=0).numpy()
    all_text_embed_reorg, all_video_embed_reorg, all_logit_scale_reorg = [], [], []
    for i in range(total_num):
        all_video_embed_reorg.append(all_video_embed[i % args.world_size][i // args.world_size])
        all_text_embed_reorg.append(all_text_embed[i % args.world_size][i // args.world_size])
        #all_logit_scale_reorg.append(all_logit_scale[i % args.world_size][i // args.world_size])
    all_text_embed = np.vstack(all_text_embed_reorg)
    all_video_embed = np.vstack(all_video_embed_reorg)
    #all_logit_scale = np.vstack(all_logit_scale_reorg)
    if args.dataset == 'ek100_mir':
        all_text_embed = all_text_embed[:9668, :]
        all_video_embed = all_video_embed[:9668, :]
        
    elif args.dataset == 'charades_ego':
        all_text_embed = all_text_embed[:9151, :]
        all_video_embed = all_video_embed[:9151, :]
        #all_logit_scale = all_logit_scale[:9151, :]
        video_list = glob.glob(osp.join(args.root, '*.mp4'))
        video_ids = pd.read_csv(args.val_metadata).values[:, 2]
        corres_narration = pd.read_csv(args.val_metadata).values[:, 5]
        sample_list = []
        sample_narration_list = []
        for video in video_list:
            narration_list = []
            video, _ = osp.splitext(osp.basename(video))
            for index, video_id in enumerate(video_ids):
                if video == video_id:
                    narration_list.append(corres_narration[index])
            if narration_list != []:
                sample_list.append(video)
                sample_narration_list.append(narration_list)
        video_indexes = [video_ids.tolist().index(elem) for elem in sample_list]
        print(len(video_indexes))
        all_video_embed = all_video_embed[video_indexes, :]
        class_embed = []
        gt_labels = []
        text_narration = pd.read_csv(args.val_metadata.replace('val', 'val_sentence')).values[:, 2]
        for narrations in sample_narration_list:
            #text_indexes = [corres_narration.tolist().index(elem) for elem in narrations]
            text_indexes = [text_narration.tolist().index(elem) for elem in narrations]
            gt_labels.append(text_indexes)
            #class_embed.append(np.mean(all_text_embed[text_indexes, :], axis=0))
        gt_label_matrix = np.zeros((len(video_indexes), len(text_narration)))
        for col in range(len(gt_label_matrix[0])):
            for row in range(len(gt_label_matrix)):
                if col in gt_labels[row]:
                    gt_label_matrix[row, col] = 1
        #all_text_embed = np.vstack(class_embed)
            


    similarity_matrix_dist = np.matmul(all_text_embed,all_video_embed.T)
    similarity_matrix = (similarity_matrix_dist + 1) / 2
    
    #dual-softmax
    # if True:
    #     similarity_matrix = softmax_numpy(similarity_matrix / 500, dim=1) * similarity_matrix
    #     similarity_matrix = softmax_numpy(similarity_matrix, dim=0)
    # else:
    if args.dataset == 'ek100_mir':
        video_id = pd.read_csv(args.val_metadata).values[:, 0]
        text_id = pd.read_csv(args.val_metadata.replace('test', 'test_sentence')).values[:, 0]
        indexes = [video_id.tolist().index(elem) for elem in text_id]
        
    elif args.dataset == 'charades_ego':
        video_text_id = pd.read_csv(args.val_metadata).values[:, 1]
        #text_id = pd.read_csv(args.val_metadata).values[:, 7]
        text_id = pd.read_csv(args.val_metadata.replace('val', 'val_sentence')).values[:, 1]
        indexes = [video_text_id.tolist().index(elem) for elem in text_id]
    
    similarity_matrix = similarity_matrix.T[:, indexes]
    print(similarity_matrix.shape)

    # if True:
    #     v_v_dist = np.matmul(all_video_embed,all_video_embed.T)
    #     t_t_dist = np.matmul(all_text_embed,all_text_embed.T)
    #     q_g_dist = similarity_matrix_dist
    #     print(q_g_dist.shape, t_t_dist.shape, v_v_dist.shape)
    #     t_v_dist = -re_ranking(q_g_dist, t_t_dist, v_v_dist).T[:, indexes]
    #similarity_matrix = naive_rerank(similarity_matrix.T).T

    rel_matrix = pd.read_pickle(args.relevancy_test)
    if args.dataset == 'ek100_mir':
        vis_map, txt_map, avg_map = get_mAP(similarity_matrix, rel_matrix)
        print('mAP: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_map, txt_map, avg_map))
        vis_nDCG, txt_nDCG, avg_nDCG = get_nDCG(similarity_matrix, rel_matrix)
        print('nDCG: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_nDCG, txt_nDCG, avg_nDCG))
    elif args.dataset == 'charades_ego':
        vis_map, txt_map, avg_map = get_charades_map(similarity_matrix, gt_label_matrix)
        print('mAP: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_map, txt_map, avg_map))

    if args.submit_file and args.dataset == 'ek100_mir':
        create_and_save_dict(similarity_matrix, text_id, video_id)
    elif args.submit_file and args.dataset == 'charades_ego':
        create_and_save_dict(similarity_matrix, text_id, gt_label_matrix)
    # return {**{k: v.avg for k, v in metrics.items()},
    #         'vis_map': vis_map, 'txt_map': txt_map, 'avg_map': avg_map,
    #         'vis_ndcg': vis_nDCG, 'txt_ndcg': txt_nDCG, 'avg_ndcg': avg_nDCG}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LAVILA training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)