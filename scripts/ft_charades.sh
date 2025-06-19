#!/bin/bash
export TORCH_USE_CUDA_DSA=1
# abort entire script on error
set -e

DATA_ROOT=/home/wangxiaoqi/CharadesEgo_v1_480
AVION_ROOT=..
EXP_PATH=/home/wangxiaoqi/AVION/experiments/ft_charades_evab
if [ ! -d $EXP_PATH ]; then
    mkdir $EXP_PATH
fi

#cp this shell file to experiment folder
SCRIPT_PATH="$0"
SCRIPT_NAME=$(basename "$SCRIPT_PATH")
cp "$SCRIPT_PATH" "$EXP_PATH/$SCRIPT_NAME"

CUDA_VISIBLE_DEVICES='0,1,2,3' PYTHONPATH=../third_party/decord/python/ NCCL_P2P_LEVEL=NVL NCCL_ALGO=Tree torchrun --nproc_per_node=4 scripts/finetune_charades.py\
    --root $DATA_ROOT/origin/ \
    --dataset charades_ego \
    --train-metadata $DATA_ROOT/annotations/metadata_train.csv \
    --val-metadata $DATA_ROOT/annotations/metadata_val.csv \
    --relevancy-train $DATA_ROOT/annotations/relevancy_train.pkl \
    --relevancy-test $DATA_ROOT/annotations/relevancy_val.pkl \
    --output-dir ./ \
    --use-flash-attn \
    --grad-checkpointing \
    --model EVA_B_16 \
    --pretrain-zoo open_clip \
    --optimizer lamb \
    --use-zero \
    --use-fast-conv1 \
    --batch-size 32 \
    --epochs 10 \
    --eval-freq 1 \
    --lr 3e-5 \
    --lr-start 1e-6 \
    --lr-end 5e-7 \
    --loss-margin 0.3 \
    --loss-thres 0.1 \
    --seed 3407 \
    --fused-decode-crop \
    --dist-backend 'nccl' \
    --use-multi-epochs-loader \
    --pretrain-model ../pretrain/eva_pretrain_evab_wllm343.pt \
    --output-dir $EXP_PATH 2>&1 | tee $EXP_PATH/log.txt