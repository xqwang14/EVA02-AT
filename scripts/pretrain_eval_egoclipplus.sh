#!/bin/bash
export TORCH_USE_CUDA_DSA=1
# abort entire script on error
set -e

EGO_ROOT=/home/wangxiaoqi/ego4d
DATA_ROOT=/home/wangxiaoqi/ego4d/downscaled
VAL_ROOT=/home/wangxiaoqi/EK100_320p_15sec_30fps_libx264
AVION_ROOT=..
EXP_PATH=../experiments/ego4d_pretrain_eval
if [ ! -d $EXP_PATH ]; then
    mkdir $EXP_PATH
fi

#cp this shell file to experiment folder
SCRIPT_PATH="$0"
SCRIPT_NAME=$(basename "$SCRIPT_PATH")
cp "$SCRIPT_PATH" "$EXP_PATH/$SCRIPT_NAME"

CUDA_VISIBLE_DEVICES='0,1,2,3' TORCH_DISTRIBUTED_DEBUG=DETAIL PYTHONPATH=../third_party/decord/python/ NCCL_P2P_LEVEL=NVL torchrun --nproc_per_node=4 ../train/ego4d_pretrain.py \
    --train-metadata $EGO_ROOT/annotations/ego4d_train.rephraser.no_punkt_top3.pkl\
    --train-metadata-aux $EGO_ROOT/annotations/ego4d_train.narrator_63690737.return_10.pkl \
    --val-metadata $VAL_ROOT/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv \
    --relevancy-path $VAL_ROOT/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl \
    --pretrain-path $AVION_ROOT/pretrain/eva02_large_patch14_clip_224.merged2b_s4b_b131k/open_clip_pytorch_model.bin \
    --root $DATA_ROOT/ \
    --root-val $VAL_ROOT \
    --pretrain-zoo open_clip \
    --output-dir ./ \
    --model EVA_L_14 \
    --optimizer adamw \
    --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --use-zero \
    --lr 3e-5 \
    --lr-start 3e-5 \
    --lr-end 3e-5 \
    --fix-lr \
    --clip-length 4 \
    --batch-size 128 \
    --project-embed-dim 256 \
    --seed 3407 \
    --freeze-temperature \
    --fused-decode-crop \
    --output-dir $EXP_PATH 2>&1 | tee $EXP_PATH/log.txt