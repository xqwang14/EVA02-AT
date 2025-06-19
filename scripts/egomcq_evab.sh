#!/bin/bash
export TORCH_USE_CUDA_DSA=1
# abort entire script on error
set -e

EGO_ROOT=/home/wangxiaoqi/ego4d
DATA_ROOT=/home/wangxiaoqi/ego4d/downscaled
VAL_ROOT=/home/wangxiaoqi/EK100_320p_15sec_30fps_libx264
AVION_ROOT=..
EXP_PATH=../experiments/evab_egomcq
if [ ! -d $EXP_PATH ]; then
    mkdir $EXP_PATH
fi

#cp this shell file to experiment folder
SCRIPT_PATH="$0"
SCRIPT_NAME=$(basename "$SCRIPT_PATH")
cp "$SCRIPT_PATH" "$EXP_PATH/$SCRIPT_NAME"
# python scripts/main_lavila_finetune_mir.py\
#     --root $DATA_ROOT/ \
#     --train-metadata $DATA_ROOT/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_train.csv \
#     --val-metadata $DATA_ROOT/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv \
#     --relevancy-path $DATA_ROOT/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl \
#     --output-dir ./ \
#     --video-chunk-length 15 --use-flash-attn \
#     --grad-checkpointing \
#     --use-fast-conv1 \
#     --batch-size 64 \
#     --no-gather-with-grad \
#     --fused-decode-crop \
#     --pretrain-model $AVION_ROOT/experiments/AVION_pretrain_lavila_vitb_best.pt \
#     --output-dir $EXP_PATH 2>&1 | tee $EXP_PATH/log.txt

CUDA_VISIBLE_DEVICES='1,3' PYTHONPATH=../third_party/decord/python/ NCCL_P2P_LEVEL=NVL torchrun --nproc_per_node=2 ../train/test_mir.py\
    --dataset ego4d_mcq \
    --root $DATA_ROOT/ \
    --val-metadata $EGO_ROOT/annotations/egomcq.json \
    --output-dir ./ \
    --use-flash-attn \
    --video-chunk-length 15 \
    --clip-length 4 \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 64 \
    --fused-decode-crop \
    --submit-file \
    --dist-backend 'nccl' \
    --use-multi-epochs-loader \
    --pretrain-model ../pretrain/eva_pretrain_evab_wllm343.pt \
    --output-dir $EXP_PATH 2>&1 | tee $EXP_PATH/log.txt