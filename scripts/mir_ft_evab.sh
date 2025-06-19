#!/bin/bash
export TORCH_USE_CUDA_DSA=1
# abort entire script on error
set -e

DATA_ROOT=/home/wangxiaoqi/EK100_320p_15sec_30fps_libx264
EXP_PATH=../experiments/ego4d_ft_evab_test
if [ ! -d $EXP_PATH ]; then
    mkdir $EXP_PATH
fi

#cp this shell file to experiment folder
SCRIPT_PATH="$0"
SCRIPT_NAME=$(basename "$SCRIPT_PATH")
cp "$SCRIPT_PATH" "$EXP_PATH/$SCRIPT_NAME"

CUDA_VISIBLE_DEVICES='0,1,2,3' PYTHONPATH=../third_party/decord/python/ NCCL_P2P_LEVEL=NVL NCCL_ALGO=Tree torchrun --nproc_per_node=4 ../train/mir_ft_sms.py\
    --root $DATA_ROOT/ \
    --train-metadata $DATA_ROOT/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_train.csv \
    --val-metadata $DATA_ROOT/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv \
    --relevancy-train $DATA_ROOT/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_train.pkl \
    --relevancy-test $DATA_ROOT/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl \
    --output-dir ./ \
    --video-chunk-length 15 --use-flash-attn \
    --grad-checkpointing \
    --model EVA_B_16 \
    --pretrain-zoo open_clip \
    --use-fast-conv1 \
    --batch-size 64 \
    --use-zero \
    --epochs 100 \
    --lr 2e-5 \
    --lr-start 1e-6 \
    --lr-end 2e-6 \
    --loss-margin 0.6 \
    --loss-thres 0.1 \
    --seed 3407 \
    --fused-decode-crop \
    --dist-backend 'nccl' \
    --use-multi-epochs-loader \
    --pretrain-model ../pretrain/eva_pretrain_evab_wllm343.pt \
    --output-dir $EXP_PATH 2>&1 | tee $EXP_PATH/log.txt