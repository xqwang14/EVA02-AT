#!/bin/bash
export TORCH_USE_CUDA_DSA=1
# abort entire script on error
set -e

DATA_ROOT=/home/wangxiaoqi/CharadesEgo_v1_480
AVION_ROOT=/home/wangxiaoqi/SMS_eva/
EXP_PATH=/home/wangxiaoqi/experiments/demo_charades
if [ ! -d $EXP_PATH ]; then
    mkdir $EXP_PATH
fi

#cp this shell file to experiment folder
SCRIPT_PATH="$0"
SCRIPT_NAME=$(basename "$SCRIPT_PATH")
cp "$SCRIPT_PATH" "$EXP_PATH/$SCRIPT_NAME"

CUDA_VISIBLE_DEVICES='0,1,2,3' PYTHONPATH=../third_party/decord/python/ NCCL_P2P_LEVEL=NVL torchrun --nproc_per_node=4 ../train/test_mir.py\
    --dataset charades_ego \
    --root $DATA_ROOT/ \
    --train-metadata $DATA_ROOT/annotations/metadata_train.csv \
    --val-metadata $DATA_ROOT/annotations/metadata_val.csv \
    --relevancy-train $DATA_ROOT/annotations/relevancy_val.pkl \
    --relevancy-test $DATA_ROOT/annotations/relevancy_val.pkl \
    --output-dir ./ \
    --use-flash-attn \
    --clip-length 16 \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 32 \
    --fused-decode-crop \
    --submit-file \
    --dist-backend 'nccl' \
    --use-multi-epochs-loader \
    --pretrain-model /home/wangxiaoqi/experiments/evaL_pretrain5/checkpoint.pt \
    --output-dir $EXP_PATH 2>&1 | tee $EXP_PATH/log.txt