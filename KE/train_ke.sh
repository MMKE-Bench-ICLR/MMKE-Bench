#/bin/bash

# USAGE: bash train_ke.sh [GPU_ID] [MODEL_NAME]
# EXAMPLE: bash train_ke.sh 0 blip2

# MODEL_NAME=[blip2, minigpt4, llava]
# TYPE = [entity,visual,user]

GPU=$1
MODEL=$2
TYPE=$3
export CUDA_VISIBLE_DEVICES=$GPU
time=$(date "+%Y%m%d_%H%M%S")
python KE/train_ke.py \
    --model_name $MODEL \
    --data_type $TYPE \
    --gpus 1 \
    --num_workers 0 \
    --batch_size 1 \
    --max_steps 20000 \
    --divergences lp \
    2>&1 | tee models/$MODEL/$time\_train_log_$MODEL.txt
