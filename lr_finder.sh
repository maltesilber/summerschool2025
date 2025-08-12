#!/bin/bash

DATA_ROOT="/home/malte/datasets/FungiImages"
OUTPUT_BASE="./vit-base-checkpoints"
LOGGING_DIR="./logs"
BATCH_SIZE=128
EPOCHS=10
LR_SCHEDULER_TYPE="cosine"
WARMUP_RATIO=0.01
LOGGING_STEPS=50
FP16_FLAG=""

# Uncomment if mixed precision is desired
# FP16_FLAG="--fp16"

# Define the learning rates and weight decays to test
LRs=(3e-5 1e-4)
WDs=(1e-4 1e-2)

# Iterate over learning rates and weight decays
for LR in "${LRs[@]}"; do
  for WD in "${WDs[@]}"; do
    OUTPUT_DIR="${OUTPUT_BASE}/lr_${LR}_wd_${WD}"
    python train_script.py \
      --data_root $DATA_ROOT \
      --output_dir $OUTPUT_DIR \
      --logging_dir $LOGGING_DIR \
      --batch_size $BATCH_SIZE \
      --epochs $EPOCHS \
      --lr $LR \
      --lr_scheduler_type $LR_SCHEDULER_TYPE \
      --weight_decay $WD \
      --warmup_ratio $WARMUP_RATIO \
      --logging_steps $LOGGING_STEPS \
      $FP16_FLAG
  done
done
