#!/usr/bin/env bash 

set -euo pipefail

GPU_ID=0
CONFIG_PTH=./4_target_llava15_flickr30k.yaml
POISON_SAVE_PTH="../runs/4target_llava15_flickr30k/optim_patch"

if [ -d "$POISON_SAVE_PTH" ]; then
    echo "Error: save path $POISON_SAVE_PTH already exists."
    exit 1
fi

mkdir -p "$POISON_SAVE_PTH"

export PYTHONPATH=$(pwd)
export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES="$GPU_ID" python ./optimize_trigger.py \
  --config "$CONFIG_PTH" \
  --poison_save_pth "$POISON_SAVE_PTH" \
  > "$POISON_SAVE_PTH/runlog.txt" 2>&1
