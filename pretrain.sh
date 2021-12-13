#!/bin/bash

#inside the container

OUTPUT_DIR=/pretrain
# 8 GPU
#horovodrun -np 8 python src/pretrain/run_pretrain.py \
#    --config src/configs/pretrain_image_text_base_resnet50_mlm_itm.json \
#    --output_dir $OUTPUT_DIR

# 1 GPU
horovodrun -np 1 python -u src/pretrain/run_pretrain.py \
    --config src/configs/pretrain_image_text_base_resnet50_mlm_itm.json \
    --gradient_accumulation_steps 16 \
#    &> pretrain_cpu.log &

