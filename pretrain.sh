#!/bin/bash

#inside the container

OUTPUT_DIR=/pretrain

# 1 replica
nohup python -u src/pretrain/run_pretrain.py \
    --config src/configs/pretrain_image_text_base_resnet50_mlm_itm.json \
    &> pretrain_m2000_bs1.log &

