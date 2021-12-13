#!/bin/bash

CONFIG_PATH=src/configs/msrvtt_ret_base_resnet50.json
OUTPUT_DIR=/msrvtt_ret
# for single CPU

nohup python -u src/tasks/run_video_retrieval.py \
	    --config $CONFIG_PATH \
	    --output_dir $OUTPUT_DIR \
	    --train_batch_size 8 \
	    --gradient_accumulation_steps 8 \
	    &>> msrvtt_ret_cpu.log &

