#!/bin/bash

MODEL_DIR=./models/pruned_models

for model in $(ls $MODEL_DIR); do
    python -u latency-benchmark.py \
        --model_path $MODEL_DIR/$model --method pretrained \
        --warmup 3 --benchmark 5 --max_new_tokens 250 \
        --log_file ${model}_log.txt \
        | tee logs/${model}_terminal.txt
    
    sleep 120
done