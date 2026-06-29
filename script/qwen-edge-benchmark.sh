#!/bin/bash

mkdir -p logs

# This script runs the Qwen-Edge benchmark tests.
sleep 60
python -u qwen-edge-benchmark.py \
    --model_path Qwen/Qwen2.5-1.5B --method pretrained \
    --warmup 3 --benchmark 10 --max_new_tokens 200 \
    --log_file qwen_pretrained_log.txt \
    | tee logs/qwen_pretrained_terminal.txt
sleep 60

python -u qwen-edge-benchmark.py \
    --model_path models/qwen_prune_log_optimized/pytorch_model.bin --method pruned \
    --warmup 3 --benchmark 10 --max_new_tokens 200 \
    --log_file qwen_pruned_log.txt \
    | tee logs/qwen_pruned_terminal.txt
sleep 60

python -u qwen-edge-benchmark.py \
    --model_path models/qwen_prune_log_optimized/pytorch_model.bin --method pruned-lora \
    --adapter models/qwen_optimized_healed_v2 \
    --warmup 3 --benchmark 10 --max_new_tokens 200 \
    --log_file qwen_pruned_lora_log.txt \
    | tee logs/qwen_pruned_lora_terminal.txt