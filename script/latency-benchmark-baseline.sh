#!/bin/bash

MODEL_DIR=(
    "Qwen/Qwen2.5-1.5B"
    "meta-llama/Llama-3.2-1B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)
LOG_DIR="logs/baseline"

# Ensure logs directory exists
mkdir -p $LOG_DIR

# Iterate through all items in MODEL_DIR
for MODEL in ${MODEL_DIR[@]}; do
    
    model_name=$(basename "$MODEL")
    echo "=========================================="
    echo "Benchmarking: $MODEL (Method: Pretrained)"
    echo "Model Path: $MODEL"
    echo "=========================================="
    
    # Run benchmark
    python -u latency-benchmark.py \
        --model_path "$MODEL" \
        --method pretrained \
        --warmup 3 \
        --input_len 512 \
        --output_len 256 \
        --num_samples 20 \
        --log_file "${model_name}_log.txt" \
        | tee "${LOG_DIR}/${model_name}_terminal.txt"
    
    echo "Benchmark completed for: $model_name"
    echo "Waiting 60 seconds before next benchmark..."
    sleep 60
done