#!/bin/bash

MODEL_DIR=./models/pruned-models-v2

# Ensure logs directory exists
mkdir -p logs

# Function to benchmark a model
benchmark_model() {
    local model_path=$1
    local method=$2
    local custom_name=${3:-""}
    
    # Use custom name if provided, otherwise extract from path
    local model_name="${custom_name:-$(basename "$model_path")}"
    
    echo "=========================================="
    echo "Benchmarking: $model_name (Method: $method)"
    echo "Model Path: $model_path"
    echo "=========================================="
    
    # Run benchmark
    python -u latency-benchmark.py \
        --model_path "$model_path" \
        --method "$method" \
        --warmup 3 \
        --input_len 512 \
        --output_len 256 \
        --num_samples 20 \
        --log_file "${model_name}_log.txt" \
        | tee "logs/${model_name}_terminal.txt"
    
    echo "Benchmark completed for: $model_name"
    echo "Waiting 60 seconds before next benchmark..."
    sleep 60
}

# Iterate through all items in MODEL_DIR
for item in $(ls -d "$MODEL_DIR"/*/ 2>/dev/null); do
    item_name=$(basename "$item")
    
    # Check if this is the llm-pruner-models folder
    if [[ "$item_name" == "llm-pruner-models" ]]; then
        echo "Found llm-pruner-models folder, iterating with --method pruned"
        
        # Iterate through subdirectories in llm-pruner-models
        for model_path in $(ls -d "$item"*/); do
            if [[ -e "$model_path" ]]; then
                dir_name=$(basename "$model_path")
                benchmark_model "${model_path}pytorch_model.bin" "pruned" "$dir_name"
            fi
        done
    else
        # For all other directories, use --method pretrained
        if [[ -e "$item" ]]; then
            # benchmark_model "$item" "pretrained"
            echo "passed"
        fi
    fi
done

echo "=========================================="
echo "All benchmarks completed!"
echo "Results saved to logs/csv_results/"
echo "=========================================="