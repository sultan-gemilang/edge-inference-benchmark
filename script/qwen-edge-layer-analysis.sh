#!/bin/bash

MODEL="Qwen/Qwen2.5-1.5B"
LOG_DIR="logs/qwen2-layer-analysis"
MODEL_PATH="models/qwen2_layers"

mkdir -p $LOG_DIR/mlp/log
mkdir -p $LOG_DIR/mlp/terminal

mkdir -p $LOG_DIR/attn/log
mkdir -p $LOG_DIR/attn/terminal

# This script runs the Qwen-Edge benchmark layer analysis tests.
echo "Running Qwen-Edge benchmark layer analysis tests..."

# MLP analysis
echo "Running MLP layer analysis..."
for i in {0..27}
do
    echo "Running pruned $MODEL_PATH MLP layer $i analysis..."
    
    python -u qwen-edge-benchmark.py \
    --model_path $MODEL_PATH/mlp/mlp_layer_$i/pytorch_model.bin --method pruned \
    --warmup 3 --benchmark 50 --max_new_tokens 200 \
    --log_file qwen2-layer-analysis/mlp/log/qwen2_mlp_layer_log_$i.txt \
    | tee $LOG_DIR/mlp/terminal/qwen2_mlp_layer_terminal_$i.txt
    
    sleep 60
done

# ATTN analysis
echo "Running ATTN layer analysis..."
for i in {0..27}
do
    echo "Running pruned $MODEL_PATH ATTN layer $i analysis..."
    
    python -u qwen-edge-benchmark.py \
    --model_path $MODEL_PATH/attn/attn_layer_$i/pytorch_model.bin --method pruned \
    --warmup 3 --benchmark 50 --max_new_tokens 200 \
    --log_file qwen2-layer-analysis/attn/log/qwen2_attn_layer_log_$i.txt \
    | tee $LOG_DIR/attn/terminal/qwen2_attn_layer_terminal_$i.txt
    
    sleep 60
done