#!/bin/bash
# Benchmark all sparsegpt/wanda/magnitude pruned models in compressed-models/pruned-models-v2/
# using vLLM runtime. Skips llm-pruner-models (.pt format, incompatible with vLLM).
# Uses the toy-torch-vllm conda env (separate from toy-torch) to avoid breaking
# the existing pruned/.pt model loading which depends on toy-torch's PyTorch version.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$REPO_ROOT/compressed-models/pruned-models-v2"
PYTHON="$HOME/anaconda3/envs/toy-torch-vllm/bin/python"

mkdir -p "$REPO_ROOT/logs"

benchmark_model_vllm() {
    local model_path=$1
    local model_name
    model_name=$(basename "$model_path")

    echo "=========================================="
    echo "Benchmarking (vllm): $model_name"
    echo "Model Path: $model_path"
    echo "=========================================="

    "$PYTHON" -u "$REPO_ROOT/latency-benchmark.py" \
        --model_path "$model_path" \
        --method vllm \
        --warmup 3 \
        --input_len 512 \
        --output_len 256 \
        --num_samples 20 \
        --gpu_mem_util 0.55 \
        --max_model_len 1024 \
        --enforce_eager \
        --log_file "${model_name}_vllm_log.txt" \
        | tee "$REPO_ROOT/logs/${model_name}_vllm_terminal.txt"

    echo "Benchmark completed for: $model_name"
    echo "Waiting 60 seconds before next benchmark..."
    sleep 60
}

for item in "$MODEL_DIR"/*/; do
    item_name=$(basename "$item")

    # Skip llm-pruner-models: those are .pt serialized dicts, not HF format
    if [[ "$item_name" == "llm-pruner-models" ]]; then
        echo "Skipping llm-pruner-models (not HuggingFace format)."
        continue
    fi

    # Skip non-directory entries
    if [[ ! -d "$item" ]]; then
        continue
    fi

    # Only benchmark dirs that contain model.safetensors (confirmed HF format)
    if [[ -f "${item}model.safetensors" ]]; then
        benchmark_model_vllm "$item"
    else
        echo "Skipping $item_name: no model.safetensors found."
    fi
done

echo "=========================================="
echo "All vLLM benchmarks completed!"
echo "Results saved to logs/csv_results/"
echo "=========================================="
