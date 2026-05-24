import time
import subprocess
import torch
import psutil
import os
import numpy as np
import argparse
import gc
import csv
from pathlib import Path

from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from peft import PeftModel

# --- Configuration ---
LOG_PATH = "logs"
CSV_PATH = os.path.join(LOG_PATH, "csv_results")
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)
if not os.path.exists(CSV_PATH):
    os.makedirs(CSV_PATH)

PROMPT = "Write a short poem about a legendary Japanese motorcycle."

class LLMLatencyBenchmark:
    def __init__(self):
        self.metrics = {
            "load_time": 0,
            "tokenization": [],
            "ttft": [],        # Time To First Token
            "tpot": [],        # Tokens Per Output Token
            "tgt": [],         # Total Generation Time
            "tps": [],         # Tokens Per Second (Decode)
            "detokenization": [],
            "peak_vram": []    
        }

        self.model = None
        self.tokenizer = None 
        self.tegra_process = None
        self.model_path = None
        self.method = None

    def start_hardware_logger(self):
        """Starts tegrastats in the background."""
        LOG_FILE = os.path.join(LOG_PATH, args.log_file)
        
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        print(f"Starting hardware logger -> {LOG_FILE}")
        # Note: On some Jetsons, tegrastats might need sudo. 
        # If this fails, run the python script with sudo.
        cmd = ["tegrastats", "--logfile", LOG_FILE]
        self.tegra_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop_hardware_logger(self):
        """Stops tegrastats."""
        if self.tegra_process:
            self.tegra_process.terminate()
            self.tegra_process.wait()
            print("Hardware logger stopped.")
    
    def get_model_identifier(self):
        """Extract model name and info from model_path."""
        # Remove trailing slashes
        path = self.model_path.rstrip('/')
        
        # Handle HuggingFace paths like "Qwen/Qwen2.5-1.5B" or local paths
        if "/" in path:
            model_name = os.path.basename(path)
        else:
            model_name = os.path.basename(path)
        
        # Remove .pt extension if present
        if model_name.endswith('.pt'):
            model_name = model_name[:-3]
        
        # Try to extract sparsity info if present (e.g., "sparsity_50" in path)
        sparsity = ""
        if "sparsity" in path.lower():
            # Extract from the full path to get sparsity number
            parts = path.split("_")
            for i, part in enumerate(parts):
                if part.lower() == "sparsity" and i + 1 < len(parts):
                    sparsity_num = parts[i+1].split("/")[0]  # Handle paths with slashes
                    if sparsity_num.isdigit():
                        sparsity = f"_sparsity_{sparsity_num}"
                    break
        
        return model_name, sparsity
    
    def export_to_csv(self):
        """Export benchmark metrics to CSV files organized by model and method."""
        model_name, sparsity = self.get_model_identifier()
        
        # Create model-specific directory
        model_dir = os.path.join(CSV_PATH, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Filename: model_method_sparsity_metrics.csv
        filename = f"{model_name}_{self.method}{sparsity}_metrics.csv"
        csv_file = os.path.join(model_dir, filename)
        
        # Prepare data for CSV - metrics per round
        num_rounds = len(self.metrics["tokenization"])
        
        # Write detailed metrics CSV
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            headers = [
                "Round",
                "Tokenization_Time(s)",
                "TTFT(s)",
                "TPOT(s)",
                "Total_Gen_Time(s)",
                "Throughput(t/s)",
                "Detokenization_Time(s)",
                "Peak_VRAM(GB)"
            ]
            writer.writerow(headers)
            
            # Data rows
            for i in range(num_rounds):
                row = [
                    i + 1,
                    self.metrics["tokenization"][i] if i < len(self.metrics["tokenization"]) else "",
                    self.metrics["ttft"][i] if i < len(self.metrics["ttft"]) else "",
                    self.metrics["tpot"][i] if i < len(self.metrics["tpot"]) else "",
                    self.metrics["tgt"][i] if i < len(self.metrics["tgt"]) else "",
                    self.metrics["tps"][i] if i < len(self.metrics["tps"]) else "",
                    self.metrics["detokenization"][i] if i < len(self.metrics["detokenization"]) else "",
                    self.metrics["peak_vram"][i] if i < len(self.metrics["peak_vram"]) else ""
                ]
                writer.writerow(row)
        
        print(f"Detailed metrics CSV saved to: {os.path.abspath(csv_file)}")
        
        # Write summary statistics CSV
        summary_file = os.path.join(model_dir, f"{model_name}_{self.method}{sparsity}_summary.csv")
        
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Summary headers
            writer.writerow(["Metric", "Mean", "Std Dev", "Min", "Max"])
            
            # Summary data
            def write_summary_row(name, values):
                if values:
                    writer.writerow([
                        name,
                        f"{np.mean(values):.6f}",
                        f"{np.std(values):.6f}",
                        f"{np.min(values):.6f}",
                        f"{np.max(values):.6f}"
                    ])
            
            writer.writerow(["Load_Time(s)", f"{self.metrics['load_time']:.6f}", "", "", ""])
            write_summary_row("Tokenization(s)", self.metrics["tokenization"])
            write_summary_row("TTFT(s)", self.metrics["ttft"])
            write_summary_row("TPOT(s)", self.metrics["tpot"])
            write_summary_row("Total_Gen_Time(s)", self.metrics["tgt"])
            write_summary_row("Throughput(t/s)", self.metrics["tps"])
            write_summary_row("Detokenization(s)", self.metrics["detokenization"])
            write_summary_row("Peak_VRAM(GB)", self.metrics["peak_vram"])
        
        print(f"Summary statistics CSV saved to: {os.path.abspath(summary_file)}")
    
    def load_model_tokenizer(self, model_path, method, adapter_path=None):
        """
        Flexible loader for different pruning/quantization methods.
        """
        self.model_path = model_path
        self.method = method
        
        print(f"\n=== Phase 1: Loading Model ({method}) ===")
        print(f"Model Path: {model_path}")
        print(f"Method: {method}")
        print(f"Adapter Path: {adapter_path if adapter_path else 'N/A'}")
        
        # Clear cache before loading (Safe now because self.model is init to None)
        if self.model:
            del self.model
            torch.cuda.empty_cache()
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        try:
            if method == "pretrained":
                # Qwen2.5 benefits from float16
                self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map="auto")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                # Note: device_map="auto" usually handles .to("cuda"), but explicit is fine if single GPU
                # self.model.to("cuda") 
                print(f"Model loaded from pretrained checkpoint.")
                
            elif method == "pruned":
                # Assuming your .pt file is a dict {'model': obj, 'tokenizer': obj}
                pruned_dict = torch.load(model_path, map_location="cpu", weights_only=False)
                self.tokenizer = pruned_dict['tokenizer']
                self.model = pruned_dict['model']
                self.model.to("cuda")
                print(f"Model loaded from pruned checkpoint.")
                
            elif method == "pruned-lora":
                pruned_dict = torch.load(model_path, map_location="cpu", weights_only=False)
                self.tokenizer = pruned_dict['tokenizer']
                base_model = pruned_dict['model']
                
                print("Loading LoRA adapter...")
                self.model = PeftModel.from_pretrained(base_model, adapter_path, torch_dtype=torch.float16)
                # Merge adapter for accurate benchmarking (removes inference overhead)
                self.model = self.model.merge_and_unload()
                self.model.to("cuda")
                print(f"Model loaded from pruned checkpoint with LoRA adapter.")
            else:
                raise ValueError(f"Unknown loading method: {method}")
            
            gc.collect()
            torch.cuda.empty_cache()
            
            # <--- FIX 2: Qwen 2.5 Pad Token Fix
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            torch.cuda.synchronize()
            self.metrics["load_time"] = time.perf_counter() - start_time
            print(f"Model loaded in {self.metrics['load_time']:.2f} seconds.")
        
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e           

    def run_benchmark(self, warmup_rounds, benchmark_rounds):
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded before benchmarking.")
        
        try:
            # --- warmup rounds ---
            print(f"\n=== Phase 2: Warmup {warmup_rounds} Rounds ===")
            dummy_input = self.tokenizer("Hello", return_tensors="pt").to("cuda")
            for _ in range(warmup_rounds):
                with torch.no_grad():
                    _ = self.model.generate(**dummy_input, max_new_tokens=10)
            print("Warmup completed.")
            
            # --- start logger ---
            self.start_hardware_logger()
            
            # --- benchmark loop ---
            print(f"\n=== Phase 3: Benchmarking {benchmark_rounds} Rounds ===")
            
            for i in range(benchmark_rounds):
                print(f"Round {i+1}/{benchmark_rounds}...", end="", flush=True)
                
                # A. Tokenization
                torch.cuda.synchronize()
                tt0 = time.perf_counter()
                inputs = self.tokenizer(PROMPT, return_tensors="pt").to("cuda")
                torch.cuda.synchronize()
                tt1 = time.perf_counter()
                self.metrics["tokenization"].append(tt1 - tt0)
                
                # B. Generation (TTFT & TPS)
                streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)
                gen_kwargs = dict(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id # Explicitly pass pad token
                )
                
                # Start generation in a separate thread so we can time the first token
                thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
                
                torch.cuda.synchronize()
                t_gen_start = time.perf_counter()
                thread.start()
                
                # Wait for First Token (TTFT)
                generated_text = ""
                try:
                    # 'next' blocks until the first token arrives
                    first_token = next(iter(streamer))
                    torch.cuda.synchronize()
                    t_first = time.perf_counter()
                    
                    self.metrics["ttft"].append(t_first - t_gen_start)
                    generated_text += first_token
                    
                    # Consume rest of stream to finish generation
                    for token in streamer:
                        generated_text += token
                    
                    torch.cuda.synchronize()
                    t_gen_end = time.perf_counter()
                    
                    # Metrics Calculation
                    tgt = t_gen_end - t_gen_start
                    self.metrics["tgt"].append(tgt)
                    
                    # Throughput: Output Tokens / Total Generation Time
                    num_tokens = len(self.tokenizer.encode(generated_text))
                    self.metrics["tps"].append(num_tokens / tgt if tgt > 0 else 0)
                    
                    if num_tokens > 1:
                        decoding_time = t_gen_end - t_first
                        tpot = decoding_time / (num_tokens - 1)
                        self.metrics["tpot"].append(tpot)
                    else:
                        self.metrics["tpot"].append(0.0)

                except StopIteration:
                    print(" [Error: No tokens generated] ", end="")
                finally:
                    thread.join()
                    
                # C. Detokenization
                torch.cuda.synchronize()
                d0 = time.perf_counter()
                text = self.tokenizer.decode(self.tokenizer.encode(generated_text))
                torch.cuda.synchronize()
                d1 = time.perf_counter()
                self.metrics["detokenization"].append(d1 - d0)
                
                print(f" Generated text: {text[:50]}...", end="")
                
                # D. VRAM Check
                peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
                self.metrics["peak_vram"].append(peak_mem)
                
                print(f" Done. (TPS: {self.metrics['tps'][-1]:.2f})")
                
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user!")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
        finally:
            self.stop_hardware_logger()
            self.print_report()
                
    def print_report(self):
        print("\n" + "="*50)
        print(f"       BENCHMARK REPORT: {PROMPT[:30]}...")
        print("="*50)
        
        def print_metric(name, values, unit="s"):
            if not values: return
            avg = np.mean(values)
            mini = np.min(values)
            maxi = np.max(values)
            stdv = np.std(values)
            print(f"{name:<20} | Avg: {avg:.4f}{unit} | Min: {mini:.4f}{unit} | Max: {maxi:.4f}{unit} | Std: {stdv:.4f}{unit}")

        print(f"Model Load Time:     {self.metrics['load_time']:.4f} s")
        print("-" * 50)
        print_metric("TTFT (First Token)", self.metrics["ttft"])
        print_metric("TPOT", self.metrics["tpot"])
        print_metric("Total Gen Time", self.metrics["tgt"])
        print_metric("Throughput", self.metrics["tps"], unit=" t/s")
        print_metric("Tokenization", self.metrics["tokenization"])
        print_metric("Detokenization", self.metrics["detokenization"])
        print("-" * 50)
        if self.metrics["peak_vram"]:
            print(f"Peak VRAM Usage:     {max(self.metrics['peak_vram']):.2f} GB")
        print("="*50)
        LOG_FILE = os.path.join(LOG_PATH, args.log_file)
        print(f"Hardware log saved to: {os.path.abspath(LOG_FILE)}")
        
        # Export to CSV
        self.export_to_csv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, default="Qwen/Qwen2.5-1.5B", help="Path or HF ID of the model")
    parser.add_argument("--method", type=str, default="pretrained", choices=["pretrained", "pruned", "pruned-lora"])
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter (optional)")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup rounds")
    parser.add_argument("--benchmark", type=int, default=10, help="Number of benchmark rounds")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Maximum new tokens to generate")
    parser.add_argument("--log_file", type=str, default="benchmark_log.txt", help="Path to save hardware log")
    
    args = parser.parse_args()

    bench = LLMLatencyBenchmark()
    
    try:
        bench.load_model_tokenizer(model_path=args.model_path, method=args.method, adapter_path=args.adapter)
        bench.run_benchmark(warmup_rounds=args.warmup, benchmark_rounds=args.benchmark)
    except Exception as e:
        print(f"Test Failed: {e}")