import time
import subprocess
import torch
import os
import numpy as np
import argparse
import gc
import csv
from pathlib import Path

from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from datasets import load_dataset

# --- Configuration ---
LOG_PATH = "logs"
CSV_PATH = os.path.join(LOG_PATH, "csv_results")
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(CSV_PATH, exist_ok=True)

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
            "peak_vram": [],    
            "input_tokens": [], # NEW: Tracks exact input size per round
            "output_tokens": [] # Tracks generation token size
        }

        self.model = None
        self.tokenizer = None 
        self.tegra_process = None
        self.model_path = None
        self.method = None
        self.prompts = []       # Holds the dynamically loaded dataset

    def start_hardware_logger(self):
        """Starts tegrastats in the background."""
        LOG_FILE = os.path.join(LOG_PATH, args.log_file)
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        print(f"Starting hardware logger -> {LOG_FILE}")
        cmd = ["tegrastats", "--logfile", LOG_FILE]
        self.tegra_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop_hardware_logger(self):
        """Stops tegrastats."""
        if self.tegra_process:
            self.tegra_process.terminate()
            self.tegra_process.wait()
            print("Hardware logger stopped.")
    
    def get_model_identifier(self):
        path = self.model_path.rstrip('/')
        model_name = os.path.basename(path)
        if model_name.endswith('.pt'):
            model_name = model_name[:-3]
        
        sparsity = ""
        if "sparsity" in path.lower():
            parts = path.split("_")
            for i, part in enumerate(parts):
                if part.lower() == "sparsity" and i + 1 < len(parts):
                    sparsity_num = parts[i+1].split("/")[0]
                    if sparsity_num.isdigit():
                        sparsity = f"_sparsity_{sparsity_num}"
                    break
        return model_name, sparsity
    
    def export_to_csv(self):
        model_name, sparsity = self.get_model_identifier()
        model_dir = os.path.join(CSV_PATH, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        filename = f"{model_name}_{self.method}{sparsity}_metrics.csv"
        csv_file = os.path.join(model_dir, filename)
        
        num_rounds = len(self.metrics["tokenization"])
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = [
                "Round", "Input_Tokens", "Output_Tokens", "Tokenization_Time(s)",
                "TTFT(s)", "TPOT(s)", "Total_Gen_Time(s)", "Throughput(t/s)",
                "Detokenization_Time(s)", "Peak_VRAM(GB)"
            ]
            writer.writerow(headers)
            
            for i in range(num_rounds):
                row = [
                    i + 1,
                    self.metrics["input_tokens"][i] if i < len(self.metrics["input_tokens"]) else "",
                    self.metrics["output_tokens"][i] if i < len(self.metrics["output_tokens"]) else "",
                    self.metrics["tokenization"][i] if i < len(self.metrics["tokenization"]) else "",
                    self.metrics["ttft"][i] if i < len(self.metrics["ttft"]) else "",
                    self.metrics["tpot"][i] if i < len(self.metrics["tpot"]) else "",
                    self.metrics["tgt"][i] if i < len(self.metrics["tgt"]) else "",
                    self.metrics["tps"][i] if i < len(self.metrics["tps"]) else "",
                    self.metrics["detokenization"][i] if i < len(self.metrics["detokenization"]) else "",
                    self.metrics["peak_vram"][i] if i < len(self.metrics["peak_vram"]) else ""
                ]
                writer.writerow(row)
        
        summary_file = os.path.join(model_dir, f"{model_name}_{self.method}{sparsity}_summary.csv")
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Mean", "Std Dev", "Min", "Max"])
            
            def write_summary_row(name, values):
                if values:
                    writer.writerow([
                        name, f"{np.mean(values):.6f}", f"{np.std(values):.6f}",
                        f"{np.min(values):.6f}", f"{np.max(values):.6f}"
                    ])
            
            writer.writerow(["Load_Time(s)", f"{self.metrics['load_time']:.6f}", "", "", ""])
            write_summary_row("Input_Tokens", self.metrics["input_tokens"])
            write_summary_row("Output_Tokens", self.metrics["output_tokens"])
            write_summary_row("Tokenization(s)", self.metrics["tokenization"])
            write_summary_row("TTFT(s)", self.metrics["ttft"])
            write_summary_row("TPOT(s)", self.metrics["tpot"])
            write_summary_row("Total_Gen_Time(s)", self.metrics["tgt"])
            write_summary_row("Throughput(t/s)", self.metrics["tps"])
            write_summary_row("Detokenization(s)", self.metrics["detokenization"])
            write_summary_row("Peak_VRAM(GB)", self.metrics["peak_vram"])
            
    def load_model_tokenizer(self, model_path, method, adapter_path=None):
        self.model_path = model_path
        self.method = method
        
        print(f"\n=== Phase 1: Loading Model ({method}) ===")
        if self.model:
            del self.model
            torch.cuda.empty_cache()
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        try:
            if method == "pretrained":
                self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map="auto")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            elif method == "pruned":
                pruned_dict = torch.load(model_path, map_location="cpu", weights_only=False)
                self.tokenizer = pruned_dict['tokenizer']
                self.model = pruned_dict['model']
                self.model.to("cuda")
            elif method == "pruned-lora":
                pruned_dict = torch.load(model_path, map_location="cpu", weights_only=False)
                self.tokenizer = pruned_dict['tokenizer']
                base_model = pruned_dict['model']
                self.model = PeftModel.from_pretrained(base_model, adapter_path, torch_dtype=torch.float16)
                self.model = self.model.merge_and_unload()
                self.model.to("cuda")
            else:
                raise ValueError(f"Unknown loading method: {method}")
            
            gc.collect()
            torch.cuda.empty_cache()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            torch.cuda.synchronize()
            self.metrics["load_time"] = time.perf_counter() - start_time
            print(f"Model loaded in {self.metrics['load_time']:.2f} seconds.")
        
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e           

    def load_calibration_dataset(self, num_samples, input_length):
        """Downloading WikiText2, padding/truncating to exact length."""
        print(f"\n=== Phase 2: Loading Dataset ({num_samples} samples, strict length: {input_length}) ===")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=False)
        
        self.prompts = []
        for item in dataset:
            text = item['text']
            if len(text.strip()) == 0:
                continue
                
            tokens = self.tokenizer(text, return_tensors="pt")
            if tokens.input_ids.shape[1] >= input_length:
                # Truncate to exact length
                truncated_ids = tokens.input_ids[:, :input_length]
                prompt_text = self.tokenizer.decode(truncated_ids[0], skip_special_tokens=True)
                self.prompts.append(prompt_text)
                
            if len(self.prompts) >= num_samples:
                break
                
        print(f"Successfully loaded {len(self.prompts)} standardized prompts.")

    def run_benchmark(self, warmup_rounds, output_length):
        if not self.model or not self.tokenizer or not self.prompts:
            raise RuntimeError("Model, tokenizer, and dataset must be loaded before benchmarking.")
        
        try:
            print(f"\n=== Phase 3: Warmup {warmup_rounds} Rounds ===")
            dummy_input = self.tokenizer("Hello, world!", return_tensors="pt").to("cuda")
            for _ in range(warmup_rounds):
                with torch.no_grad():
                    _ = self.model.generate(**dummy_input, max_new_tokens=10)
            print("Warmup completed.")
            
            self.start_hardware_logger()
            benchmark_rounds = len(self.prompts)
            
            print(f"\n=== Phase 4: Benchmarking {benchmark_rounds} Rounds ===")
            
            for i, prompt_text in enumerate(self.prompts):
                print(f"Round {i+1}/{benchmark_rounds}...", end="", flush=True)
                
                # A. Tokenization
                torch.cuda.synchronize()
                tt0 = time.perf_counter()
                inputs = self.tokenizer(prompt_text, return_tensors="pt").to("cuda")
                torch.cuda.synchronize()
                tt1 = time.perf_counter()
                
                self.metrics["tokenization"].append(tt1 - tt0)
                self.metrics["input_tokens"].append(inputs.input_ids.shape[1])
                
                # B. Generation (Strict Workload Enforcement)
                streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)
                gen_kwargs = dict(
                    **inputs,
                    streamer=streamer,
                    min_new_tokens=output_length, # FORCES EXACT WORKLOAD
                    max_new_tokens=output_length, # FORCES EXACT WORKLOAD
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
                
                torch.cuda.synchronize()
                t_gen_start = time.perf_counter()
                thread.start()
                
                generated_text = ""
                try:
                    first_token = next(iter(streamer))
                    torch.cuda.synchronize()
                    t_first = time.perf_counter()
                    
                    self.metrics["ttft"].append(t_first - t_gen_start)
                    generated_text += first_token
                    
                    for token in streamer:
                        generated_text += token
                    
                    torch.cuda.synchronize()
                    t_gen_end = time.perf_counter()
                    
                    tgt = t_gen_end - t_gen_start
                    self.metrics["tgt"].append(tgt)
                    
                    num_tokens = len(self.tokenizer.encode(generated_text))
                    self.metrics["output_tokens"].append(num_tokens)
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
                
                # D. VRAM Check
                peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
                self.metrics["peak_vram"].append(peak_mem)
                
                print(f" Done. (Input: {self.metrics['input_tokens'][-1]}, Output: {num_tokens}, TPS: {self.metrics['tps'][-1]:.2f})")
                
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user!")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
        finally:
            self.stop_hardware_logger()
            self.print_report()
                
    def print_report(self):
        print("\n" + "="*50)
        print("       BENCHMARK REPORT")
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
        print_metric("Input Tokens", self.metrics["input_tokens"], unit=" tokens")
        print_metric("Output Tokens", self.metrics["output_tokens"], unit=" tokens")
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
        
        self.export_to_csv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, default="Qwen/Qwen2.5-1.5B", help="Path or HF ID of the model")
    parser.add_argument("--method", type=str, default="pretrained", choices=["pretrained", "pruned", "pruned-lora"])
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter (optional)")
    
    # New matrix arguments
    parser.add_argument("--input_len", type=int, default=512, help="Exact number of input tokens per prompt")
    parser.add_argument("--output_len", type=int, default=128, help="Exact number of output tokens to generate")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of dataset samples to benchmark")
    
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup rounds")
    parser.add_argument("--log_file", type=str, default="benchmark_log.txt", help="Path to save hardware log")
    
    args = parser.parse_args()

    bench = LLMLatencyBenchmark()
    
    try:
        bench.load_model_tokenizer(model_path=args.model_path, method=args.method, adapter_path=args.adapter)
        bench.load_calibration_dataset(num_samples=args.num_samples, input_length=args.input_len)
        bench.run_benchmark(warmup_rounds=args.warmup, output_length=args.output_len)
    except Exception as e:
        print(f"Test Failed: {e}")