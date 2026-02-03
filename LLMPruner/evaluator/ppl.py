import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
import logging

# Matikan warning tokenizer yang berisik soal sequence length
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

class PPLMetric:
    def __init__(self, model, tokenizer, datasets, max_seq_len, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.max_seq_len = max_seq_len
        self.device = device
    
    def __str__(self):
        res = ""
        for dataset in self.datasets:
            ppl = self.evaluate(dataset)
            res += f"\n {dataset} PPL: {ppl:.2f}"
        return res

    def evaluate(self, dataset_name):
        # 1. Load Dataset
        if dataset_name == 'wikitext2':
            test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        elif dataset_name == 'ptb':
            try:
                test = load_dataset('ptb_text_only', 'penn_treebank', split='test')
            except:
                test = load_dataset('ptb_text_only', split='test')
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not supported")

        # 2. [FIX] Deteksi Nama Kolom (text vs sentence)
        if 'text' in test.column_names:
            col_name = 'text'
        elif 'sentence' in test.column_names:
            col_name = 'sentence'
        else:
            # Fallback: ambil kolom pertama apapun namanya
            col_name = test.column_names[0]
            print(f"Warning: Column 'text'/'sentence' not found. Using '{col_name}'")

        # 3. Tokenisasi
        # Kita gabungkan semua text jadi satu string raksasa
        raw_text = "\n\n".join(test[col_name])
        
        # truncation=False: Biarkan panjangnya melebihi batas model (nanti kita potong-potong sendiri)
        # verbose=False: Supaya tidak muncul warning "Token indices sequence length..."
        encodings = self.tokenizer(raw_text, return_tensors='pt', truncation=False, verbose=False)

        # 4. Sliding Window Evaluation (Hemat Memori)
        # Variable 'encodings' TETAP DI CPU, jangan dipindah ke .to(device)
        
        max_length = self.max_seq_len
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        
        # Loop chunk-by-chunk
        pbar = tqdm(range(0, seq_len, stride), desc=f"Eval {dataset_name}")
        
        for begin_loc in pbar:
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            
            # Ambil potongan kecil dari CPU
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            
            # Pindahkan potongan kecil ke GPU
            input_ids = input_ids.to(self.device)
            
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100 # Masking context history

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        
        if len(nlls) > 0:
            ppl = torch.exp(torch.stack(nlls).mean())
            return ppl.item()
        else:
            return float('inf')