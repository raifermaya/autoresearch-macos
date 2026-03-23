"""
Self-harm classification training script. Single-GPU, single-file.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import time
from dataclasses import dataclass, asdict

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

def verify_macos_env():
    if sys.platform != "darwin":
        raise RuntimeError(f"This script requires macOS with Metal. Detected platform: {sys.platform}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS (Metal Performance Shaders) is not available.")
    print("Environment verified: macOS with Metal (MPS) available.")
    print()

verify_macos_env()

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_f05

# ---------------------------------------------------------------------------
# Classifier Model
# ---------------------------------------------------------------------------

@dataclass
class ClassifierConfig:
    sequence_len: int = 512
    vocab_size: int = 8192
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    num_classes: int = 2
    dropout: float = 0.1


class Classifier(nn.Module):
    """CNN-based text classifier - fast and good at detecting local patterns."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Multi-scale CNN: detect patterns of different sizes (odd kernels for same padding)
        self.convs = nn.ModuleList([
            nn.Conv1d(config.n_embd, 200, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 9]  # various n-gram sizes
        ])

        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(200 * 4, config.num_classes)  # 4 conv outputs

    def forward(self, input_ids, attention_mask=None):
        x = self.wte(input_ids)  # [B, T, C]
        x = x.transpose(1, 2)  # [B, C, T] for conv1d

        # Apply each conv and max pool
        conv_outputs = []
        for conv in self.convs:
            h = F.gelu(conv(x))  # [B, 200, T]
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(1).float()  # [B, 1, T]
                h = h.masked_fill(mask == 0, float('-inf'))
            h = h.max(dim=2)[0]  # [B, 200]
            conv_outputs.append(h)

        pooled = torch.cat(conv_outputs, dim=1)  # [B, 800]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

DEPTH = 3
N_HEAD = 6
N_EMBD = 384
DROPOUT = 0.1
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.2
NUM_CLASSES = 2

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.set_float32_matmul_precision("high")

device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_type)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

config = ClassifierConfig(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    n_layer=DEPTH,
    n_head=N_HEAD,
    n_embd=N_EMBD,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT,
)
print(f"Model config: {asdict(config)}")

model = Classifier(config).to(device)
num_params = model.num_params()
print(f"Parameters: {num_params:,}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999),
)

train_loader = make_dataloader(tokenizer, BATCH_SIZE, "train")

print(f"Time budget: {TIME_BUDGET}s")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
total_training_time = 0
step = 0
smooth_loss = 0

def sync_device(device_type):
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()

model.train()
for input_ids, attention_mask, labels in train_loader:
    sync_device(device_type)
    t0 = time.time()

    # LR schedule with cosine annealing
    import math
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    if progress < WARMUP_RATIO:
        lr_mult = progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    else:
        decay_progress = (progress - WARMUP_RATIO) / (1.0 - WARMUP_RATIO)
        lr_mult = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    lr = LEARNING_RATE * max(lr_mult, 0.1)
    for g in optimizer.param_groups:
        g['lr'] = lr

    # Forward
    logits = model(input_ids, attention_mask)
    loss = F.cross_entropy(logits, labels)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    sync_device(device_type)
    t1 = time.time()
    dt = t1 - t0

    if step > 5:
        total_training_time += dt

    # Logging
    loss_f = loss.item()
    smooth_loss = 0.9 * smooth_loss + 0.1 * loss_f
    debiased_loss = smooth_loss / (1 - 0.9**(step + 1))
    pct_done = 100 * progress
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_loss:.4f} | lr: {lr:.2e} | dt: {dt*1000:.0f}ms | remaining: {remaining:.0f}s    ", end="", flush=True)

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()

    step += 1

    if step > 5 and total_training_time >= TIME_BUDGET:
        break

print()

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

model.eval()
metrics = evaluate_f05(model, tokenizer, BATCH_SIZE)

t_end = time.time()
if device_type == "cuda":
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
else:
    peak_vram_mb = 0.0

print("---")
print(f"f05:              {metrics['f05']:.6f}")
print(f"precision:        {metrics['precision']:.6f}")
print(f"recall:           {metrics['recall']:.6f}")
print(f"accuracy:         {metrics['accuracy']:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
