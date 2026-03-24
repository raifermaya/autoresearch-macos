"""
Self-harm classification training script using XLM-RoBERTa.
Usage: uv run train.py [--undersample] [--undersample-ratio 1.5] [--class-weight 1.0] ...
"""

import argparse

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import time
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaTokenizer, XLMRobertaModel

def verify_macos_env():
    if sys.platform != "darwin":
        raise RuntimeError(f"This script requires macOS with Metal. Detected platform: {sys.platform}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS (Metal Performance Shaders) is not available.")
    print("Environment verified: macOS with Metal (MPS) available.")
    print()

verify_macos_env()

from prepare import MAX_SEQ_LEN, TIME_BUDGET, load_data, train_val_split, evaluate_f05

# ---------------------------------------------------------------------------
# Command-line arguments (for autoresearch agent control)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Train XLM-RoBERTa classifier")
# Class imbalance handling
parser.add_argument("--undersample", action="store_true", default=True, help="Undersample majority class")
parser.add_argument("--no-undersample", dest="undersample", action="store_false", help="Disable undersampling")
parser.add_argument("--undersample-ratio", type=float, default=1.5, help="Ratio of neg:pos after undersampling")
parser.add_argument("--focal-loss", action="store_true", default=True, help="Use focal loss")
parser.add_argument("--no-focal-loss", dest="focal_loss", action="store_false", help="Use cross-entropy instead")
parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma parameter")
parser.add_argument("--class-weight", type=float, default=1.0, help="Weight for positive class in loss")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Focal Loss for class imbalance
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # class weights
        self.gamma = gamma  # focusing parameter

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ---------------------------------------------------------------------------
# XLM-RoBERTa Tokenizer Wrapper
# ---------------------------------------------------------------------------

class XLMRobertaTokenizerWrapper:
    """Wrapper to make XLM-RoBERTa tokenizer compatible with prepare.py interface."""

    def __init__(self, model_name="xlm-roberta-base"):
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def get_pad_token_id(self):
        return self.pad_token_id

    def get_cls_token_id(self):
        return self.cls_token_id

    def encode(self, text, max_length=MAX_SEQ_LEN):
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )
        return encoded['input_ids']

    def encode_batch(self, texts, max_length=MAX_SEQ_LEN):
        return [self.encode(text, max_length) for text in texts]

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# XLM-RoBERTa Classifier
# ---------------------------------------------------------------------------

class XLMRobertaClassifier(nn.Module):
    """XLM-RoBERTa for sequence classification with optional freezing."""

    def __init__(self, model_name="xlm-roberta-base", num_classes=2, dropout=0.1, freeze_base=True, unfreeze_top_n=2):
        super().__init__()
        self.roberta = XLMRobertaModel.from_pretrained(model_name)

        # Freeze base model for faster training, but unfreeze top N layers
        if freeze_base:
            for param in self.roberta.parameters():
                param.requires_grad = False
            # Unfreeze top N encoder layers
            num_layers = len(self.roberta.encoder.layer)
            for i in range(num_layers - unfreeze_top_n, num_layers):
                for param in self.roberta.encoder.layer[i].parameters():
                    param.requires_grad = True
            print(f"Base model partially frozen - top {unfreeze_top_n} layers + classifier trainable")

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Custom DataLoader
# ---------------------------------------------------------------------------

def make_xlm_dataloader(tokenizer_wrapper, batch_size, split, val_ratio=0.1, undersample=True, undersample_ratio=1.5):
    """Create dataloader using XLM-RoBERTa tokenizer with optional undersampling."""
    texts, labels = load_data()
    (train_texts, train_labels), (val_texts, val_labels) = train_val_split(texts, labels, val_ratio)

    if split == "train":
        data_texts, data_labels = train_texts, train_labels

        # Undersample majority class for training
        if undersample:
            import random
            random.seed(42)

            pos_indices = [i for i, l in enumerate(data_labels) if l == 1]
            neg_indices = [i for i, l in enumerate(data_labels) if l == 0]

            # Undersample negatives to match positives (or slight oversample ratio)
            target_neg = int(len(pos_indices) * undersample_ratio)
            sampled_neg = random.sample(neg_indices, min(target_neg, len(neg_indices)))

            balanced_indices = pos_indices + sampled_neg
            random.shuffle(balanced_indices)

            data_texts = [data_texts[i] for i in balanced_indices]
            data_labels = [data_labels[i] for i in balanced_indices]
            print(f"Undersampled training data: {len(data_texts)} samples (pos={len(pos_indices)}, neg={len(sampled_neg)})")
    else:
        data_texts, data_labels = val_texts, val_labels

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    import random
    indices = list(range(len(data_texts)))

    while True:
        if split == "train":
            random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            if len(batch_indices) < batch_size and split == "train":
                continue

            batch_texts = [data_texts[j] for j in batch_indices]
            batch_labels = [data_labels[j] for j in batch_indices]

            input_ids = tokenizer_wrapper.encode_batch(batch_texts)
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
            attention_mask = (input_ids != tokenizer_wrapper.get_pad_token_id()).long()
            labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)

            yield input_ids, attention_mask, labels_tensor

        if split == "val":
            break


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

MODEL_NAME = "xlm-roberta-base"
DROPOUT = 0.1
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4  # effective batch = 32
CLASSIFIER_LR = 1e-3    # high LR for classifier head
BASE_LR = 2e-5          # low LR for base (if unfrozen)
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
NUM_CLASSES = 2
FREEZE_BASE = True      # freeze base model for speed
UNFREEZE_TOP_N = 4  # unfreeze top N encoder layers

# Class imbalance handling (controlled via command-line args)
UNDERSAMPLE = args.undersample
UNDERSAMPLE_RATIO = args.undersample_ratio
USE_FOCAL_LOSS = args.focal_loss
FOCAL_GAMMA = args.focal_gamma
CLASS_WEIGHT_POSITIVE = args.class_weight

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.set_float32_matmul_precision("high")

device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_type)

print(f"Loading XLM-RoBERTa tokenizer and model...")
tokenizer = XLMRobertaTokenizerWrapper(MODEL_NAME)
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

model = XLMRobertaClassifier(MODEL_NAME, NUM_CLASSES, DROPOUT, freeze_base=FREEZE_BASE, unfreeze_top_n=UNFREEZE_TOP_N).to(device)
num_params = model.num_params()
trainable_params = model.num_trainable_params()
print(f"Total parameters: {num_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Optimizer for all trainable parameters
trainable_params_list = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(
    trainable_params_list,
    lr=CLASSIFIER_LR,
    weight_decay=WEIGHT_DECAY,
)

# Loss function with class weights
class_weights = torch.tensor([1.0, CLASS_WEIGHT_POSITIVE], device=device)
if USE_FOCAL_LOSS:
    criterion = FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA)
    print(f"Using Focal Loss with gamma={FOCAL_GAMMA}, class_weights={class_weights.tolist()}")
else:
    criterion = lambda logits, labels: F.cross_entropy(logits, labels, weight=class_weights)
    print(f"Using weighted CrossEntropy, class_weights={class_weights.tolist()}")

train_loader = make_xlm_dataloader(tokenizer, BATCH_SIZE, "train", undersample=UNDERSAMPLE, undersample_ratio=UNDERSAMPLE_RATIO)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Batch size: {BATCH_SIZE}, Accumulation steps: {ACCUMULATION_STEPS}, Effective batch: {BATCH_SIZE * ACCUMULATION_STEPS}")

# ---------------------------------------------------------------------------
# Training loop with gradient accumulation
# ---------------------------------------------------------------------------

t_start_training = time.time()
total_training_time = 0
step = 0
accum_step = 0
smooth_loss = 0

def sync_device(device_type):
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()

model.train()
optimizer.zero_grad()

for input_ids, attention_mask, labels in train_loader:
    sync_device(device_type)
    t0 = time.time()

    # LR schedule with warmup and cosine annealing
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    if progress < WARMUP_RATIO:
        lr_mult = progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    else:
        decay_progress = (progress - WARMUP_RATIO) / (1.0 - WARMUP_RATIO)
        lr_mult = 0.5 * (1.0 + math.cos(math.pi * decay_progress))

    current_lr = CLASSIFIER_LR * max(lr_mult, 0.01)
    for g in optimizer.param_groups:
        g['lr'] = current_lr

    # Forward
    logits = model(input_ids, attention_mask)
    loss = criterion(logits, labels) / ACCUMULATION_STEPS

    # Backward (accumulate gradients)
    loss.backward()

    accum_step += 1

    # Update weights after accumulation
    if accum_step % ACCUMULATION_STEPS == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        step += 1

    sync_device(device_type)
    t1 = time.time()
    dt = t1 - t0

    if accum_step > 5:
        total_training_time += dt

    # Logging
    loss_f = loss.item() * ACCUMULATION_STEPS
    smooth_loss = 0.9 * smooth_loss + 0.1 * loss_f
    debiased_loss = smooth_loss / (1 - 0.9**(accum_step + 1))
    pct_done = 100 * progress
    remaining = max(0, TIME_BUDGET - total_training_time)

    if accum_step % ACCUMULATION_STEPS == 0:
        print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_loss:.4f} | lr: {current_lr:.2e} | dt: {dt*1000:.0f}ms | remaining: {remaining:.0f}s    ", end="", flush=True)

    if accum_step == 1:
        gc.collect()
        gc.freeze()
        gc.disable()

    if accum_step > 5 and total_training_time >= TIME_BUDGET:
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
print(f"trainable_params: {trainable_params:,}")
print(f"model:            {MODEL_NAME}")
print(f"freeze_base:      {FREEZE_BASE}")
print(f"focal_loss:       {USE_FOCAL_LOSS}")
print(f"focal_gamma:      {FOCAL_GAMMA}")
print(f"class_weight:     {CLASS_WEIGHT_POSITIVE}")
print(f"undersample:      {UNDERSAMPLE}")
print(f"undersample_ratio:{UNDERSAMPLE_RATIO}")
