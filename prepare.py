"""
Data preparation and evaluation for self-harm classification.

Usage:
    python prepare.py                  # prepare data and tokenizer
    python prepare.py --test-split 0.1 # use 10% for validation

Data is loaded from train_data.csv in the repo root.
Tokenizer is stored in ~/.cache/autoresearch/.
"""

import os
import sys
import time
import math
import argparse
import pickle
from collections import Counter

import pandas as pd
import rustbpe
import tiktoken
import torch
import torch.nn.functional as F

def verify_macos_env():
    import sys
    if sys.platform != "darwin":
        raise RuntimeError(f"This script requires macOS with Metal. Detected platform: {sys.platform}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS (Metal Performance Shaders) is not available. Ensure you are running on Apple Silicon with a compatible PyTorch build.")
    print("Environment verified: macOS detected with Metal (MPS) hardware acceleration available.")
    print()

verify_macos_env()

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 512        # max sequence length for classification
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
DATA_FILE = os.path.join(os.path.dirname(__file__), "train_data.csv")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
VOCAB_SIZE = 8192

# BPE split pattern (GPT-4 style)
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = ["<|pad|>", "<|cls|>", "<|sep|>", "<|unk|>"]
PAD_TOKEN = "<|pad|>"
CLS_TOKEN = "<|cls|>"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_cached_data = None

def load_data():
    """Load and cache the dataset from CSV."""
    global _cached_data
    if _cached_data is not None:
        return _cached_data

    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)

    # Use 'text' column, fall back to 'clean-text' if needed
    text_col = 'text' if 'text' in df.columns else 'clean-text'

    # Filter to rows with valid text and label
    df = df[[text_col, 'label']].dropna()
    df = df[df[text_col].str.len() > 0]

    texts = df[text_col].tolist()
    labels = df['label'].astype(int).tolist()

    print(f"Loaded {len(texts)} samples")
    print(f"Label distribution: {Counter(labels)}")

    _cached_data = (texts, labels)
    return _cached_data


def train_val_split(texts, labels, val_ratio=0.1, seed=42):
    """Split data into train and validation sets."""
    import random
    random.seed(seed)

    indices = list(range(len(texts)))
    random.shuffle(indices)

    val_size = int(len(indices) * val_ratio)
    val_indices = set(indices[:val_size])

    train_texts, train_labels = [], []
    val_texts, val_labels = [], []

    for i, (text, label) in enumerate(zip(texts, labels)):
        if i in val_indices:
            val_texts.append(text)
            val_labels.append(label)
        else:
            train_texts.append(text)
            train_labels.append(label)

    return (train_texts, train_labels), (val_texts, val_labels)


# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def text_iterator(texts, max_chars=100_000_000):
    """Yield documents for tokenizer training."""
    nchars = 0
    for text in texts:
        nchars += len(text)
        yield text
        if nchars >= max_chars:
            return


def train_tokenizer(texts):
    """Train BPE tokenizer using rustbpe, save as tiktoken pickle."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")

    if os.path.exists(tokenizer_pkl):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(texts), vocab_size_no_special, pattern=SPLIT_PATTERN)

    # Build tiktoken encoding from trained merges
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    # Save tokenizer
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    # Sanity check
    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper."""

    def __init__(self, enc):
        self.enc = enc
        self.pad_token_id = enc.encode_single_token(PAD_TOKEN)
        self.cls_token_id = enc.encode_single_token(CLS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_pad_token_id(self):
        return self.pad_token_id

    def get_cls_token_id(self):
        return self.cls_token_id

    def encode(self, text, max_length=MAX_SEQ_LEN):
        """Encode text with CLS token, truncate/pad to max_length."""
        ids = [self.cls_token_id] + self.enc.encode_ordinary(text)
        if len(ids) > max_length:
            ids = ids[:max_length]
        else:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        return ids

    def encode_batch(self, texts, max_length=MAX_SEQ_LEN):
        """Encode a batch of texts."""
        return [self.encode(text, max_length) for text in texts]

    def decode(self, ids):
        # Filter out special tokens for decoding
        ids = [i for i in ids if i not in (self.pad_token_id, self.cls_token_id)]
        return self.enc.decode(ids)


def make_dataloader(tokenizer, batch_size, split, val_ratio=0.1):
    """
    Create a dataloader for classification.
    Yields (input_ids, attention_mask, labels) batches.
    """
    texts, labels = load_data()
    (train_texts, train_labels), (val_texts, val_labels) = train_val_split(texts, labels, val_ratio)

    if split == "train":
        data_texts, data_labels = train_texts, train_labels
    else:
        data_texts, data_labels = val_texts, val_labels

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    import random
    indices = list(range(len(data_texts)))

    while True:
        if split == "train":
            random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            if len(batch_indices) < batch_size and split == "train":
                continue  # skip incomplete batches during training

            batch_texts = [data_texts[j] for j in batch_indices]
            batch_labels = [data_labels[j] for j in batch_indices]

            # Tokenize
            input_ids = tokenizer.encode_batch(batch_texts)
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = (input_ids != tokenizer.get_pad_token_id()).long()

            labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)

            yield input_ids, attention_mask, labels_tensor

        if split == "val":
            break  # only one pass for validation


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def fbeta_score(precision, recall, beta=0.5):
    """Calculate F-beta score. F0.5 weights precision 2x more than recall."""
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


@torch.no_grad()
def evaluate_f05(model, tokenizer, batch_size):
    """
    Evaluate model using F0.5 score for binary classification.
    F0.5 weights precision 2x more than recall.

    Returns dict with f05, precision, recall, accuracy.
    """
    model.eval()

    val_loader = make_dataloader(tokenizer, batch_size, "val")

    all_preds = []
    all_labels = []

    for input_ids, attention_mask, labels in val_loader:
        logits = model(input_ids, attention_mask)
        preds = (logits[:, 1] > logits[:, 0]).long()  # predict class 1 if logit[1] > logit[0]

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # Calculate metrics
    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / len(all_labels) if len(all_labels) > 0 else 0.0
    f05 = fbeta_score(precision, recall, beta=0.5)

    return {
        "f05": f05,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for self-harm classification")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio (default: 0.1)")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Load data
    texts, labels = load_data()
    print()

    # Step 2: Train tokenizer
    train_tokenizer(texts)
    print()

    # Step 3: Show split info
    (train_texts, train_labels), (val_texts, val_labels) = train_val_split(texts, labels, args.val_ratio)
    print(f"Train set: {len(train_texts)} samples")
    print(f"Val set: {len(val_texts)} samples")
    print()
    print("Done! Ready to train.")
