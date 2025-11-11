#!/usr/bin/env python3
"""
Training worker: f32 → bf16 initialization.

Initializes embeddings in float32, then converts to bfloat16.
Records embeddings only (no gradients/momentum).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments

# Model architecture
VOCAB_SIZE = 128
HIDDEN_DIM = 64
N_LAYER = 2
N_HEAD = 2
MAX_SEQ_LEN = 128

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01

# Initialization
INIT_SIGMA = 1e-5

# Data
CORPUS_PATH = "../data/training_corpus.txt"


class ByteDataset(Dataset):
    def __init__(self, corpus_tensor, max_seq_len):
        self.corpus = corpus_tensor
        self.max_seq_len = max_seq_len

    def __len__(self):
        return max(0, len(self.corpus) - self.max_seq_len)

    def __getitem__(self, idx):
        chunk = self.corpus[idx : idx + self.max_seq_len + 1]
        return {
            'input_ids': chunk[:-1],
            'labels': chunk[1:]
        }


class EmbeddingRecorder:
    """Records embedding snapshots at each step."""

    def __init__(self, vocab_size, hidden_dim, num_steps, device):
        self.device = device
        # Allocate storage on GPU
        self.embeddings = torch.zeros(
            (num_steps + 1, vocab_size, hidden_dim),
            dtype=torch.bfloat16,
            device=device
        )
        self.current_step = 0

    def record_initial(self, model):
        """Record step 0."""
        self.embeddings[0] = model.transformer.wte.weight.data.clone()
        self.current_step = 1

    def record_step(self, model):
        """Record current step."""
        if self.current_step < len(self.embeddings):
            self.embeddings[self.current_step] = model.transformer.wte.weight.data.clone()
            self.current_step += 1


class RecordingTrainer(Trainer):
    """Custom trainer that records embeddings."""

    def __init__(self, recorder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder = recorder

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to record embeddings after each step."""
        loss = super().training_step(model, inputs, num_items_in_batch)
        self.recorder.record_step(model)
        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--test-id", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()

    # Set device (always GPU 0 for this experiment)
    if not torch.cuda.is_available():
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cuda:0')

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Enable TF32 if CUDA
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Setup output
    output_dir = Path("../data") / f"experiment_test{args.test_id:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.output}.safetensors"

    print(f"{'='*80}")
    print(f"F32→BF16 Training Worker")
    print(f"{'='*80}")
    print(f"Test ID: {args.test_id}")
    print(f"Seed: {args.seed}")
    print(f"Steps: {args.steps}")
    print(f"Output: {output_file}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    # Load corpus
    print(f"Loading corpus...")
    with open(CORPUS_PATH, 'r', encoding='ascii') as f:
        corpus_text = f.read()

    corpus_bytes = [b for b in corpus_text.encode('ascii') if b < VOCAB_SIZE]
    corpus_tensor = torch.tensor(corpus_bytes, dtype=torch.long, device=device)

    unique_bytes = set(corpus_bytes)
    dead_token_ids = sorted(set(range(VOCAB_SIZE)) - unique_bytes)
    live_token_ids = sorted(unique_bytes)

    print(f"  Live tokens: {len(live_token_ids)} / {VOCAB_SIZE}")
    print(f"  Dead tokens: {len(dead_token_ids)} / {VOCAB_SIZE}\n")

    # Create dataset
    dataset = ByteDataset(corpus_tensor, MAX_SEQ_LEN)

    # Create model
    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=MAX_SEQ_LEN,
        n_embd=HIDDEN_DIM,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        tie_word_embeddings=True,
    )

    model = GPT2LMHeadModel(config)
    model = model.to(torch.bfloat16).to(device)

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters\n")

    # F32→BF16 initialization
    print(f"Initializing: f32 (σ={INIT_SIGMA:.2e}) → bf16")
    with torch.no_grad():
        # Generate in float32
        random_vector = torch.randn(HIDDEN_DIM, dtype=torch.float32, device=device)
        random_vector = random_vector / random_vector.norm()
        noise = torch.randn(VOCAB_SIZE, HIDDEN_DIM, dtype=torch.float32, device=device) * INIT_SIGMA
        init_f32 = random_vector + noise

        # Convert to bfloat16 AFTER addition
        init_bf16 = init_f32.to(torch.bfloat16)
        model.transformer.wte.weight[:] = init_bf16

    print(f"  ✓ Initialized\n")

    # Setup recorder
    recorder = EmbeddingRecorder(VOCAB_SIZE, HIDDEN_DIM, args.steps, device)
    recorder.record_initial(model)

    print(f"Embedding history: {recorder.embeddings.shape} "
          f"({recorder.embeddings.numel() * recorder.embeddings.element_size() / 1e6:.1f} MB)\n")

    # Training args
    training_args = TrainingArguments(
        output_dir=f"/tmp/training_output_{args.output}",
        max_steps=args.steps,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        logging_steps=args.steps + 1,  # No logging
        save_steps=args.steps + 1,  # No checkpoints
        save_total_limit=0,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        bf16=True,
        seed=args.seed,
        report_to="none",
        disable_tqdm=True,
    )

    # Create trainer
    trainer = RecordingTrainer(
        recorder=recorder,
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Train
    print(f"{'='*80}")
    print(f"Training...")
    print(f"{'='*80}\n")

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"✓ Training complete ({elapsed:.1f}s, {args.steps/elapsed:.1f} it/s)")
    print(f"{'='*80}\n")

    # Save
    print(f"Saving embeddings...")
    save_dict = {
        'test_id': torch.tensor(args.test_id, dtype=torch.int32),
        'seed': torch.tensor(args.seed, dtype=torch.int32),
        'init_method': torch.tensor(0, dtype=torch.int32),  # 0=f32→bf16
        'steps': torch.tensor(args.steps, dtype=torch.int32),
        'dead_token_ids': torch.tensor(dead_token_ids, dtype=torch.long),
        'live_token_ids': torch.tensor(live_token_ids, dtype=torch.long),
        'embeddings': recorder.embeddings.cpu(),  # Move to CPU for saving
        'init_sigma': torch.tensor(INIT_SIGMA, dtype=torch.float32),
    }

    save_file(save_dict, output_file)
    file_size_mb = output_file.stat().st_size / 1e6

    print(f"  ✓ Saved: {output_file}")
    print(f"  Size: {file_size_mb:.1f} MB\n")


if __name__ == "__main__":
    main()
