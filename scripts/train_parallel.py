#!/usr/bin/env python3
"""
Parallel training script for embedding evolution experiments.

Optimized for maximum vector-steps/sec throughput on high-end GPUs by running
multiple tiny models in parallel.

Usage:
    python train_parallel.py --seed 42 --output run_001
    python train_parallel.py --seed 43 --output run_002
    # etc.
"""

import argparse
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, TrainerCallback
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from safetensors.torch import save_file
import time

# ============================================================================
# HYPERPARAMETERS - Optimized for maximum vector-steps/sec
# ============================================================================

# Model architecture (tiny for maximum throughput)
VOCAB_SIZE = 128           # ASCII byte vocabulary
HIDDEN_DIM = 64            # Embedding dimension
N_LAYER = 2                # Transformer layers
N_HEAD = 2                 # Attention heads
MAX_SEQ_LEN = 128          # Context window

# Training
BATCH_SIZE = 512           # Per-device batch size
GRADIENT_ACCUMULATION = 1  # Effective batch = BATCH_SIZE × this
NUM_TRAIN_STEPS = 100000   # Total training steps
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01

# Initialization
INIT_MODE = "qwen"         # "normal" or "qwen"

# Checkpointing
SAVE_EVERY_N_STEPS = 1000  # Snapshot frequency

# Data loading
NUM_WORKERS = 0            # MUST be 0 for GPU dataset

# Data
CORPUS_PATH = "../data/training_corpus.txt"

# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Train tiny transformer for embedding evolution")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--output", type=str, required=True, help="Output directory name (e.g., run_001)")
    parser.add_argument("--steps", type=int, default=NUM_TRAIN_STEPS, help="Training steps")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    return parser.parse_args()


class GPUByteDataset(Dataset):
    """Zero-copy dataset with corpus pre-loaded to GPU."""
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


class EmbeddingSnapshotCallback(TrainerCallback):
    """Captures embedding snapshots during training."""
    def __init__(self, embedding_history, output_path, save_every_n, run_id):
        self.embedding_history = embedding_history
        self.output_path = output_path
        self.save_every_n = save_every_n
        self.run_id = run_id
        self.last_time = time.time()
        self.steps_since_print = 0

        output_path.parent.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step

        # Store in memory
        self.embedding_history[step] = model.transformer.wte.weight.data.clone()

        # Save to disk periodically
        should_save = (step % self.save_every_n == 0) or (step == args.max_steps)
        if should_save:
            save_file(
                {'embedding_history': self.embedding_history[:step+1].cpu()},
                self.output_path
            )

        # Print every 1000 steps
        self.steps_since_print += 1
        if step % 1000 == 0 and step > 0:
            elapsed = time.time() - self.last_time
            throughput = self.steps_since_print / elapsed
            vector_steps_per_sec = throughput * VOCAB_SIZE

            embeddings = self.embedding_history[step]
            centroid_norm = embeddings.mean(dim=0).norm().item()

            marker = "[SAVED]" if should_save else ""
            print(f"[{self.run_id}] [{step:5d}] {throughput:6.1f} it/s ({vector_steps_per_sec:8.0f} vec-step/s) | centroid: {centroid_norm:.6f} {marker}")

            self.last_time = time.time()
            self.steps_since_print = 0

        return control


def main():
    args = parse_args()

    # Set device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Enable TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Setup output
    output_dir = Path(f"../data/embeddings_{VOCAB_SIZE}vocab_{INIT_MODE}init_{args.output}")
    output_file = output_dir / "embedding_evolution.safetensors"

    print(f"\n{'='*80}")
    print(f"Run ID: {args.output}")
    print(f"Seed: {args.seed}")
    print(f"GPU: {args.gpu} ({torch.cuda.get_device_name(device)})")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    # Load corpus
    print(f"Loading corpus...")
    with open(CORPUS_PATH, 'r', encoding='ascii') as f:
        corpus_text = f.read()

    corpus_bytes = [b for b in corpus_text.encode('ascii') if b < VOCAB_SIZE]
    corpus_tensor = torch.tensor(corpus_bytes, dtype=torch.long, device=device)

    unique_bytes = len(set(corpus_bytes))
    dead_tokens = VOCAB_SIZE - unique_bytes

    print(f"  Total bytes: {len(corpus_bytes):,}")
    print(f"  Unique: {unique_bytes} / {VOCAB_SIZE}")
    print(f"  Dead tokens: {dead_tokens}")
    print(f"  Corpus on GPU: {corpus_tensor.numel() * corpus_tensor.element_size() / 1e6:.2f} MB\n")

    # Create dataset
    dataset = GPUByteDataset(corpus_tensor, MAX_SEQ_LEN)
    print(f"Dataset: {len(dataset):,} examples\n")

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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters\n")

    # Initialize embeddings
    if INIT_MODE == "qwen":
        print(f"Qwen initialization (singular unit vector)")
        with torch.no_grad():
            random_vector = torch.randn(HIDDEN_DIM, device=device)
            # random_vector = random_vector / random_vector.norm()
            model.transformer.wte.weight[:] = random_vector
        print(f"  All {VOCAB_SIZE} tokens → unit vector\n")
    else:
        print(f"Normal initialization\n")

    # Allocate embedding history
    embedding_history = torch.zeros(
        (args.steps + 1, VOCAB_SIZE, HIDDEN_DIM),
        dtype=torch.bfloat16,
        device=device
    )
    embedding_history[0] = model.transformer.wte.weight.data.clone()

    print(f"Embedding history: {embedding_history.shape} ({embedding_history.numel() * embedding_history.element_size() / 1e6:.1f} MB)\n")

    # Training args
    training_args = TrainingArguments(
        output_dir=f"./training_output_{args.output}",
        max_steps=args.steps,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        logging_steps=100,
        save_steps=args.steps + 1,  # Disable model checkpoints
        save_total_limit=0,
        dataloader_num_workers=NUM_WORKERS,
        dataloader_pin_memory=False,
        bf16=True,
        tf32=True,
        seed=args.seed,
        report_to="none",
        disable_tqdm=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[EmbeddingSnapshotCallback(
            embedding_history,
            output_file,
            SAVE_EVERY_N_STEPS,
            args.output
        )],
    )

    # Train
    print(f"{'='*80}")
    print(f"Starting training...")
    print(f"{'='*80}\n")

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    avg_throughput = args.steps / elapsed
    avg_vector_steps = avg_throughput * VOCAB_SIZE

    print(f"\n{'='*80}")
    print(f"✓ Training complete")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Throughput: {avg_throughput:.1f} it/s")
    print(f"  Vector-steps/sec: {avg_vector_steps:,.0f}")
    print(f"  Output: {output_file}")
    print(f"{'='*80}\n")

    # Final save
    save_file({'embedding_history': embedding_history}, output_file)


if __name__ == "__main__":
    main()
