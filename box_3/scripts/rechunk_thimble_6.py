#!/usr/bin/env python3
"""
Rechunk Thimble 6 HDF5 for Fast Temporal Access

The original thimble_6.h5 is chunked (1, 10000, 64) - optimized for writing
one timestep at a time. This makes loading the full temporal sequence slow.

This script rechunks to (1500, 10000, 64) - ~2GB chunks (well under HDF5's 4GB limit).
No compression. Maximum speed for loading full tensors.

Usage:
    uv run python box_3/scripts/rechunk_thimble_6.py

Output:
    box_3/tensors/Thimble/thimble_6_chunky.h5
"""

import h5py
import numpy as np
from pathlib import Path
import time

# Paths
INPUT_PATH = Path("box_3/tensors/Thimble/thimble_6.h5")
OUTPUT_PATH = Path("box_3/tensors/Thimble/thimble_6_chunky.h5")

# Chunking strategy for temporal tensors
# Each temporal tensor: (6001, 10000, 64) float16 = 7.68 GB total
# Chunk size: (1500, 10000, 64) float16 = 1.92 GB per chunk < 4GB HDF5 limit
# Result: 5 chunks per tensor (1500×4 + 1001 for last chunk)
TEMPORAL_CHUNK = (1250, 10000, 64)

# Compression strategy
# Try lightweight compression: gzip level 1 (fastest)
# If compression ratio ~2:1 with minimal performance hit, worth it
# Set to None for no compression (max speed)
COMPRESSION = None
COMPRESSION_LEVEL = 1  # 1 = fastest, 9 = best compression

def rechunk_dataset(input_file, output_file, dataset_name, chunk_shape):
    """
    Load a dataset from input file, write to output file with new chunking.
    Memory-efficient: only one tensor in RAM at a time.
    """
    print(f"\nProcessing {dataset_name}...")

    # Load from input
    load_start = time.time()
    with h5py.File(input_file, 'r') as f:
        print(f"  Loading from disk...")
        data = f[dataset_name][:]
        original_shape = data.shape
        original_dtype = data.dtype

    load_time = time.time() - load_start
    size_mb = data.nbytes / 1e6
    print(f"  Loaded {original_shape} {original_dtype} ({size_mb:.1f} MB in {load_time:.1f}s)")

    # Write to output with new chunking
    write_start = time.time()
    with h5py.File(output_file, 'a') as f:
        if chunk_shape is None:
            print(f"  Writing contiguous (no chunking)...")
        else:
            print(f"  Writing with chunking {chunk_shape}...")

        f.create_dataset(
            dataset_name,
            data=data,
            chunks=chunk_shape,
            compression=COMPRESSION,
            compression_opts=COMPRESSION_LEVEL if COMPRESSION else None,
            dtype=original_dtype
        )

    write_time = time.time() - write_start
    print(f"  ✓ {dataset_name} complete (wrote in {write_time:.1f}s)")

    # Explicitly free memory before next iteration
    del data

def main():
    print("=" * 60)
    print("Rechunking Thimble 6 for Whimsy and Speed 🧀")
    print("=" * 60)

    # Remove output file if it exists (fresh start)
    if OUTPUT_PATH.exists():
        print(f"\nRemoving existing {OUTPUT_PATH.name}...")
        OUTPUT_PATH.unlink()

    # Copy file-level attributes first
    print("\n--- Copying File Attributes ---")
    with h5py.File(INPUT_PATH, 'r') as f_in:
        with h5py.File(OUTPUT_PATH, 'a') as f_out:
            # Copy all root-level attributes
            for key, value in f_in.attrs.items():
                f_out.attrs[key] = value
                print(f"  {key}: {value}")

    # Temporal tensors: chunk as ~2GB blocks (well under 4GB HDF5 limit)
    temporal_datasets = ['W', 'grad_W', 'momentum_W', 'variance_W']

    print("\n--- Temporal Tensors (~2GB chunks) ---")
    print(f"Chunk shape: {TEMPORAL_CHUNK} = ~1.92 GB per chunk")
    if COMPRESSION:
        print(f"Compression: {COMPRESSION} level {COMPRESSION_LEVEL}")
    for dataset_name in temporal_datasets:
        rechunk_dataset(INPUT_PATH, OUTPUT_PATH, dataset_name, TEMPORAL_CHUNK)

    # Metadata: no chunking (contiguous storage, small data)
    metadata_datasets = ['dead_ids', 'dead_mask', 'live_ids', 'live_mask', 'losses']

    print("\n--- Metadata (contiguous) ---")
    for dataset_name in metadata_datasets:
        rechunk_dataset(INPUT_PATH, OUTPUT_PATH, dataset_name, None)

    print("\n" + "=" * 60)
    print("✓ Rechunking complete!")
    print(f"✓ Output: {OUTPUT_PATH}")
    print("=" * 60)

    # Verify file sizes
    input_size = INPUT_PATH.stat().st_size / 1e9
    output_size = OUTPUT_PATH.stat().st_size / 1e9
    print(f"\nFile sizes:")
    print(f"  Original:  {input_size:.2f} GB")
    print(f"  Rechunked: {output_size:.2f} GB")
    print(f"  Ratio:     {output_size/input_size:.2f}x")

    if COMPRESSION:
        compression_ratio = input_size / output_size if output_size > 0 else 0
        print(f"\nCompression ({COMPRESSION} level {COMPRESSION_LEVEL}):")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Space saved: {(input_size - output_size):.2f} GB ({(1 - output_size/input_size)*100:.1f}%)")
    else:
        print(f"\n(Larger size expected - no compression, optimized for speed)")

if __name__ == "__main__":
    main()
