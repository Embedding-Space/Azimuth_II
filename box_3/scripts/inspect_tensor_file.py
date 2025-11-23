#!/usr/bin/env python3
"""
Tensor File Inspector

Prints a detailed inventory of tensors in .safetensors or .h5 files.
Shows: name, shape, dtype, size in MB, and (for HDF5) chunking info.

Usage:
    uv run python box_3/scripts/inspect_tensor_file.py <filepath>

Examples:
    uv run python box_3/scripts/inspect_tensor_file.py box_3/tensors/Thimble/thimble_6.h5
    uv run python box_3/scripts/inspect_tensor_file.py box_3/tensors/Qwen3-4B-Instruct-2507/W.safetensors
"""

import sys
from pathlib import Path


def format_size(nbytes):
    """Convert bytes to human-readable format."""
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024**2:
        return f"{nbytes / 1024:.2f} KB"
    elif nbytes < 1024**3:
        return f"{nbytes / 1024**2:.2f} MB"
    else:
        return f"{nbytes / 1024**3:.2f} GB"


def format_shape(shape):
    """Format shape tuple for display."""
    return f"({', '.join(str(s) for s in shape)})"


def inspect_safetensors(filepath):
    """Inspect a .safetensors file."""
    from safetensors import safe_open
    import numpy as np

    print(f"\n{'='*70}")
    print(f"SafeTensors File: {filepath.name}")
    print(f"{'='*70}\n")

    tensors = []
    total_size = 0

    with safe_open(filepath, framework="numpy") as f:
        for name in f.keys():
            # Get metadata without loading the tensor (avoids bfloat16 issues)
            slice_obj = f.get_slice(name)
            shape = slice_obj.get_shape()
            dtype_str = str(slice_obj.get_dtype())

            # Calculate size from shape and dtype
            # Map dtype strings to byte sizes
            dtype_sizes = {
                'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2,
                'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1,
                'U64': 8, 'U32': 4, 'U16': 2, 'U8': 1,
                'BOOL': 1
            }

            dtype_size = dtype_sizes.get(dtype_str, 4)  # Default to 4 if unknown
            nbytes = int(np.prod(shape) * dtype_size)
            total_size += nbytes

            tensors.append({
                'name': name,
                'shape': shape,
                'dtype': dtype_str,
                'size': nbytes
            })

    # Print table header
    print(f"{'Tensor Name':<30} {'Shape':<25} {'Dtype':<12} {'Size':<12}")
    print(f"{'-'*30} {'-'*25} {'-'*12} {'-'*12}")

    # Print each tensor
    for t in tensors:
        print(f"{t['name']:<30} {format_shape(t['shape']):<25} {t['dtype']:<12} {format_size(t['size']):<12}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"Total tensors: {len(tensors)}")
    print(f"Total size: {format_size(total_size)}")
    print(f"File size: {format_size(filepath.stat().st_size)}")
    print(f"{'='*70}\n")


def inspect_hdf5(filepath):
    """Inspect an HDF5 file."""
    import h5py
    import numpy as np

    print(f"\n{'='*90}")
    print(f"HDF5 File: {filepath.name}")
    print(f"{'='*90}\n")

    datasets = []
    total_size = 0

    with h5py.File(filepath, 'r') as f:
        for name in f.keys():
            dataset = f[name]
            shape = dataset.shape
            dtype = dataset.dtype
            nbytes = dataset.size * dataset.dtype.itemsize
            total_size += nbytes

            # Get chunking info
            chunks = dataset.chunks
            if chunks:
                chunk_str = format_shape(chunks)
                chunk_size = np.prod(chunks) * dataset.dtype.itemsize
                chunk_size_str = format_size(chunk_size)
                num_chunks = np.prod([np.ceil(s / c) for s, c in zip(shape, chunks)])
                chunk_info = f"{chunk_str} ({chunk_size_str}, {int(num_chunks)} chunks)"
            else:
                chunk_info = "contiguous"

            datasets.append({
                'name': name,
                'shape': shape,
                'dtype': str(dtype),
                'size': nbytes,
                'chunks': chunk_info
            })

    # Print table header
    print(f"{'Dataset Name':<20} {'Shape':<25} {'Dtype':<10} {'Size':<12} {'Chunking':<30}")
    print(f"{'-'*20} {'-'*25} {'-'*10} {'-'*12} {'-'*30}")

    # Print each dataset
    for d in datasets:
        print(f"{d['name']:<20} {format_shape(d['shape']):<25} {d['dtype']:<10} {format_size(d['size']):<12} {d['chunks']:<30}")

    # Print summary
    print(f"\n{'='*90}")
    print(f"Total datasets: {len(datasets)}")
    print(f"Total data size: {format_size(total_size)}")
    print(f"File size: {format_size(filepath.stat().st_size)}")
    overhead_pct = (filepath.stat().st_size - total_size) / filepath.stat().st_size * 100
    print(f"Overhead: {overhead_pct:.1f}% (metadata + compression)")
    print(f"{'='*90}\n")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    filepath = Path(sys.argv[1])

    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    suffix = filepath.suffix.lower()

    if suffix == '.safetensors':
        inspect_safetensors(filepath)
    elif suffix in ['.h5', '.hdf5']:
        inspect_hdf5(filepath)
    else:
        print(f"Error: Unsupported file type: {suffix}")
        print("Supported types: .safetensors, .h5, .hdf5")
        sys.exit(1)


if __name__ == "__main__":
    main()
