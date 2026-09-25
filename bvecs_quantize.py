#!/usr/bin/env python3
"""
Convert a .fvecs file to signed int8 .bvecs format using linear symmetric quantization.
Scale factor is determined by the maximum absolute value across the dataset.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from bvecs_writer import append_bvecs, count_bvecs
from fvecs_writer import count_fvecs

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def iter_fvecs_batches(path: Path, batch_size: int = 32768):
    """
    Stream batches of float32 vectors from an fvecs file.
    """
    raw = np.memmap(path, dtype=np.int32, mode="r")
    if raw.size == 0:
        return

    dim = int(raw[0])
    row_size = dim + 1
    num_vectors = raw.size // row_size
    matrix = raw.reshape(num_vectors, row_size)

    for start in range(0, num_vectors, batch_size):
        end = min(start + batch_size, num_vectors)
        batch_rows = matrix[start:end]
        yield np.asarray(batch_rows[:, 1:].view(np.float32), dtype=np.float32)


def compute_max_abs(fvecs_paths: list[Path], batch_size: int = 65536) -> float:
    """
    Compute the maximum absolute value across all given fvecs files.
    """
    global_max = 0.0
    for path in fvecs_paths:
        if not path.exists():
            raise FileNotFoundError(f"fvecs file not found: {path}")
        for batch in iter_fvecs_batches(path, batch_size=batch_size):
            if batch.size > 0:
                batch_max = float(np.max(np.abs(batch)))
                if batch_max > global_max:
                    global_max = batch_max
    return global_max


def quantize_fvecs_to_bvecs(
    input_fvecs: Path,
    output_bvecs: Path,
    scale_factor: float,
    batch_size: int = 32768,
    overwrite: bool = True,
) -> int:
    """
    Quantize an fvecs file to bvecs using linear symmetric quantization:
        int8_val = np.clip(np.round(float_val * scale_factor), -128, 127).astype(np.int8)
    """
    output_bvecs = Path(output_bvecs)
    if output_bvecs.exists():
        if overwrite:
            output_bvecs.unlink()
        else:
            count, _ = count_bvecs(output_bvecs)
            return count

    total_written = 0
    for batch in iter_fvecs_batches(input_fvecs, batch_size=batch_size):
        quantized = np.clip(np.round(batch * scale_factor), -128, 127).astype(np.int8)
        append_bvecs(output_bvecs, quantized)
        total_written += len(quantized)

    return total_written


def main():
    parser = argparse.ArgumentParser(description="Quantize an fvecs file to int8 bvecs")
    parser.add_argument("--input", type=Path, required=True, help="Input fvecs file")
    parser.add_argument("--output", type=Path, required=True, help="Output bvecs file")
    parser.add_argument("--meta_out", type=Path, default=None, help="Output metadata JSON file")
    parser.add_argument("--batch_size", type=int, default=32768, help="Batch size for processing")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    logger.info("Scanning max_abs across %s...", args.input)
    max_abs = compute_max_abs([args.input], batch_size=args.batch_size)
    if max_abs == 0.0:
        logger.warning("Dataset contains only exact zeros! Scale factor set to 1.0.")
        scale_factor = 1.0
    else:
        scale_factor = 127.0 / max_abs

    logger.info("Computed max_abs: %f, scale_factor: %f", max_abs, scale_factor)

    logger.info("Quantizing vectors to %s...", args.output)
    total_count = quantize_fvecs_to_bvecs(
        args.input, args.output, scale_factor, batch_size=args.batch_size, overwrite=args.overwrite
    )
    logger.info("Quantized %d vectors.", total_count)

    meta = {
        "max_abs": max_abs,
        "scale_factor": scale_factor,
        "mode": "symmetric_max_abs",
        "total_count": total_count,
    }

    if args.meta_out:
        args.meta_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.meta_out, "w") as f:
            json.dump(meta, f, indent=2)

    print(json.dumps(meta))


if __name__ == "__main__":
    main()
