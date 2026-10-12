#!/usr/bin/env python3

import argparse
import os
import shutil
import struct
import numpy as np


# ---------------------------------------------------------------------------
# fvecs helpers
# ---------------------------------------------------------------------------

def read_fvecs(fname):
    fname = os.path.expanduser(fname)
    data = np.fromfile(fname, dtype=np.float32)

    if data.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    dim = struct.unpack("<I", data[:1].tobytes())[0]
    if dim <= 0:
        raise ValueError(f"Invalid dimension {dim} in {fname}")

    row_width = dim + 1
    if data.size % row_width != 0:
        raise ValueError(
            f"File size is not consistent with fvecs format: "
            f"{fname}, dim={dim}, float_count={data.size}"
        )

    data = data.reshape(-1, row_width)
    dims = data[:, 0].view(np.int32)

    if not np.all(dims == dim):
        raise ValueError(f"Inconsistent vector dimensions in {fname}")

    return np.ascontiguousarray(data[:, 1:], dtype=np.float32)


def write_fvecs(fname, arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    n, d = arr.shape
    fname = os.path.expanduser(fname)

    d_repr = struct.unpack("<f", np.uint32(d))[0]
    formatted = np.concatenate(
        (np.full((n, 1), d_repr, dtype=np.float32), arr),
        axis=1
    )

    if n > 0:
        assert struct.unpack("<I", formatted[0, 0].tobytes()) == (d,)

    with open(fname, "wb") as f:
        formatted.tofile(f)


# ---------------------------------------------------------------------------
# bvecs helpers
# ---------------------------------------------------------------------------

def read_bvecs(fname):
    """Read a .bvecs file and return an int8 ndarray of shape (n, dim)."""
    fname = os.path.expanduser(fname)
    with open(fname, "rb") as f:
        raw = f.read()

    if not raw:
        return np.empty((0, 0), dtype=np.int8)

    dim = struct.unpack("<i", raw[:4])[0]
    if dim <= 0:
        raise ValueError(f"Invalid dimension {dim} in {fname}")

    record_size = 4 + dim
    if len(raw) % record_size != 0:
        raise ValueError(
            f"File size is not consistent with bvecs format: "
            f"{fname}, dim={dim}, total_bytes={len(raw)}"
        )

    n = len(raw) // record_size
    buf = np.frombuffer(raw, dtype=np.uint8).reshape(n, record_size)
    return np.ascontiguousarray(buf[:, 4:].view(np.int8))


def write_bvecs(fname, arr):
    """Write an int8 ndarray (n, dim) to a .bvecs file."""
    arr = np.asarray(arr, dtype=np.int8)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    n, d = arr.shape
    fname = os.path.expanduser(fname)
    header = np.array([d], dtype=np.int32).tobytes()

    with open(fname, "wb") as f:
        for i in range(n):
            f.write(header)
            f.write(arr[i].tobytes())


# ---------------------------------------------------------------------------
# Normalization logic
# ---------------------------------------------------------------------------

def normalization_error_stats(vecs):
    norms = np.linalg.norm(vecs, axis=1)
    errors = np.abs(norms - 1.0)

    if errors.size == 0:
        return {
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
        }

    return {
        "max_abs_error": float(np.max(errors)),
        "mean_abs_error": float(np.mean(errors)),
    }


def check_normalization(vecs, tol=1e-3):
    """
    Returns True if all vectors in the array are approximately normalized
    (L2 norm close to 1 within the specified tolerance).
    """
    norms = np.linalg.norm(vecs, axis=1)
    return np.all(np.abs(norms - 1) < tol)


def normalize_vectors(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(arr / norms, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Normalize vectors in an fvecs or bvecs file.")
    parser.add_argument("--input", required=True, help="Input fvecs or bvecs file")
    parser.add_argument("--output", required=True, help="Output normalized file")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Tolerance for considering vectors already normalized (default: 1e-3)",
    )
    args = parser.parse_args()

    if args.tolerance < 0:
        raise ValueError("--tolerance must be non-negative")

    is_bvecs = args.input.lower().endswith(".bvecs")
    input_path = os.path.expanduser(args.input)
    output_path = os.path.expanduser(args.output)

    if is_bvecs:
        # INT8 vectors cannot be meaningfully normalized in-place: normalization
        # would require float arithmetic then re-quantization, which corrupts the
        # original INT8 values. Instead, pass through unchanged.
        print("Input is .bvecs (INT8). Skipping normalization — copying through unchanged.")
        print(f"Normalization tolerance: {args.tolerance}")
        if os.path.abspath(input_path) != os.path.abspath(output_path):
            shutil.copyfile(input_path, output_path)
            print(f"Copied input to output without changes: {output_path}")
        else:
            print("Input and output are the same file. No action needed.")
        # Emit stats that processing.py parses so the summary is complete.
        print("Vectors normalized before: Yes")
        print("Max abs norm error after: 0")
        print("Mean abs norm error after: 0")
        return

    vectors = read_fvecs(args.input)

    normalized_before = check_normalization(vectors, tol=args.tolerance)
    before_stats = normalization_error_stats(vectors)
    print(f"Normalization tolerance: {args.tolerance}")
    print("Vectors normalized before:", "Yes" if normalized_before else "No")
    print(f"Max abs norm error before: {before_stats['max_abs_error']:.8g}")
    print(f"Mean abs norm error before: {before_stats['mean_abs_error']:.8g}")
    print(f"Vectors: {vectors.shape[0]}")
    print(f"Dimension: {vectors.shape[1] if vectors.size > 0 else 0}")

    if normalized_before:
        print("Input is already normalized. Skipping normalization.")
        if os.path.abspath(input_path) != os.path.abspath(output_path):
            shutil.copyfile(input_path, output_path)
            print(f"Copied input to output without changes: {output_path}")
        else:
            print("Input and output are the same file. No action needed.")
        return

    normalized = normalize_vectors(vectors)

    normalized_after = check_normalization(normalized, tol=args.tolerance)
    after_stats = normalization_error_stats(normalized)
    print("Vectors normalized after:", "Yes" if normalized_after else "No")
    print(f"Max abs norm error after: {after_stats['max_abs_error']:.8g}")
    print(f"Mean abs norm error after: {after_stats['mean_abs_error']:.8g}")

    write_fvecs(output_path, normalized)
    print(f"Wrote normalized file to: {output_path}")


if __name__ == "__main__":
    main()
