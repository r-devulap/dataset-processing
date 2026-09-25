#!/usr/bin/env python3

import argparse
import os
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
    # Parse using numpy: view as uint8, reshape, skip the 4-byte header per row.
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
# Shared logic
# ---------------------------------------------------------------------------

def count_zero_vectors(arr, tol=0.0):
    norms = np.linalg.norm(arr.astype(np.float32), axis=1)
    return int(np.sum(norms <= tol))


def remove_zero_vectors(arr, tol=0.0):
    norms = np.linalg.norm(arr.astype(np.float32), axis=1)
    keep_mask = norms > tol
    return np.ascontiguousarray(arr[keep_mask])


def main():
    parser = argparse.ArgumentParser(
        description="Remove vectors whose L2 norm is at or below a tolerance from an fvecs or bvecs file."
    )
    parser.add_argument("--input", required=True, help="Input fvecs or bvecs file")
    parser.add_argument("--output", required=True, help="Output file with near-zero vectors removed")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Remove vectors with L2 norm <= tolerance (default: 0.0)",
    )
    args = parser.parse_args()

    if args.tolerance < 0:
        raise ValueError("--tolerance must be non-negative")

    is_bvecs = args.input.lower().endswith(".bvecs")
    vectors = read_bvecs(args.input) if is_bvecs else read_fvecs(args.input)

    zero_count = count_zero_vectors(vectors, tol=args.tolerance)
    print(f"Zero tolerance: {args.tolerance}")
    print(f"Zero-like vectors: {zero_count} / {vectors.shape[0]}")

    cleaned = remove_zero_vectors(vectors, tol=args.tolerance)

    if cleaned.shape[0] == 0:
        raise ValueError("All vectors were zero after removal.")

    removed = vectors.shape[0] - cleaned.shape[0]
    print(f"Removed zero vectors: {removed}")
    print(f"Remaining vectors: {cleaned.shape[0]}")
    print(f"Dimension: {cleaned.shape[1]}")

    if is_bvecs:
        write_bvecs(args.output, cleaned)
    else:
        write_fvecs(args.output, cleaned.astype(np.float32))
    print(f"Wrote cleaned file to: {args.output}")


if __name__ == "__main__":
    main()
