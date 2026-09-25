#!/usr/bin/env python3

import os
import struct
from pathlib import Path
from typing import Tuple

import numpy as np


def _normalize_int8_array(arr: np.ndarray) -> np.ndarray:
    """
    Convert input to a C-contiguous int8 2D numpy array.
    """
    arr = np.asarray(arr, dtype=np.int8)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def write_bvecs(fname, arr: np.ndarray) -> None:
    """
    Write a numpy int8 array (shape: n x d) to a bvecs file.

    Each vector is stored as:
        [d (int32 little-endian), x_0 (int8), x_1 (int8), ..., x_{d-1} (int8)]

    This overwrites the destination file.
    """
    arr = _normalize_int8_array(arr)
    n, d = arr.shape

    fname = os.path.expanduser(str(fname))
    Path(fname).parent.mkdir(parents=True, exist_ok=True)

    header = np.array([d], dtype=np.int32).tobytes()
    with open(fname, "wb") as f:
        for i in range(n):
            f.write(header)
            f.write(arr[i].tobytes())


def append_bvecs(fname, arr: np.ndarray) -> None:
    """
    Append a batch of int8 vectors (shape: n x d) to an existing bvecs file.
    """
    arr = _normalize_int8_array(arr)
    n, d = arr.shape

    fname = os.path.expanduser(str(fname))
    path = Path(fname)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        existing_count, existing_dim = count_bvecs(path)
        if existing_dim != d:
            raise ValueError(
                f"Cannot append vectors of dim {d} to existing bvecs "
                f"with dim {existing_dim}: {path}"
            )

    header = np.array([d], dtype=np.int32).tobytes()
    with open(path, "ab") as f:
        for i in range(n):
            f.write(header)
            f.write(arr[i].tobytes())


def count_bvecs(fname) -> Tuple[int, int]:
    """
    Return (num_vectors, dim) for a bvecs file.
    """
    fname = os.path.expanduser(str(fname))
    path = Path(fname)

    if not path.exists():
        raise FileNotFoundError(f"bvecs file not found: {path}")

    size = path.stat().st_size
    if size == 0:
        return 0, 0

    with open(path, "rb") as f:
        header = f.read(4)
        if len(header) < 4:
            return 0, 0
        dim = struct.unpack("<i", header)[0]

    if dim <= 0:
        raise ValueError(f"Invalid dimension {dim} in {path}")

    record_size = 4 + dim
    if size % record_size != 0:
        raise ValueError(
            f"Invalid bvecs file size for {path}: "
            f"{size} bytes is not divisible by record size {record_size}"
        )

    count = size // record_size
    return int(count), dim
