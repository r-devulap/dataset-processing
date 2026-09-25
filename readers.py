import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

# Standard logging setup to match your output format
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def _validate_file_suffixes(
    input_files: List[Path],
    reader_name: str,
    allowed_suffixes: tuple[str, ...],
) -> None:
    for path in input_files:
        suffix = path.suffix.lower()
        if suffix not in allowed_suffixes:
            raise ValueError(
                f"{reader_name} expected file suffix in {allowed_suffixes}, "
                f"but got '{suffix or '<no suffix>'}' for {path}"
            )

class EmbeddingReader(ABC):
    @abstractmethod
    def iter_batches(self) -> Iterable[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> dict:
        raise NotImplementedError


class NpyEmbeddingReader(EmbeddingReader):
    def __init__(self, input_files: List[Path], batch_size: int) -> None:
        _validate_file_suffixes(input_files, "NpyEmbeddingReader", (".npy",))
        self.input_files = input_files
        self.batch_size = batch_size

    def iter_batches(self) -> Iterable[np.ndarray]:
        for path in self.input_files:
            arr = np.load(path, mmap_mode="r")
            if arr.ndim != 2:
                raise ValueError(f"{path} must contain a 2D array, got shape {arr.shape}")

            num_vectors = arr.shape[0]
            self.current_file = str(path)

            for start in range(0, num_vectors, self.batch_size):
                end = min(start + self.batch_size, num_vectors)
                yield np.asarray(arr[start:end], dtype=np.float32)

    def describe(self) -> dict:
        return {
            "reader": "npy",
            "batch_size": self.batch_size,
            "num_files": len(self.input_files),
            "files": [str(p) for p in self.input_files],
        }

class FvecsEmbeddingReader(EmbeddingReader):

    def __init__(self, input_files: List[Path], batch_size: int) -> None:
        _validate_file_suffixes(input_files, "FvecsEmbeddingReader", (".fvecs",))
        self.input_files = input_files
        self.batch_size = batch_size

    def iter_batches(self) -> Iterable[np.ndarray]:
        for path in self.input_files:
            raw = np.memmap(path, dtype=np.int32, mode="r")
            if raw.size == 0:
                raise ValueError(f"{path} is empty")

            dim = int(raw[0])
            if dim <= 0:
                raise ValueError(f"{path} has invalid dimension header: {dim}")

            row_size = dim + 1
            if raw.size % row_size != 0:
                raise ValueError(
                    f"{path} size is not a multiple of fvecs row size {row_size}"
                )

            num_vectors = raw.size // row_size
            matrix = raw.reshape(num_vectors, row_size)

            self.current_file = str(path)

            for start in range(0, num_vectors, self.batch_size):
                end = min(start + self.batch_size, num_vectors)
                batch_rows = matrix[start:end]

                if not np.all(batch_rows[:, 0] == dim):
                    raise ValueError(
                        f"{path} has inconsistent dimension header in rows {start}:{end}"
                    )

                yield np.asarray(batch_rows[:, 1:].view(np.float32), dtype=np.float32)

    def describe(self) -> dict:
        return {
            "reader": "fvecs",
            "batch_size": self.batch_size,
            "num_files": len(self.input_files),
            "files": [str(p) for p in self.input_files],
        }

class ParquetEmbeddingReader(EmbeddingReader):

    def __init__(self, input_files: List[Path], embedding_column: str, batch_size: int) -> None:
        if not embedding_column:
            raise ValueError("ParquetEmbeddingReader requires a non-empty embedding_column")
        _validate_file_suffixes(input_files, "ParquetEmbeddingReader", (".parquet",))
        self.input_files = input_files
        self.batch_size = batch_size
        self.embedding_column = embedding_column

    def iter_batches(self) -> Iterable[np.ndarray]:
        import pyarrow.parquet as pq

        for path in self.input_files:
            # use_threads=True and memory_map=True improve I/O performance
            parquet_file = pq.ParquetFile(path, memory_map=True)

            if self.embedding_column not in parquet_file.schema_arrow.names:
                raise ValueError(
                    f"{path} does not contain parquet column '{self.embedding_column}'"
                )

            for record_batch in parquet_file.iter_batches(
                    columns=[self.embedding_column],
                    batch_size=self.batch_size,
                    use_threads=True,
            ):
                # record_batch.column(0) is typically a ListArray or FixedSizeListArray
                col = record_batch.column(0)

                # .values gets the underlying flattened data; .to_numpy() is fast
                flattened_data = col.values.to_numpy(zero_copy_only=False)

                # Calculate dimensions and reshape back to 2D
                num_rows = len(record_batch)
                dim = len(flattened_data) // num_rows
                arr = flattened_data.reshape(num_rows, dim)

                self.current_file = str(path)

                yield np.ascontiguousarray(arr, dtype=np.float32)

    def describe(self) -> dict:
        return {
            "reader": "parquet",
            "embedding_column": self.embedding_column,
            "batch_size": self.batch_size,
            "num_files": len(self.input_files),
            "files": [str(p) for p in self.input_files],
        }

class Hdf5EmbeddingReader(EmbeddingReader):

    def __init__(
        self,
        input_files: List[Path],
        dataset_name: str = "train",
        batch_size: int = 32768,
    ) -> None:
        if not dataset_name:
            raise ValueError("Hdf5EmbeddingReader requires a non-empty dataset_name")
        _validate_file_suffixes(input_files, "Hdf5EmbeddingReader", (".h5", ".hdf5"))
        self.input_files = input_files
        self.dataset_name = dataset_name
        self.batch_size = batch_size

    def iter_batches(self) -> Iterable[np.ndarray]:
        import h5py

        for path in self.input_files:
            with h5py.File(path, "r") as f:
                if self.dataset_name not in f:
                    raise KeyError(
                        f"{path} does not contain HDF5 dataset '{self.dataset_name}'. "
                        f"Available keys: {list(f.keys())}"
                    )

                ds = f[self.dataset_name]
                if ds.ndim != 2:
                    raise ValueError(
                        f"{path}['{self.dataset_name}'] must be 2D, got shape {ds.shape}"
                    )

                num_vectors = ds.shape[0]
                self.current_file = str(path)

                for start in range(0, num_vectors, self.batch_size):
                    end = min(start + self.batch_size, num_vectors)
                    batch = ds[start:end]
                    yield np.asarray(batch, dtype=np.float32)

    def describe(self) -> dict:
        return {
            "reader": "hdf5",
            "dataset_name": self.dataset_name,
            "batch_size": self.batch_size,
            "num_files": len(self.input_files),
            "files": [str(p) for p in self.input_files],
        }


DEFAULT_BATCH_SIZE = 32768

def build_reader(
        source_type: str,
        input_files: List[Path],
        parquet_embedding_column: Optional[str] = None,
        hdf5_dataset_name: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
) -> EmbeddingReader:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    if source_type == "npy":
        return NpyEmbeddingReader(input_files, batch_size)

    if source_type == "parquet":
        if not parquet_embedding_column:
            raise ValueError(
                "source_type='parquet' requires parquet_embedding_column to be set"
            )
        return ParquetEmbeddingReader(input_files, parquet_embedding_column, batch_size)

    if source_type == "fvecs":
        return FvecsEmbeddingReader(input_files, batch_size)

    if source_type in ("hdf5", "h5"):
        ds_name = hdf5_dataset_name if hdf5_dataset_name else "train"
        return Hdf5EmbeddingReader(input_files, dataset_name=ds_name, batch_size=batch_size)

    raise ValueError(f"Unsupported source_type: {source_type}")
