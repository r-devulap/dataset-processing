from pathlib import Path

OUTPUT_DIR = Path("/home/ubuntu/dataset_cache/int8-datasets/processed/wikipedia-2023-11-en-mxbai-int8-1m")

FILE_PREFIX = "wikipedia_2023_11_en_mxbai_int8"
SOURCE_TYPE = "parquet"
READER_BATCH_SIZE = 32768  # Tune for RAM vs throughput.
PARQUET_EMBEDDING_COLUMN = "emb_int8"  # 1024-dim signed INT8 embeddings (mxbai-embed-large-v1)

INPUT_DIR = Path("/home/ubuntu/dataset_cache/int8-datasets/wikipedia-2023-11-en-embed-mxbai-int8-binary/data")

# ~300k vectors/file; for 1M base + 10k query we need 4 files.
SELECTION_MODE = "first_n"
FIRST_N = 4
EXPLICIT_INPUT_FILES = []

SELECT_SUBSTRINGS = []
EXCLUDE_SUBSTRINGS = []
ALLOWED_SUFFIXES = [".parquet"]

NUM_QUERY = 10000
NUM_BASE = 1000000

GT_K = 100
GT_METRIC = "ip"
GT_SHUFFLE = True
GT_GPUS = "-1"  # Set to "0" or "0,1" if you have GPUs available

ZERO_TOLERANCE = 0.0
NORMALIZATION_TOLERANCE = 1e-3

CLEANUP_INTERMEDIATE_FVECS = True
OVERWRITE = True

# Read INT8 vectors directly from source and keep them as .bvecs throughout the pipeline.
SOURCE_INT8 = True
