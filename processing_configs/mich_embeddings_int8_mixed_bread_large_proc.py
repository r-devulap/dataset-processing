from pathlib import Path

OUTPUT_DIR = Path("/home/ubuntu/dataset_cache/int8-datasets/processed/mich-embeddings-int8-mixed-bread-large-1m")

FILE_PREFIX = "mich_int8_mixed_bread_large"
SOURCE_TYPE = "parquet"
READER_BATCH_SIZE = 32768
PARQUET_EMBEDDING_COLUMN = "emb_int8"  # adjust if the column name differs

INPUT_DIR = Path("/home/ubuntu/dataset_cache/int8-datasets/mich-embeddings-int8-mixed-bread-large")

SELECTION_MODE = "first_n"
FIRST_N = 11
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
