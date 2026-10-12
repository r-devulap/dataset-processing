from pathlib import Path

OUTPUT_DIR = Path("/home/ubuntu/dataset_cache/int8-datasets/processed/wiki-multilingual-v3-int8-1m")

FILE_PREFIX = "wiki_multilingual_v3_int8"
SOURCE_TYPE = "parquet"
READER_BATCH_SIZE = 32768  # Tune for RAM vs throughput.
PARQUET_EMBEDDING_COLUMN = "emb_int8"  # 1024-dim signed INT8 embeddings

INPUT_DIR = Path("/home/ubuntu/dataset_cache/int8-datasets/wikipedia-2023-11-embed-multilingual-v3-int8-binary/en")

# ~130k vectors/file; for 1M base + 10k query we need ~8 files.
# Use first_n to avoid reading all 54M vectors unnecessarily.
# Set SELECTION_MODE = "all" and NUM_BASE = None to process the full ~54M dataset.
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
