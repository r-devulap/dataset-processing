from pathlib import Path

# ------------------------------------------------------------
# Output location
# ------------------------------------------------------------

OUTPUT_DIR = Path("/home/raghuveer/myplayground/dataset-processing/runs/sift1m_hdf5")

# ------------------------------------------------------------
# Dataset / file format metadata
# ------------------------------------------------------------

FILE_PREFIX = "sift1m"
SOURCE_TYPE = "hdf5"
HDF5_DATASET_NAME = "train"
READER_BATCH_SIZE = 32768
PARQUET_EMBEDDING_COLUMN = None

# ------------------------------------------------------------
# Local input file selection
# ------------------------------------------------------------

INPUT_DIR = Path("/home/raghuveer/myplayground/int8-datasets/sift1m")
SELECTION_MODE = "explicit"
EXPLICIT_INPUT_FILES = ["sift-128-euclidean.hdf5"]
FIRST_N = None
SELECT_SUBSTRINGS = []
EXCLUDE_SUBSTRINGS = []
ALLOWED_SUFFIXES = [".hdf5", ".h5"]

# ------------------------------------------------------------
# Requested output sizes
# ------------------------------------------------------------

# 10K queries sampled, 990K base vectors
NUM_QUERY = 10000
NUM_BASE = 990000

# ------------------------------------------------------------
# Ground truth configuration
# ------------------------------------------------------------

GT_K = 100
GT_METRIC = "l2"      # Euclidean = L2
GT_SHUFFLE = False
GT_GPUS = "-1"

# ------------------------------------------------------------
# Tolerances
# ------------------------------------------------------------

ZERO_TOLERANCE = 0.0
NORMALIZATION_TOLERANCE = 1e-3

# ------------------------------------------------------------
# Quantization configuration
# ------------------------------------------------------------

QUANTIZE_TO_INT8 = True
INT8_QUANTIZATION_MODE = "symmetric_max_abs"

# ------------------------------------------------------------
# Run behavior
# ------------------------------------------------------------

CLEANUP_INTERMEDIATE_FVECS = True
OVERWRITE = True
