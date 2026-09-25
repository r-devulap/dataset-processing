from pathlib import Path
import sys

# Change this import to match whatever dataset processing config you are actually interested in.
from processing_configs import sift1m_hdf5_proc as dataset

# ------------------------------------------------------------
# Run configuration
# ------------------------------------------------------------

RUN_DIR = dataset.OUTPUT_DIR
FILE_PREFIX = dataset.FILE_PREFIX
CLEANUP_INTERMEDIATE_FVECS = dataset.CLEANUP_INTERMEDIATE_FVECS
OVERWRITE = dataset.OVERWRITE

# ------------------------------------------------------------
# Working files. Do not update.
# ------------------------------------------------------------

RAW_BASE_FVECS = RUN_DIR / f"{FILE_PREFIX}_raw_base.fvecs"

# INT8 early-quantization path: combined bvecs right after extraction
RAW_QUANTIZED_BVECS = RUN_DIR / f"{FILE_PREFIX}_raw_base.bvecs"

# When QUANTIZE_TO_INT8=True, the intermediate cleaning/dedup/split chain
# operates on .bvecs files; otherwise it operates on .fvecs files.
# Read directly from dataset here because QUANTIZE_TO_INT8 is defined later.
_bvecs_pipeline = getattr(dataset, "QUANTIZE_TO_INT8", False)

NONZERO_BASE_FVECS  = RUN_DIR / (f"{FILE_PREFIX}_nonzero_base.bvecs"    if _bvecs_pipeline else f"{FILE_PREFIX}_nonzero_base.fvecs")
NORMALIZED_BASE_FVECS = RUN_DIR / (f"{FILE_PREFIX}_normalized_base.bvecs" if _bvecs_pipeline else f"{FILE_PREFIX}_normalized_base.fvecs")
DEDUP_BASE_FVECS    = RUN_DIR / (f"{FILE_PREFIX}_base.bvecs"             if _bvecs_pipeline else f"{FILE_PREFIX}_base.fvecs")
SPLIT_QUERY_FVECS   = RUN_DIR / (f"{FILE_PREFIX}_base_query.bvecs"       if _bvecs_pipeline else f"{FILE_PREFIX}_base_query.fvecs")
SPLIT_BASE_FVECS    = RUN_DIR / (f"{FILE_PREFIX}_base_base.bvecs"        if _bvecs_pipeline else f"{FILE_PREFIX}_base_base.fvecs")
SPLIT_QPARTS_DIR = Path(f"{DEDUP_BASE_FVECS.with_suffix('')}_qparts")
SPLIT_BPARTS_DIR = Path(f"{DEDUP_BASE_FVECS.with_suffix('')}_bparts")

# Final split outputs kept as aliases (used by GT and rename logic)
QUANTIZED_BASE_BVECS = SPLIT_BASE_FVECS
QUANTIZED_QUERY_BVECS = SPLIT_QUERY_FVECS
QUANTIZATION_META_FILE = RUN_DIR / f"{FILE_PREFIX}_quantization_meta.json"

GT_PROCESSED_BASE_FVECS = RUN_DIR / f"{FILE_PREFIX}_gt_processed_base.fvecs"
GT_PROCESSED_QUERY_FVECS = RUN_DIR / f"{FILE_PREFIX}_gt_processed_query.fvecs"
GROUND_TRUTH_FILE = RUN_DIR / "ground_truth.ivecs"

DEDUP_REPORT = RUN_DIR / f"{FILE_PREFIX}_dedup_report.txt"
DEDUP_TEMP_DIR = RUN_DIR / f"{FILE_PREFIX}_dedup_temp"

LOG_FILE = RUN_DIR / f"{FILE_PREFIX}_pipeline.log"
SUMMARY_FILE = RUN_DIR / f"{FILE_PREFIX}_summary.json"

# ------------------------------------------------------------
# Requested output sizes
# ------------------------------------------------------------

NUM_QUERY = dataset.NUM_QUERY
NUM_BASE = dataset.NUM_BASE

# ------------------------------------------------------------
# Ground truth
# ------------------------------------------------------------

GT_K = dataset.GT_K
GT_METRIC = dataset.GT_METRIC
GT_SHUFFLE = dataset.GT_SHUFFLE
GT_GPUS = dataset.GT_GPUS

FINAL_GROUND_TRUTH = RUN_DIR / f"{FILE_PREFIX}_gt_{GT_METRIC}_{GT_K}.ivecs"

# ------------------------------------------------------------
# Tolerances
# ------------------------------------------------------------

ZERO_TOLERANCE = dataset.ZERO_TOLERANCE
NORMALIZATION_TOLERANCE = dataset.NORMALIZATION_TOLERANCE

# ------------------------------------------------------------
# Quantization configuration
# ------------------------------------------------------------

QUANTIZE_TO_INT8 = getattr(dataset, "QUANTIZE_TO_INT8", False)
INT8_QUANTIZATION_MODE = getattr(dataset, "INT8_QUANTIZATION_MODE", "symmetric_max_abs")

# ------------------------------------------------------------
# Input data
# ------------------------------------------------------------

SOURCE_TYPE = dataset.SOURCE_TYPE
READER_BATCH_SIZE = dataset.READER_BATCH_SIZE
PARQUET_EMBEDDING_COLUMN = getattr(dataset, "PARQUET_EMBEDDING_COLUMN", None)
HDF5_DATASET_NAME = getattr(dataset, "HDF5_DATASET_NAME", "train")

INPUT_DIR = dataset.INPUT_DIR
SELECTION_MODE = dataset.SELECTION_MODE
FIRST_N = dataset.FIRST_N
EXPLICIT_INPUT_FILES = dataset.EXPLICIT_INPUT_FILES
SELECT_SUBSTRINGS = dataset.SELECT_SUBSTRINGS
EXCLUDE_SUBSTRINGS = dataset.EXCLUDE_SUBSTRINGS
ALLOWED_SUFFIXES = dataset.ALLOWED_SUFFIXES


def discover_input_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*") if path.is_file())


def matches_any_substring(path: str, needles: list[str]) -> bool:
    if not needles:
        return True
    return any(needle in path for needle in needles)


def matches_any_suffix(path: str, suffixes: list[str]) -> bool:
    if not suffixes:
        return True
    return any(path.endswith(suffix) for suffix in suffixes)


def filter_input_files(
    files: list[Path],
    input_dir: Path,
    select_substrings: list[str],
    exclude_substrings: list[str],
    allowed_suffixes: list[str],
) -> list[Path]:
    selected = []
    for path in files:
        rel = path.relative_to(input_dir).as_posix()

        if not matches_any_substring(rel, select_substrings):
            continue
        if exclude_substrings and any(s in rel for s in exclude_substrings):
            continue
        if not matches_any_suffix(rel, allowed_suffixes):
            continue

        selected.append(path)

    return selected


def resolve_input_files(
    input_dir: Path,
    selection_mode: str,
    first_n: int | None,
    explicit_input_files: list[str],
    select_substrings: list[str],
    exclude_substrings: list[str],
    allowed_suffixes: list[str],
) -> list[Path]:
    if selection_mode == "explicit":
        resolved = []
        for entry in explicit_input_files:
            path = Path(entry)
            if not path.is_absolute():
                path = input_dir / path
            resolved.append(path)
        return resolved

    files = discover_input_files(input_dir)

    if selection_mode == "all":
        return files

    filtered = filter_input_files(
        files,
        input_dir,
        select_substrings,
        exclude_substrings,
        allowed_suffixes,
    )

    if selection_mode == "pattern":
        return filtered

    if selection_mode == "first_n":
        if first_n is None or first_n <= 0:
            raise ValueError("FIRST_N must be a positive integer when SELECTION_MODE='first_n'")
        return filtered[:first_n]

    raise ValueError(
        "Unsupported SELECTION_MODE. Expected one of: "
        "'all', 'first_n', 'explicit', 'pattern'"
    )


INPUT_FILES = resolve_input_files(
    INPUT_DIR,
    SELECTION_MODE,
    FIRST_N,
    EXPLICIT_INPUT_FILES,
    SELECT_SUBSTRINGS,
    EXCLUDE_SUBSTRINGS,
    ALLOWED_SUFFIXES,
)

# ------------------------------------------------------------
# External stage commands
# ------------------------------------------------------------

# When QUANTIZE_TO_INT8=True, quantization runs first (single-file mode),
# and the downstream tools receive the quantized .bvecs file.
QUANTIZE_CMD = [
    sys.executable,
    "-u",
    "bvecs_quantize.py",
    "--input", str(RAW_BASE_FVECS),
    "--output", str(RAW_QUANTIZED_BVECS),
    "--meta_out", str(QUANTIZATION_META_FILE),
    "--batch_size", str(READER_BATCH_SIZE),
] if QUANTIZE_TO_INT8 else []

_REMOVE_ZEROS_INPUT = RAW_QUANTIZED_BVECS if QUANTIZE_TO_INT8 else RAW_BASE_FVECS

REMOVE_ZEROS_CMD = [
    sys.executable,
    "-u",
    "fvecs_remove_zeros.py",
    "--input", str(_REMOVE_ZEROS_INPUT),
    "--output", str(NONZERO_BASE_FVECS),
    "--tolerance", str(ZERO_TOLERANCE),
]

NORMALIZE_CMD = [
    sys.executable,
    "-u",
    "fvecs_normalize.py",
    "--input", str(NONZERO_BASE_FVECS),
    "--output", str(NORMALIZED_BASE_FVECS),
    "--tolerance", str(NORMALIZATION_TOLERANCE),
]

DEDUP_CMD = [
    sys.executable,
    "-u",
    "fvecs_deduplicator.py",
    str(NORMALIZED_BASE_FVECS),
    "--output", str(DEDUP_BASE_FVECS),
    "--report_file", str(DEDUP_REPORT),
    "--reporting_threshold", "1",
    "--chunk_size", "200000",
    "--temp_dir", str(DEDUP_TEMP_DIR),
]

SPLIT_CMD = [
    sys.executable,
    "-u",
    "fvecs_split.py",
    str(DEDUP_BASE_FVECS),
    "--num_query", str(NUM_QUERY),
    "--seed", "47",
]

_GT_BASE_INPUT = SPLIT_BASE_FVECS
_GT_QUERY_INPUT = SPLIT_QUERY_FVECS

GROUND_TRUTH_CMD = [
    sys.executable,
    "-u",
    "knn_utils.py",
    "--base", str(_GT_BASE_INPUT),
    "--query", str(_GT_QUERY_INPUT),
    "--output", str(GROUND_TRUTH_FILE),
    "--processed_base_out", str(GT_PROCESSED_BASE_FVECS),
    "--processed_query_out", str(GT_PROCESSED_QUERY_FVECS),
    "--k", str(GT_K),
    "--metric", GT_METRIC,
    "--gpus", GT_GPUS,
]

if GT_SHUFFLE:
    GROUND_TRUTH_CMD.append("--shuffle")
