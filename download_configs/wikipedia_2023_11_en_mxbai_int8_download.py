from pathlib import Path

REPO_ID = "krasserm/wikipedia-2023-11-en-embed-mxbai-int8-binary"
REPO_TYPE = "dataset"
REVISION = "main"
TOKEN = None

LOCAL_DIR = Path("/home/ubuntu/dataset_cache/int8-datasets/wikipedia-2023-11-en-embed-mxbai-int8-binary")
MAX_WORKERS = 16
DRY_RUN = False

PRINT_ALL_FILES = True
PRINT_SELECTED_FILES = True

# ~300k vectors/file; 4 files are enough for 1M base + 10k query.
SELECTION_MODE = "first_n"
FIRST_N = 4
EXPLICIT_FILES = []

SELECT_SUBSTRINGS = []
EXCLUDE_SUBSTRINGS = []
ALLOWED_SUFFIXES = [".parquet"]
