from pathlib import Path

REPO_ID = "aidos-lab/mich-embeddings-int8-mixed-bread-large"
REPO_TYPE = "dataset"
REVISION = "main"
TOKEN = None

LOCAL_DIR = Path("/home/ubuntu/dataset_cache/int8-datasets/mich-embeddings-int8-mixed-bread-large")
MAX_WORKERS = 16
DRY_RUN = False

PRINT_ALL_FILES = True
PRINT_SELECTED_FILES = True

SELECTION_MODE = "all"
FIRST_N = None
EXPLICIT_FILES = []

SELECT_SUBSTRINGS = []
EXCLUDE_SUBSTRINGS = []
ALLOWED_SUFFIXES = [".parquet"]
