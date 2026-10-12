from pathlib import Path

REPO_ID = "CohereLabs/wikipedia-2023-11-embed-multilingual-v3-int8-binary"
REPO_TYPE = "dataset"
REVISION = "main"
TOKEN = None

LOCAL_DIR = Path("/home/raghuveer/myplayground/int8-datasets/wikipedia-2023-11-embed-multilingual-v3-int8-binary")
MAX_WORKERS = 16
DRY_RUN = False

PRINT_ALL_FILES = True
PRINT_SELECTED_FILES = True

SELECTION_MODE = "all"
FIRST_N = 5000
EXPLICIT_FILES = []

SELECT_SUBSTRINGS = ["data/"]
EXCLUDE_SUBSTRINGS = []
ALLOWED_SUFFIXES = [".parquet"]
