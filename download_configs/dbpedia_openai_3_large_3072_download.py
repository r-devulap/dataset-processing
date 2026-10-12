from pathlib import Path

REPO_ID = "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M"
REPO_TYPE = "dataset"
REVISION = "main"
TOKEN = None

LOCAL_DIR = Path("/home/raghuveer/myplayground/int8-datasets/dbpedia-entities-openai3-text-embedding-3-large-3072-1M")
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
