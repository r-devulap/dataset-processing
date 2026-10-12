#!/usr/bin/env python3

from typing import Iterable, Optional
from huggingface_hub import HfApi, snapshot_download

# Change only this import to choose which repo config to run.
from download_configs import wiki_cohere_multiligual_v3_int8 as repo

# ============================================================
# Configuration
# All downloader settings come from the selected repo config.
# ============================================================

REPO_ID = repo.REPO_ID
REPO_TYPE = repo.REPO_TYPE
REVISION = repo.REVISION
TOKEN = repo.TOKEN

LOCAL_DIR = repo.LOCAL_DIR
MAX_WORKERS = repo.MAX_WORKERS
DRY_RUN = repo.DRY_RUN

PRINT_ALL_FILES = repo.PRINT_ALL_FILES
PRINT_SELECTED_FILES = repo.PRINT_SELECTED_FILES

# Selection mode:
#   "all"
#       Download every file in the repo.
#       Example:
#           SELECTION_MODE = "all"
#
#   "first_n"
#       Download the first N files after optional pattern filtering.
#       Useful when you want a small prefix of a large repo.
#       Example:
#           SELECTION_MODE = "first_n"
#           FIRST_N = 12
#           ALLOWED_SUFFIXES = [".npy"]
#
#   "explicit"
#       Download exactly the listed repo-relative file paths.
#       Example:
#           SELECTION_MODE = "explicit"
#           EXPLICIT_FILES = [
#               "data/en/embs/emb_000.npy",
#               "data/en/embs/emb_001.npy",
#           ]
#
#   "pattern"
#       Download files selected by substring/suffix filters.
#       Example:
#           SELECTION_MODE = "pattern"
#           SELECT_SUBSTRINGS = ["data/en/embs/"]
#           EXCLUDE_SUBSTRINGS = ["train"]
#           ALLOWED_SUFFIXES = [".parquet"]
SELECTION_MODE = repo.SELECTION_MODE

# Used only when SELECTION_MODE == "first_n".
FIRST_N = repo.FIRST_N

# Used only when SELECTION_MODE == "explicit".
EXPLICIT_FILES = repo.EXPLICIT_FILES

# Used only when SELECTION_MODE == "first_n" or "pattern".
SELECT_SUBSTRINGS = repo.SELECT_SUBSTRINGS
EXCLUDE_SUBSTRINGS = repo.EXCLUDE_SUBSTRINGS
ALLOWED_SUFFIXES = repo.ALLOWED_SUFFIXES

# ============================================================
# Functions
# ============================================================

def list_repo_files(
    repo_id: str,
    repo_type: str = "dataset",
    revision: str = "main",
    token: Optional[str] = None,
) -> list[str]:
    api = HfApi(token=token)
    return sorted(
        api.list_repo_files(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
        )
    )


def matches_any_substring(path: str, needles: Iterable[str]) -> bool:
    needles = list(needles)
    if not needles:
        return True
    return any(s in path for s in needles)


def matches_any_suffix(path: str, suffixes: Iterable[str]) -> bool:
    suffixes = list(suffixes)
    if not suffixes:
        return True
    return any(path.endswith(s) for s in suffixes)


def filter_files(
    files: list[str],
    select_substrings: Optional[Iterable[str]] = None,
    exclude_substrings: Optional[Iterable[str]] = None,
    allowed_suffixes: Optional[Iterable[str]] = None,
) -> list[str]:
    select_substrings = list(select_substrings or [])
    exclude_substrings = list(exclude_substrings or [])
    allowed_suffixes = list(allowed_suffixes or [])

    selected = []
    for path in files:
        if not matches_any_substring(path, select_substrings):
            continue
        if exclude_substrings and any(s in path for s in exclude_substrings):
            continue
        if not matches_any_suffix(path, allowed_suffixes):
            continue
        selected.append(path)

    return selected


def select_files(
    files: list[str],
    selection_mode: str,
    first_n: Optional[int] = None,
    explicit_files: Optional[Iterable[str]] = None,
    select_substrings: Optional[Iterable[str]] = None,
    exclude_substrings: Optional[Iterable[str]] = None,
    allowed_suffixes: Optional[Iterable[str]] = None,
) -> list[str]:
    explicit_files = list(explicit_files or [])

    if selection_mode == "all":
        return list(files)

    if selection_mode == "explicit":
        missing = [path for path in explicit_files if path not in files]
        if missing:
            raise FileNotFoundError(
                "The following EXPLICIT_FILES were not found in the repo:\n"
                + "\n".join(missing)
            )
        return explicit_files

    filtered = filter_files(
        files=files,
        select_substrings=select_substrings,
        exclude_substrings=exclude_substrings,
        allowed_suffixes=allowed_suffixes,
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


def download_selected_files(
    repo_id: str,
    selected_files: list[str],
    local_dir,
    repo_type: str = "dataset",
    revision: str = "main",
    token: Optional[str] = None,
    max_workers: int = 8,
) -> str:
    local_dir.mkdir(parents=True, exist_ok=True)

    return snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=selected_files,
        token=token,
        max_workers=max_workers,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    files = list_repo_files(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        revision=REVISION,
        token=TOKEN,
    )

    print(f"\nFound {len(files)} files in {REPO_TYPE} repo '{REPO_ID}'.\n")

    if PRINT_ALL_FILES:
        print("All files:")
        for f in files:
            print(f)

    selected = select_files(
        files=files,
        selection_mode=SELECTION_MODE,
        first_n=FIRST_N,
        explicit_files=EXPLICIT_FILES,
        select_substrings=SELECT_SUBSTRINGS,
        exclude_substrings=EXCLUDE_SUBSTRINGS,
        allowed_suffixes=ALLOWED_SUFFIXES,
    )

    print(f"\nSelected {len(selected)} file(s).\n")

    if PRINT_SELECTED_FILES:
        for f in selected:
            print(f)

    if not selected:
        print("\nNo files matched the configured selection.")
        return

    if DRY_RUN:
        print("\nDRY_RUN is True. No download performed.")
        return

    print(f"\nDownloading with max_workers={MAX_WORKERS} ...")
    snapshot_path = download_selected_files(
        repo_id=REPO_ID,
        selected_files=selected,
        local_dir=LOCAL_DIR,
        repo_type=REPO_TYPE,
        revision=REVISION,
        token=TOKEN,
        max_workers=MAX_WORKERS,
    )

    print(f"\nDone. Files downloaded under: {snapshot_path}")


if __name__ == "__main__":
    main()
