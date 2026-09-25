#!/usr/bin/env python3

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

from config import (
    CLEANUP_INTERMEDIATE_FVECS,
    DEDUP_BASE_FVECS,
    DEDUP_TEMP_DIR,
    DEDUP_CMD,
    DEDUP_REPORT,
    FILE_PREFIX,
    FINAL_GROUND_TRUTH,
    GROUND_TRUTH_CMD,
    GROUND_TRUTH_FILE,
    GT_K,
    GT_METRIC,
    GT_PROCESSED_BASE_FVECS,
    GT_PROCESSED_QUERY_FVECS,
    GT_SHUFFLE,
    GT_GPUS,
    HDF5_DATASET_NAME,
    INPUT_FILES,
    INT8_QUANTIZATION_MODE,
    LOG_FILE,
    NORMALIZATION_TOLERANCE,
    NORMALIZE_CMD,
    NONZERO_BASE_FVECS,
    NUM_BASE,
    NUM_QUERY,
    OVERWRITE,
    PARQUET_EMBEDDING_COLUMN,
    QUANTIZE_CMD,
    QUANTIZE_TO_INT8,
    QUANTIZED_BASE_BVECS,
    QUANTIZED_QUERY_BVECS,
    QUANTIZATION_META_FILE,
    RAW_BASE_FVECS,
    RAW_QUANTIZED_BVECS,
    REMOVE_ZEROS_CMD,
    RUN_DIR,
    SOURCE_TYPE,
    SPLIT_BASE_FVECS,
    SPLIT_BPARTS_DIR,
    SPLIT_CMD,
    SPLIT_QPARTS_DIR,
    SPLIT_QUERY_FVECS,
    SUMMARY_FILE,
    ZERO_TOLERANCE,
    NORMALIZED_BASE_FVECS,
)
from bvecs_writer import count_bvecs
from fvecs_writer import append_fvecs, count_fvecs
from ivecs_check import read_ivecs_info
from config import READER_BATCH_SIZE
from readers import build_reader


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("hf_dataset_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def validate_input_files() -> None:
    if not INPUT_FILES:
        raise FileNotFoundError("No input files were selected for processing.")

    missing = [p for p in INPUT_FILES if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing input files:\n" + "\n".join(str(p) for p in missing)
        )


def extract_base_vectors(logger: logging.Logger) -> dict:
    RAW_BASE_FVECS.parent.mkdir(parents=True, exist_ok=True)

    if RAW_BASE_FVECS.exists():
        if OVERWRITE:
            logger.info("Deleting existing output: %s", RAW_BASE_FVECS)
            RAW_BASE_FVECS.unlink()
        else:
            count, dim = count_output_file(RAW_BASE_FVECS)
            logger.info(
                "Output already exists and OVERWRITE=False, skipping extraction: "
                "vectors=%d dim=%d file=%s",
                count,
                dim,
                RAW_BASE_FVECS,
            )
            return {
                "skipped": True,
                "file_prefix": FILE_PREFIX,
                "source_type": SOURCE_TYPE,
                "parquet_embedding_column": PARQUET_EMBEDDING_COLUMN,
                "vectors_written": count,
                "dim": dim,
                "output_file": str(RAW_BASE_FVECS),
            }

    validate_input_files()

    reader = build_reader(
        SOURCE_TYPE,
        INPUT_FILES,
        parquet_embedding_column=PARQUET_EMBEDDING_COLUMN,
        hdf5_dataset_name=HDF5_DATASET_NAME,
        batch_size=READER_BATCH_SIZE,
    )
    logger.info("Reader description: %s", reader.describe())

    requested_initial_vectors = None if NUM_BASE is None else (NUM_BASE + NUM_QUERY)
    logger.info("Requested initial extraction target: %s", requested_initial_vectors)

    total_vectors = 0
    dim = None
    batch_count = 0
    limit_reached = False

    for batch_idx, batch in enumerate(reader.iter_batches()):
        if batch.ndim != 2:
            raise ValueError(f"Batch {batch_idx} is not 2D: shape={batch.shape}")

        batch_vectors, batch_dim = batch.shape

        if dim is None:
            dim = batch_dim
        elif batch_dim != dim:
            raise ValueError(
                f"Inconsistent vector dimensionality: expected {dim}, "
                f"got {batch_dim} in batch {batch_idx}"
            )

        if requested_initial_vectors is not None:
            remaining = requested_initial_vectors - total_vectors
            if remaining <= 0:
                limit_reached = True
                break
            if batch_vectors > remaining:
                batch = batch[:remaining]
                batch_vectors = batch.shape[0]

        append_fvecs(RAW_BASE_FVECS, batch)

        total_vectors += batch_vectors
        batch_count += 1

        current_file = getattr(reader, "current_file", None)
        if current_file is None and batch_idx < len(INPUT_FILES):
            current_file = INPUT_FILES[batch_idx]

        file_label = Path(current_file).name if current_file is not None else "<unknown>"

        logger.info(
            "Progress: batch=%d file=%s batch_vectors=%d total_vectors=%d dim=%d",
            batch_idx,
            file_label,
            batch_vectors,
            total_vectors,
            dim,
        )

        if requested_initial_vectors is not None and total_vectors >= requested_initial_vectors:
            limit_reached = True
            break

    final_count, final_dim = count_output_file(RAW_BASE_FVECS)

    if requested_initial_vectors is not None and final_count < requested_initial_vectors:
        raise ValueError(
            f"Requested NUM_BASE={NUM_BASE} and NUM_QUERY={NUM_QUERY}, "
            f"so at least {requested_initial_vectors} input vectors are required, "
            f"but only {final_count} were available from the selected input files."
        )

    if final_count != total_vectors or final_dim != dim:
        raise RuntimeError(
            f"Post-write verification failed: "
            f"expected count={total_vectors}, dim={dim}; "
            f"got count={final_count}, dim={final_dim}"
        )

    return {
        "skipped": False,
        "file_prefix": FILE_PREFIX,
        "source_type": SOURCE_TYPE,
        "parquet_embedding_column": PARQUET_EMBEDDING_COLUMN,
        "input_files": [str(p) for p in INPUT_FILES],
        "num_input_files": len(INPUT_FILES),
        "num_batches": batch_count,
        "vectors_written": total_vectors,
        "dim": dim,
        "requested_initial_vectors": requested_initial_vectors,
        "limit_reached": limit_reached,
        "output_file": str(RAW_BASE_FVECS),
    }


def run_external_stage(
    logger: logging.Logger,
    stage_name: str,
    command: list[str],
    expected_outputs: Path | list[Path] | None = None,
) -> dict:
    if expected_outputs is None:
        outputs = []
    elif isinstance(expected_outputs, Path):
        outputs = [expected_outputs]
    else:
        outputs = list(expected_outputs)

    if outputs and all(p.exists() for p in outputs) and not OVERWRITE:
        logger.info(
            "Output already exists and OVERWRITE=False, skipping stage '%s': %s",
            stage_name,
            [str(p) for p in outputs],
        )
        result = {
            "stage": stage_name,
            "skipped": True,
            "output_files": [str(p) for p in outputs],
            "outputs": {},
        }

        for p in outputs:
            try:
                count, dim = count_output_file(p)
                result["outputs"][str(p)] = {
                    "vectors_written": count,
                    "dim": dim,
                }
            except Exception as e:
                result["outputs"][str(p)] = {
                    "count_error": repr(e),
                }

        return result

    logger.info("Starting external stage: %s", stage_name)
    logger.info("Command: %s", command)

    start = time.time()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    stage_metadata = {}

    assert process.stdout is not None
    for line in process.stdout:
        text = line.rstrip()
        logger.info("[%s] %s", stage_name, text)

        if stage_name == "remove_zeros" and text.startswith("Zero tolerance:"):
            stage_metadata["zero_tolerance"] = float(text.split(":", 1)[1].strip())

        if stage_name == "normalize":
            if text.startswith("Normalization tolerance:"):
                stage_metadata["normalization_tolerance"] = float(text.split(":", 1)[1].strip())
            elif text.startswith("Max abs norm error after:"):
                stage_metadata["max_abs_norm_error_after"] = float(text.split(":", 1)[1].strip())
            elif text.startswith("Mean abs norm error after:"):
                stage_metadata["mean_abs_norm_error_after"] = float(text.split(":", 1)[1].strip())

    returncode = process.wait()
    elapsed = time.time() - start

    if returncode != 0:
        raise RuntimeError(
            f"Stage '{stage_name}' failed with return code {returncode}"
        )

    result = {
        "stage": stage_name,
        "skipped": False,
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "output_files": [str(p) for p in outputs],
        "outputs": {},
    }

    if stage_metadata:
        result["metadata"] = stage_metadata

    for p in outputs:
        if not p.exists():
            raise RuntimeError(
                f"Stage '{stage_name}' succeeded but expected output does not exist: {p}"
            )
        count, dim = count_output_file(p)
        result["outputs"][str(p)] = {
            "vectors_written": count,
            "dim": dim,
        }

    logger.info("Finished external stage: %s in %.3f seconds", stage_name, elapsed)
    return result


def count_output_file(path: Path) -> tuple[int, int]:
    suffix = path.suffix.lower()

    if suffix == ".fvecs":
        return count_fvecs(path)

    if suffix == ".bvecs":
        return count_bvecs(path)

    if suffix == ".ivecs":
        _, num_rows, row_length = read_ivecs_info(str(path))
        if row_length is None:
            return 0, 0
        return num_rows, row_length

    raise ValueError(f"Unsupported output file type for counting: {path}")


def safe_delete(path: Path, logger: logging.Logger) -> None:
    if path.exists():
        path.unlink()
        logger.info("Deleted intermediate file: %s", path)


def safe_delete_dir(path: Path, logger: logging.Logger) -> None:
    if path.exists():
        shutil.rmtree(path)
        logger.info("Deleted temporary directory: %s", path)


def safe_rename(src: Path, dst: Path, logger: logging.Logger) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Cannot rename missing file: {src}")
    if dst.exists():
        dst.unlink()
    src.rename(dst)
    logger.info("Renamed final artifact: %s -> %s", src, dst)


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(LOG_FILE)

    logger.info("Starting pipeline run")
    logger.info("Run directory: %s", RUN_DIR)
    logger.info("File prefix: %s", FILE_PREFIX)
    logger.info("Source type: %s", SOURCE_TYPE)
    logger.info("Parquet embedding column: %s", PARQUET_EMBEDDING_COLUMN)
    logger.info("Remove zeros tolerance: %s", ZERO_TOLERANCE)
    logger.info("Normalization tolerance: %s", NORMALIZATION_TOLERANCE)
    logger.info("Quantize to INT8: %s", QUANTIZE_TO_INT8)

    start = time.time()
    success = True
    error = None

    summary = {
        "success": True,
        "elapsed_seconds": 0.0,
        "file_prefix": FILE_PREFIX,
        "tolerances": {
            "remove_zeros": ZERO_TOLERANCE,
            "normalization": NORMALIZATION_TOLERANCE,
        },
        "stages": {},
        "error": None,
    }

    try:
        summary["stages"]["extract_base"] = extract_base_vectors(logger)

        if QUANTIZE_TO_INT8:
            # --- Quantize first, before any cleaning ---
            summary["stages"]["quantize_int8"] = run_external_stage(
                logger,
                "quantize_int8",
                QUANTIZE_CMD,
                expected_outputs=RAW_QUANTIZED_BVECS,
            )
            if QUANTIZATION_META_FILE.exists():
                with open(QUANTIZATION_META_FILE, "r") as f:
                    summary["stages"]["quantize_int8"]["meta"] = json.load(f)
            if CLEANUP_INTERMEDIATE_FVECS:
                safe_delete(RAW_BASE_FVECS, logger)

        summary["stages"]["remove_zeros"] = run_external_stage(
            logger,
            "remove_zeros",
            REMOVE_ZEROS_CMD,
            expected_outputs=NONZERO_BASE_FVECS,
        )
        if CLEANUP_INTERMEDIATE_FVECS:
            if QUANTIZE_TO_INT8:
                safe_delete(RAW_QUANTIZED_BVECS, logger)
            else:
                safe_delete(RAW_BASE_FVECS, logger)

        summary["stages"]["normalize"] = run_external_stage(
            logger,
            "normalize",
            NORMALIZE_CMD,
            expected_outputs=NORMALIZED_BASE_FVECS,
        )

        normalize_metadata = summary["stages"]["normalize"].get("metadata", {})
        if "mean_abs_norm_error_after" in normalize_metadata:
            logger.info(
                "Normalization mean abs norm error after: %s",
                normalize_metadata["mean_abs_norm_error_after"],
            )
        if "max_abs_norm_error_after" in normalize_metadata:
            logger.info(
                "Normalization max abs norm error after: %s",
                normalize_metadata["max_abs_norm_error_after"],
            )

        if CLEANUP_INTERMEDIATE_FVECS:
            safe_delete(NONZERO_BASE_FVECS, logger)

        summary["stages"]["dedup"] = run_external_stage(
            logger,
            "dedup",
            DEDUP_CMD,
            expected_outputs=DEDUP_BASE_FVECS,
        )
        summary["stages"]["dedup"]["report_file"] = str(DEDUP_REPORT)
        if CLEANUP_INTERMEDIATE_FVECS:
            safe_delete(NORMALIZED_BASE_FVECS, logger)

        if DEDUP_TEMP_DIR.exists():
            safe_delete_dir(DEDUP_TEMP_DIR, logger)

        summary["stages"]["split_queries"] = run_external_stage(
            logger,
            "split_queries",
            SPLIT_CMD,
            expected_outputs=[SPLIT_QUERY_FVECS, SPLIT_BASE_FVECS],
        )
        if CLEANUP_INTERMEDIATE_FVECS:
            safe_delete(DEDUP_BASE_FVECS, logger)

        if SPLIT_QPARTS_DIR.exists():
            safe_delete_dir(SPLIT_QPARTS_DIR, logger)
        if SPLIT_BPARTS_DIR.exists():
            safe_delete_dir(SPLIT_BPARTS_DIR, logger)

        summary["stages"]["ground_truth"] = run_external_stage(
            logger,
            "ground_truth",
            GROUND_TRUTH_CMD,
            expected_outputs=GROUND_TRUTH_FILE,
        )

        summary["stages"]["ground_truth"]["metric"] = GT_METRIC
        summary["stages"]["ground_truth"]["k"] = GT_K
        summary["stages"]["ground_truth"]["shuffle"] = GT_SHUFFLE
        summary["stages"]["ground_truth"]["gpus"] = GT_GPUS

        if QUANTIZE_TO_INT8:
            base_source = QUANTIZED_BASE_BVECS
            query_source = QUANTIZED_QUERY_BVECS
            ext = ".bvecs"
        else:
            if GT_PROCESSED_BASE_FVECS.exists() and GT_PROCESSED_QUERY_FVECS.exists():
                base_source = GT_PROCESSED_BASE_FVECS
                query_source = GT_PROCESSED_QUERY_FVECS
            else:
                base_source = SPLIT_BASE_FVECS
                query_source = SPLIT_QUERY_FVECS
            ext = ".fvecs"

        actual_query_count, _ = count_output_file(query_source)
        actual_base_count, _ = count_output_file(base_source)

        final_base_file = RUN_DIR / f"{FILE_PREFIX}_base_{actual_base_count}{ext}"
        final_query_file = RUN_DIR / f"{FILE_PREFIX}_query_{actual_query_count}{ext}"

        safe_rename(base_source, final_base_file, logger)
        safe_rename(query_source, final_query_file, logger)
        safe_rename(GROUND_TRUTH_FILE, FINAL_GROUND_TRUTH, logger)

        if CLEANUP_INTERMEDIATE_FVECS and not QUANTIZE_TO_INT8:
            if base_source != SPLIT_BASE_FVECS:
                safe_delete(SPLIT_BASE_FVECS, logger)
            if query_source != SPLIT_QUERY_FVECS:
                safe_delete(SPLIT_QUERY_FVECS, logger)

        summary["requested_counts"] = {
            "base": NUM_BASE,
            "query": NUM_QUERY,
        }

        summary["final_counts"] = {
            "base": actual_base_count,
            "query": actual_query_count,
        }

        summary["final_artifacts"] = {
            "base": str(final_base_file),
            "query": str(final_query_file),
            "ground_truth": str(FINAL_GROUND_TRUTH),
        }

    except Exception as e:
        success = False
        error = repr(e)
        logger.exception("Pipeline failed")

    finally:
        elapsed = time.time() - start
        summary["success"] = success
        summary["elapsed_seconds"] = elapsed
        summary["error"] = error

        with SUMMARY_FILE.open("w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Finished pipeline run in %.3f seconds", elapsed)
        logger.info("Summary written to %s", SUMMARY_FILE)

        if not success:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
