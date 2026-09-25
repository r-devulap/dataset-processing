# Vector Embedding Dataset Processing Pipeline

A reusable Python pipeline for turning downloaded embedding datasets into clean ANN benchmark artifacts.

### Current Outputs
- **Base vectors**: `.fvecs` (or `.bvecs` when INT8 quantization is enabled)
- **Query vectors**: `.fvecs` (or `.bvecs` when INT8 quantization is enabled)
- **Ground truth**: `.ivecs`

The pipeline is designed for large embedding datasets and supports a staged workflow with logs, a run summary, cleanup of intermediate files, and dataset-specific configuration.

---

## Current Scope

This repository currently supports:
- **.npy embedding inputs**
- **parquet embedding inputs**, where a configured parquet column contains one embedding vector per row as a list of floating-point values
- **.fvecs embedding inputs**, for cases where embeddings are already stored in ANN-style `.fvecs` files and should be processed directly
- **.hdf5 / .h5 embedding inputs**, where a configured dataset key is read directly from the HDF5 file

---

## What This Project Does

Given one or more embedding files, this project can:
1. Extract vectors into a single base `.fvecs` file
2. (Optional) Quantize float32 vectors to signed INT8 `.bvecs` — runs **first**, before any cleaning, so all downstream stages operate on quantized data
3. Remove exact zero vectors
4. Normalize vectors when needed
5. Deduplicate vectors
6. Sample query vectors without replacement from the cleaned vector set
7. Generate exact ground-truth nearest neighbors for the final base/query split
8. Log progress, output stats, and errors at each stage
9. Clean up intermediate files after successful downstream stages

---

## Getting Started

### Example Run
After editing `config.py`, run the pipeline from the repository root with:

    python3 processing.py

---

## Configuration (`config.py`)

At a minimum, you should set the following values in your dataset config under `processing_configs/` before running the pipeline. The top of `config.py` selects which dataset config to use:

    # config.py
    from processing_configs import sift1m_hdf5_proc as dataset

### Required Settings

| Setting | Description |
| :--- | :--- |
| **OUTPUT_DIR** | Path to the run output directory |
| **FILE_PREFIX** | Common prefix for output artifacts |
| **NUM_BASE** | Requested final number of base vectors |
| **NUM_QUERY** | Requested final number of query vectors |
| **SOURCE_TYPE** | Input format: `"npy"`, `"parquet"`, `"fvecs"`, or `"hdf5"` |

### Ground Truth Settings
- **GT_K**: Number of nearest neighbors to compute
- **GT_METRIC**: `"ip"` or `"l2"`
- **GT_SHUFFLE**: Whether to let knn_utils.py shuffle before ground truth generation
- **GT_GPUS**: `"-1"` for CPU, or values such as `"0"` or `"0,1"` for GPU execution

### HDF5 Settings

Required when `SOURCE_TYPE = "hdf5"`.

| Setting | Default | Description |
| :--- | :--- | :--- |
| **HDF5_DATASET_NAME** | `"train"` | Dataset key to read inside the HDF5 file |

### INT8 Quantization Settings

These are optional. When omitted, the pipeline defaults to float32 `.fvecs` outputs.

| Setting | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **QUANTIZE_TO_INT8** | `bool` | `False` | When `True`, quantizes the raw extracted `.fvecs` to signed INT8 `.bvecs` as the **first** pipeline step. All subsequent stages (zero removal, normalization, deduplication, split, ground truth) operate on the `.bvecs` data |
| **INT8_QUANTIZATION_MODE** | `str` | `"symmetric_max_abs"` | Quantization mode. `"symmetric_max_abs"` scales by `127.0 / max_abs` across the full dataset |

---

## Minimal Example Configuration

    from pathlib import Path
    from processing_configs import sift1m_hdf5_proc as dataset

    OUTPUT_DIR = dataset.OUTPUT_DIR
    FILE_PREFIX = dataset.FILE_PREFIX
    CLEANUP_INTERMEDIATE_FVECS = True
    OVERWRITE = False

    NUM_BASE = dataset.NUM_BASE
    NUM_QUERY = dataset.NUM_QUERY

    GT_K = 100
    GT_METRIC = "l2"
    GT_SHUFFLE = False
    GT_GPUS = "-1"

    SOURCE_TYPE = dataset.SOURCE_TYPE
    HDF5_DATASET_NAME = dataset.HDF5_DATASET_NAME

    QUANTIZE_TO_INT8 = True
    INT8_QUANTIZATION_MODE = "symmetric_max_abs"

---

## Notes on Output

The pipeline writes outputs into the configured `OUTPUT_DIR`. Final artifacts are named using the `FILE_PREFIX`, actual vector counts, and ground truth parameters.

When `QUANTIZE_TO_INT8 = False` (default):
- `<prefix>_base_<actual_count>.fvecs`
- `<prefix>_query_<actual_count>.fvecs`
- `<prefix>_gt_<metric>_<k>.ivecs`

When `QUANTIZE_TO_INT8 = True`:
- `<prefix>_base_<actual_count>.bvecs`
- `<prefix>_query_<actual_count>.bvecs`
- `<prefix>_gt_<metric>_<k>.ivecs`
