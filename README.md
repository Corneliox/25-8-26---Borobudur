# Borobudur Stone Relief Standardized Preservation & CLAHE Pipeline

Official reproducibility repository for the IJIGSP revised manuscript.

## 1. Environment & Hardware Specifications (Reproducibility Paradata)
- **Local Image Processing & Controlled Ablation Workstation:** Python `3.12.5`, OpenCV `4.10.0`, NumPy `1.26+`, SciPy `1.16.1`, scikit-image `0.24.0`, PyTorch `2.5.1+cu118`, Ultralytics `8.4.120`, NVIDIA GeForce RTX 2050 (4 GB VRAM).
- **Cloud Deep Learning Benchmark (Kaggle):** `Intel(R) Xeon(R) CPU @ 2.00GHz` (4 vCPUs), `2 × NVIDIA Tesla T4`, `31.35 GB RAM`, Python `3.12.12`.
- **Deterministic Settings:** OpenCV CLAHE is strictly deterministic (`clipLimit=3.0`, `tileGridSize=(32,32)`). All downstream YOLO and Mask R-CNN experiments used fixed `seed=0` (`deterministic=True`).

## 2. Repository Contents
- `scripts/clahe_enhancement_pipeline.py`: Standardized grayscale + CLAHE preprocessing pipeline.
- `scripts/compute_iqa_metrics.py`: IQA metrics computation (Tenengrad, Laplacian, Tonal Spread, RMS Contrast, Shannon Entropy).
- `scripts/run_statistical_validation.py`: Paired t-test, Cohen's *d*, and 95% CI statistical validation.
- `scripts/solar_and_gsd_analysis.py`: NOAA solar altitude/azimuth trajectory (`2025-08-27`, Borobudur) and optical GSD calculations.
- `scripts/benchmark_enhancements.py` & `02b_recompute_s2_exact_benchmark.py`: 6-method enhancement comparison (`Raw`, `Global HE`, `Unclipped AHE`, `MSR Retinex`, `Lecca 2023 Color-CLAHE`, `Proposed CLAHE`), noise floor analysis, and non-circular grid search.
- `scripts/03_build_paired_dl_datasets.py`, `04_run_downstream_dl.py`, `05_run_maskrcnn_downstream.py`: Paired Raw vs. CLAHE downstream training & evaluation (`YOLOv8m-seg`, `YOLO11m-seg`, `Mask R-CNN ResNet50-FPN`).
- `datasets/`: Complete 55-tile EXIF audit, solar geometry, 6-method IQA metrics, statistical summaries, and downstream DL histories.
- `samples/`: 3 representative raw & CLAHE-enhanced relief tile pairs covering all three working distances (`1`: `230 cm`, `21a`: `153.33 cm`, `16a`: `73.66 cm`).
- `figures_300dpi/`: Publication-ready `300 dpi` analytical plots.

## 3. Quick Reproduction
```bash
pip install -r scripts/requirements.txt
python scripts/clahe_enhancement_pipeline.py --input samples/raw --output samples/output_check
python scripts/run_statistical_validation.py datasets/Supplementary_Data_S2_IQA_Metrics_55_Panels.csv
```
