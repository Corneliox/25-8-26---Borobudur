"""
Downstream Deep Learning Validation (Major Issue #1 & #4):
Controlled Raw vs. CLAHE comparison across two model families/generations:
1. YOLOv8m-seg (Box Object Detection + Mask Instance Segmentation)
2. YOLO11m-seg (Box Object Detection + Mask Instance Segmentation)
Strictly controlled: identical splits (159 train / 57 val patches from 55 tiles),
seed=0, deterministic=True, imgsz=640, epochs=60, AdamW.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

OUT_DIR = r"D:\~Ideas n Innovation\25-8-26 --Borobudur\Revision\experiments"
DS_BASE = os.path.join(OUT_DIR, "dl_datasets")
RUNS_DIR = os.path.join(OUT_DIR, "dl_runs")
os.makedirs(RUNS_DIR, exist_ok=True)

MODELS = {
    "YOLOv8m-seg": r"D:\~Ideas n Innovation\25-8-26 --Borobudur\~Final\Kaggle\Ver 18\yolov8m-seg.pt",
    "YOLO11m-seg": r"D:\~Ideas n Innovation\25-8-26 --Borobudur\~Final\Kaggle\Ver 18\yolo11m-seg.pt",
}
DATASETS = {
    "Raw": os.path.join(DS_BASE, "ds_raw", "data.yaml"),
    "CLAHE": os.path.join(DS_BASE, "ds_clahe", "data.yaml"),
}

def main():
    summary_rows = []
    curves = {}

    for m_name, m_weights in MODELS.items():
        for d_name, d_yaml in DATASETS.items():
            run_name = f"{m_name}_{d_name}"
            print(f"\n>>> Training {run_name} <<<", flush=True)
            model = YOLO(m_weights)
            model.train(
                data=d_yaml,
                epochs=50,
                patience=20,
                imgsz=640,
                batch=4,
                device=0,
                workers=0,
                optimizer="AdamW",
                lr0=0.001,
                hsv_h=0.0,
                hsv_s=0.0,
                hsv_v=0.0,
                seed=0,
                deterministic=True,
                amp=True,
                project=RUNS_DIR,
                name=run_name,
                exist_ok=True,
                verbose=False,
            )
            res_csv = os.path.join(RUNS_DIR, run_name, "results.csv")
            df = pd.read_csv(res_csv)
            df.columns = [c.strip() for c in df.columns]
            curves[run_name] = df
            best_idx = df["metrics/mAP50(M)"].idxmax()
            best_row = df.loc[best_idx]
            summary_rows.append({
                "Model": m_name,
                "Preprocessing": d_name,
                "Best_Epoch": int(best_row["epoch"]),
                "Box_Precision": float(best_row["metrics/precision(B)"]),
                "Box_Recall": float(best_row["metrics/recall(B)"]),
                "Box_mAP50": float(best_row["metrics/mAP50(B)"]),
                "Box_mAP50_95": float(best_row["metrics/mAP50-95(B)"]),
                "Mask_Precision": float(best_row["metrics/precision(M)"]),
                "Mask_Recall": float(best_row["metrics/recall(M)"]),
                "Mask_mAP50": float(best_row["metrics/mAP50(M)"]),
                "Mask_mAP50_95": float(best_row["metrics/mAP50-95(M)"]),
                "Final_Epoch_Box_mAP50": float(df["metrics/mAP50(B)"].iloc[-1]),
                "Final_Epoch_Mask_mAP50": float(df["metrics/mAP50(M)"].iloc[-1]),
            })

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(os.path.join(OUT_DIR, "dl_downstream_summary.csv"), index=False)
    print("\n=== DOWNSTREAM DL SUMMARY ===")
    print(sum_df.to_string(index=False))

    # Plot 300 dpi convergence curves (Box mAP50, Mask mAP50, and Val Loss)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=300)
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    styles = {
        "YOLOv8m-seg_Raw": ("#7f7f7f", "--", "YOLOv8m-seg (Raw)"),
        "YOLOv8m-seg_CLAHE": ("#1f77b4", "-", "YOLOv8m-seg (CLAHE)"),
        "YOLO11m-seg_Raw": ("#d62728", "--", "YOLO11m-seg (Raw)"),
        "YOLO11m-seg_CLAHE": ("#2ca02c", "-", "YOLO11m-seg (CLAHE)"),
    }
    for k, df in curves.items():
        col, ls, lbl = styles[k]
        axes[0].plot(df["epoch"], df["metrics/mAP50(B)"], color=col, linestyle=ls, lw=1.8, label=lbl)
        axes[1].plot(df["epoch"], df["metrics/mAP50(M)"], color=col, linestyle=ls, lw=1.8, label=lbl)
        val_loss = df["val/box_loss"] + df["val/seg_loss"] + df["val/cls_loss"]
        axes[2].plot(df["epoch"], val_loss, color=col, linestyle=ls, lw=1.8, label=lbl)

    axes[0].set_title("(a) Detection Convergence (Box mAP@0.5)", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("mAP@0.5 (Box)")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=7.5)

    axes[1].set_title("(b) Segmentation Convergence (Mask mAP@0.5)", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP@0.5 (Mask)")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=7.5)

    axes[2].set_title("(c) Validation Composite Loss", fontsize=10, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Val Loss (Box + Seg + Cls)")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(fontsize=7.5)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Fig_Downstream_DL_Convergence_300dpi.png"), dpi=300)
    plt.close(fig)
    print("Saved DL downstream summary and 300-dpi convergence plot!")

if __name__ == "__main__":
    main()
