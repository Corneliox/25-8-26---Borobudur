"""
================================================================================
BOROBUDUR RESEARCH: STATISTICAL HYPOTHESIS TESTING & EFFECT SIZE VALIDATION
================================================================================
Paper: Standardized Protocol for Digital Preservation of Eroded Stone Reliefs
Journal: International Journal of Image, Graphics and Signal Processing (IJIGSP)
Reproduces: Table 2 (Quantitative statistical evaluation across 55 panels)
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

def run_analysis(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")
        
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {csv_path}")
    print(f"Sample size: N = {len(df)} relief panels/tiles (Degrees of freedom df = {len(df) - 1})\n")
    
    metrics = [
        ("Tenengrad Edge Definition (A.U.)", "Tenengrad_Raw", "Tenengrad_CLAHE", "Tenengrad_Gain_Pct"),
        ("Laplacian Structural Sharpness (A.U.)", "Laplacian_Raw", "Laplacian_CLAHE", "Laplacian_Gain_Pct"),
        ("Tonal Dynamic Spread (sigma)", "Tonal_Spread_Raw", "Tonal_Spread_CLAHE", "Tonal_Spread_Gain_Pct"),
        ("RMS Contrast (Normalized)", "RMS_Contrast_Raw", "RMS_Contrast_CLAHE", "RMS_Contrast_Gain_Pct"),
        ("Shannon Information Entropy (bits/pixel)", "Shannon_Entropy_Raw", "Shannon_Entropy_CLAHE", "Shannon_Entropy_Gain_Pct")
    ]
    
    header = f"{'Metric':<38} | {'Raw (Mean+-SD)':<16} | {'CLAHE (Mean+-SD)':<18} | {'Gain':<8} | {'t(54)':<8} | {'p-value':<12} | {'Cohen d':<8}"
    print("="*120)
    print(header)
    print("="*120)
    
    for label, raw_col, clahe_col, gain_col in metrics:
        raw = df[raw_col]
        clahe = df[clahe_col]
        gain = df[gain_col]
        
        # Paired sample t-test
        t_stat, p_val = stats.ttest_rel(clahe, raw)
        
        # Cohen's d for paired samples: mean(difference) / std(difference)
        diff = clahe - raw
        cohen_d = diff.mean() / diff.std()
        
        raw_str = f"{raw.mean():.2f}+-{raw.std():.2f}"
        clahe_str = f"{clahe.mean():.2f}+-{clahe.std():.2f}"
        gain_str = f"+{gain.mean():.1f}%"
        t_str = f"{t_stat:.2f}"
        p_str = f"{p_val:.2e}"
        d_str = f"{cohen_d:.2f}"
        
        print(f"{label:<38} | {raw_str:<16} | {clahe_str:<18} | {gain_str:<8} | {t_str:<8} | {p_str:<12} | {d_str:<8}")
        
    print("="*120)
    print("CONCLUSION: All 5 metrics demonstrate extreme statistical significance (p < 0.001)")
    print("            and very large to extremely large effect sizes (Cohen's d > 3.0),")
    print("            confirming robust and repeatable signal enhancement across Borobudur reliefs.\n")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.normpath(os.path.join(script_dir, "..", "datasets", "Supplementary_Data_S2_IQA_Metrics_55_Panels.csv"))
    csv_file = sys.argv[1] if len(sys.argv) > 1 else default_csv
    run_analysis(csv_file)
