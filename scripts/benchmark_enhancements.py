"""
Comprehensive Image Enhancement Benchmark, No-Reference IQA, Noise Analysis,
CLAHE Grid Search (Non-Circular Metrics), and Stratified Statistics
across the 55 Borobudur Relief Tiles (FullOriginal).
"""
import os
import glob
import csv
import cv2
import numpy as np
import pandas as pd
from scipy import stats
from skimage.restoration import estimate_sigma
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORIG_DIR = r"D:\~Ideas n Innovation\25-8-26 --Borobudur\dataset\Full\FullOriginal"
EXIF_CSV = r"D:\~Ideas n Innovation\25-8-26 --Borobudur\Revision\experiments\exif_audit_55_tiles.csv"
OUT_DIR = r"D:\~Ideas n Innovation\25-8-26 --Borobudur\Revision\experiments"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Exact IQA metric functions matching compute_iqa_metrics.py + Noise & BRISQUE-style naturalness
def calc_tenengrad(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.sqrt(np.mean(gx**2 + gy**2)))

def calc_laplacian(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def calc_tonal_spread(gray):
    return float(gray.std())

def calc_rms_contrast(gray):
    return float((gray.astype(np.float64) / 255.0).std())

def calc_shannon_entropy(gray):
    hist, _ = np.histogram(gray.ravel(), bins=256, range=[0, 256])
    p = hist / float(gray.size)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def calc_noise_sigma(gray):
    # Immerkaer / Wavelet median absolute deviation noise floor (fast, standard)
    return float(estimate_sigma(gray, channel_axis=None))

def calc_enr(gray, noise_sigma):
    # Edge-to-Noise Ratio: Tenengrad / noise_sigma
    ten = calc_tenengrad(gray)
    return float(ten / max(noise_sigma, 1e-6))

def calc_psnr(ref_gray, enh_gray):
    mse = np.mean((ref_gray.astype(np.float64) - enh_gray.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))

# Enhancement methods
def apply_he(gray):
    return cv2.equalizeHist(gray)

def apply_ahe(gray, tile=(32, 32)):
    # Unconstrained AHE simulated via very high clipLimit
    clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=tile)
    return clahe.apply(gray)

def apply_msr(gray, sigmas=(15, 80, 250)):
    # Multi-Scale Retinex on grayscale luminance, normalized to 0..255
    img_f = gray.astype(np.float64) + 1.0
    msr = np.zeros_like(img_f)
    for s in sigmas:
        blur = cv2.GaussianBlur(img_f, (0, 0), s)
        msr += np.log10(img_f) - np.log10(blur + 1.0)
    msr /= len(sigmas)
    # Robust 1%-99% contrast stretch to uint8
    lo, hi = np.percentile(msr, (1.0, 99.0))
    if hi <= lo:
        return gray.copy()
    out = np.clip((msr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out

def apply_lecca_meek_clahe(bgr):
    # Lecca (2023) MEEK baseline: RGB/LAB-L CLAHE (standard 8x8, clip=2.0) preserving chromatic space,
    # evaluated in luminance for fair IQA comparison
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enh = clahe.apply(l)
    lab_enh = cv2.merge([l_enh, a, b])
    bgr_enh = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr_enh, cv2.COLOR_BGR2GRAY)

def apply_proposed_clahe(gray, clip_limit=3.0, tile=(32, 32)):
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile).apply(gray)

# Load 55 image paths
img_paths = sorted(glob.glob(os.path.join(ORIG_DIR, "**", "*.png"), recursive=True))
print(f"Loaded {len(img_paths)} FullOriginal images.")
exif_df = pd.read_csv(EXIF_CSV)
exif_map = exif_df.set_index("Tile_ID").to_dict(orient="index")

# A. Benchmark all 6 methods across 55 images (downsample 0.5x ONLY if needed? NO: run at FULL resolution 6024x4024?
# Wait: 55 images at 6024x4024 x 6 methods: let's check runtime! For fast, exact verification, let's check full-res vs S2 CSV!)
records = []
for idx, p in enumerate(img_paths, 1):
    tid = os.path.splitext(os.path.basename(p))[0]
    bgr = cv2.imread(p)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    variants = {
        "Raw": gray,
        "Global_HE": apply_he(gray),
        "AHE_Unclipped": apply_ahe(gray, (32, 32)),
        "MSR_Retinex": apply_msr(gray),
        "Lecca_2023_ColorCLAHE": apply_lecca_meek_clahe(bgr),
        "Proposed_CLAHE": apply_proposed_clahe(gray, 3.0, (32, 32)),
    }
    meta = exif_map.get(tid, {})
    for mname, img_v in variants.items():
        ns = calc_noise_sigma(img_v)
        ten = calc_tenengrad(img_v)
        records.append({
            "Tile_ID": tid,
            "Method": mname,
            "ISO": meta.get("ISO", np.nan),
            "Camera": meta.get("Camera_Model", ""),
            "Tenengrad": ten,
            "Laplacian": calc_laplacian(img_v),
            "Tonal_Spread": calc_tonal_spread(img_v),
            "RMS_Contrast": calc_rms_contrast(img_v),
            "Shannon_Entropy": calc_shannon_entropy(img_v),
            "Noise_Sigma": ns,
            "ENR": ten / max(ns, 1e-6),
            "PSNR_vs_Raw": calc_psnr(gray, img_v),
        })
    if idx % 10 == 0 or idx == len(img_paths):
        print(f"  Processed {idx}/{len(img_paths)} tiles...", flush=True)

df = pd.DataFrame(records)
df.to_csv(os.path.join(OUT_DIR, "benchmark_6methods_55tiles.csv"), index=False)

# Summary table across 6 methods
summary_rows = []
for mname in ["Raw", "Global_HE", "AHE_Unclipped", "MSR_Retinex", "Lecca_2023_ColorCLAHE", "Proposed_CLAHE"]:
    sub = df[df["Method"] == mname]
    row = {"Method": mname}
    for col in ["Tenengrad", "Laplacian", "Tonal_Spread", "RMS_Contrast", "Shannon_Entropy", "Noise_Sigma", "ENR", "PSNR_vs_Raw"]:
        mean = sub[col].mean()
        sd = sub[col].std(ddof=1)
        se = sd / np.sqrt(len(sub))
        ci95 = stats.t.ppf(0.975, df=len(sub)-1) * se
        row[f"{col}_Mean"] = mean
        row[f"{col}_SD"] = sd
        row[f"{col}_SE"] = se
        row[f"{col}_CI95"] = ci95
    summary_rows.append(row)

sum_df = pd.DataFrame(summary_rows)
sum_df.to_csv(os.path.join(OUT_DIR, "benchmark_summary_stats.csv"), index=False)

print("\n=== BENCHMARK SUMMARY (N=55, Mean +/- SD) ===")
for _, r in sum_df.iterrows():
    print(f"{r['Method']:<22} | Ten={r['Tenengrad_Mean']:6.2f}±{r['Tenengrad_SD']:5.2f} | "
          f"Lap={r['Laplacian_Mean']:6.2f}±{r['Laplacian_SD']:6.2f} | "
          f"Ent={r['Shannon_Entropy_Mean']:.3f} | Noise={r['Noise_Sigma_Mean']:.2f} | ENR={r['ENR_Mean']:.2f}")

# B. Paired stats Raw vs Proposed_CLAHE with 95% CI, SE, Mean Per-Tile Gain, and Ratio of Means
raw_df = df[df["Method"] == "Raw"].set_index("Tile_ID")
prop_df = df[df["Method"] == "Proposed_CLAHE"].set_index("Tile_ID")
print("\n=== PAIRED STATS: RAW vs PROPOSED CLAHE (N=55) ===")
for col in ["Tenengrad", "Laplacian", "Tonal_Spread", "RMS_Contrast", "Shannon_Entropy", "Noise_Sigma", "ENR"]:
    a = raw_df[col]
    b = prop_df[col]
    diff = b - a
    rel_gain = ((b - a) / a) * 100.0
    t_val, p_val = stats.ttest_rel(b, a)
    d_val = diff.mean() / diff.std(ddof=1)
    se_gain = rel_gain.std(ddof=1) / np.sqrt(len(rel_gain))
    ci_gain = stats.t.ppf(0.975, df=len(rel_gain)-1) * se_gain
    ratio_means = ((b.mean() - a.mean()) / a.mean()) * 100.0
    print(f"{col:<16} | Raw={a.mean():.2f}±{a.std():.2f} | CLAHE={b.mean():.2f}±{b.std():.2f} | "
          f"PerTileGain={rel_gain.mean():+.1f}% (±{ci_gain:.1f}% 95%CI, SE={se_gain:.2f}%) | "
          f"RatioMeans={ratio_means:+.1f}% | t={t_val:.2f}, p={p_val:.2e}, d={d_val:.2f}")

# C. Stratified Analysis by ISO & Surface Baseline Contrast Tertiles
merged = prop_df.copy()
for col in ["Tenengrad", "Laplacian", "Noise_Sigma", "ENR"]:
    merged[f"{col}_Raw"] = raw_df[col]
    merged[f"{col}_GainPct"] = ((prop_df[col] - raw_df[col]) / raw_df[col]) * 100.0

merged["ISO_Group"] = pd.cut(merged["ISO"], bins=[0, 125, 250, 1000], labels=["Low (ISO 50-125)", "Medium (ISO 160-250)", "High (ISO 400-800)"])
print("\n=== NOISE & SHARPNESS BY ISO GROUP ===")
print(merged.groupby("ISO_Group", observed=False)[["Noise_Sigma_Raw", "Noise_Sigma", "Noise_Sigma_GainPct", "ENR_Raw", "ENR", "Tenengrad_GainPct"]].mean().round(2))

merged["Surface_Contrast_Tier"] = pd.qcut(raw_df["RMS_Contrast"], q=3, labels=["Low-Contrast/Weathered (Worst)", "Moderate Contrast", "High-Contrast (Best)"])
print("\n=== STRATIFIED BY BASELINE SURFACE CONTRAST (EROSION/WEATHERING PROXY) ===")
print(merged.groupby("Surface_Contrast_Tier", observed=False)[["Tenengrad_Raw", "Tenengrad", "Tenengrad_GainPct", "Laplacian_Raw", "Laplacian", "Laplacian_GainPct"]].mean().round(2))

# D. CLAHE Grid Search on representative subset (or all 55? Let's run on 12 stratified tiles across all 3 distances/ISO for speed & accuracy)
sample_ids = ["1", "2", "10", "16a", "16b", "21a", "24a", "26", "31a", "34", "38a", "38c"]
clip_vals = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
grid_vals = [4, 8, 16, 32]
gs_rows = []
sample_grays = {}
for sid in sample_ids:
    p = os.path.join(ORIG_DIR, f"{sid}.png")
    if not os.path.exists(p):
        matches = glob.glob(os.path.join(ORIG_DIR, "**", f"{sid}.png"), recursive=True)
        p = matches[0]
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    # Downscale 2x for fast 36-config grid search consistency
    sample_grays[sid] = cv2.resize(im, (im.shape[1]//2, im.shape[0]//2), interpolation=cv2.INTER_AREA)

for g in grid_vals:
    for c in clip_vals:
        tens, laps, ents, noises, enrs, psnrs = [], [], [], [], [], []
        for sid, im in sample_grays.items():
            enh = cv2.createCLAHE(clipLimit=c, tileGridSize=(g, g)).apply(im)
            ns = calc_noise_sigma(enh)
            t = calc_tenengrad(enh)
            tens.append(t)
            laps.append(calc_laplacian(enh))
            ents.append(calc_shannon_entropy(enh))
            noises.append(ns)
            enrs.append(t / max(ns, 1e-6))
            psnrs.append(calc_psnr(im, enh))
        gs_rows.append({
            "tileGridSize": g, "clipLimit": c,
            "Tenengrad": np.mean(tens), "Laplacian": np.mean(laps),
            "Entropy": np.mean(ents), "Noise_Sigma": np.mean(noises),
            "ENR": np.mean(enrs), "PSNR_Fidelity": np.mean(psnrs)
        })

gs_df = pd.DataFrame(gs_rows)
gs_df.to_csv(os.path.join(OUT_DIR, "clahe_grid_search_noncircular.csv"), index=False)

# Plot 300 dpi Grid Search figure (Edge-to-Noise Ratio + Noise Amplification vs ClipLimit)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=300)
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
colors = {4: "#7f7f7f", 8: "#2ca02c", 16: "#ff7f0e", 32: "#1f77b4"}
for g in grid_vals:
    sub = gs_df[gs_df["tileGridSize"] == g]
    ax1.plot(sub["clipLimit"], sub["Tenengrad"], marker="o", ms=4, lw=1.8, color=colors[g], label=f"{g}×{g} grid")
    ax2.plot(sub["clipLimit"], sub["Noise_Sigma"], marker="s", ms=4, lw=1.8, color=colors[g], label=f"{g}×{g} grid")
ax1.axvline(3.0, color="red", linestyle="--", lw=1.2, label="Selected clipLimit=3.0")
ax2.axvline(3.0, color="red", linestyle="--", lw=1.2, label="Selected clipLimit=3.0")
ax1.set_title("(a) Edge Gradient (Tenengrad) vs. Clip Limit", fontsize=10, fontweight="bold")
ax1.set_xlabel("CLAHE Clip Limit", fontsize=9)
ax1.set_ylabel("Mean Tenengrad Gradient (A.U.)", fontsize=9)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(fontsize=8)
ax2.set_title("(b) Estimated Noise Floor (σ_noise) vs. Clip Limit", fontsize=10, fontweight="bold")
ax2.set_xlabel("CLAHE Clip Limit", fontsize=9)
ax2.set_ylabel("Wavelet Noise Sigma (A.U.)", fontsize=9)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "Fig_CLAHE_Tradeoff_300dpi.png"), dpi=300)
plt.close(fig)
print("\nSaved benchmark CSVs and 300-dpi tradeoff plot successfully!")
