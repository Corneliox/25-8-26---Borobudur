"""
Recompute 6-method benchmark using the exact S2/Table 2 definitions:
- Tenengrad = mean(sqrt(Gx^2 + Gy^2))
- RMS Contrast = std(I) / mean(I)  (Coefficient of Variation / Normalized RMS)
- Laplacian = var(Laplacian(I))
- Tonal Spread = std(I)
- Shannon Entropy = -sum(p * log2(p))
And merge Noise_Sigma from benchmark_6methods_55tiles.csv + generate 300 dpi figures.
"""
import os, glob
import cv2
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORIG_DIR = r"D:\~Ideas n Innovation\25-8-26 --Borobudur\dataset\Full\FullOriginal"
OUT_DIR = r"D:\~Ideas n Innovation\25-8-26 --Borobudur\Revision\experiments"
PREV_CSV = os.path.join(OUT_DIR, "benchmark_6methods_55tiles.csv")

prev_df = pd.read_csv(PREV_CSV)
noise_lookup = prev_df.set_index(["Tile_ID", "Method"])["Noise_Sigma"].to_dict()

def calc_tenengrad_s2(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx**2 + gy**2)))

def calc_rms_s2(gray):
    m = float(gray.mean())
    return float(gray.std() / m) if m > 0 else 0.0

def calc_laplacian(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def calc_tonal(gray):
    return float(gray.std())

def calc_entropy(gray):
    hist, _ = np.histogram(gray.ravel(), bins=256, range=[0, 256])
    p = hist / float(gray.size)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def apply_msr(gray, sigmas=(15, 80, 250)):
    h, w = gray.shape
    img_f = gray.astype(np.float32) + 1.0
    msr = np.zeros_like(img_f)
    for s in sigmas:
        if s >= 50:
            # Downsample 4x for large-scale illumination Gaussian estimation (exact low-freq envelope, 16x faster)
            small = cv2.resize(img_f, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
            blur_small = cv2.GaussianBlur(small, (0, 0), s / 4.0)
            blur = cv2.resize(blur_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            blur = cv2.GaussianBlur(img_f, (0, 0), s)
        msr += np.log10(img_f) - np.log10(blur + 1.0)
    msr /= len(sigmas)
    lo, hi = np.percentile(msr, (1.0, 99.0))
    return np.clip((msr - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)

def apply_lecca(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_enh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    bgr_enh = cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr_enh, cv2.COLOR_BGR2GRAY)

rows = []
img_paths = sorted(glob.glob(os.path.join(ORIG_DIR, "**", "*.png"), recursive=True))
for idx, p in enumerate(img_paths, 1):
    tid = os.path.splitext(os.path.basename(p))[0]
    bgr = cv2.imread(p)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    variants = {
        "Raw": gray,
        "Global_HE": cv2.equalizeHist(gray),
        "AHE_Unclipped": cv2.createCLAHE(clipLimit=100.0, tileGridSize=(32, 32)).apply(gray),
        "MSR_Retinex": apply_msr(gray),
        "Lecca_2023_ColorCLAHE": apply_lecca(bgr),
        "Proposed_CLAHE": cv2.createCLAHE(clipLimit=3.0, tileGridSize=(32, 32)).apply(gray),
    }
    for mname, img_v in variants.items():
        ten = calc_tenengrad_s2(img_v)
        ns = noise_lookup.get((tid, mname), np.nan)
        rows.append({
            "Tile_ID": tid,
            "Method": mname,
            "Tenengrad_S2": ten,
            "Laplacian": calc_laplacian(img_v),
            "Tonal_Spread": calc_tonal(img_v),
            "RMS_Contrast_S2": calc_rms_s2(img_v),
            "Shannon_Entropy": calc_entropy(img_v),
            "Noise_Sigma": ns,
            "ENR_S2": ten / ns if ns else np.nan,
        })
    if idx % 10 == 0 or idx == len(img_paths):
        print(f"Processed {idx}/{len(img_paths)} tiles", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "benchmark_6methods_55tiles_s2exact.csv"), index=False)

print("=== S2-EXACT 6-METHOD BENCHMARK (N=55) ===")
for mname in ["Raw", "Global_HE", "AHE_Unclipped", "MSR_Retinex", "Lecca_2023_ColorCLAHE", "Proposed_CLAHE"]:
    sub = df[df["Method"] == mname]
    print(f"{mname:<22} | Ten={sub['Tenengrad_S2'].mean():6.2f}±{sub['Tenengrad_S2'].std():5.2f} | "
          f"Lap={sub['Laplacian'].mean():6.2f}±{sub['Laplacian'].std():6.2f} | "
          f"Tonal={sub['Tonal_Spread'].mean():5.2f}±{sub['Tonal_Spread'].std():4.2f} | "
          f"RMS={sub['RMS_Contrast_S2'].mean():.2f}±{sub['RMS_Contrast_S2'].std():.2f} | "
          f"Ent={sub['Shannon_Entropy'].mean():.2f}±{sub['Shannon_Entropy'].std():.2f} | "
          f"Noise={sub['Noise_Sigma'].mean():.2f} | ENR={sub['ENR_S2'].mean():.2f}")

raw_df = df[df["Method"] == "Raw"].set_index("Tile_ID")
prop_df = df[df["Method"] == "Proposed_CLAHE"].set_index("Tile_ID")
print("\n=== TABLE 2 EXACT REPLICATION WITH SE & 95% CI (N=55) ===")
for col in ["Tenengrad_S2", "Laplacian", "Tonal_Spread", "RMS_Contrast_S2", "Shannon_Entropy"]:
    a, b = raw_df[col], prop_df[col]
    rel = ((b - a) / a) * 100.0
    se = rel.std(ddof=1) / np.sqrt(len(rel))
    ci95 = stats.t.ppf(0.975, df=len(rel)-1) * se
    t_val, p_val = stats.ttest_rel(b, a)
    d_val = (b - a).mean() / (b - a).std(ddof=1)
    ratio_m = ((b.mean() - a.mean()) / a.mean()) * 100.0
    print(f"{col:<16} | Raw={a.mean():.2f}±{a.std():.2f} | CLAHE={b.mean():.2f}±{b.std():.2f} | "
          f"Gain={rel.mean():+.1f}% (SE={se:.2f}%, 95%CI=±{ci95:.1f}%) | RatioMeans={ratio_m:+.1f}% | "
          f"t={t_val:.2f}, p={p_val:.2e}, d={d_val:.2f}")
