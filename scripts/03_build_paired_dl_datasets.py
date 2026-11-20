"""
Build strictly paired Raw vs. CLAHE instance segmentation datasets from all 55 Borobudur tiles
using Roboflow polygon annotations (Borobudur-Kelompok-1-3 + Borobudur-Kelompok-2-1),
with deterministic 3x2 overlapping tiling (2400x2400 patches) and zero tile-level leakage.
"""
import os
import glob
import shutil
import cv2
import numpy as np

ORIG_DIR = r"D:\~Ideas n Innovation\25-8-26 --Borobudur\dataset\Full\FullOriginal"
RF_DIRS = [
    r"D:\~Ideas n Innovation\25-8-26 --Borobudur\~Final\Kaggle\Ver 39\Borobudur-Kelompok-1-3",
    r"D:\~Ideas n Innovation\25-8-26 --Borobudur\~Final\Kaggle\Ver 39\Borobudur-Kelompok-2-1",
]
OUT_BASE = r"D:\~Ideas n Innovation\25-8-26 --Borobudur\Revision\experiments\dl_datasets"

CLASSES = ['face', 'foot', 'hand', 'lower_arm', 'lower_leg', 'torso', 'upper_arm', 'upper_leg']

# Map original image paths by Tile_ID
orig_map = {}
for p in glob.glob(os.path.join(ORIG_DIR, "**", "*.png"), recursive=True):
    tid = os.path.splitext(os.path.basename(p))[0]
    orig_map[tid] = p

# Collect all 55 tile annotations and splits (train: 38, valid+test -> val: 17)
# Keeping Roboflow's train split (19+19=38) and combining valid+test (2+6+3+6=17) as held-out val/test
# Wait: let's keep train (38), valid (5), test (12) AND merge valid+test if desired, or keep train/val/test!
tile_items = []
for rf_dir in RF_DIRS:
    for sp in ["train", "valid", "test"]:
        out_sp = "val" if sp in ("valid", "test") else "train"
        for lbl_p in sorted(glob.glob(os.path.join(rf_dir, sp, "labels", "*.txt"))):
            tid = os.path.basename(lbl_p).split("_processed")[0]
            polys = []
            for line in open(lbl_p).read().splitlines():
                parts = line.strip().split()
                if len(parts) >= 7:
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    pts = np.array(coords, dtype=np.float64).reshape(-1, 2)
                    polys.append((cls_id, pts))
            tile_items.append((tid, out_sp, polys))

print(f"Total tiles collected: {len(tile_items)} (train={sum(1 for x in tile_items if x[1]=='train')}, val={sum(1 for x in tile_items if x[1]=='val')})")

# Prepare directories for Raw and CLAHE
for ds_name in ["ds_raw", "ds_clahe"]:
    ds_root = os.path.join(OUT_BASE, ds_name)
    if os.path.exists(ds_root):
        shutil.rmtree(ds_root)
    for sp in ["train", "val"]:
        os.makedirs(os.path.join(ds_root, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(ds_root, "labels", sp), exist_ok=True)
    yaml_path = os.path.join(ds_root, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {ds_root}\ntrain: images/train\nval: images/val\nnc: {len(CLASSES)}\nnames: {CLASSES}\n")

# Deterministic 3x2 grid windows on 6024 x 4024
# x starts: [0, 1812, 3624] with width 2400 -> covers [0..2400], [1812..4212], [3624..6024]
# y starts: [0, 1624] with height 2400 -> covers [0..2400], [1624..4024]
W, H = 6024, 4024
PW, PH = 2400, 2400
X_STARTS = [0, 1812, 3624]
Y_STARTS = [0, 1624]
TARGET_SZ = 640  # Clean 640x640 patches for fast paired DL comparison

def clip_polygon_to_box(pts_px, x0, y0, pw, ph):
    # Check bounding box overlap first, then clip coordinates into patch [0..pw, 0..ph]
    xs, ys = pts_px[:, 0], pts_px[:, 1]
    bx0, bx1 = xs.min(), xs.max()
    by0, by1 = ys.min(), ys.max()
    ix0, iy0 = max(bx0, x0), max(by0, y0)
    ix1, iy1 = min(bx1, x0 + pw), min(by1, y0 + ph)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    inter_area = (ix1 - ix0) * (iy1 - iy0)
    orig_area = max((bx1 - bx0) * (by1 - by0), 1.0)
    if inter_area / orig_area < 0.35:  # Keep instances with >=35% bbox inside patch
        return None
    # Shift and clip vertices
    nx = np.clip((xs - x0) / pw, 0.0, 1.0)
    ny = np.clip((ys - y0) / ph, 0.0, 1.0)
    if (nx.max() - nx.min()) < 0.01 or (ny.max() - ny.min()) < 0.01:
        return None
    return np.stack([nx, ny], axis=1)

clahe_obj = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(32, 32))
counts = {"train": 0, "val": 0}
inst_counts = {"train": 0, "val": 0}

for idx, (tid, sp, polys) in enumerate(tile_items, 1):
    bgr = cv2.imread(orig_map[tid])
    assert bgr.shape[:2] == (H, W), f"Unexpected shape {bgr.shape} for {tid}"
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    enh = clahe_obj.apply(gray)
    enh_bgr = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)

    for yi, y0 in enumerate(Y_STARTS):
        for xi, x0 in enumerate(X_STARTS):
            patch_polys = []
            for cls_id, pts_norm in polys:
                pts_px = pts_norm.copy()
                pts_px[:, 0] *= W
                pts_px[:, 1] *= H
                clipped = clip_polygon_to_box(pts_px, x0, y0, PW, PH)
                if clipped is not None:
                    patch_polys.append((cls_id, clipped))
            if not patch_polys:
                continue  # Skip empty stone-border patches without annotations

            patch_name = f"{tid}_r{yi}_c{xi}"
            raw_crop = cv2.resize(bgr[y0:y0+PH, x0:x0+PW], (TARGET_SZ, TARGET_SZ), interpolation=cv2.INTER_AREA)
            clahe_crop = cv2.resize(enh_bgr[y0:y0+PH, x0:x0+PW], (TARGET_SZ, TARGET_SZ), interpolation=cv2.INTER_AREA)

            cv2.imwrite(os.path.join(OUT_BASE, "ds_raw", "images", sp, patch_name + ".jpg"), raw_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(OUT_BASE, "ds_clahe", "images", sp, patch_name + ".jpg"), clahe_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            lbl_lines = []
            for cls_id, cpts in patch_polys:
                coords_str = " ".join(f"{v:.6f}" for v in cpts.flatten())
                lbl_lines.append(f"{cls_id} {coords_str}")
            lbl_txt = "\n".join(lbl_lines) + "\n"

            with open(os.path.join(OUT_BASE, "ds_raw", "labels", sp, patch_name + ".txt"), "w") as f:
                f.write(lbl_txt)
            with open(os.path.join(OUT_BASE, "ds_clahe", "labels", sp, patch_name + ".txt"), "w") as f:
                f.write(lbl_txt)

            counts[sp] += 1
            inst_counts[sp] += len(patch_polys)

print(f"Built paired datasets! Patches: {counts}, Instances: {inst_counts}")
