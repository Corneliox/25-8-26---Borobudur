"""
Downstream Two-Stage Instance Segmentation Validation (Major Issue #1):
Mask R-CNN (ResNet-50-FPN) comparing Raw vs. CLAHE on the exact same
paired Borobudur relief patch dataset (159 train / 57 val patches, 8 classes),
with fixed random seed (seed=0) and per-epoch training loss + validation mIoU / mAP@0.5.
"""
import os
import glob
import random
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = r"D:\~Ideas n Innovation\25-8-26 --Borobudur\Revision\experiments"
DS_BASE = os.path.join(OUT_DIR, "dl_datasets")
NUM_CLASSES = 9  # 8 relief body-part classes + 1 background

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class BorobudurSegDataset(Dataset):
    def __init__(self, ds_root, split="train"):
        self.img_dir = os.path.join(ds_root, "images", split)
        self.lbl_dir = os.path.join(ds_root, "labels", split)
        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_p = self.img_paths[idx]
        bgr = cv2.imread(img_p)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        lbl_p = os.path.join(self.lbl_dir, os.path.splitext(os.path.basename(img_p))[0] + ".txt")
        boxes, labels, masks = [], [], []
        if os.path.exists(lbl_p):
            for line in open(lbl_p).read().splitlines():
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                cls_id = int(parts[0]) + 1  # 0 is background
                pts = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
                pts[:, 0] *= w
                pts[:, 1] *= h
                pts_int = np.round(pts).astype(np.int32)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [pts_int], 1)
                ys, xs = np.where(mask > 0)
                if len(xs) == 0 or len(ys) == 0:
                    continue
                x0, x1 = float(xs.min()), float(xs.max())
                y0, y1 = float(ys.min()), float(ys.max())
                if x1 - x0 < 2.0 or y1 - y0 < 2.0:
                    continue
                boxes.append([x0, y0, x1, y1])
                labels.append(cls_id)
                masks.append(mask)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "masks": torch.tensor(np.array(masks), dtype=torch.uint8) if masks else torch.zeros((0, h, w), dtype=torch.uint8),
        }
        return img_t, target

def collate_fn(batch):
    return tuple(zip(*batch))

def build_model():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, NUM_CLASSES)
    return model

def compute_eval_metrics(model, val_loader, device, iou_thresh=0.5):
    model.eval()
    tp_box, fp_box, fn_box = 0, 0, 0
    tp_mask, fp_mask, fn_mask = 0, 0, 0
    matched_ious = []
    # For per-class AP@0.5
    preds_by_cls = {c: [] for c in range(1, NUM_CLASSES)}
    gts_by_cls = {c: 0 for c in range(1, NUM_CLASSES)}

    with torch.no_grad():
        for images, targets in val_loader:
            images = [im.to(device) for im in images]
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                gt_boxes = tgt["boxes"].numpy()
                gt_labels = tgt["labels"].numpy()
                gt_masks = tgt["masks"].numpy() > 0
                for gl in gt_labels:
                    gts_by_cls[int(gl)] += 1

                scores = out["scores"].cpu().numpy()
                keep = scores >= 0.25
                p_boxes = out["boxes"].cpu().numpy()[keep]
                p_labels = out["labels"].cpu().numpy()[keep]
                p_scores = scores[keep]
                p_masks = (out["masks"].cpu().numpy()[keep, 0] > 0.5)

                order = np.argsort(-p_scores)
                used_gt_box = set()
                used_gt_mask = set()

                for pi in order:
                    pl = int(p_labels[pi])
                    pb = p_boxes[pi]
                    pm = p_masks[pi]
                    ps = float(p_scores[pi])

                    # Best matching GT of same class
                    best_biou, best_bi = 0.0, -1
                    best_miou, best_mi = 0.0, -1
                    for gi in range(len(gt_boxes)):
                        if int(gt_labels[gi]) != pl:
                            continue
                        gb = gt_boxes[gi]
                        ix0, iy0 = max(pb[0], gb[0]), max(pb[1], gb[1])
                        ix1, iy1 = min(pb[2], gb[2]), min(pb[3], gb[3])
                        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                        union = max((pb[2]-pb[0])*(pb[3]-pb[1]) + (gb[2]-gb[0])*(gb[3]-gb[1]) - inter, 1e-6)
                        biou = inter / union
                        if biou > best_biou and gi not in used_gt_box:
                            best_biou, best_bi = biou, gi

                        gm = gt_masks[gi]
                        minter = np.logical_and(pm, gm).sum()
                        munion = max(np.logical_or(pm, gm).sum(), 1)
                        miou = minter / munion
                        if miou > best_miou and gi not in used_gt_mask:
                            best_miou, best_mi = miou, gi

                    is_tp_b = 1 if best_biou >= iou_thresh and best_bi >= 0 else 0
                    if is_tp_b:
                        used_gt_box.add(best_bi)
                        tp_box += 1
                    else:
                        fp_box += 1

                    is_tp_m = 1 if best_miou >= iou_thresh and best_mi >= 0 else 0
                    if is_tp_m:
                        used_gt_mask.add(best_mi)
                        tp_mask += 1
                        matched_ious.append(best_miou)
                    else:
                        fp_mask += 1

                    preds_by_cls[pl].append((ps, is_tp_b, is_tp_m))

    total_gt = sum(gts_by_cls.values())
    fn_box = max(total_gt - tp_box, 0)
    fn_mask = max(total_gt - tp_mask, 0)

    def calc_map(idx_flag):
        aps = []
        for c in range(1, NUM_CLASSES):
            n_gt = gts_by_cls[c]
            if n_gt == 0:
                continue
            plist = sorted(preds_by_cls[c], key=lambda x: -x[0])
            if not plist:
                aps.append(0.0)
                continue
            tps = np.cumsum([x[idx_flag] for x in plist])
            fps = np.cumsum([1 - x[idx_flag] for x in plist])
            rec = tps / n_gt
            prec = tps / np.maximum(tps + fps, 1e-6)
            # 101-point COCO AP50
            mrec = np.concatenate(([0.0], rec, [1.0]))
            mpre = np.concatenate(([0.0], prec, [0.0]))
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            r_grid = np.linspace(0, 1, 101)
            ap = np.mean([mpre[mrec >= r].max() if np.any(mrec >= r) else 0.0 for r in r_grid])
            aps.append(ap)
        return float(np.mean(aps)) if aps else 0.0

    # Global foreground pixel mIoU across val set
    box_map50 = calc_map(1)
    mask_map50 = calc_map(2)
    mean_inst_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    box_f1 = 2 * tp_box / max(2 * tp_box + fp_box + fn_box, 1)
    mask_f1 = 2 * tp_mask / max(2 * tp_mask + fp_mask + fn_mask, 1)
    return {
        "Box_mAP50": box_map50,
        "Mask_mAP50": mask_map50,
        "Mean_Matched_IoU": mean_inst_iou,
        "Box_F1": box_f1,
        "Mask_F1": mask_f1,
    }

def train_maskrcnn(ds_name, epochs=25):
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_root = os.path.join(DS_BASE, f"ds_{ds_name.lower()}")
    train_ds = BorobudurSegDataset(ds_root, "train")
    val_ds = BorobudurSegDataset(ds_root, "val")
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)

    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for images, targets in train_loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            ep_loss += float(losses.item())
        avg_loss = ep_loss / max(len(train_loader), 1)
        metrics = compute_eval_metrics(model, val_loader, device)
        row = {"epoch": ep, "train_loss": avg_loss, **metrics}
        hist.append(row)
        print(f"[MaskRCNN-{ds_name}] Ep {ep:02d}/{epochs} | loss={avg_loss:.4f} | "
              f"Box_mAP50={metrics['Box_mAP50']:.4f} | Mask_mAP50={metrics['Mask_mAP50']:.4f} | IoU={metrics['Mean_Matched_IoU']:.4f}", flush=True)

    df = pd.DataFrame(hist)
    df.to_csv(os.path.join(OUT_DIR, f"maskrcnn_{ds_name.lower()}_history.csv"), index=False)
    return df

if __name__ == "__main__":
    df_raw = train_maskrcnn("Raw", epochs=20)
    df_clahe = train_maskrcnn("CLAHE", epochs=20)

    # Plot 300 dpi Mask R-CNN comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.2), dpi=300)
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    ax1.plot(df_raw["epoch"], df_raw["train_loss"], "--", color="#7f7f7f", lw=1.8, label="Mask R-CNN (Raw)")
    ax1.plot(df_clahe["epoch"], df_clahe["train_loss"], "-", color="#1f77b4", lw=2.0, label="Mask R-CNN (CLAHE)")
    ax1.set_title("(a) Mask R-CNN Training Loss Convergence", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Composite Training Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=8)

    ax2.plot(df_raw["epoch"], df_raw["Mask_mAP50"], "--", color="#7f7f7f", lw=1.8, label="Mask mAP@0.5 (Raw)")
    ax2.plot(df_clahe["epoch"], df_clahe["Mask_mAP50"], "-", color="#1f77b4", lw=2.0, label="Mask mAP@0.5 (CLAHE)")
    ax2.plot(df_raw["epoch"], df_raw["Box_mAP50"], ":", color="#d62728", lw=1.6, label="Box mAP@0.5 (Raw)")
    ax2.plot(df_clahe["epoch"], df_clahe["Box_mAP50"], "-", color="#2ca02c", lw=1.8, label="Box mAP@0.5 (CLAHE)")
    ax2.set_title("(b) Mask R-CNN Validation Accuracy (mAP@0.5)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mAP@0.5")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Fig_MaskRCNN_Raw_vs_CLAHE_300dpi.png"), dpi=300)
    plt.close(fig)
    print("Mask R-CNN evaluation completed and saved!")
