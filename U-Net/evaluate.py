"""
evaluate_models.py  —  Evaluation for UNet / FAT-Net / MALUNet / AoP-GAU-Net
==============================================================================
FIXES vs originals (both versions)
-----------------------------------
  FIX-A  [CRITICAL] _load() uses strict=False + key-unwrapping so checkpoints
         saved by train_*.py (which include optimizer/scheduler keys) load
         cleanly without a RuntimeError.
  FIX-B  [CRITICAL] Blue-channel guard in SegDataset mask decoding prevents
         cyan/yellow pixels being misclassified as FH or PS.
  FIX-C  [CRITICAL] SegDataset also supports grayscale-index masks (values 0/1/2)
         not just colour-coded masks.
  FIX-D  [MEDIUM] pred tensor uses .detach() before .numpy() to avoid silent
         failures on tensors with a grad_fn.
  FIX-E  [MEDIUM] evaluate_one tracks total foreground predicted; warns if
         the model is predicting nothing but background (dead checkpoint).
  FIX-F  [MINOR] SegDataset __getitem__ limits recursion to 10 retries.
  FIX-G  Dataset now also accepts pre-split directory (test/ folder) via
         --test_split_dir, in addition to the flat --img_dir / --mask_dir.

Usage examples
--------------
  # Flat directory:
  python evaluate_models.py --img_dir path/images --mask_dir path/masks \
      --unet_ckpt unet_best.pth --fatnet_ckpt fatnet_best.pth \
      --malunet_ckpt malunet_best.pth --aopgau_ckpt runs/.../best.pth

  # Pre-split test directory:
  python evaluate_models.py --test_split_dir path/to/dataset/test \
      --unet_ckpt unet_best.pth ...

  # Efficiency benchmark only:
  python evaluate_models.py --benchmark_only
"""

import os
import sys
import time
import argparse
import csv
import math
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── model imports ─────────────────────────────────────────────────────────────
try:
    from new_model       import UNet
    from fatnet_model    import FAT_Net
    from malunet_model   import MALUNet
    from model_optimized import AoPGAUNet, build_model as build_aopgau
except ImportError as e:
    print(f"[WARN] Could not import one or more models: {e}")
    print("       Place model files in the same directory or adjust the imports.")


# ══════════════════════════════════════════════════════════════════════════════
# 0. DATASET
# ══════════════════════════════════════════════════════════════════════════════

class SegDataset(Dataset):
    """
    Loads grayscale ultrasound images + masks.

    FIX-B: blue-channel guard added to prevent cyan/yellow misclassification.
    FIX-C: auto-detects grayscale-index masks (values 0/1/2) vs colour masks.
    FIX-F: recursion capped at 10 retries.

    Supports TWO directory layouts:
      (a) flat: img_dir/ + mask_dir/
      (b) split: split_root/images/ + split_root/masks/
    """
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(self, img_dir: str = "", mask_dir: str = "",
                 split_root: str = "", size: int = 256):
        self.size = size

        if split_root:
            root     = Path(split_root)
            img_dir  = str(root / "images")
            mask_dir = str(root / "masks")

        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.names    = sorted(
            f for f in os.listdir(img_dir)
            if Path(f).suffix.lower() in self.IMG_EXTS
        )

    def __len__(self) -> int:
        return len(self.names)

    def _load_mask(self, path: str) -> np.ndarray:
        """FIX-C: auto-detect mask type."""
        msk = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if msk is None:
            return None

        # Grayscale index mask
        if msk.ndim == 2 and msk.max() <= 2:
            return msk.astype(np.uint8)

        # Colour-coded BGR mask
        if msk.ndim == 3:
            B_ch = msk[:, :, 0]
            G_ch = msk[:, :, 1]
            R_ch = msk[:, :, 2]
            label = np.zeros(msk.shape[:2], dtype=np.uint8)
            # FIX-B: blue-channel guard
            label[(G_ch > 200) & (R_ch < 50) & (B_ch < 50)] = 1  # Green → FH
            label[(R_ch > 200) & (G_ch < 50) & (B_ch < 50)] = 2  # Red   → PS
            return label

        # Single-channel grayscale (binary legacy format)
        label = np.zeros_like(msk, dtype=np.uint8)
        label[msk > 127] = 1
        return label

    def __getitem__(self, idx: int, _retries: int = 0):
        if _retries >= 10:  # FIX-F
            raise RuntimeError(
                f"SegDataset: 10 consecutive corrupt/missing files. "
                f"Check --img_dir and --mask_dir.")
        name      = self.names[idx]
        img_path  = os.path.join(self.img_dir,  name)
        mask_path = os.path.join(self.mask_dir, name)

        img  = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return self.__getitem__((idx + 1) % len(self.names), _retries + 1)

        label = self._load_mask(mask_path)
        if label is None:
            return self.__getitem__((idx + 1) % len(self.names), _retries + 1)

        img   = cv2.resize(img,   (self.size, self.size))
        label = cv2.resize(label, (self.size, self.size),
                           interpolation=cv2.INTER_NEAREST)
        label = np.clip(label, 0, 2)

        img_f = img.astype(np.float32) / 255.0
        return torch.from_numpy(img_f).unsqueeze(0), torch.from_numpy(label).long()


# ══════════════════════════════════════════════════════════════════════════════
# 1. PER-MODEL PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_batch(imgs_01: torch.Tensor, model_key: str) -> torch.Tensor:
    """imgs_01 : (B, 1, H, W) float32 in [0, 1]"""
    if model_key == "unet":
        return imgs_01                            # [0,1] grayscale 1-channel

    elif model_key == "fatnet":
        img_n = imgs_01 * 2.0 - 1.0              # [-1,+1]
        return img_n.repeat(1, 3, 1, 1)          # 3-channel

    elif model_key == "malunet":
        return imgs_01.repeat(1, 3, 1, 1)        # [0,1] 3-channel

    elif model_key == "aopgau":
        img_n = imgs_01 * 2.0 - 1.0              # [-1,+1]
        return img_n.repeat(1, 3, 1, 1)          # 3-channel

    return imgs_01


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def _load(model: nn.Module, path: str) -> None:
    """
    FIX-A: strict=False + unwraps checkpoint wrapper keys so checkpoints
    saved by train_*.py (containing optimizer/scheduler/epoch) load cleanly.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for k in ("model", "model_state_dict", "state_dict", "net"):
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] _load: {len(missing)} missing key(s): "
              f"{missing[:3]}{'...' if len(missing) > 3 else ''}")
    if unexpected:
        print(f"  [WARN] _load: {len(unexpected)} unexpected key(s): "
              f"{unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")


def build_models(args) -> Dict[str, nn.Module]:
    models = {}

    m = UNet(in_channels=1, out_channels=3)
    if args.unet_ckpt and os.path.isfile(args.unet_ckpt):
        _load(m, args.unet_ckpt)
        print(f"[✓] UNet loaded from {args.unet_ckpt}")
    else:
        print("[~] UNet: random weights (no checkpoint provided)")
    models["unet"] = m

    m = FAT_Net(n_channels=3, n_classes=3)
    if args.fatnet_ckpt and os.path.isfile(args.fatnet_ckpt):
        _load(m, args.fatnet_ckpt)
        print(f"[✓] FAT-Net loaded from {args.fatnet_ckpt}")
    else:
        print("[~] FAT-Net: random weights")
    models["fatnet"] = m

    m = MALUNet(num_classes=3, input_channels=3)
    if args.malunet_ckpt and os.path.isfile(args.malunet_ckpt):
        _load(m, args.malunet_ckpt)
        print(f"[✓] MALUNet loaded from {args.malunet_ckpt}")
    else:
        print("[~] MALUNet: random weights")
    models["malunet"] = m

    m = build_aopgau(in_channels=3, num_classes=3, base_ch=32,
                     use_dw=True, use_att_pool=True)
    if args.aopgau_ckpt and os.path.isfile(args.aopgau_ckpt):
        _load(m, args.aopgau_ckpt)
        print(f"[✓] AoP-GAU-Net loaded from {args.aopgau_ckpt}")
    else:
        print("[~] AoP-GAU-Net: random weights")
    models["aopgau"] = m

    return models


# ══════════════════════════════════════════════════════════════════════════════
# 3. METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_confusion(pred: np.ndarray, gt: np.ndarray,
                      n_classes: int) -> np.ndarray:
    cm   = np.zeros((n_classes, n_classes), dtype=np.int64)
    mask = (gt >= 0) & (gt < n_classes)
    np.add.at(cm, (gt[mask], pred[mask]), 1)
    return cm


def iou_from_cm(cm: np.ndarray) -> np.ndarray:
    inter = np.diag(cm)
    union = cm.sum(1) + cm.sum(0) - inter
    return np.where(union > 0, inter / union, np.nan)


def dice_from_cm(cm: np.ndarray) -> np.ndarray:
    inter = np.diag(cm)
    denom = cm.sum(1) + cm.sum(0)
    return np.where(denom > 0, 2 * inter / denom, np.nan)


def precision_recall_f1_from_cm(cm: np.ndarray):
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.where((tp + fp) > 0, tp / (tp + fp), np.nan)
    rec  = np.where((tp + fn) > 0, tp / (tp + fn), np.nan)
    f1   = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), np.nan)
    return prec, rec, f1


def specificity_from_cm(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    tn = cm.sum() - tp - fp - fn
    return np.where((tn + fp) > 0, tn / (tn + fp), np.nan)


def hausdorff95(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    from scipy.ndimage import distance_transform_edt
    pred_bin = pred_bin.astype(bool)
    gt_bin   = gt_bin.astype(bool)
    if not pred_bin.any() or not gt_bin.any():
        return np.nan
    d_pred_to_gt = distance_transform_edt(~gt_bin)[pred_bin]
    d_gt_to_pred = distance_transform_edt(~pred_bin)[gt_bin]
    return float(max(np.percentile(d_pred_to_gt, 95),
                     np.percentile(d_gt_to_pred, 95)))


def boundary_f1(pred_bin: np.ndarray, gt_bin: np.ndarray,
                threshold: int = 2) -> float:
    from scipy.ndimage import distance_transform_edt, binary_erosion
    pred_bin = pred_bin.astype(bool)
    gt_bin   = gt_bin.astype(bool)

    def get_boundary(m): return m & ~binary_erosion(m)
    pb = get_boundary(pred_bin)
    gb = get_boundary(gt_bin)

    if not pb.any() and not gb.any(): return 1.0
    if not pb.any() or  not gb.any(): return 0.0

    dist_to_pb = distance_transform_edt(~pb)
    dist_to_gb = distance_transform_edt(~gb)

    pb_arr = pb.astype(np.float32)
    gb_arr = gb.astype(np.float32)

    prec = (pb_arr[dist_to_gb <= threshold].sum()) / (pb_arr.sum() + 1e-8)
    rec  = (gb_arr[dist_to_pb <= threshold].sum()) / (gb_arr.sum() + 1e-8)

    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 4. EFFICIENCY METRICS
# ══════════════════════════════════════════════════════════════════════════════

def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def compute_gflops(model: nn.Module, input_shape: Tuple, device) -> float:
    try:
        from thop import profile as thop_profile
        dummy = torch.randn(*input_shape, device=device)
        with torch.no_grad():
            macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        return macs * 2 / 1e9
    except Exception:
        pass

    total_flops = [0]
    def conv_hook(m, inp, out):
        b, c_in, _, _ = inp[0].shape
        c_out, h_out, w_out = out.shape[1:]
        kh, kw = (m.kernel_size if hasattr(m.kernel_size, "__iter__")
                  else (m.kernel_size, m.kernel_size))
        total_flops[0] += 2 * b * c_in * c_out * kh * kw * h_out * w_out // m.groups

    hooks = [m.register_forward_hook(conv_hook)
             for m in model.modules() if isinstance(m, nn.Conv2d)]
    dummy = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        try: _ = model(dummy)
        except Exception: pass
    for h in hooks: h.remove()
    return total_flops[0] / 1e9


def measure_latency(model: nn.Module, input_shape: Tuple,
                    device, n_warmup: int = 10,
                    n_runs: int = 50) -> Tuple[float, float]:
    dummy = torch.randn(*input_shape, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            try: _ = model(dummy)
            except Exception: pass
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            try: _ = model(dummy)
            except Exception: pass
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    mean_s = np.mean(times)
    return mean_s * 1000, input_shape[0] / mean_s


def measure_gpu_memory(model: nn.Module, input_shape: Tuple, device) -> float:
    if device.type != "cuda":
        return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    dummy = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        try: _ = model(dummy)
        except Exception: pass
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


# ══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

NUM_CLASSES   = 3
CLASS_NAMES   = ["Background", "FH", "PS"]
DISPLAY_NAMES = {
    "unet":    "UNet",
    "fatnet":  "FAT-Net",
    "malunet": "MALUNet",
    "aopgau":  "AoP-GAU-Net",
}
MODEL_SIZE = {"unet": 256, "fatnet": 128, "malunet": 256, "aopgau": 256}
MODEL_INCH = {"unet":   1, "fatnet":   3, "malunet":   3, "aopgau":   3}


def evaluate_one(model: nn.Module, key: str,
                 loader: DataLoader, device,
                 size: int) -> Dict:
    model.eval()
    model.to(device)

    cm_total  = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    hd95_list = {1: [], 2: []}
    bf1_list  = {1: [], 2: []}
    total_fg_predicted = 0

    with torch.no_grad():
        for imgs_01, labels in loader:
            if imgs_01.shape[-1] != size:
                imgs_01 = F.interpolate(imgs_01, size=(size, size),
                                        mode="bilinear", align_corners=False)
                labels  = F.interpolate(labels.unsqueeze(1).float(),
                                        size=(size, size),
                                        mode="nearest").squeeze(1).long()

            inp = preprocess_batch(imgs_01, key).to(device)
            lbl = labels.to(device)

            out = model(inp)
            if isinstance(out, (tuple, list)):
                out = out[0]

            # FIX-D: .detach() before .numpy()
            pred = out.argmax(dim=1).detach().cpu().numpy()
            gt   = lbl.detach().cpu().numpy()

            total_fg_predicted += int((pred > 0).sum())

            for b in range(pred.shape[0]):
                cm_total += compute_confusion(pred[b], gt[b], NUM_CLASSES)
                for cls in [1, 2]:
                    pm = (pred[b] == cls)
                    gm = (gt[b]   == cls)
                    if gm.sum() > 0:
                        hd95_list[cls].append(hausdorff95(pm, gm))
                        bf1_list[cls].append(boundary_f1(pm, gm))

    # FIX-E: warn if model never predicted any foreground
    if total_fg_predicted == 0:
        print(f"\n  [WARN] {DISPLAY_NAMES[key]}: predicted ZERO foreground pixels "
              f"— checkpoint may not have converged or uses wrong label schema.")

    iou   = iou_from_cm(cm_total)
    dice  = dice_from_cm(cm_total)
    prec, rec, f1 = precision_recall_f1_from_cm(cm_total)
    spec  = specificity_from_cm(cm_total)

    total_px   = cm_total.sum()
    correct_px = np.diag(cm_total).sum()
    pixel_acc  = correct_px / total_px if total_px > 0 else 0.0

    per_class_acc = np.where(cm_total.sum(1) > 0,
                             np.diag(cm_total) / cm_total.sum(1), np.nan)
    mean_acc = float(np.nanmean(per_class_acc))

    return {
        "Dice_BG": float(dice[0]),  "Dice_FH": float(dice[1]),
        "Dice_PS": float(dice[2]),  "mDice":   float(np.nanmean(dice)),
        "FG_Dice": float(np.nanmean(dice[1:])),

        "IoU_BG":  float(iou[0]),   "IoU_FH":  float(iou[1]),
        "IoU_PS":  float(iou[2]),   "mIoU":    float(np.nanmean(iou)),
        "FG_IoU":  float(np.nanmean(iou[1:])),

        "PixelAcc":    float(pixel_acc),
        "MeanAcc":     float(mean_acc),
        "Precision":   float(np.nanmean(prec[1:])),
        "Recall":      float(np.nanmean(rec[1:])),
        "F1":          float(np.nanmean(f1[1:])),
        "Specificity": float(np.nanmean(spec[1:])),

        "HD95_FH":   float(np.nanmean(hd95_list[1])) if hd95_list[1] else float("nan"),
        "HD95_PS":   float(np.nanmean(hd95_list[2])) if hd95_list[2] else float("nan"),
        "HD95_mean": float(np.nanmean([x for sub in hd95_list.values() for x in sub]))
                     if any(hd95_list.values()) else float("nan"),

        "BF1_FH":    float(np.nanmean(bf1_list[1])) if bf1_list[1] else float("nan"),
        "BF1_PS":    float(np.nanmean(bf1_list[2])) if bf1_list[2] else float("nan"),
        "BF1_mean":  float(np.nanmean([x for sub in bf1_list.values() for x in sub]))
                     if any(bf1_list.values()) else float("nan"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. HTML REPORT
# ══════════════════════════════════════════════════════════════════════════════

LOWER_IS_BETTER = {
    "HD95_FH", "HD95_PS", "HD95_mean",
    "Params_M", "GFLOPs", "Latency_ms", "GPU_MB",
}

METRIC_GROUPS = {
    "Dice Score":                    ["Dice_BG", "Dice_FH", "Dice_PS", "mDice", "FG_Dice"],
    "IoU / Jaccard":                 ["IoU_BG",  "IoU_FH",  "IoU_PS",  "mIoU",  "FG_IoU"],
    "Accuracy":                      ["PixelAcc", "MeanAcc"],
    "Precision / Recall / F1":       ["Precision", "Recall", "F1"],
    "Specificity":                   ["Specificity"],
    "Hausdorff Distance (↓ better)": ["HD95_FH", "HD95_PS", "HD95_mean"],
    "Boundary F1":                   ["BF1_FH",  "BF1_PS",  "BF1_mean"],
    "Efficiency":                    ["Params_M", "GFLOPs", "Latency_ms",
                                      "Throughput_ips", "GPU_MB"],
}

METRIC_DISPLAY = {
    "Dice_BG": "Dice BG", "Dice_FH": "Dice FH", "Dice_PS": "Dice PS",
    "mDice": "mDice", "FG_Dice": "FG Dice",
    "IoU_BG": "IoU BG", "IoU_FH": "IoU FH", "IoU_PS": "IoU PS",
    "mIoU": "mIoU", "FG_IoU": "FG IoU",
    "PixelAcc": "Pixel Acc.", "MeanAcc": "Mean Acc.",
    "Precision": "Precision", "Recall": "Recall", "F1": "F1-Score",
    "Specificity": "Specificity",
    "HD95_FH": "HD95 FH (px)", "HD95_PS": "HD95 PS (px)",
    "HD95_mean": "HD95 Mean (px)",
    "BF1_FH": "Boundary F1 FH", "BF1_PS": "Boundary F1 PS",
    "BF1_mean": "Boundary F1 Mean",
    "Params_M": "Params (M)", "GFLOPs": "GFLOPs",
    "Latency_ms": "Latency (ms)", "Throughput_ips": "Throughput (img/s)",
    "GPU_MB": "GPU Mem (MB)",
}

MODEL_COLORS = {
    "unet":    "#4A90D9",
    "fatnet":  "#E07B39",
    "malunet": "#50A850",
    "aopgau":  "#9B59B6",
}


def fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, float):
        if abs(v) >= 100: return f"{v:.1f}"
        if abs(v) >= 10:  return f"{v:.2f}"
        return f"{v:.4f}"
    return str(v)


def build_html_report(all_results: Dict[str, Dict],
                      model_keys: List[str]) -> str:
    best_model: Dict[str, str] = {}
    for group_metrics in METRIC_GROUPS.values():
        for m in group_metrics:
            vals = {k: all_results[k].get(m, float("nan"))
                    for k in model_keys
                    if not math.isnan(all_results[k].get(m, float("nan")))}
            if not vals: continue
            best_k = (min(vals, key=vals.get) if m in LOWER_IS_BETTER
                      else max(vals, key=vals.get))
            best_model[m] = best_k

    win_count = {k: 0 for k in model_keys}
    for bk in best_model.values():
        win_count[bk] += 1

    css = textwrap.dedent("""
    <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5;
           color: #1a1a2e; padding: 30px; }
    h1 { text-align: center; margin-bottom: 6px; font-size: 2em; color: #1a1a2e; }
    .subtitle { text-align: center; color: #666; margin-bottom: 30px; font-size: 0.95em; }
    .win-cards { display: flex; justify-content: center; gap: 18px;
                 flex-wrap: wrap; margin-bottom: 36px; }
    .win-card { border-radius: 12px; padding: 16px 24px; color: white;
                text-align: center; min-width: 150px;
                box-shadow: 0 4px 12px rgba(0,0,0,.15); }
    .win-card .model-name { font-size: 1.1em; font-weight: 700; }
    .win-card .win-num    { font-size: 2.2em; font-weight: 800; line-height: 1.1; }
    .win-card .win-label  { font-size: 0.8em; opacity: .85; }
    .section-header { background: #1a1a2e; color: white; padding: 10px 18px;
                      border-radius: 8px 8px 0 0; font-weight: 700;
                      font-size: 1.05em; margin-top: 30px; }
    .tbl-wrap { overflow-x: auto; border-radius: 0 0 8px 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,.08); margin-bottom: 4px; }
    table { width: 100%; border-collapse: collapse; background: white; }
    thead tr { background: #f5f7fa; }
    th { padding: 12px 16px; text-align: center; font-size: .85em;
         font-weight: 700; border-bottom: 2px solid #e0e4ea; white-space: nowrap; }
    th.metric-col { text-align: left; }
    td { padding: 10px 16px; text-align: center; font-size: .92em;
         border-bottom: 1px solid #f0f2f5; white-space: nowrap; }
    td.metric-label { text-align: left; font-weight: 500; color: #444; }
    tr:hover td { background: #fafbfd; }
    .model-pill { display: inline-block; border-radius: 20px; padding: 4px 14px;
                  color: white; font-weight: 700; font-size: .9em; }
    .best { font-weight: 800; border-radius: 6px; padding: 4px 10px; color: white; }
    .badge { display: inline-block; border-radius: 20px; padding: 2px 8px;
             font-size: .75em; font-weight: 700; color: white; margin-left: 4px;
             vertical-align: middle; }
    footer { text-align: center; color: #888; margin-top: 40px; font-size: .85em; }
    </style>
    """)

    def pill(key):
        c = MODEL_COLORS[key]
        return (f'<span class="model-pill" style="background:{c}">'
                f'{DISPLAY_NAMES[key]}</span>')

    def cell(key, metric, value):
        v_fmt = fmt(value)
        if best_model.get(metric) == key and v_fmt != "—":
            c = MODEL_COLORS[key]
            return (f'<td><span class="best" style="background:{c}">'
                    f'{v_fmt}</span></td>')
        return f"<td>{v_fmt}</td>"

    rows = ['<!DOCTYPE html><html lang="en"><head>',
            '<meta charset="UTF-8">',
            '<title>Model Evaluation Report</title>',
            css,
            '</head><body>',
            '<h1>🔬 Segmentation Model Evaluation Report</h1>',
            '<p class="subtitle">UNet &nbsp;|&nbsp; FAT-Net &nbsp;|&nbsp; '
            'MALUNet &nbsp;|&nbsp; AoP-GAU-Net &nbsp;—&nbsp; '
            'Fetal Head Ultrasound Segmentation (3-class: BG / FH / PS)</p>']

    rows.append('<div class="win-cards">')
    for k in model_keys:
        c = MODEL_COLORS[k]
        rows.append(
            f'<div class="win-card" style="background:{c}">'
            f'<div class="model-name">{DISPLAY_NAMES[k]}</div>'
            f'<div class="win-num">{win_count[k]}</div>'
            f'<div class="win-label">metrics won</div>'
            f'</div>'
        )
    rows.append('</div>')

    for group, metrics in METRIC_GROUPS.items():
        rows.append(f'<div class="section-header">{group}</div>')
        rows.append('<div class="tbl-wrap"><table>')
        rows.append('<thead><tr><th class="metric-col">Metric</th>')
        for k in model_keys:
            rows.append(f'<th>{pill(k)}</th>')
        rows.append('</tr></thead><tbody>')
        for m in metrics:
            rows.append('<tr>')
            label = METRIC_DISPLAY.get(m, m)
            badge = (' <span class="badge" style="background:#e74c3c">↓ lower</span>'
                     if m in LOWER_IS_BETTER else "")
            rows.append(f'<td class="metric-label">{label}{badge}</td>')
            for k in model_keys:
                v = all_results[k].get(m, float("nan"))
                rows.append(cell(k, m, v))
            rows.append('</tr>')
        rows.append('</tbody></table></div>')

    import datetime
    rows.append(
        f'<footer>Generated {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")} '
        f'| Bold coloured cells = best performer for that metric</footer>'
    )
    rows.append('</body></html>')
    return "\n".join(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 7. CSV EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_csv(all_results: Dict[str, Dict], model_keys: List[str],
             path: str) -> None:
    all_metrics = [m for ms in METRIC_GROUPS.values() for m in ms]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric"] + [DISPLAY_NAMES[k] for k in model_keys])
        for m in all_metrics:
            row = [METRIC_DISPLAY.get(m, m)]
            for k in model_keys:
                row.append(fmt(all_results[k].get(m, float("nan"))))
            writer.writerow(row)
    print(f"[✓] CSV saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate UNet / FAT-Net / MALUNet / AoP-GAU-Net")
    # FIX-G: supports both flat dir and pre-split test dir
    p.add_argument("--img_dir",          default="",  help="Flat image directory")
    p.add_argument("--mask_dir",         default="",  help="Flat mask directory")
    p.add_argument("--test_split_dir",   default="",
                   help="Pre-split test dir (e.g. dataset/test); "
                        "overrides --img_dir / --mask_dir")
    p.add_argument("--unet_ckpt",    default="", help="UNet checkpoint (.pth)")
    p.add_argument("--fatnet_ckpt",  default="", help="FAT-Net checkpoint (.pth)")
    p.add_argument("--malunet_ckpt", default="", help="MALUNet checkpoint (.pth)")
    p.add_argument("--aopgau_ckpt",  default="", help="AoP-GAU-Net checkpoint (.pth)")
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--num_workers",  type=int, default=0)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir",      default="eval_output")
    p.add_argument("--benchmark_only", action="store_true")
    return p.parse_args()


def main():
    import argparse as _ap
    args   = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION — UNet / FAT-Net / MALUNet / AoP-GAU-Net")
    print("=" * 60)
    print(f"  Device    : {device}")
    print(f"  Output dir: {args.out_dir}")
    print()

    models     = build_models(args)
    model_keys = list(models.keys())

    # FIX-G: resolve dataset source
    if args.test_split_dir and os.path.isdir(args.test_split_dir):
        has_dataset = True
        dataset = SegDataset(split_root=args.test_split_dir, size=256)
    elif args.img_dir and args.mask_dir \
            and os.path.isdir(args.img_dir) and os.path.isdir(args.mask_dir):
        has_dataset = True
        dataset = SegDataset(img_dir=args.img_dir, mask_dir=args.mask_dir, size=256)
    else:
        has_dataset = False
        dataset = None

    if has_dataset and not args.benchmark_only:
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"))
        print(f"[✓] Dataset: {len(dataset)} images")
    else:
        loader = None
        if not args.benchmark_only:
            print("[~] No dataset provided — skipping segmentation metrics.")

    all_results: Dict[str, Dict] = {k: {} for k in model_keys}

    if loader is not None:
        for key, model in models.items():
            print(f"\n[→] Evaluating {DISPLAY_NAMES[key]} (segmentation) …")
            sz = MODEL_SIZE[key]
            seg_metrics = evaluate_one(model, key, loader, device, sz)
            all_results[key].update(seg_metrics)
            print(f"    mDice={seg_metrics['mDice']:.4f}  "
                  f"mIoU={seg_metrics['mIoU']:.4f}  "
                  f"PixAcc={seg_metrics['PixelAcc']:.4f}")

    print("\n[→] Computing efficiency metrics …")
    for key, model in models.items():
        model.eval().to(device)
        sz, inch = MODEL_SIZE[key], MODEL_INCH[key]
        shape    = (1, inch, sz, sz)

        params       = count_params(model)
        gflops       = compute_gflops(model, shape, device)
        lat_ms, tput = measure_latency(model, shape, device)
        gpu_mb       = measure_gpu_memory(model, shape, device)

        all_results[key].update({
            "Params_M":       params,
            "GFLOPs":         gflops,
            "Latency_ms":     lat_ms,
            "Throughput_ips": tput,
            "GPU_MB":         gpu_mb,
        })
        print(f"    {DISPLAY_NAMES[key]:15s}  "
              f"Params={params:.2f}M  GFLOPs={gflops:.2f}  "
              f"Latency={lat_ms:.1f}ms  Throughput={tput:.1f} img/s  "
              f"GPU={gpu_mb:.0f}MB")

    csv_path = os.path.join(args.out_dir, "evaluation_results.csv")
    save_csv(all_results, model_keys, csv_path)

    html      = build_html_report(all_results, model_keys)
    html_path = os.path.join(args.out_dir, "evaluation_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[✓] HTML report saved → {html_path}")

    # ── Terminal summary ──────────────────────────────────────────────────────
    seg_m = ["mDice", "FG_Dice", "mIoU", "FG_IoU", "PixelAcc", "F1",
             "HD95_mean", "BF1_mean"]
    eff_m = ["Params_M", "GFLOPs", "Latency_ms", "Throughput_ips", "GPU_MB"]

    col_w  = 13
    header = f"{'Metric':<24}" + "".join(
        f"{DISPLAY_NAMES[k]:>{col_w}}" for k in model_keys)
    sep    = "-" * len(header)

    def _print_section(title, metrics):
        has_any = any(not math.isnan(all_results[k].get(m, float("nan")))
                      for m in metrics for k in model_keys)
        if not has_any:
            print(f"\n  [{title}] — skipped (no dataset provided)")
            return
        print(f"\n  {title}")
        print(header); print(sep)
        for m in metrics:
            vals  = {k: all_results[k].get(m, float("nan")) for k in model_keys}
            valid = {k: v for k, v in vals.items() if not math.isnan(v)}
            if not valid: continue
            best_k = (min(valid, key=valid.get) if m in LOWER_IS_BETTER
                      else max(valid, key=valid.get))
            lb  = " ↓" if m in LOWER_IS_BETTER else "  "
            row = f"  {METRIC_DISPLAY.get(m,m) + lb:<22}"
            for k in model_keys:
                mark = "*" if k == best_k else " "
                row += f"{mark + fmt(vals[k]):>{col_w}}"
            print(row)

    print("\n" + "=" * 60 + "\n  SUMMARY\n" + "=" * 60)
    _print_section("Segmentation Metrics", seg_m)
    _print_section("Efficiency  (lower is better ↓)", eff_m)
    print("\n  * = best performer for that metric")
    print(f"\nOutputs written to: {os.path.abspath(args.out_dir)}/")


if __name__ == "__main__":
    import argparse
    main()