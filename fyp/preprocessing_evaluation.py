"""
preprocessing_evaluation.py
-----------------------------
Quantitative evaluation of the OCTA vessel segmentation preprocessing
pipeline for the diabetic retinopathy classifier.

Pipeline stages (applied to OCTA images only):
    original → grayscale → Gaussian blur → CLAHE → adaptive threshold
    → morphological cleanup → skeletonize → vessel mask overlay

Metrics (no ground truth required):
    1. SNR = μ_signal / σ_noise
    2. CNR = (μ_vessel − μ_background) / σ_background
    3. VD  = N_vessel / N_total
    4. SLD = N_skeleton / N_total
    5. FD  = box-counting fractal dimension

If ground-truth vessel masks are found:
    6. Dice     = 2|A∩B| / (|A| + |B|)
    7. Jaccard  = |A∩B| / |A∪B|

Medical context: these metrics quantify how well the preprocessing step
preserves clinically relevant vessel structure information from OCTA scans.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV (cv2) not available — preprocessing will be limited.")

try:
    from skimage.morphology import skeletonize
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from scipy.stats import mannwhitneyu
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Ground truth mask detection
# ---------------------------------------------------------------------------

def check_ground_truth_masks(dataset_root: Path) -> bool:
    """
    Inspect the dataset directory for vessel ground-truth mask files.

    Searches for common mask folder patterns: 'masks', 'vessel_masks',
    'ground_truth', 'gt_masks', and files with '_mask' suffix.

    Parameters
    ----------
    dataset_root : Path — root of the dataset directory

    Returns
    -------
    bool — True if mask files are found, False otherwise.
           Reports clearly to stdout what was found.
    """
    root = Path(dataset_root)
    mask_patterns = [
        "masks", "vessel_masks", "ground_truth", "gt", "gt_masks",
        "annotations", "labels"
    ]
    mask_suffixes = ["_mask.png", "_mask.tif", "_mask.bmp", "_gt.png"]

    found_dirs: List[Path] = []
    found_files: List[Path] = []

    for pattern in mask_patterns:
        for match in root.rglob(pattern):
            if match.is_dir():
                found_dirs.append(match)

    for suffix in mask_suffixes:
        found_files.extend(root.rglob(f"*{suffix}"))

    if found_dirs or found_files:
        print("\n[preprocessing] Ground-truth mask detection: FOUND")
        for d in found_dirs:
            print(f"  Mask directory: {d}")
        for f in found_files[:5]:
            print(f"  Mask file: {f}")
        if len(found_files) > 5:
            print(f"  ... and {len(found_files) - 5} more mask files")
        return True
    else:
        print("\n[preprocessing] Ground-truth mask detection: NOT FOUND")
        print("  No vessel segmentation masks were found in the dataset.")
        print("  Quantitative evaluation will use no-reference metrics only.")
        return False


# ---------------------------------------------------------------------------
# Core preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_octa(img_gray: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Apply the full OCTA preprocessing pipeline and return each stage.

    Stages: grayscale → Gaussian blur → CLAHE → adaptive threshold
            → morphological cleanup → skeletonize → vessel overlay

    Parameters
    ----------
    img_gray : np.ndarray (H, W) uint8 — grayscale OCTA image

    Returns
    -------
    dict mapping stage name → image array
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for preprocessing.")

    stages: Dict[str, np.ndarray] = {}
    stages["original"] = img_gray.copy()

    # 1. Grayscale (already done, but keep for pipeline consistency)
    stages["grayscale"] = img_gray.copy()

    # 2. Gaussian blur — reduces noise before CLAHE
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    stages["gaussian_blur"] = blurred

    # 3. CLAHE — contrast limited adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(blurred)
    stages["clahe"] = clahe_img

    # 4. Adaptive threshold — local thresholding suits uneven illumination
    thresh = cv2.adaptiveThreshold(
        clahe_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, C=2,
    )
    stages["adaptive_threshold"] = thresh

    # 5. Morphological cleanup — remove small noise, fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    stages["morphological_cleanup"] = cleaned

    # 6. Skeletonize
    if SKIMAGE_AVAILABLE:
        binary = (cleaned > 0).astype(bool)
        skeleton = (skeletonize(binary) * 255).astype(np.uint8)
        stages["skeleton"] = skeleton
    else:
        stages["skeleton"] = cleaned.copy()

    # 7. Vessel mask overlay (colour)
    overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    overlay[cleaned > 0] = [255, 80, 80]  # highlight vessels in red
    stages["vessel_overlay"] = overlay

    return stages


# ---------------------------------------------------------------------------
# Quantitative metrics
# ---------------------------------------------------------------------------

def compute_snr(img_gray: np.ndarray, vessel_mask: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio: SNR = μ_signal / σ_noise.

    Signal = mean intensity of vessel pixels.
    Noise  = standard deviation of background pixels.
    """
    try:
        vessel_pixels = img_gray[vessel_mask > 0].astype(float)
        bg_pixels = img_gray[vessel_mask == 0].astype(float)
        if len(vessel_pixels) == 0 or len(bg_pixels) == 0:
            return float("nan")
        mu_signal = vessel_pixels.mean()
        sigma_noise = bg_pixels.std()
        if sigma_noise < 1e-8:
            return float("nan")
        return float(mu_signal / sigma_noise)
    except Exception as e:
        warnings.warn(f"SNR computation failed: {e}")
        return float("nan")


def compute_cnr(img_gray: np.ndarray, vessel_mask: np.ndarray) -> float:
    """
    Contrast-to-Noise Ratio: CNR = (μ_vessel − μ_background) / σ_background.
    """
    try:
        vessel_pixels = img_gray[vessel_mask > 0].astype(float)
        bg_pixels = img_gray[vessel_mask == 0].astype(float)
        if len(vessel_pixels) == 0 or len(bg_pixels) == 0:
            return float("nan")
        mu_vessel = vessel_pixels.mean()
        mu_bg = bg_pixels.mean()
        sigma_bg = bg_pixels.std()
        if sigma_bg < 1e-8:
            return float("nan")
        return float((mu_vessel - mu_bg) / sigma_bg)
    except Exception as e:
        warnings.warn(f"CNR computation failed: {e}")
        return float("nan")


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Dice coefficient: Dice = 2|A∩B| / (|A| + |B|).

    Parameters
    ----------
    pred_mask : np.ndarray — predicted binary mask
    gt_mask   : np.ndarray — ground-truth binary mask

    Returns
    -------
    float — Dice ∈ [0, 1]
    """
    try:
        pred_bool = pred_mask > 0
        gt_bool = gt_mask > 0
        intersection = (pred_bool & gt_bool).sum()
        denom = pred_bool.sum() + gt_bool.sum()
        if denom == 0:
            return float("nan")
        return float(2 * intersection / denom)
    except Exception as e:
        warnings.warn(f"Dice computation failed: {e}")
        return float("nan")


def compute_jaccard(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Jaccard (IoU): Jaccard = |A∩B| / |A∪B|.
    """
    try:
        pred_bool = pred_mask > 0
        gt_bool = gt_mask > 0
        intersection = (pred_bool & gt_bool).sum()
        union = (pred_bool | gt_bool).sum()
        if union == 0:
            return float("nan")
        return float(intersection / union)
    except Exception as e:
        warnings.warn(f"Jaccard computation failed: {e}")
        return float("nan")


def _fractal_dimension_mask(binary: np.ndarray) -> float:
    """Box-counting fractal dimension for a binary mask."""
    try:
        if not binary.any():
            return float("nan")
        min_dim = min(binary.shape)
        sizes = 2 ** np.arange(1, int(math.log2(min_dim)))
        if len(sizes) < 2:
            return float("nan")
        counts = []
        for size in sizes:
            h, w = binary.shape
            ph = math.ceil(h / size) * size
            pw = math.ceil(w / size) * size
            padded = np.zeros((ph, pw), dtype=bool)
            padded[:h, :w] = binary
            boxes = padded.reshape(ph // size, size, pw // size, size)
            counts.append(boxes.any(axis=(1, 3)).sum())
        counts = np.array(counts, dtype=float)
        valid = counts > 0
        if valid.sum() < 2:
            return float("nan")
        coeffs = np.polyfit(np.log(1.0 / sizes[valid]), np.log(counts[valid]), 1)
        return float(coeffs[0])
    except Exception as e:
        warnings.warn(f"FD computation failed: {e}")
        return float("nan")


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_preprocessing(
    dataset_root: str,
    save_dir: str,
    gt_available: Optional[bool] = None,
) -> None:
    """
    Process all OCTA images, compute per-image metrics, and export results.

    Exports:
    - preprocessing_metrics.csv   — per-image metrics
    - preprocessing_summary.csv   — class-wise mean ± std
    - boxplots/violin plots comparing healthy vs DR

    Parameters
    ----------
    dataset_root : str           — root of the dataset
    save_dir     : str           — output directory
    gt_available : bool, optional — if None, auto-detected
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    root = Path(dataset_root)

    if gt_available is None:
        gt_available = check_ground_truth_masks(root)

    label_dirs = {"healthy": 0, "dr": 1}
    for alt in ("diabetic_retinopathy", "DR"):
        if (root / alt).exists() and "dr" not in label_dirs:
            label_dirs[alt] = 1

    records = []
    for class_name, label in label_dirs.items():
        octa_dir = root / class_name / "OCTA"
        if not octa_dir.exists():
            continue
        img_paths = sorted(
            list(octa_dir.glob("*.png")) + list(octa_dir.glob("*.jpg"))
            + list(octa_dir.glob("*.tif")) + list(octa_dir.glob("*.bmp"))
        )
        for img_path in img_paths:
            try:
                if CV2_AVAILABLE:
                    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                elif PIL_AVAILABLE:
                    img_gray = np.array(Image.open(img_path).convert("L"))
                else:
                    continue

                stages = preprocess_octa(img_gray)
                vessel_mask = stages["morphological_cleanup"]

                snr = compute_snr(img_gray, vessel_mask)
                cnr = compute_cnr(img_gray, vessel_mask)
                from feature_extraction import (
                    compute_vessel_density, compute_skeleton_line_density,
                    _fractal_dimension_mask,
                )
                vd = compute_vessel_density(vessel_mask)
                sld = compute_skeleton_line_density(vessel_mask)
                fd = _fractal_dimension_mask((vessel_mask > 0))

                rec = {
                    "image_path": str(img_path),
                    "class": class_name,
                    "label": label,
                    "snr": snr, "cnr": cnr, "vd": vd, "sld": sld, "fd": fd,
                }

                # Ground truth metrics (only if masks available)
                if gt_available:
                    mask_path = img_path.parent.parent / "masks" / img_path.name
                    if mask_path.exists():
                        if CV2_AVAILABLE:
                            gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        else:
                            gt = np.array(Image.open(mask_path).convert("L"))
                        rec["dice"] = compute_dice(vessel_mask, gt)
                        rec["jaccard"] = compute_jaccard(vessel_mask, gt)

                records.append(rec)
            except Exception as e:
                warnings.warn(f"Preprocessing evaluation failed for {img_path}: {e}")

    if not records:
        print("[preprocessing] No OCTA images found for evaluation.")
        return

    # Export per-image metrics
    import csv as _csv
    metric_csv = save_path / "preprocessing_metrics.csv"
    with open(metric_csv, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"[preprocessing] Saved: {metric_csv}")

    # Class-wise summary
    _export_summary(records, save_path)

    # Plots
    _plot_preprocessing_comparison(records, save_path)


def _export_summary(records: list, save_path: Path) -> None:
    """Export class-wise mean ± std summary CSV."""
    import csv as _csv
    metric_cols = ["snr", "cnr", "vd", "sld", "fd"]
    classes = list({r["class"] for r in records})
    summary_rows = []
    for cls in classes:
        row = {"class": cls}
        cls_recs = [r for r in records if r["class"] == cls]
        for col in metric_cols:
            vals = [r[col] for r in cls_recs if col in r and not math.isnan(float(r[col]))]
            if vals:
                row[f"{col}_mean"] = float(np.mean(vals))
                row[f"{col}_std"] = float(np.std(vals))
            else:
                row[f"{col}_mean"] = float("nan")
                row[f"{col}_std"] = float("nan")
        summary_rows.append(row)

    summary_csv = save_path / "preprocessing_summary.csv"
    if summary_rows:
        with open(summary_csv, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"[preprocessing] Summary saved: {summary_csv}")


def _plot_preprocessing_comparison(records: list, save_path: Path) -> None:
    """Boxplot/violin comparison between healthy and DR."""
    if not MPL_AVAILABLE:
        return
    metric_cols = ["snr", "cnr", "vd", "sld", "fd"]
    n = len(metric_cols)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]
    for i, col in enumerate(metric_cols):
        ax = axes[i]
        healthy_vals = [r[col] for r in records if r["label"] == 0 and col in r
                        and not math.isnan(float(r[col]))]
        dr_vals = [r[col] for r in records if r["label"] == 1 and col in r
                   and not math.isnan(float(r[col]))]
        if not healthy_vals and not dr_vals:
            ax.set_visible(False)
            continue
        ax.boxplot([healthy_vals, dr_vals], labels=["Healthy", "DR"])
        ax.set_title(col.upper())
        if SCIPY_AVAILABLE and len(healthy_vals) > 0 and len(dr_vals) > 0:
            try:
                _, pval = mannwhitneyu(healthy_vals, dr_vals, alternative="two-sided")
                ax.set_xlabel(f"p={pval:.3f}")
            except Exception:
                pass

    plt.suptitle("Preprocessing Metrics: Healthy vs DR", fontsize=13)
    plt.tight_layout()
    out = save_path / "preprocessing_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[preprocessing] Plot saved: {out}")


# ---------------------------------------------------------------------------
# 8-stage visualization
# ---------------------------------------------------------------------------

def visualize_preprocessing_stages(
    octa_path: str,
    save_path: str,
) -> None:
    """
    Generate an 8-panel figure showing each preprocessing stage:
    original → grayscale → Gaussian blur → CLAHE → threshold
    → cleanup → skeleton → vessel mask overlay.

    Parameters
    ----------
    octa_path : str — path to a single OCTA image file
    save_path : str — output PNG path
    """
    if not CV2_AVAILABLE:
        warnings.warn("OpenCV required for preprocessing visualization.")
        return
    if not MPL_AVAILABLE:
        warnings.warn("Matplotlib required for preprocessing visualization.")
        return

    try:
        img_gray = cv2.imread(str(octa_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            warnings.warn(f"Could not load image: {octa_path}")
            return
        stages = preprocess_octa(img_gray)
        stage_names = [
            "original", "grayscale", "gaussian_blur", "clahe",
            "adaptive_threshold", "morphological_cleanup", "skeleton",
            "vessel_overlay",
        ]
        titles = [
            "1. Original", "2. Grayscale", "3. Gaussian Blur", "4. CLAHE",
            "5. Adaptive Threshold", "6. Morphological Cleanup",
            "7. Skeleton", "8. Vessel Overlay",
        ]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        for i, (key, title) in enumerate(zip(stage_names, titles)):
            if key in stages:
                img = stages[key]
                if img.ndim == 3:
                    axes[i].imshow(img)
                else:
                    axes[i].imshow(img, cmap="gray")
                axes[i].set_title(title, fontsize=9)
            axes[i].axis("off")

        plt.suptitle("OCTA Preprocessing Pipeline Stages", fontsize=13)
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[preprocessing] Stage visualization saved: {save_path}")
    except Exception as e:
        warnings.warn(f"Preprocessing stage visualization failed: {e}")


# ---------------------------------------------------------------------------
# U-Net feasibility report generator
# ---------------------------------------------------------------------------

def generate_unet_feasibility_report(save_dir: str) -> None:
    """
    Generate a markdown report explaining why supervised U-Net training is
    not implemented (no ground-truth masks in the dataset).

    Parameters
    ----------
    save_dir : str — directory where the report markdown will be saved
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    report_path = save_path / "unet_feasibility.md"

    content = """# U-Net Vessel Segmentation: Feasibility Report

## Summary

Supervised U-Net training was **not implemented** because no pixel-level vessel
segmentation ground-truth labels were found in the dataset.

## Why U-Net Requires Manual Labels

U-Net is a supervised segmentation network that learns to predict a binary vessel
mask for each input pixel.  To train it, the model requires:

1. **Input images** — OCTA scans (available ✓)
2. **Ground-truth masks** — manually annotated pixel-level vessel labels (**not found ✗**)

Without masks, the network has no target output to learn from and cannot be
trained.

## What Was Found in the Dataset

- OCT images: paired scans for healthy subjects and DR patients.
- OCTA images: co-registered angiography images.
- **No vessel segmentation masks were detected** in any subfolder of the dataset.

## What Was Implemented Instead

A **classical preprocessing pipeline** was applied to OCTA images:

```
Grayscale → Gaussian Blur → CLAHE → Adaptive Threshold
→ Morphological Cleanup → Skeletonize → Vessel Mask
```

This pipeline was evaluated using **no-reference quantitative metrics**:
- SNR, CNR, Vessel Density, Skeleton Line Density, Fractal Dimension.

## Future Work

To enable supervised U-Net training, the following steps are required:

1. Obtain manual pixel-level vessel annotations from a clinical expert.
2. Or use a **pretrained public model** (e.g., trained on DRIVE / STARE / CHASE-DB1)
   and apply it as a zero-shot segmenter.  Results should be validated carefully
   as domain shift may affect accuracy.

## References

- Ronneberger, O. et al. (2015). *U-Net: Convolutional Networks for Biomedical
  Image Segmentation*. MICCAI.
- Staal, J. et al. (2004). *Ridge based vessel segmentation in color images of
  the retina*. IEEE TMI. (DRIVE dataset)

---
*This report was auto-generated by `preprocessing_evaluation.py`.*
"""
    report_path.write_text(content)
    print(f"[preprocessing] U-Net feasibility report: {report_path}")
