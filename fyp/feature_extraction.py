"""
feature_extraction.py
---------------------
Handcrafted vascular feature extraction from binary OCTA vessel masks.

Features implemented:
    1. VD   — Vessel Density       = N_vessel / N_total
    2. SLD  — Skeleton Line Density = N_skeleton / N_total
    3. FAZ  — FAZ Circularity      = 4π·A / P²
    4. TI   — Tortuosity Index     = mean(L_curve / L_straight) over branches
    5. VP   — Vessel Perimeter     = total contour perimeter
    6. FD   — Fractal Dimension    = box-counting method

All features return NaN safely if computation fails.

Medical context:
- In DR, vessel density typically DECREASES due to capillary dropout.
- FAZ (foveal avascular zone) enlarges and becomes less circular in DR.
- Tortuosity INCREASES in DR due to neovascularisation / altered flow.
- Fractal dimension characterises the complexity of the vascular network.
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
    warnings.warn("OpenCV (cv2) not available — vessel perimeter will be NaN.")

try:
    from skimage.morphology import skeletonize, thin
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not available — SLD and tortuosity will be NaN.")

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
# Preprocessing helper: binarize an OCTA image into a vessel mask
# ---------------------------------------------------------------------------

def binarize_octa(img_array: np.ndarray, threshold: int = 30) -> np.ndarray:
    """
    Produce a binary vessel mask from a grayscale OCTA image.

    Parameters
    ----------
    img_array : np.ndarray (H, W) uint8 — grayscale OCTA image
    threshold : int — intensity threshold (vessels are bright pixels)

    Returns
    -------
    mask : np.ndarray (H, W) uint8 with values 0 or 255
    """
    if img_array.ndim == 3:
        img_array = img_array[:, :, 0]  # take first channel if RGB
    _, mask = cv2.threshold(img_array, threshold, 255, cv2.THRESH_BINARY) \
        if CV2_AVAILABLE else (None, (img_array > threshold).astype(np.uint8) * 255)
    return mask


# ---------------------------------------------------------------------------
# Feature 1: Vessel Density (VD)
# ---------------------------------------------------------------------------

def compute_vessel_density(vessel_mask: np.ndarray) -> float:
    """
    Vessel Density (VD) = N_vessel_pixels / N_total_pixels.

    In DR, VD typically decreases due to capillary non-perfusion.

    Parameters
    ----------
    vessel_mask : np.ndarray — binary mask (non-zero = vessel pixel)

    Returns
    -------
    float — vessel density ∈ [0, 1]
    """
    try:
        n_total = vessel_mask.size
        n_vessel = int((vessel_mask > 0).sum())
        return n_vessel / max(n_total, 1)
    except Exception as e:
        warnings.warn(f"VD computation failed: {e}")
        return float("nan")


# ---------------------------------------------------------------------------
# Feature 2: Skeleton Line Density (SLD)
# ---------------------------------------------------------------------------

def compute_skeleton_line_density(vessel_mask: np.ndarray) -> float:
    """
    Skeleton Line Density (SLD) = N_skeleton_pixels / N_total_pixels.

    Captures the geometric extent of the vascular network, independent
    of vessel calibre.

    Parameters
    ----------
    vessel_mask : np.ndarray — binary mask

    Returns
    -------
    float — SLD ∈ [0, 1], or NaN if skimage is unavailable
    """
    if not SKIMAGE_AVAILABLE:
        return float("nan")
    try:
        binary = (vessel_mask > 0).astype(bool)
        skeleton = skeletonize(binary)
        n_skel = int(skeleton.sum())
        return n_skel / max(vessel_mask.size, 1)
    except Exception as e:
        warnings.warn(f"SLD computation failed: {e}")
        return float("nan")


# ---------------------------------------------------------------------------
# Feature 3: FAZ Circularity
# ---------------------------------------------------------------------------

# Circularity formula: C = 4π·A / P²
# Perfect circle → C = 1.  More irregular → C < 1.
# FAZ enlarges and becomes less circular in DR.

_MIN_FAZ_AREA = 50   # pixels — avoid noise detections

def compute_faz_circularity(vessel_mask: np.ndarray) -> float:
    """
    FAZ Circularity = 4π·A / P² where A = FAZ area, P = FAZ perimeter.

    The FAZ (foveal avascular zone) is detected as the largest connected
    component in the central region of the INVERTED vessel mask.

    Returns NaN (with warning) if detection fails or FAZ is too small.

    Parameters
    ----------
    vessel_mask : np.ndarray — binary vessel mask (non-zero = vessel)

    Returns
    -------
    float — circularity ∈ (0, 1], or NaN
    """
    if not CV2_AVAILABLE:
        return float("nan")
    try:
        # Invert: background (non-vessel) becomes foreground
        inverted = cv2.bitwise_not(vessel_mask)
        h, w = inverted.shape[:2]
        # Central region: inner 50% of image
        cy, cx = h // 2, w // 2
        ch, cw = h // 4, w // 4
        central = np.zeros_like(inverted)
        central[cy - ch: cy + ch, cx - cw: cx + cw] = \
            inverted[cy - ch: cy + ch, cx - cw: cx + cw]

        n_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(
            central, connectivity=8
        )
        if n_labels < 2:
            warnings.warn("FAZ detection: no connected components found.")
            return float("nan")

        # Largest component (excluding background label 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = int(np.argmax(areas)) + 1  # +1 because we excluded 0
        faz_area = int(areas[largest_idx - 1])

        if faz_area < _MIN_FAZ_AREA:
            warnings.warn(
                f"FAZ detection: largest component area {faz_area} < "
                f"minimum {_MIN_FAZ_AREA}. Returning NaN."
            )
            return float("nan")

        faz_mask = (labels_im == largest_idx).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            faz_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return float("nan")
        perimeter = cv2.arcLength(contours[0], closed=True)
        if perimeter < 1e-6:
            return float("nan")
        # Circularity = 4π·A / P²
        circularity = (4 * math.pi * faz_area) / (perimeter ** 2)
        return float(min(circularity, 1.0))  # clamp to 1.0 due to discretisation

    except Exception as e:
        warnings.warn(f"FAZ circularity computation failed: {e}")
        return float("nan")


# ---------------------------------------------------------------------------
# Feature 4: Tortuosity Index (TI)
# ---------------------------------------------------------------------------

_MIN_BRANCH_LEN = 20   # pixels — filter short branches
_MIN_STRAIGHT = 3      # pixels — filter degenerate branches

def _trace_branches(skeleton: np.ndarray) -> List[np.ndarray]:
    """
    Extract individual branch pixel sequences from a skeletonised image.

    Returns a list of arrays, each shaped (N, 2) with (row, col) coordinates.
    """
    if not SKIMAGE_AVAILABLE:
        return []
    try:
        from skimage.morphology import label as sk_label
        labeled = sk_label(skeleton)
        branches = []
        for lbl in range(1, labeled.max() + 1):
            coords = np.argwhere(labeled == lbl)
            if len(coords) >= _MIN_BRANCH_LEN:
                branches.append(coords)
        return branches
    except Exception:
        return []


def compute_tortuosity_index(vessel_mask: np.ndarray) -> float:
    """
    Tortuosity Index (TI) = mean(L_curve / L_straight) over valid branches.

    L_curve   = total arc length (sum of pixel-to-pixel distances)
    L_straight = Euclidean distance between branch endpoints

    Branches shorter than MIN_BRANCH_LEN or with L_straight < MIN_STRAIGHT
    are filtered to avoid degenerate / noisy estimates.

    Returns NaN if no valid branches exist.

    Parameters
    ----------
    vessel_mask : np.ndarray — binary vessel mask

    Returns
    -------
    float — mean tortuosity index, or NaN
    """
    if not SKIMAGE_AVAILABLE:
        return float("nan")
    try:
        binary = (vessel_mask > 0).astype(bool)
        skeleton = skeletonize(binary)
        branches = _trace_branches(skeleton)

        ratios = []
        for branch in branches:
            if len(branch) < _MIN_BRANCH_LEN:
                continue
            # Sort coordinates along the branch (approximate by index order)
            # Arc length: sum of consecutive Euclidean distances
            diffs = np.diff(branch, axis=0).astype(float)
            l_curve = float(np.sum(np.linalg.norm(diffs, axis=1)))
            # Straight-line distance between first and last pixel
            l_straight = float(np.linalg.norm(branch[-1] - branch[0]))
            if l_straight < _MIN_STRAIGHT:
                continue
            ratios.append(l_curve / l_straight)

        if not ratios:
            warnings.warn("Tortuosity: no valid branches found. Returning NaN.")
            return float("nan")
        return float(np.mean(ratios))

    except Exception as e:
        warnings.warn(f"Tortuosity computation failed: {e}")
        return float("nan")


# ---------------------------------------------------------------------------
# Feature 5: Vessel Perimeter (VP)
# ---------------------------------------------------------------------------

def compute_vessel_perimeter(vessel_mask: np.ndarray) -> float:
    """
    Vessel Perimeter (VP) = total contour perimeter from all vessel regions.

    Uses cv2.arcLength on each detected contour.

    Parameters
    ----------
    vessel_mask : np.ndarray — binary vessel mask

    Returns
    -------
    float — total perimeter in pixels, or NaN
    """
    if not CV2_AVAILABLE:
        return float("nan")
    try:
        binary_u8 = (vessel_mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            binary_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        total_perimeter = sum(cv2.arcLength(c, closed=True) for c in contours)
        return float(total_perimeter)
    except Exception as e:
        warnings.warn(f"Vessel perimeter computation failed: {e}")
        return float("nan")


# ---------------------------------------------------------------------------
# Feature 6: Fractal Dimension (FD) — box-counting method
# ---------------------------------------------------------------------------

def compute_fractal_dimension(vessel_mask: np.ndarray) -> float:
    """
    Fractal Dimension (FD) — box-counting method.

    Counts the number of boxes N(r) that contain at least one vessel pixel
    at decreasing box sizes r.  FD = -slope of log N(r) vs log(1/r).

    A more complex vascular network has a higher FD.  FD typically decreases
    in DR due to capillary loss.

    Parameters
    ----------
    vessel_mask : np.ndarray — binary vessel mask

    Returns
    -------
    float — fractal dimension estimate, or NaN
    """
    try:
        binary = (vessel_mask > 0)
        if not binary.any():
            return float("nan")

        # Box sizes as powers of 2
        min_dim = min(binary.shape)
        sizes = 2 ** np.arange(1, int(math.log2(min_dim)))
        if len(sizes) < 2:
            return float("nan")

        counts = []
        for size in sizes:
            # Pad image so dimensions are divisible by size
            h, w = binary.shape
            ph = math.ceil(h / size) * size
            pw = math.ceil(w / size) * size
            padded = np.zeros((ph, pw), dtype=bool)
            padded[:h, :w] = binary
            # Reshape and count non-empty boxes
            boxes = padded.reshape(
                ph // size, size, pw // size, size
            )
            count = boxes.any(axis=(1, 3)).sum()
            counts.append(count)

        counts = np.array(counts, dtype=float)
        sizes = np.array(sizes, dtype=float)
        # Remove zero counts to avoid log(0)
        valid = counts > 0
        if valid.sum() < 2:
            return float("nan")
        # Fit line: log(N) = FD * log(1/r) + const
        coeffs = np.polyfit(np.log(1.0 / sizes[valid]), np.log(counts[valid]), 1)
        return float(coeffs[0])

    except Exception as e:
        warnings.warn(f"Fractal dimension computation failed: {e}")
        return float("nan")


# ---------------------------------------------------------------------------
# Composite: extract all features at once
# ---------------------------------------------------------------------------

def extract_all_features(vessel_mask: np.ndarray) -> Dict[str, float]:
    """
    Compute all 6 handcrafted vascular features from a binary vessel mask.

    Parameters
    ----------
    vessel_mask : np.ndarray — binary vessel mask (non-zero = vessel pixel)

    Returns
    -------
    dict with keys: vd, sld, faz_circularity, tortuosity, vessel_perimeter,
                    fractal_dimension
    """
    return {
        "vd": compute_vessel_density(vessel_mask),
        "sld": compute_skeleton_line_density(vessel_mask),
        "faz_circularity": compute_faz_circularity(vessel_mask),
        "tortuosity": compute_tortuosity_index(vessel_mask),
        "vessel_perimeter": compute_vessel_perimeter(vessel_mask),
        "fractal_dimension": compute_fractal_dimension(vessel_mask),
    }


# ---------------------------------------------------------------------------
# Dataset-level extraction
# ---------------------------------------------------------------------------

def extract_features_dataset(
    dataset_root: str,
    save_dir: str,
    threshold: int = 30,
) -> None:
    """
    Process all OCTA images in the dataset, compute features, and export
    results as CSV files and comparison plots.

    Expects directory structure:
        <dataset_root>/
            healthy/OCTA/
            dr/OCTA/         (or diabetic_retinopathy/OCTA/)

    Parameters
    ----------
    dataset_root : str — root of the dataset
    save_dir     : str — output directory for CSV and plots
    threshold    : int — binarization threshold for vessel mask
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    root = Path(dataset_root)
    label_dirs = {"healthy": 0, "dr": 1}
    # Support alternate naming
    for alt in ("diabetic_retinopathy", "DR"):
        if (root / alt).exists() and "dr" not in label_dirs:
            label_dirs[alt] = 1

    records = []
    for class_name, label in label_dirs.items():
        octa_dir = root / class_name / "OCTA"
        if not octa_dir.exists():
            continue
        img_paths = list(octa_dir.glob("*.png")) + list(octa_dir.glob("*.jpg")) \
                    + list(octa_dir.glob("*.tif")) + list(octa_dir.glob("*.bmp"))
        for img_path in img_paths:
            try:
                img_array = _load_grayscale(img_path)
                mask = binarize_octa(img_array, threshold)
                feats = extract_all_features(mask)
                feats["image_path"] = str(img_path)
                feats["label"] = label
                feats["class"] = class_name
                records.append(feats)
            except Exception as e:
                warnings.warn(f"Feature extraction failed for {img_path}: {e}")

    if not records:
        print("[feature_extraction] No OCTA images found.")
        return

    # Export CSV
    if PANDAS_AVAILABLE:
        import pandas as pd
        df = pd.DataFrame(records)
        csv_path = save_path / "feature_extraction.csv"
        df.to_csv(csv_path, index=False)
        print(f"[feature_extraction] Saved: {csv_path}")
        _plot_feature_comparison(df, save_path)
        _plot_correlation_heatmap(df, save_path)
    else:
        import csv as _csv
        csv_path = save_path / "feature_extraction.csv"
        with open(csv_path, "w", newline="") as f:
            if records:
                writer = _csv.DictWriter(f, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
        print(f"[feature_extraction] Saved: {csv_path}")


def _load_grayscale(img_path: Path) -> np.ndarray:
    """Load an image as a grayscale numpy array."""
    if CV2_AVAILABLE:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"cv2 could not read {img_path}")
        return img
    if PIL_AVAILABLE:
        img = Image.open(img_path).convert("L")
        return np.array(img)
    raise ImportError("Neither cv2 nor Pillow is available.")


def _plot_feature_comparison(df, save_path: Path) -> None:
    """Generate boxplot comparison between healthy and DR for each feature."""
    if not MPL_AVAILABLE or not PANDAS_AVAILABLE:
        return
    import pandas as pd
    feature_cols = ["vd", "sld", "faz_circularity", "tortuosity",
                    "vessel_perimeter", "fractal_dimension"]
    available = [c for c in feature_cols if c in df.columns]

    if SCIPY_AVAILABLE:
        from scipy.stats import mannwhitneyu

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(available):
        ax = axes[i]
        healthy_vals = df[df["label"] == 0][col].dropna().values
        dr_vals = df[df["label"] == 1][col].dropna().values
        ax.boxplot([healthy_vals, dr_vals], labels=["Healthy", "DR"])
        ax.set_title(col)
        ax.set_ylabel("Value")
        if SCIPY_AVAILABLE and len(healthy_vals) > 0 and len(dr_vals) > 0:
            try:
                _, pval = mannwhitneyu(healthy_vals, dr_vals, alternative="two-sided")
                ax.set_xlabel(f"Mann-Whitney p={pval:.3f}")
            except Exception:
                pass

    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Vascular Feature Comparison: Healthy vs DR", fontsize=13)
    plt.tight_layout()
    out = save_path / "feature_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[feature_extraction] Saved: {out}")


def _plot_correlation_heatmap(df, save_path: Path) -> None:
    """Feature correlation heatmap."""
    if not MPL_AVAILABLE or not PANDAS_AVAILABLE:
        return
    feature_cols = ["vd", "sld", "faz_circularity", "tortuosity",
                    "vessel_perimeter", "fractal_dimension"]
    available = [c for c in feature_cols if c in df.columns]
    if len(available) < 2:
        return
    corr = df[available].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(available)))
    ax.set_yticks(range(len(available)))
    ax.set_xticklabels(available, rotation=45, ha="right")
    ax.set_yticklabels(available)
    for ii in range(len(available)):
        for jj in range(len(available)):
            ax.text(jj, ii, f"{corr.iloc[ii, jj]:.2f}",
                    ha="center", va="center", fontsize=7)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    out = save_path / "feature_correlation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[feature_extraction] Saved: {out}")


# ---------------------------------------------------------------------------
# Visual debugging overlays
# ---------------------------------------------------------------------------

def visualize_feature_overlays(
    octa_path: str,
    vessel_mask: np.ndarray,
    save_path: str,
) -> None:
    """
    Generate a 3-panel figure: vessel mask overlay | skeleton overlay |
    FAZ overlay.

    Parameters
    ----------
    octa_path   : str        — path to the original OCTA image
    vessel_mask : np.ndarray — binary vessel mask
    save_path   : str        — output PNG path
    """
    if not MPL_AVAILABLE or not PIL_AVAILABLE:
        warnings.warn("Matplotlib / Pillow required for overlay visualization.")
        return
    try:
        orig = np.array(Image.open(octa_path).convert("RGB"))
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Panel 1: vessel mask overlay
        axes[0].imshow(orig)
        axes[0].imshow(vessel_mask, alpha=0.4, cmap="Reds")
        axes[0].set_title("Vessel Mask Overlay")
        axes[0].axis("off")

        # Panel 2: skeleton overlay
        if SKIMAGE_AVAILABLE:
            skeleton = skeletonize((vessel_mask > 0).astype(bool))
            axes[1].imshow(orig)
            skel_overlay = np.zeros((*orig.shape[:2], 4), dtype=np.float32)
            skel_overlay[skeleton] = [0, 1, 0, 0.8]
            axes[1].imshow(skel_overlay)
            axes[1].set_title("Skeleton Overlay")
        else:
            axes[1].imshow(orig)
            axes[1].set_title("Skeleton (skimage unavailable)")
        axes[1].axis("off")

        # Panel 3: FAZ overlay (approximate)
        axes[2].imshow(orig)
        if CV2_AVAILABLE:
            h, w = vessel_mask.shape[:2]
            cy, cx = h // 2, w // 2
            ch, cw = h // 4, w // 4
            faz_box = np.zeros((*orig.shape[:2], 4), dtype=np.float32)
            faz_box[cy - ch: cy + ch, cx - cw: cx + cw] = [0, 0, 1, 0.2]
            axes[2].imshow(faz_box)
        axes[2].set_title("Central FAZ Region")
        axes[2].axis("off")

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[feature_extraction] Saved: {save_path}")
    except Exception as e:
        warnings.warn(f"Feature overlay visualization failed: {e}")
