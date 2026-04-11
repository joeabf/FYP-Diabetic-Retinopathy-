"""
setup_project.py
-----------------
Project setup script for the FYP Diabetic Retinopathy classifier.

Creates the required output directory structure and generates reference
markdown documents:
    - outputs/reports/formulas_references.md
    - outputs/reports/advisor_methodology.md

Run this once before starting experiments:
    python fyp/setup_project.py
"""

from __future__ import annotations

import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Output directory structure
# ---------------------------------------------------------------------------

OUTPUT_DIRS = [
    "outputs/figures",
    "outputs/tables",
    "outputs/reports",
    "outputs/checkpoints",
    "outputs/plots",       # compatibility with older save_plots calls
]


def create_output_dirs(base: str = ".") -> None:
    """
    Create the standard output directory structure.

    Parameters
    ----------
    base : str — base directory (default: current working directory)
    """
    for rel_dir in OUTPUT_DIRS:
        full_path = Path(base) / rel_dir
        full_path.mkdir(parents=True, exist_ok=True)
    print("[setup] Output directories created:")
    for rel_dir in OUTPUT_DIRS:
        print(f"  {Path(base) / rel_dir}")


# ---------------------------------------------------------------------------
# Formulas and references document
# ---------------------------------------------------------------------------

FORMULAS_REFERENCES_MD = r"""# Metric and Feature Formulas with References

All equations and features used in the diabetic retinopathy (DR) classifier
are documented here.  Clinical relevance for DR is included where applicable.

---

## 1. Classification Metrics

### ROC AUC
- **Equation**: Area under the Receiver Operating Characteristic curve
  (integral of TPR vs FPR over all thresholds).
- **Interpretation**: Threshold-independent measure of discriminative ability.
  AUC = 1.0 → perfect; AUC = 0.5 → no discriminative power.
- **Clinical meaning**: For DR screening, AUC quantifies the overall ability
  to separate healthy from diseased patients.
- **Reference**: Hanley & McNeil (1982). *The meaning and use of the area
  under a receiver operating characteristic (ROC) curve*. Radiology. [VERIFY]

### PR AUC (Average Precision)
- **Equation**: Area under the Precision-Recall curve.
  AP = Σ (R_n − R_{n-1}) · P_n
- **Interpretation**: More informative than ROC AUC for imbalanced datasets.
- **Clinical meaning**: Reflects the trade-off between catching DR cases
  (recall) and avoiding unnecessary referrals (precision).
- **Reference**: Davis & Goadrich (2006). *The relationship between
  Precision-Recall and ROC curves*. ICML. [VERIFY]

### F1 Score
- **Equation**: F1 = 2·TP / (2·TP + FP + FN)
- **Interpretation**: Harmonic mean of precision and recall.
- **Clinical meaning**: Balanced measure for imbalanced class distributions.

### Balanced Accuracy
- **Equation**: BAcc = (Recall + Specificity) / 2
- **Interpretation**: Accounts for imbalance; 0.5 = random, 1.0 = perfect.

### Sensitivity (Recall)
- **Equation**: Recall = TP / (TP + FN)
- **Clinical meaning**: Proportion of DR patients correctly identified.
  In a screening context, high recall is critical to avoid missed diagnoses.

### Specificity
- **Equation**: Specificity = TN / (TN + FP)
- **Clinical meaning**: Proportion of healthy patients correctly identified.

---

## 2. Vascular Features from OCTA

### Vessel Density (VD)
- **Equation**: VD = N_vessel_pixels / N_total_pixels
- **Interpretation**: Proportion of image area covered by vessels.
- **Clinical meaning**: VD decreases in DR due to capillary non-perfusion
  (ischaemia). A lower VD is associated with disease severity.
- **Reference**: Spaide et al. (2018). *Optical coherence tomography
  angiography*. Prog Retin Eye Res. [VERIFY]

### Skeleton Line Density (SLD)
- **Equation**: SLD = N_skeleton_pixels / N_total_pixels
  (skeleton computed via medial-axis / thinning)
- **Interpretation**: Geometric extent of the vascular network,
  independent of vessel calibre.
- **Clinical meaning**: Reduced SLD indicates fewer vessel branches.
- **Reference**: Reif et al. (2014). *Quantifying optical microangiography
  images obtained from a spectral domain optical coherence tomography system*.
  IJBO. [VERIFY]

### FAZ Circularity
- **Equation**: C = 4π·A / P²
  where A = FAZ area (pixels), P = FAZ perimeter (pixels).
  Perfect circle: C = 1.  More irregular: C < 1.
- **Interpretation**: Measures how circular the foveal avascular zone is.
- **Clinical meaning**: FAZ enlarges and becomes less circular in DR due to
  capillary dropout around the fovea.
- **Reference**: Samara et al. (2017). *Correlation of foveal avascular zone
  size with foveal morphology in normal eyes using optical coherence
  tomography angiography*. Retina. [VERIFY]

### Tortuosity Index (TI)
- **Equation**: TI = mean(L_curve / L_straight) over valid vessel branches
  where L_curve = total arc length of branch,
        L_straight = Euclidean distance between endpoints.
  Minimum branch length filter: 20 pixels.  L_straight < 3 px filtered out.
- **Interpretation**: TI = 1.0 → straight vessels.  TI > 1.0 → curved.
- **Clinical meaning**: Vascular tortuosity increases in DR due to
  neovascularisation and altered blood flow regulation.
- **Reference**: Grisan et al. (2003). *A divide et impera strategy for
  automatic classification of retinal vessels into arteries and veins*.
  EMBC. [VERIFY]

### Vessel Perimeter (VP)
- **Equation**: VP = total contour perimeter (cv2.arcLength) over all
  connected vessel regions.
- **Interpretation**: Total boundary length of all detected vessel segments.
- **Clinical meaning**: Correlates with vascular complexity.

### Fractal Dimension (FD)
- **Equation**: FD = −slope of log N(r) vs log(1/r)
  using box-counting method:
  N(r) = number of boxes of side r containing ≥1 vessel pixel.
- **Interpretation**: FD ≈ 1 → simple curves; FD → 2 → space-filling.
  Normal retinal vasculature: FD ≈ 1.6–1.7.
- **Clinical meaning**: FD decreases in DR as the vascular network
  simplifies due to capillary dropout.
- **Reference**: Daxer (1993). *The fractal geometry of proliferative
  diabetic retinopathy: implications for the diagnosis and the process
  of retinal vasculogenesis*. Curr Eye Res. [VERIFY]

---

## 3. Preprocessing Quality Metrics

### SNR (Signal-to-Noise Ratio)
- **Equation**: SNR = μ_signal / σ_noise
  where μ_signal = mean intensity of vessel pixels,
        σ_noise = std of background pixels.
- **Clinical meaning**: Higher SNR → clearer vessel signal.

### CNR (Contrast-to-Noise Ratio)
- **Equation**: CNR = (μ_vessel − μ_background) / σ_background
- **Interpretation**: How well vessels are separated from background.
- **Reference**: Bhatt et al. (2013). *Evaluation of image quality metrics
  for diagnostic ultrasound*. JTMBE. [VERIFY]

---

## 4. Segmentation Metrics (if ground truth available)

### Dice Coefficient
- **Equation**: Dice = 2|A∩B| / (|A| + |B|)
- **Interpretation**: Overlap measure ∈ [0, 1].  1.0 = perfect overlap.

### Jaccard Index (IoU)
- **Equation**: Jaccard = |A∩B| / |A∪B|
- **Interpretation**: Intersection over Union.  Related to Dice:
  Dice = 2·Jaccard / (1 + Jaccard)

---

*[VERIFY] = placeholder citation — confirm exact reference before submission.*
"""


def write_formulas_references(save_dir: str) -> None:
    """
    Write the formulas and references markdown file.

    Parameters
    ----------
    save_dir : str — directory where the file will be saved
    """
    path = Path(save_dir) / "formulas_references.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FORMULAS_REFERENCES_MD)
    print(f"[setup] Formulas and references: {path}")


# ---------------------------------------------------------------------------
# Methodology document
# ---------------------------------------------------------------------------

METHODOLOGY_MD = """# Methodology Summary

## Overview
Binary classification of OCT + OCTA image pairs into:
- **Class 0** — Healthy
- **Class 1** — Diabetic Retinopathy (DR)

## Architecture
- **Two ResNet-18 backbones** (ImageNet pretrained) — one per modality.
- Feature vectors (512-d each) are concatenated → 1024-d.
- Fusion classifier: Linear(1024→256) → ReLU → Dropout(0.4) → Linear(256→1).
- Training strategy: Phase 1 (backbone frozen, head trained),
  Phase 2 (all layers fine-tuned at 1/10 LR).

## Dataset
- Paired OCT and OCTA images.
- Labels: healthy (0), DR (1).
- Patient-level stratified split: 70% train / 15% val / 15% test.
- No patient appears in more than one split.

## Class Imbalance Handling
- `WeightedRandomSampler` (oversamples minority class in training).
- `BCEWithLogitsLoss` with `pos_weight = N_healthy / N_DR`.
- Optional: Binary Focal Loss (`--loss-fn focal`).

## Augmentation
- Modular `PairedAugmentPolicy` with synchronized OCT/OCTA transforms.
- Intensity transforms (brightness/contrast) applied to OCT only.
- Ablation study: none / basic_all / basic_dr_only.

## Threshold Selection
- Threshold is optimised on the **validation set** to maximise specificity
  subject to recall ≥ 0.90 (screening-oriented).
- The same threshold is applied unchanged to the **test set**.

## Evaluation
- **Threshold-independent**: ROC AUC, PR AUC (from probability scores).
- **Threshold-dependent**: accuracy, recall, specificity, F1, balanced accuracy.
- Cross-validation: stratified k-fold at patient level.

## Handcrafted Features (OCTA)
Vessel Density, Skeleton Line Density, FAZ Circularity, Tortuosity Index,
Vessel Perimeter, Fractal Dimension.

## Preprocessing (OCTA)
Grayscale → Gaussian Blur → CLAHE → Adaptive Threshold → Morphological
Cleanup → Skeletonize → Vessel Overlay.

## Limitations
- Small dataset (< 200 patients); results are preliminary.
- No ground-truth vessel masks → unsupervised vessel extraction only.
- U-Net segmentation not implemented (requires pixel-level annotations).
"""


def write_methodology(save_dir: str) -> None:
    """
    Write the methodology markdown file.

    Parameters
    ----------
    save_dir : str — directory where the file will be saved
    """
    path = Path(save_dir) / "advisor_methodology.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(METHODOLOGY_MD)
    print(f"[setup] Methodology summary: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Create directories and generate all reference documents."""
    parser = argparse.ArgumentParser(
        description="Set up output directories and generate reference docs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-dir", default=".",
                        help="Base directory for outputs")
    args = parser.parse_args()

    base = args.base_dir
    create_output_dirs(base)
    reports_dir = str(Path(base) / "outputs" / "reports")
    write_formulas_references(reports_dir)
    write_methodology(reports_dir)
    print("\n[setup] Project setup complete.")


if __name__ == "__main__":
    main()
