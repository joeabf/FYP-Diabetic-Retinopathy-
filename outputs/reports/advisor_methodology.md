# Methodology Summary

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
