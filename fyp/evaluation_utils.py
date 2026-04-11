"""
evaluation_utils.py
-------------------
Evaluation utilities for the dual-input OCT/OCTA diabetic retinopathy
classifier.

Key design principles:
- ROC AUC and PR AUC are computed from continuous probability scores,
  NOT from binary predictions.
- Threshold-dependent metrics (accuracy, F1, …) are always computed
  separately, AFTER threshold selection.
- Validation threshold is found on the validation set; it is then applied
  unchanged to the held-out test set to avoid data leakage.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# NumPy 2.0 renamed np.trapz → np.trapezoid; support both versions
_np_trapz = getattr(np, "trapezoid", None) or np.trapz

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    warnings.warn("Matplotlib not available — plots will be skipped.")

try:
    from sklearn.metrics import (
        roc_auc_score,
        roc_curve,
        precision_recall_curve,
        average_precision_score,
        confusion_matrix as sk_confusion_matrix,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available — using manual metric computation.")


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def _binary_metrics_at_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Compute threshold-dependent binary classification metrics.

    Parameters
    ----------
    labels    : 1-D array of ground-truth labels (0 / 1).
    probs     : 1-D array of predicted positive-class probabilities [0, 1].
    threshold : Decision threshold.

    Returns
    -------
    dict with keys: accuracy, precision, recall, specificity, f1,
                    balanced_accuracy, tp, tn, fp, fn
    """
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    accuracy = (tp + tn) / max(len(labels), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)       # sensitivity
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    balanced_accuracy = (recall + specificity) / 2.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def _roc_auc_manual(labels: np.ndarray, probs: np.ndarray) -> float:
    """Manual trapezoidal ROC AUC when sklearn is unavailable."""
    thresholds = np.linspace(0, 1, 101)[::-1]
    fprs, tprs = [], []
    for t in thresholds:
        m = _binary_metrics_at_threshold(labels, probs, t)
        tprs.append(m["recall"])
        fprs.append(1.0 - m["specificity"])
    # Trapezoidal integration
    fprs_arr = np.array(fprs)
    tprs_arr = np.array(tprs)
    idx = np.argsort(fprs_arr)
    return float(_np_trapz(tprs_arr[idx], fprs_arr[idx]))


def _pr_auc_manual(labels: np.ndarray, probs: np.ndarray) -> float:
    """Manual trapezoidal PR AUC when sklearn is unavailable."""
    thresholds = np.linspace(0, 1, 101)[::-1]
    precisions, recalls = [], []
    for t in thresholds:
        m = _binary_metrics_at_threshold(labels, probs, t)
        precisions.append(m["precision"])
        recalls.append(m["recall"])
    recalls_arr = np.array(recalls)
    precisions_arr = np.array(precisions)
    idx = np.argsort(recalls_arr)
    return float(_np_trapz(precisions_arr[idx], recalls_arr[idx]))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_full_metrics(
    labels_np: np.ndarray,
    probs_np: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """
    Compute a full set of binary classification metrics.

    Clearly separates threshold-INDEPENDENT metrics (ROC AUC, PR AUC)
    from threshold-DEPENDENT metrics (accuracy, recall, …).

    Parameters
    ----------
    labels_np : np.ndarray — ground-truth labels (0 / 1)
    probs_np  : np.ndarray — predicted probabilities (continuous, [0, 1])
    threshold : float      — decision threshold applied to probabilities

    Returns
    -------
    dict with all metrics plus ``threshold`` key.
    """
    # -- Threshold-independent ------------------------------------------
    if SKLEARN_AVAILABLE:
        roc_auc = float(roc_auc_score(labels_np, probs_np))
        pr_auc = float(average_precision_score(labels_np, probs_np))
    else:
        roc_auc = _roc_auc_manual(labels_np, probs_np)
        pr_auc = _pr_auc_manual(labels_np, probs_np)

    # -- Threshold-dependent --------------------------------------------
    thresh_metrics = _binary_metrics_at_threshold(labels_np, probs_np, threshold)

    return {
        "threshold": threshold,
        # Threshold-independent
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        # Threshold-dependent
        **thresh_metrics,
    }


def find_screening_threshold(
    labels_np: np.ndarray,
    probs_np: np.ndarray,
    min_recall: float = 0.90,
) -> Tuple[float, Dict]:
    """
    Find the operating threshold optimised for clinical screening:

    - Maximise specificity subject to recall >= ``min_recall``
    - Fallback: maximise balanced_accuracy + F1 if the recall constraint
      cannot be satisfied (data or model limitation).

    This function should be called on the **validation set only**.
    The returned threshold is then applied to the test set unchanged.

    Parameters
    ----------
    labels_np  : np.ndarray — ground-truth labels (0 / 1)
    probs_np   : np.ndarray — predicted probabilities
    min_recall : float      — minimum required sensitivity (default 0.90)

    Returns
    -------
    best_threshold : float
    metrics        : dict — metrics at best_threshold
    """
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_threshold = 0.5
    best_score = -1.0
    fallback_threshold = 0.5
    fallback_score = -1.0
    constraint_met = False

    for t in thresholds:
        m = _binary_metrics_at_threshold(labels_np, probs_np, t)
        # Primary: recall constraint satisfied → maximise specificity
        if m["recall"] >= min_recall:
            if not constraint_met or m["specificity"] > best_score:
                best_threshold = t
                best_score = m["specificity"]
                constraint_met = True
        # Fallback score: balanced_accuracy + F1
        fb_score = m["balanced_accuracy"] + m["f1"]
        if fb_score > fallback_score:
            fallback_threshold = t
            fallback_score = fb_score

    if not constraint_met:
        warnings.warn(
            f"Cannot achieve recall >= {min_recall:.2f} at any threshold. "
            f"Falling back to balanced_accuracy + F1 optimisation. "
            f"Results are preliminary due to small dataset.",
            stacklevel=2,
        )
        best_threshold = fallback_threshold

    metrics = compute_full_metrics(labels_np, probs_np, best_threshold)
    return best_threshold, metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_roc_curve(
    labels_np: np.ndarray,
    probs_np: np.ndarray,
    threshold: float,
    save_path: str,
    title: str = "ROC Curve",
) -> None:
    """
    Plot and save the ROC curve.

    Uses sklearn.metrics.roc_curve when available.  Saves FPR/TPR/thresholds
    arrays to a companion ``.npz`` file for reproducibility.

    Exports both PNG and PDF at 300 DPI.

    Parameters
    ----------
    labels_np  : np.ndarray — ground-truth labels
    probs_np   : np.ndarray — predicted probabilities
    threshold  : float      — operating point to mark on the curve
    save_path  : str        — base output path (extension will be replaced)
    title      : str        — figure title
    """
    if not MPL_AVAILABLE:
        warnings.warn("Matplotlib not available — skipping ROC plot.")
        return

    # Compute curve
    if SKLEARN_AVAILABLE:
        fpr, tpr, thresholds_arr = roc_curve(labels_np, probs_np)
        auc_val = roc_auc_score(labels_np, probs_np)
    else:
        ts = np.linspace(0, 1, 101)[::-1]
        fpr, tpr = [], []
        for t in ts:
            m = _binary_metrics_at_threshold(labels_np, probs_np, t)
            fpr.append(1.0 - m["specificity"])
            tpr.append(m["recall"])
        fpr = np.array(fpr)
        tpr = np.array(tpr)
        thresholds_arr = ts
        auc_val = _roc_auc_manual(labels_np, probs_np)

    # Operating point at selected threshold
    op_metrics = _binary_metrics_at_threshold(labels_np, probs_np, threshold)
    op_fpr = 1.0 - op_metrics["specificity"]
    op_tpr = op_metrics["recall"]

    base = Path(save_path).with_suffix("")
    # Save arrays for reproducibility
    np.savez(str(base) + "_roc_data.npz", fpr=fpr, tpr=tpr, thresholds=thresholds_arr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"ROC (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="No skill")
    ax.scatter([op_fpr], [op_tpr], color="red", zorder=5, s=80,
               label=f"Operating point (thr={threshold:.2f})")
    ax.set_xlabel("False Positive Rate (1 – Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    # Disclaimer for small datasets
    ax.annotate(
        "Preliminary — small dataset",
        xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=7, color="grey",
    )
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        plt.savefig(str(base) + ext, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] ROC curve saved to {base}.png / .pdf")


def plot_pr_curve(
    labels_np: np.ndarray,
    probs_np: np.ndarray,
    threshold: float,
    save_path: str,
    title: str = "Precision-Recall Curve",
) -> None:
    """
    Plot and save the Precision-Recall curve.

    Marks the no-skill baseline (class prevalence).

    Parameters
    ----------
    labels_np  : np.ndarray — ground-truth labels
    probs_np   : np.ndarray — predicted probabilities
    threshold  : float      — operating point to mark
    save_path  : str        — base output path
    title      : str        — figure title
    """
    if not MPL_AVAILABLE:
        warnings.warn("Matplotlib not available — skipping PR plot.")
        return

    prevalence = labels_np.mean()

    if SKLEARN_AVAILABLE:
        precision, recall, _ = precision_recall_curve(labels_np, probs_np)
        pr_auc = average_precision_score(labels_np, probs_np)
    else:
        ts = np.linspace(0, 1, 101)[::-1]
        precision, recall = [], []
        for t in ts:
            m = _binary_metrics_at_threshold(labels_np, probs_np, t)
            precision.append(m["precision"])
            recall.append(m["recall"])
        precision = np.array(precision)
        recall = np.array(recall)
        pr_auc = _pr_auc_manual(labels_np, probs_np)

    op_m = _binary_metrics_at_threshold(labels_np, probs_np, threshold)

    base = Path(save_path).with_suffix("")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="darkorange", lw=2,
            label=f"PR curve (AUC = {pr_auc:.3f})")
    ax.axhline(y=prevalence, color="k", linestyle="--", lw=1,
               label=f"No skill (prevalence = {prevalence:.2f})")
    ax.scatter([op_m["recall"]], [op_m["precision"]], color="red",
               zorder=5, s=80, label=f"Operating point (thr={threshold:.2f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.annotate(
        "Preliminary — small dataset",
        xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=7, color="grey",
    )
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        plt.savefig(str(base) + ext, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] PR curve saved to {base}.png / .pdf")


def plot_confusion_matrix(
    labels_np: np.ndarray,
    preds_np: np.ndarray,
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot and save a labelled confusion matrix.

    Parameters
    ----------
    labels_np : np.ndarray — ground-truth labels
    preds_np  : np.ndarray — binary predictions
    save_path : str        — base output path
    title     : str        — figure title
    """
    if not MPL_AVAILABLE:
        warnings.warn("Matplotlib not available — skipping confusion matrix plot.")
        return

    if SKLEARN_AVAILABLE:
        cm = sk_confusion_matrix(labels_np, preds_np)
    else:
        tn = int(((preds_np == 0) & (labels_np == 0)).sum())
        fp = int(((preds_np == 1) & (labels_np == 0)).sum())
        fn = int(((preds_np == 0) & (labels_np == 1)).sum())
        tp = int(((preds_np == 1) & (labels_np == 1)).sum())
        cm = np.array([[tn, fp], [fn, tp]])

    class_names = ["Healthy (0)", "DR (1)"]
    base = Path(save_path).with_suffix("")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    thresh_cm = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh_cm else "black")
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        plt.savefig(str(base) + ext, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] Confusion matrix saved to {base}.png / .pdf")


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_imbalance_report(
    all_samples,
    train_samples,
    val_samples,
    test_samples,
) -> None:
    """
    Print a class imbalance report for each data split.

    Parameters
    ----------
    all_samples   : list of Sample named-tuples (must have .label attribute)
    train_samples : list of training samples
    val_samples   : list of validation samples
    test_samples  : list of test samples
    """
    def _counts(samples):
        n_healthy = sum(1 for s in samples if s.label == 0)
        n_dr = sum(1 for s in samples if s.label == 1)
        ratio = n_healthy / max(n_dr, 1)
        return n_healthy, n_dr, ratio

    all_h, all_d, all_r = _counts(all_samples)
    tr_h, tr_d, tr_r = _counts(train_samples)
    va_h, va_d, va_r = _counts(val_samples)
    te_h, te_d, te_r = _counts(test_samples)

    print("\n" + "=" * 55)
    print("CLASS IMBALANCE REPORT")
    print("=" * 55)
    print(f"{'Split':<12} {'Healthy':>8} {'DR':>8} {'Ratio H:D':>12}")
    print("-" * 55)
    print(f"{'All':<12} {all_h:>8} {all_d:>8} {all_r:>12.2f}")
    print(f"{'Train':<12} {tr_h:>8} {tr_d:>8} {tr_r:>12.2f}")
    print(f"{'Val':<12} {va_h:>8} {va_d:>8} {va_r:>12.2f}")
    print(f"{'Test':<12} {te_h:>8} {te_d:>8} {te_r:>12.2f}")
    print("-" * 55)
    print("Strategy: WeightedRandomSampler + BCEWithLogitsLoss(pos_weight)")
    print("=" * 55 + "\n")


def export_error_cases(
    samples,
    labels_np: np.ndarray,
    preds_np: np.ndarray,
    probs_np: np.ndarray,
    save_path: str,
) -> None:
    """
    Export a CSV listing every sample with its prediction outcome
    (TP / TN / FP / FN).

    Parameters
    ----------
    samples   : list of Sample named-tuples (label, oct_path, octa_path)
    labels_np : np.ndarray — ground-truth labels
    preds_np  : np.ndarray — binary predictions
    probs_np  : np.ndarray — predicted probabilities
    save_path : str        — output CSV path
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, s in enumerate(samples):
        true_lbl = int(labels_np[i])
        pred_lbl = int(preds_np[i])
        prob = float(probs_np[i])
        if true_lbl == 1 and pred_lbl == 1:
            outcome = "TP"
        elif true_lbl == 0 and pred_lbl == 0:
            outcome = "TN"
        elif true_lbl == 0 and pred_lbl == 1:
            outcome = "FP"
        else:
            outcome = "FN"
        rows.append({
            "sample_id": i,
            "patient_id": getattr(s, "patient_id", ""),
            "oct_path": str(getattr(s, "oct_path", "")),
            "octa_path": str(getattr(s, "octa_path", "")),
            "true_label": true_lbl,
            "predicted_label": pred_lbl,
            "probability": prob,
            "outcome": outcome,
        })

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval] Error cases saved to {save_path}")
