"""
run_augmentation_ablation.py
-----------------------------
Augmentation ablation study runner.

Runs training under 3 augmentation conditions on the SAME data split
with the SAME random seed:
    1. none        — no augmentation
    2. basic_all   — conservative augmentation on all training samples
    3. basic_dr_only — conservative augmentation on DR samples only

For each condition, reports:
    accuracy, precision, recall, specificity, F1, balanced accuracy,
    ROC AUC, PR AUC, confusion matrix, selected threshold

Outputs:
    outputs/tables/augmentation_ablation_results.csv
    outputs/tables/augmentation_ablation_results.json
    outputs/figures/augmentation_ablation_comparison.png
    outputs/figures/augmentation_demo_*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, WeightedRandomSampler
except ImportError:
    print("ERROR: PyTorch is required.")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

# Local modules — must be importable from the same directory
try:
    from augmentation_policies import (
        PairedAugmentPolicy,
        get_augmentation_policy,
        visualize_augmentation_comparison,
        AugmentationMode,
    )
    AUG_AVAILABLE = True
except ImportError:
    AUG_AVAILABLE = False
    warnings.warn("augmentation_policies.py not found.")

try:
    from train_dual_input_dr import (
        build_paired_samples,
        stratified_split,
        OCTOCTADatasetV2,
        DualResNet,
        BinaryFocalLoss,
        run_epoch,
        set_seed,
        _binary_metrics,
        _roc_auc,
        _pr_auc,
        _find_best_threshold,
    )
    TRAIN_AVAILABLE = True
except ImportError as e:
    TRAIN_AVAILABLE = False
    warnings.warn(f"train_dual_input_dr.py import failed: {e}")

try:
    from evaluation_utils import compute_full_metrics
    EVAL_UTILS_AVAILABLE = True
except ImportError:
    EVAL_UTILS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Ablation conditions
# ---------------------------------------------------------------------------

ABLATION_CONDITIONS = ["none", "basic_all", "basic_dr_only"]


# ---------------------------------------------------------------------------
# Per-condition training run
# ---------------------------------------------------------------------------

def _run_single_condition(
    train_samples,
    val_samples,
    augmentation_mode: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict:
    """
    Train and evaluate a single ablation condition.

    Uses the same data split and seed for every condition to ensure
    fair comparison.

    Parameters
    ----------
    train_samples    : list of Sample named-tuples
    val_samples      : list of Sample named-tuples
    augmentation_mode: str — one of ABLATION_CONDITIONS
    args             : parsed CLI arguments
    device           : torch.device

    Returns
    -------
    dict with all metrics for this condition
    """
    set_seed(args.seed)

    # Dataset
    if AUG_AVAILABLE:
        policy = get_augmentation_policy(augmentation_mode)
    else:
        policy = None

    train_ds = OCTOCTADatasetV2(train_samples, policy=policy)
    val_ds = OCTOCTADatasetV2(val_samples, policy=None)  # no aug at eval

    # Weighted sampler
    class_counts = [sum(1 for s in train_samples if s.label == c) for c in [0, 1]]
    sample_weights = [1.0 / max(class_counts[s.label], 1) for s in train_samples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    # Model
    model = DualResNet(freeze_backbone=True).to(device)
    pos_w = class_counts[0] / max(class_counts[1], 1)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w], device=device)
    )
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    best_bacc = -1.0
    best_probs: List[float] = []
    best_labels: List[int] = []
    best_state = None
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        if epoch == args.unfreeze_epoch:
            model.unfreeze_backbones()
            optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1)
        run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        _, v_lbl, v_prb = run_epoch(
            model, val_loader, criterion, None, device, train=False
        )
        vm = _binary_metrics(v_lbl, v_prb)
        if vm["balanced_accuracy"] > best_bacc:
            best_bacc = vm["balanced_accuracy"]
            best_probs, best_labels = list(v_prb), list(v_lbl)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= args.patience:
            break

    # Threshold from val set
    thr = _find_best_threshold(best_labels, best_probs, args.min_recall)
    metrics = _binary_metrics(best_labels, best_probs, thr)
    metrics["roc_auc"] = _roc_auc(best_labels, best_probs)
    metrics["pr_auc"] = _pr_auc(best_labels, best_probs)
    metrics["threshold"] = thr
    metrics["augmentation_mode"] = augmentation_mode

    return metrics


# ---------------------------------------------------------------------------
# Results export
# ---------------------------------------------------------------------------

def _save_results(results: List[Dict], out_dir: Path) -> None:
    """Save ablation results to CSV and JSON."""
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = tables_dir / "augmentation_ablation_results.csv"
    with open(csv_path, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"[ablation] CSV saved: {csv_path}")

    # JSON
    json_path = tables_dir / "augmentation_ablation_results.json"
    clean_results = []
    for r in results:
        clean_results.append(
            {k: (float(v) if isinstance(v, (float, np.floating)) else v)
             for k, v in r.items()}
        )
    with open(json_path, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"[ablation] JSON saved: {json_path}")


# ---------------------------------------------------------------------------
# Comparison bar chart
# ---------------------------------------------------------------------------

def _plot_comparison(results: List[Dict], out_dir: Path) -> None:
    """Generate side-by-side bar chart comparing augmentation conditions."""
    if not MPL_AVAILABLE:
        return

    metric_keys = ["recall", "specificity", "f1", "balanced_accuracy",
                   "roc_auc", "pr_auc"]
    conditions = [r["augmentation_mode"] for r in results]
    n_metrics = len(metric_keys)
    n_conditions = len(conditions)

    x = np.arange(n_metrics)
    width = 0.25
    colors = ["steelblue", "darkorange", "forestgreen", "red"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, r in enumerate(results):
        vals = [r.get(k, 0.0) for k in metric_keys]
        ax.bar(x + i * width, vals, width, label=r["augmentation_mode"],
               color=colors[i % len(colors)], alpha=0.85)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Augmentation Ablation Study: Comparison")
    ax.set_xticks(x + width * (n_conditions - 1) / 2)
    ax.set_xticklabels(metric_keys, rotation=20, ha="right")
    ax.set_ylim([0, 1.15])
    ax.legend()
    plt.tight_layout()

    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    out_file = figs_dir / "augmentation_ablation_comparison.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ablation] Comparison chart saved: {out_file}")


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_ablation(args: argparse.Namespace) -> None:
    """
    Main ablation study: trains under each condition and reports results.

    Parameters
    ----------
    args : argparse.Namespace — parsed CLI arguments
    """
    if not TRAIN_AVAILABLE:
        print("ERROR: train_dual_input_dr.py must be importable.")
        return

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.output_dir)

    # Build SAME split for all conditions
    all_samples = build_paired_samples(args.data_root)
    train_samples, val_samples, test_samples = stratified_split(
        all_samples, seed=args.seed
    )
    print(f"Split — train: {len(train_samples)}, val: {len(val_samples)}, "
          f"test: {len(test_samples)}")

    # Optional demo figures
    if AUG_AVAILABLE and train_samples:
        first_dr = next((s for s in train_samples if s.label == 1), None)
        if first_dr:
            visualize_augmentation_comparison(
                str(first_dr.oct_path),
                str(first_dr.octa_path),
                save_dir=str(out_dir / "figures"),
                n_samples=3,
                label=1,
            )

    # Run each condition
    results = []
    for mode in ABLATION_CONDITIONS:
        print(f"\n{'=' * 50}")
        print(f"Running ablation condition: {mode}")
        try:
            metrics = _run_single_condition(
                train_samples, val_samples, mode, args, device
            )
            results.append(metrics)
            print(
                f"  F1={metrics['f1']:.3f} Recall={metrics['recall']:.3f} "
                f"AUC={metrics['roc_auc']:.3f} thr={metrics['threshold']:.3f}"
            )
        except Exception as e:
            warnings.warn(f"Ablation condition '{mode}' failed: {e}")

    # Save results
    _save_results(results, out_dir)
    _plot_comparison(results, out_dir)

    # Print summary table
    print(f"\n{'=' * 60}")
    print("AUGMENTATION ABLATION SUMMARY")
    print(f"{'Mode':<20} {'Recall':>8} {'Spec':>8} {'F1':>8} "
          f"{'BAcc':>8} {'AUC':>8} {'PRAUC':>8}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['augmentation_mode']:<20} "
            f"{r.get('recall', 0.0):>8.4f} "
            f"{r.get('specificity', 0.0):>8.4f} "
            f"{r.get('f1', 0.0):>8.4f} "
            f"{r.get('balanced_accuracy', 0.0):>8.4f} "
            f"{r.get('roc_auc', 0.0):>8.4f} "
            f"{r.get('pr_auc', 0.0):>8.4f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Augmentation Ablation Study for DR Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", default="fyp/Final_Dataset")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--unfreeze-epoch", type=int, default=5)
    p.add_argument("--min-recall", type=float, default=0.90)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
