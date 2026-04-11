"""
train_dual_input_dr.py
-----------------------
Dual-input (OCT + OCTA) binary classifier for Diabetic Retinopathy detection.

Architecture: Two ResNet-18 backbones (one per modality) whose features are
fused and passed through a classifier head.

Design decisions:
- Patient-level stratified splitting (no patient appears in >1 split)
- WeightedRandomSampler + BCEWithLogitsLoss(pos_weight) for class imbalance
- Threshold tuning on VALIDATION set only; test set is untouched until final eval
- ROC AUC and PR AUC computed from continuous probabilities, not binary preds
- Augmentation is synchronized across OCT/OCTA pairs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import warnings
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# NumPy 2.0 renamed np.trapz → np.trapezoid; support both versions
_np_trapz = getattr(np, "trapezoid", None) or np.trapz

# ---------------------------------------------------------------------------
# Optional / required imports
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Provide stub base classes so class definitions don't fail at import time
    class Dataset:  # type: ignore[no-redef]
        pass
    class nn:  # type: ignore[no-redef]
        Module = object
        Linear = object
        ReLU = object
        Dropout = object
        Sequential = object
        Identity = object
    if __name__ == "__main__":
        print("ERROR: PyTorch is required. Install with: pip install torch torchvision")
        sys.exit(1)

try:
    import torchvision.models as tv_models
    import torchvision.transforms as transforms
    TV_AVAILABLE = True
except ImportError:
    TV_AVAILABLE = False
    if __name__ == "__main__":
        print("ERROR: torchvision is required. Install with: pip install torchvision")
        sys.exit(1)

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    if __name__ == "__main__":
        print("ERROR: Pillow is required. Install with: pip install Pillow")
        sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    warnings.warn("Matplotlib not available — plots will be skipped.")

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available — using manual AUC computation.")

# Local modules
try:
    from augmentation_policies import PairedAugmentPolicy, get_augmentation_policy
    AUG_AVAILABLE = True
except ImportError:
    AUG_AVAILABLE = False
    warnings.warn("augmentation_policies.py not found — augmentation disabled.")

try:
    from evaluation_utils import (
        compute_full_metrics,
        find_screening_threshold,
        plot_roc_curve,
        plot_pr_curve,
        plot_confusion_matrix,
        print_imbalance_report,
        export_error_cases,
    )
    EVAL_UTILS_AVAILABLE = True
except ImportError:
    EVAL_UTILS_AVAILABLE = False
    warnings.warn("evaluation_utils.py not found — using built-in metrics.")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

Sample = namedtuple("Sample", ["oct_path", "octa_path", "label", "patient_id"])


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def build_paired_samples(data_root: str) -> List[Sample]:
    """
    Build a list of (OCT, OCTA, label, patient_id) named-tuples from the
    dataset directory.

    Expected directory structure::

        <data_root>/
            healthy/
                OCT/  <patient_id>.png
                OCTA/ <patient_id>.png
            dr/
                OCT/  <patient_id>.png
                OCTA/ <patient_id>.png

    The patient_id is derived from the filename stem.

    Parameters
    ----------
    data_root : str — path to the dataset root

    Returns
    -------
    list of Sample named-tuples
    """
    root = Path(data_root)
    # Support alternate DR folder names
    label_dirs = []
    for cls, lbl in [("healthy", 0), ("dr", 1),
                     ("diabetic_retinopathy", 1), ("DR", 1)]:
        cdir = root / cls
        if cdir.exists():
            label_dirs.append((cdir, lbl))

    samples: List[Sample] = []
    for class_dir, label in label_dirs:
        oct_dir = class_dir / "OCT"
        octa_dir = class_dir / "OCTA"
        if not oct_dir.exists() or not octa_dir.exists():
            raise FileNotFoundError(
                f"Expected folders '{oct_dir}' and '{octa_dir}' to exist. "
                f"Please check your --data-root path."
            )
        oct_files = {p.stem: p for p in oct_dir.glob("*")
                     if p.suffix.lower() in (".png", ".jpg", ".tif", ".bmp")}
        for stem, oct_path in sorted(oct_files.items()):
            octa_path = None
            for ext in (".png", ".jpg", ".tif", ".bmp"):
                candidate = octa_dir / (stem + ext)
                if candidate.exists():
                    octa_path = candidate
                    break
            if octa_path is None:
                continue
            samples.append(Sample(oct_path, octa_path, label, stem))
        cls_name = class_dir.name
        print(f"[{cls_name}] paired samples: {len([s for s in samples if s.label == label])} "
              f"across {len(set(s.patient_id for s in samples if s.label == label))} patients")

    return samples


def stratified_split(
    samples: List[Sample],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """
    Patient-level stratified train/val/test split.

    Guarantees no patient appears in more than one split.

    Parameters
    ----------
    samples    : list of Sample named-tuples
    val_ratio  : fraction for validation
    test_ratio : fraction for test
    seed       : random seed

    Returns
    -------
    (train_samples, val_samples, test_samples)
    """
    rng = random.Random(seed)

    # Group patient IDs by label
    by_label: Dict[int, List[str]] = defaultdict(list)
    for s in samples:
        by_label[s.label].append(s.patient_id)

    train_ids, val_ids, test_ids = set(), set(), set()
    for label, pids in by_label.items():
        unique = list(dict.fromkeys(pids))  # preserve order, deduplicate
        rng.shuffle(unique)
        n = len(unique)
        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        n_train = n - n_val - n_test

        test_ids.update(unique[:n_test])
        val_ids.update(unique[n_test: n_test + n_val])
        train_ids.update(unique[n_test + n_val:])

        print(f"label={label} patients={n} train={n_train} val={n_val} test={n_test}")

    train_samples = [s for s in samples if s.patient_id in train_ids]
    val_samples = [s for s in samples if s.patient_id in val_ids]
    test_samples = [s for s in samples if s.patient_id in test_ids]

    return train_samples, val_samples, test_samples


def print_class_imbalance_report(
    all_samples: List[Sample],
    train_samples: List[Sample],
    val_samples: List[Sample],
    test_samples: List[Sample],
) -> None:
    """Print a class imbalance report for each data split."""
    if EVAL_UTILS_AVAILABLE:
        print_imbalance_report(all_samples, train_samples, val_samples, test_samples)
        return

    def _counts(ss):
        n0 = sum(1 for s in ss if s.label == 0)
        n1 = sum(1 for s in ss if s.label == 1)
        return n0, n1, n0 / max(n1, 1)

    print("\n" + "=" * 55)
    print("CLASS IMBALANCE REPORT")
    print("=" * 55)
    print(f"{'Split':<12} {'Healthy':>8} {'DR':>8} {'Ratio H:D':>12}")
    print("-" * 55)
    for name, ss in [("All", all_samples), ("Train", train_samples),
                     ("Val", val_samples), ("Test", test_samples)]:
        h, d, r = _counts(ss)
        print(f"{name:<12} {h:>8} {d:>8} {r:>12.2f}")
    print("-" * 55)
    print("Strategy: WeightedRandomSampler + BCEWithLogitsLoss(pos_weight)")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Dataset class with augmentation support
# ---------------------------------------------------------------------------

# Base transforms (resize + to-tensor + normalise)
_IMG_MEAN = [0.485, 0.456, 0.406]
_IMG_STD = [0.229, 0.224, 0.225]
_IMG_SIZE = 224

if TV_AVAILABLE:
    _BASE_TRANSFORMS = transforms.Compose([
        transforms.Resize((_IMG_SIZE, _IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMG_MEAN, std=_IMG_STD),
    ])
else:
    _BASE_TRANSFORMS = None


class OCTOCTADataset(Dataset):
    """
    Dataset for paired OCT + OCTA images with optional augmentation.

    Parameters
    ----------
    samples    : list of Sample named-tuples
    augment    : bool — simple flag for backward compatibility
    """

    def __init__(self, samples: List[Sample], augment: bool = False) -> None:
        self.samples = samples
        self.augment = augment
        if augment:
            self._aug_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
            ])
        else:
            self._aug_transforms = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        s = self.samples[idx]
        oct_img = Image.open(s.oct_path).convert("RGB")
        octa_img = Image.open(s.octa_path).convert("RGB")

        if self.augment and self._aug_transforms is not None:
            seed = random.randint(0, 2 ** 31)
            random.seed(seed)
            torch.manual_seed(seed)
            oct_img = self._aug_transforms(oct_img)
            random.seed(seed)
            torch.manual_seed(seed)
            octa_img = self._aug_transforms(octa_img)

        oct_tensor = _BASE_TRANSFORMS(oct_img)
        octa_tensor = _BASE_TRANSFORMS(octa_img)
        return oct_tensor, octa_tensor, s.label


class OCTOCTADatasetV2(Dataset):
    """
    Dataset for paired OCT + OCTA images using the new PairedAugmentPolicy.

    This version passes the sample label to the augmentation policy so that
    DR-only policies can apply augmentation conditionally.

    Parameters
    ----------
    samples : list of Sample named-tuples
    policy  : PairedAugmentPolicy or None
    """

    def __init__(
        self,
        samples: List[Sample],
        policy: Optional["PairedAugmentPolicy"] = None,
    ) -> None:
        self.samples = samples
        self.policy = policy

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        s = self.samples[idx]
        oct_img = Image.open(s.oct_path).convert("RGB")
        octa_img = Image.open(s.octa_path).convert("RGB")

        if self.policy is not None:
            oct_img, octa_img = self.policy(oct_img, octa_img, label=s.label)

        oct_tensor = _BASE_TRANSFORMS(oct_img)
        octa_tensor = _BASE_TRANSFORMS(octa_img)
        return oct_tensor, octa_tensor, s.label


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the model's estimated probability for the correct class.

    gamma  : focusing parameter (default 2.0)
    alpha  : weight for the positive class (equivalent to pos_weight in BCE)

    Medical context: focal loss down-weights easy (well-classified) examples
    and focuses learning on hard misclassifications, which is beneficial when
    DR samples are scarce.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (N,) raw output logits from the model
        targets : (N,) binary targets {0, 1}
        """
        probs = torch.sigmoid(logits)
        # p_t = prob if target==1, else (1 - prob)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        # alpha_t = alpha for positive class, 1 for negative class
        alpha_t = self.alpha * targets + (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = -alpha_t * focal_weight * torch.log(p_t + 1e-8)
        return loss.mean()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DualResNet(nn.Module):
    """
    Dual-stream ResNet-18 for paired OCT + OCTA classification.

    Two independent ResNet-18 backbones process OCT and OCTA separately.
    Their feature vectors are concatenated and passed through a classifier head.

    Parameters
    ----------
    freeze_backbone : bool — if True, backbone weights are initially frozen
                             (useful for a warm-up phase before fine-tuning)
    """

    def __init__(self, freeze_backbone: bool = True) -> None:
        super().__init__()

        # ResNet-18 backbone for OCT
        backbone_oct = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        feat_dim = backbone_oct.fc.in_features  # 512
        backbone_oct.fc = nn.Identity()  # remove classification head
        self.backbone_oct = backbone_oct

        # ResNet-18 backbone for OCTA
        backbone_octa = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        backbone_octa.fc = nn.Identity()
        self.backbone_octa = backbone_octa

        # Fusion classifier head: concat both 512-d vectors → 1024 → 1
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),  # raw logit (no sigmoid — use BCEWithLogitsLoss)
        )

        if freeze_backbone:
            for param in self.backbone_oct.parameters():
                param.requires_grad = False
            for param in self.backbone_octa.parameters():
                param.requires_grad = False

    def unfreeze_backbones(self) -> None:
        """Unfreeze backbone parameters for phase-2 fine-tuning."""
        for param in self.backbone_oct.parameters():
            param.requires_grad = True
        for param in self.backbone_octa.parameters():
            param.requires_grad = True

    def forward(self, oct_x: torch.Tensor, octa_x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        oct_x  : (N, 3, H, W) — OCT batch
        octa_x : (N, 3, H, W) — OCTA batch

        Returns
        -------
        logits : (N,) — raw logit per sample
        """
        feat_oct = self.backbone_oct(oct_x)   # (N, 512)
        feat_octa = self.backbone_octa(octa_x)  # (N, 512)
        fused = torch.cat([feat_oct, feat_octa], dim=1)  # (N, 1024)
        return self.classifier(fused).squeeze(1)  # (N,)


# ---------------------------------------------------------------------------
# Metric helpers (fallback when evaluation_utils not available)
# ---------------------------------------------------------------------------

def _binary_metrics(labels, probs, threshold=0.5):
    """Compute basic metrics at a given threshold."""
    preds = (np.array(probs) >= threshold).astype(int)
    labels = np.array(labels)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    recall = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    bacc = (recall + spec) / 2.0
    return {"recall": recall, "specificity": spec, "precision": prec,
            "f1": f1, "balanced_accuracy": bacc,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _roc_auc(labels, probs):
    """Compute ROC AUC."""
    if SKLEARN_AVAILABLE:
        return float(roc_auc_score(labels, probs))
    ts = np.linspace(0, 1, 101)[::-1]
    fprs, tprs = [], []
    for t in ts:
        m = _binary_metrics(labels, probs, t)
        tprs.append(m["recall"])
        fprs.append(1.0 - m["specificity"])
    fprs_a, tprs_a = np.array(fprs), np.array(tprs)
    idx = np.argsort(fprs_a)
    return float(_np_trapz(tprs_a[idx], fprs_a[idx]))


def _pr_auc(labels, probs):
    """Compute PR AUC."""
    if SKLEARN_AVAILABLE:
        return float(average_precision_score(labels, probs))
    ts = np.linspace(0, 1, 101)[::-1]
    precs, recs = [], []
    for t in ts:
        m = _binary_metrics(labels, probs, t)
        precs.append(m["precision"])
        recs.append(m["recall"])
    recs_a, precs_a = np.array(recs), np.array(precs)
    idx = np.argsort(recs_a)
    return float(_np_trapz(precs_a[idx], recs_a[idx]))


def _find_best_threshold(labels, probs, min_recall=0.90):
    """Find threshold maximising specificity subject to recall >= min_recall."""
    if EVAL_UTILS_AVAILABLE:
        t, _ = find_screening_threshold(np.array(labels), np.array(probs), min_recall)
        return t
    best_t, best_spec = 0.5, -1.0
    for t in np.arange(0.01, 1.00, 0.01):
        m = _binary_metrics(labels, probs, t)
        if m["recall"] >= min_recall and m["specificity"] > best_spec:
            best_t, best_spec = t, m["specificity"]
    return best_t


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer: Optional[optim.Optimizer],
    device: torch.device,
    train: bool = True,
) -> Tuple[float, List[int], List[float]]:
    """
    Run one training or evaluation epoch.

    Parameters
    ----------
    model     : nn.Module
    loader    : DataLoader
    criterion : loss function
    optimizer : optimizer (None for evaluation)
    device    : torch.device
    train     : bool — if True, runs backward pass

    Returns
    -------
    (avg_loss, all_labels, all_probs)
    """
    model.train(train)
    total_loss = 0.0
    all_labels: List[int] = []
    all_probs: List[float] = []

    with torch.set_grad_enabled(train):
        for oct_x, octa_x, labels in loader:
            oct_x = oct_x.to(device)
            octa_x = octa_x.to(device)
            labels_float = labels.float().to(device)

            logits = model(oct_x, octa_x)
            loss = criterion(logits, labels_float)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.tolist())

    avg_loss = total_loss / max(len(all_labels), 1)
    return avg_loss, all_labels, all_probs


# ---------------------------------------------------------------------------
# Plot saving
# ---------------------------------------------------------------------------

def save_plots(
    train_losses: List[float],
    val_metrics_history: List[Dict],
    save_dir: str,
    prefix: str = "",
) -> None:
    """
    Save training curves and validation metric plots.

    Parameters
    ----------
    train_losses         : list of per-epoch training losses
    val_metrics_history  : list of metric dicts per epoch
    save_dir             : output directory
    prefix               : filename prefix
    """
    if not MPL_AVAILABLE:
        return
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(train_losses) + 1))

    # Loss curve
    fig, ax = plt.subplots()
    ax.plot(epochs, train_losses, label="Train loss")
    val_losses = [m.get("val_loss", float("nan")) for m in val_metrics_history]
    if any(not math.isnan(v) for v in val_losses):
        ax.plot(epochs, val_losses, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path / f"{prefix}loss_curve.png", dpi=150)
    plt.close(fig)

    # Validation metrics
    metric_keys = ["f1", "recall", "specificity", "balanced_accuracy"]
    fig, ax = plt.subplots()
    for key in metric_keys:
        vals = [m.get(key, float("nan")) for m in val_metrics_history]
        ax.plot(epochs, vals, label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.set_title("Validation Metrics")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path / f"{prefix}val_metrics.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def stratified_kfold_patients(
    samples: List[Sample],
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[List[Sample], List[Sample]]]:
    """
    Generate stratified K-fold splits at the patient level.

    Guarantees no patient leaks across folds.

    Parameters
    ----------
    samples : list of Sample
    n_folds : int
    seed    : int

    Returns
    -------
    list of (train_samples, val_samples) tuples
    """
    rng = random.Random(seed)
    by_label: Dict[int, List[str]] = defaultdict(list)
    for s in samples:
        by_label[s.label].append(s.patient_id)

    # Deduplicate and shuffle patient IDs per class
    class_folds: Dict[int, List[List[str]]] = {}
    for label, pids in by_label.items():
        unique = list(dict.fromkeys(pids))
        rng.shuffle(unique)
        n = len(unique)
        fold_size = n // n_folds
        folds = [unique[i * fold_size: (i + 1) * fold_size] for i in range(n_folds)]
        # Distribute remainder
        for i, pid in enumerate(unique[n_folds * fold_size:]):
            folds[i].append(pid)
        class_folds[label] = folds

    splits = []
    for fold_idx in range(n_folds):
        val_ids: set = set()
        train_ids: set = set()
        for label, folds in class_folds.items():
            val_ids.update(folds[fold_idx])
            for i, f in enumerate(folds):
                if i != fold_idx:
                    train_ids.update(f)
        train_s = [s for s in samples if s.patient_id in train_ids]
        val_s = [s for s in samples if s.patient_id in val_ids]
        splits.append((train_s, val_s))

    return splits


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_model(args: argparse.Namespace) -> Dict:
    """
    Full training pipeline: load data → train → evaluate → save results.

    Parameters
    ----------
    args : argparse.Namespace — parsed CLI arguments

    Returns
    -------
    dict with final test metrics
    """
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Build data ----------------------------------------------------
    all_samples = build_paired_samples(args.data_root)
    train_samples, val_samples, test_samples = stratified_split(
        all_samples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print_class_imbalance_report(all_samples, train_samples, val_samples, test_samples)

    # --- Augmentation policy ------------------------------------------
    aug_mode = args.augmentation_mode
    if args.disable_augmentation:
        aug_mode = "none"
    print(f"Using OCTOCTADatasetV2 with augmentation_mode='{aug_mode}'")

    if AUG_AVAILABLE:
        train_policy = get_augmentation_policy(aug_mode, seed=None)
        val_policy = None  # no augmentation at eval time
    else:
        train_policy = None
        val_policy = None

    train_ds = OCTOCTADatasetV2(train_samples, policy=train_policy)
    val_ds = OCTOCTADatasetV2(val_samples, policy=val_policy)
    test_ds = OCTOCTADatasetV2(test_samples, policy=None)

    # --- Weighted sampler ---------------------------------------------
    class_counts = [
        sum(1 for s in train_samples if s.label == c) for c in [0, 1]
    ]
    sample_weights = [
        1.0 / max(class_counts[s.label], 1) for s in train_samples
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    # --- Model and loss -----------------------------------------------
    model = DualResNet(freeze_backbone=True).to(device)

    n_healthy = class_counts[0]
    n_dr = class_counts[1]
    pos_weight_val = n_healthy / max(n_dr, 1)
    print(f"BCEWithLogitsLoss pos_weight = {pos_weight_val:.4f}")
    pos_weight = torch.tensor([pos_weight_val], device=device)

    if args.loss_fn == "focal":
        criterion = BinaryFocalLoss(gamma=2.0, alpha=pos_weight_val)
        print("Using BinaryFocalLoss")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("Using BCEWithLogitsLoss")

    # Phase 1 optimizer: only classifier head
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=False,
    )

    # --- Training loop ------------------------------------------------
    best_val_bacc = -1.0
    best_val_probs: List[float] = []
    best_val_labels: List[int] = []
    best_epoch = 0
    patience_count = 0

    train_losses: List[float] = []
    val_metrics_history: List[Dict] = []
    best_state: Optional[Dict] = None

    for epoch in range(1, args.epochs + 1):
        # Phase 2: unfreeze backbones after warm-up
        if epoch == args.unfreeze_epoch:
            model.unfreeze_backbones()
            optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1)
            print(f"Epoch {epoch:02d} | Backbones unfrozen (phase-2 fine-tuning)")

        # Train
        train_loss, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        train_losses.append(train_loss)

        # Validate
        _, val_labels, val_probs = run_epoch(
            model, val_loader, criterion, None, device, train=False
        )
        val_m = _binary_metrics(val_labels, val_probs, threshold=0.5)
        val_auc = _roc_auc(val_labels, val_probs)
        val_pr_auc = _pr_auc(val_labels, val_probs)
        val_m["roc_auc"] = val_auc
        val_m["pr_auc"] = val_pr_auc
        val_m["val_loss"] = train_loss
        val_metrics_history.append(val_m)

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} "
            f"| val_F1={val_m['f1']:.3f} | val_Recall={val_m['recall']:.3f} "
            f"| val_Spec={val_m['specificity']:.3f} "
            f"| val_BAcc={val_m['balanced_accuracy']:.3f}"
        )

        scheduler.step(val_m["balanced_accuracy"])

        # Save best model
        if val_m["balanced_accuracy"] > best_val_bacc:
            best_val_bacc = val_m["balanced_accuracy"]
            best_val_probs = list(val_probs)
            best_val_labels = list(val_labels)
            best_epoch = epoch
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1

        if patience_count >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # --- Threshold tuning on VALIDATION SET ONLY ----------------------
    best_threshold = _find_best_threshold(
        best_val_labels, best_val_probs, args.min_recall
    )
    print(f"\nBest threshold (from val set, min_recall={args.min_recall:.2f}): "
          f"{best_threshold:.3f}")

    # --- Final test evaluation ----------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_labels, test_probs = run_epoch(
        model, test_loader, criterion, None, device, train=False
    )
    test_preds = (np.array(test_probs) >= best_threshold).astype(int)
    test_m = _binary_metrics(test_labels, test_probs, best_threshold)
    test_m["roc_auc"] = _roc_auc(test_labels, test_probs)
    test_m["pr_auc"] = _pr_auc(test_labels, test_probs)
    test_m["threshold"] = best_threshold
    test_m["best_val_epoch"] = best_epoch

    print("\n=== Test Results ===")
    for k, v in test_m.items():
        if isinstance(v, float):
            print(f"  {k:<25} {v:.4f}")
        else:
            print(f"  {k:<25} {v}")

    # --- Save outputs -------------------------------------------------
    out_dir = Path(args.output_dir)
    plots_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    save_plots(train_losses, val_metrics_history, str(plots_dir))

    if EVAL_UTILS_AVAILABLE:
        plot_roc_curve(
            np.array(test_labels), np.array(test_probs),
            best_threshold, str(plots_dir / "roc_curve.png"),
            title="Test ROC Curve",
        )
        plot_pr_curve(
            np.array(test_labels), np.array(test_probs),
            best_threshold, str(plots_dir / "pr_curve.png"),
            title="Test PR Curve",
        )
        plot_confusion_matrix(
            np.array(test_labels), test_preds,
            str(plots_dir / "confusion_matrix.png"),
            title="Test Confusion Matrix",
        )
        export_error_cases(
            test_samples, np.array(test_labels), test_preds,
            np.array(test_probs), str(tables_dir / "error_cases.csv"),
        )

    # Save JSON metrics
    metrics_out = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                   for k, v in test_m.items()}
    json_path = tables_dir / "test_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nTest metrics saved to: {json_path}")

    # Save model checkpoint
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, ckpt_dir / "best_model.pt")

    return test_m


# ---------------------------------------------------------------------------
# Cross-validation entry point
# ---------------------------------------------------------------------------

def run_cross_validation(args: argparse.Namespace) -> None:
    """
    Run k-fold cross-validation at the patient level.

    For each fold: train → val threshold tuning → val evaluation.
    Reports fold-by-fold metrics and mean ± std.

    Parameters
    ----------
    args : argparse.Namespace
    """
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_samples = build_paired_samples(args.data_root)
    folds = stratified_kfold_patients(all_samples, n_folds=args.cv_folds, seed=args.seed)

    aug_mode = "none" if args.disable_augmentation else args.augmentation_mode
    fold_results = []

    for fold_idx, (train_s, val_s) in enumerate(folds):
        print(f"\n{'=' * 50}")
        print(f"Cross-validation fold {fold_idx + 1}/{args.cv_folds}")
        print(f"  train={len(train_s)}, val={len(val_s)}")

        if AUG_AVAILABLE:
            train_policy = get_augmentation_policy(aug_mode)
        else:
            train_policy = None

        train_ds = OCTOCTADatasetV2(train_s, policy=train_policy)
        val_ds = OCTOCTADatasetV2(val_s, policy=None)

        class_counts = [sum(1 for s in train_s if s.label == c) for c in [0, 1]]
        sample_weights = [1.0 / max(class_counts[s.label], 1) for s in train_s]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), True)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers)

        model = DualResNet(freeze_backbone=True).to(device)
        pos_w = class_counts[0] / max(class_counts[1], 1)
        if args.loss_fn == "focal":
            criterion = BinaryFocalLoss(gamma=2.0, alpha=pos_w)
        else:
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_w], device=device)
            )
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

        best_bacc, best_probs, best_labels, patience_count = -1.0, [], [], 0
        best_state = None
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
        fm = _binary_metrics(best_labels, best_probs, thr)
        fm["roc_auc"] = _roc_auc(best_labels, best_probs)
        fm["pr_auc"] = _pr_auc(best_labels, best_probs)
        fm["threshold"] = thr
        fm["fold"] = fold_idx + 1
        fold_results.append(fm)
        print(f"  Fold {fold_idx + 1}: F1={fm['f1']:.3f} Recall={fm['recall']:.3f} "
              f"AUC={fm['roc_auc']:.3f} thr={thr:.3f}")

    # Aggregate
    metric_keys = ["recall", "specificity", "f1", "balanced_accuracy",
                   "roc_auc", "pr_auc", "precision"]
    print(f"\n{'=' * 50}")
    print(f"Cross-Validation Summary ({args.cv_folds} folds)")
    print(f"{'Metric':<25} {'Mean':>8} {'± Std':>8}")
    print("-" * 45)
    for key in metric_keys:
        vals = [r[key] for r in fold_results if key in r]
        if vals:
            print(f"  {key:<23} {np.mean(vals):>8.4f} ± {np.std(vals):.4f}")

    # Save to CSV
    out_dir = Path(args.output_dir)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / "cv_fold_results.csv"
    with open(csv_path, "w", newline="") as f:
        if fold_results:
            writer = csv.DictWriter(f, fieldnames=fold_results[0].keys())
            writer.writeheader()
            writer.writerows(fold_results)
    print(f"\nCV fold results saved to: {csv_path}")

    # Bar chart
    if MPL_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 5))
        means = [np.mean([r[k] for r in fold_results if k in r]) for k in metric_keys]
        stds = [np.std([r[k] for r in fold_results if k in r]) for k in metric_keys]
        x = np.arange(len(metric_keys))
        ax.bar(x, means, yerr=stds, capsize=4, color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_keys, rotation=30, ha="right")
        ax.set_ylabel("Score")
        ax.set_title(f"Cross-Validation Summary ({args.cv_folds} folds)")
        ax.set_ylim([0, 1.1])
        plt.tight_layout()
        fig_path = out_dir / "figures" / "cv_summary_barplot.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"CV bar chart saved to: {fig_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for training."""
    p = argparse.ArgumentParser(
        description="Dual-Input OCT/OCTA Diabetic Retinopathy Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--data-root", default="fyp/Final_Dataset",
                   help="Path to dataset root directory")
    p.add_argument("--output-dir", default="outputs",
                   help="Directory for all saved outputs")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=7,
                   help="Early stopping patience (epochs)")
    p.add_argument("--unfreeze-epoch", type=int, default=5,
                   help="Epoch at which to unfreeze backbone weights")

    # Augmentation
    p.add_argument(
        "--augmentation-mode",
        default="basic_all",
        choices=["none", "basic_all", "basic_dr_only", "extended_dr_only"],
        help="Augmentation policy to use for training",
    )
    p.add_argument(
        "--disable-augmentation",
        action="store_true",
        help="[Backward compat] Equivalent to --augmentation-mode none.",
    )

    # Loss
    p.add_argument("--loss-fn", default="bce", choices=["bce", "focal"],
                   help="Loss function: bce or focal")

    # Evaluation
    p.add_argument("--min-recall", type=float, default=0.90,
                   help="Minimum recall for threshold selection (screening mode)")

    # Cross-validation
    p.add_argument("--cv", action="store_true", help="Run cross-validation")
    p.add_argument("--cv-folds", type=int, default=5)

    return p


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.cv:
        run_cross_validation(args)
    else:
        train_model(args)


if __name__ == "__main__":
    main()
