"""
augmentation_policies.py
------------------------
Modular augmentation policy system for paired OCT/OCTA images.

CRITICAL design constraints:
- OCT and OCTA augmentations must always use the SAME random state (synchronized).
- Intensity transforms (brightness/contrast) must ONLY be applied to OCT,
  NOT to OCTA, because OCTA vessel intensity encodes physiological information.
- Augmentation must be conservative and retinal/OCTA-safe.
"""

from __future__ import annotations

import random
import warnings
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports with graceful fallbacks
# ---------------------------------------------------------------------------
try:
    from PIL import Image, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    warnings.warn("Pillow not available. Augmentation will not work.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    warnings.warn("Matplotlib not available. Visualization will be skipped.")

try:
    import torch
    import torchvision.transforms.functional as TF
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch/torchvision not available.")


# ---------------------------------------------------------------------------
# Augmentation mode enum
# ---------------------------------------------------------------------------

class AugmentationMode(Enum):
    """
    Available augmentation policies for the DR classifier.

    NONE            — no augmentation (evaluation / baseline).
    BASIC_ALL       — apply conservative augmentation to all training samples.
    BASIC_DR_ONLY   — apply conservative augmentation only to DR (label==1) samples.
    EXTENDED_DR_ONLY— stronger augmentation applied only to DR samples to
                      further counter class imbalance.
    """
    NONE = "none"
    BASIC_ALL = "basic_all"
    BASIC_DR_ONLY = "basic_dr_only"
    EXTENDED_DR_ONLY = "extended_dr_only"


# ---------------------------------------------------------------------------
# Conservative augmentation parameters
# These values were chosen to be safe for clinical retinal/OCTA images.
# ---------------------------------------------------------------------------

BASIC_PARAMS = {
    "hflip_p": 0.5,
    "rotation_degrees": 8,        # ±8° — preserves retinal anatomy
    "brightness_limit": 0.15,     # ±15% — OCT only
    "contrast_limit": 0.15,       # ±15% — OCT only
    "translate_frac": 0.05,       # max 5% translation
    "scale_range": (0.95, 1.05),  # ±5% scale
    "noise_std": 0.0,             # disabled by default in basic
}

EXTENDED_PARAMS = {
    "hflip_p": 0.5,
    "rotation_degrees": 12,       # slightly wider rotation
    "brightness_limit": 0.20,
    "contrast_limit": 0.20,
    "translate_frac": 0.08,
    "scale_range": (0.92, 1.08),
    "noise_std": 0.02,            # very mild Gaussian noise
}


# ---------------------------------------------------------------------------
# Core paired augmentation class
# ---------------------------------------------------------------------------

class PairedAugmentPolicy:
    """
    Applies the same geometric transformation to a paired (OCT, OCTA) image
    and applies intensity transforms ONLY to OCT.

    Parameters
    ----------
    mode : AugmentationMode
        The augmentation policy to use.
    seed : int, optional
        Fixed seed for reproducibility.  If None, a random seed is used each
        call (standard training behaviour).

    Usage
    -----
    policy = PairedAugmentPolicy(AugmentationMode.BASIC_ALL)
    oct_aug, octa_aug = policy(oct_pil, octa_pil, label=0)

    When ``label == 0`` and mode is ``BASIC_DR_ONLY`` or
    ``EXTENDED_DR_ONLY``, the original images are returned unchanged.
    """

    def __init__(
        self,
        mode: AugmentationMode = AugmentationMode.BASIC_ALL,
        seed: Optional[int] = None,
    ) -> None:
        self.mode = mode
        self.seed = seed
        # Select parameter set
        if mode == AugmentationMode.EXTENDED_DR_ONLY:
            self.params = EXTENDED_PARAMS
        else:
            self.params = BASIC_PARAMS

    # ------------------------------------------------------------------
    def __call__(
        self,
        oct_img,  # PIL.Image
        octa_img,  # PIL.Image
        label: int = 0,
    ) -> Tuple:
        """
        Apply augmentation to a paired (OCT, OCTA) image.

        Parameters
        ----------
        oct_img  : PIL.Image  — OCT scan
        octa_img : PIL.Image  — co-registered OCTA scan
        label    : int        — 0 = healthy, 1 = DR

        Returns
        -------
        (aug_oct, aug_octa) : Tuple[PIL.Image, PIL.Image]
        """
        if self.mode == AugmentationMode.NONE:
            return oct_img, octa_img

        if self.mode in (AugmentationMode.BASIC_DR_ONLY,
                         AugmentationMode.EXTENDED_DR_ONLY) and label != 1:
            return oct_img, octa_img

        # Fix a shared random seed so OCT and OCTA receive identical
        # geometric transforms.
        rng_seed = self.seed if self.seed is not None else random.randint(0, 2**31)
        random.seed(rng_seed)
        np.random.seed(rng_seed)

        # 1. Horizontal flip (same for both)
        if random.random() < self.params["hflip_p"]:
            oct_img = oct_img.transpose(Image.FLIP_LEFT_RIGHT)
            octa_img = octa_img.transpose(Image.FLIP_LEFT_RIGHT)

        # 2. Small rotation (same for both)
        angle = random.uniform(
            -self.params["rotation_degrees"],
            self.params["rotation_degrees"],
        )
        oct_img = oct_img.rotate(angle, resample=Image.BILINEAR, expand=False)
        octa_img = octa_img.rotate(angle, resample=Image.BILINEAR, expand=False)

        # 3. Small affine: translation + scale (same for both)
        w, h = oct_img.size
        tx = random.uniform(-self.params["translate_frac"], self.params["translate_frac"]) * w
        ty = random.uniform(-self.params["translate_frac"], self.params["translate_frac"]) * h
        scale = random.uniform(*self.params["scale_range"])

        if TORCH_AVAILABLE:
            oct_img = TF.affine(
                oct_img, angle=0, translate=[int(tx), int(ty)],
                scale=scale, shear=0,
            )
            octa_img = TF.affine(
                octa_img, angle=0, translate=[int(tx), int(ty)],
                scale=scale, shear=0,
            )

        # 4. Brightness / contrast — OCT ONLY (not OCTA)
        if self.params["brightness_limit"] > 0:
            bfactor = 1.0 + random.uniform(
                -self.params["brightness_limit"],
                self.params["brightness_limit"],
            )
            oct_img = ImageEnhance.Brightness(oct_img).enhance(bfactor)

        if self.params["contrast_limit"] > 0:
            cfactor = 1.0 + random.uniform(
                -self.params["contrast_limit"],
                self.params["contrast_limit"],
            )
            oct_img = ImageEnhance.Contrast(oct_img).enhance(cfactor)

        # 5. Very mild Gaussian noise — OCT ONLY
        if self.params["noise_std"] > 0:
            arr = np.array(oct_img).astype(np.float32)
            noise = np.random.normal(0, self.params["noise_std"] * 255, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            oct_img = Image.fromarray(arr)

        return oct_img, octa_img

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"PairedAugmentPolicy(mode={self.mode.value})"


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_augmentation_policy(mode: str, seed: Optional[int] = None) -> PairedAugmentPolicy:
    """
    Factory function that converts a string mode name to a
    ``PairedAugmentPolicy`` instance.

    Parameters
    ----------
    mode : str
        One of: ``"none"``, ``"basic_all"``, ``"basic_dr_only"``,
        ``"extended_dr_only"``.
    seed : int, optional
        Random seed for deterministic augmentation.

    Returns
    -------
    PairedAugmentPolicy

    Raises
    ------
    ValueError
        If ``mode`` is not a recognised augmentation policy.
    """
    try:
        aug_mode = AugmentationMode(mode)
    except ValueError:
        valid = [m.value for m in AugmentationMode]
        raise ValueError(
            f"Unknown augmentation mode '{mode}'. Valid choices: {valid}"
        )
    return PairedAugmentPolicy(mode=aug_mode, seed=seed)


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

def visualize_augmentation_comparison(
    oct_path: str,
    octa_path: str,
    save_dir: str,
    n_samples: int = 4,
    label: int = 1,
) -> None:
    """
    Generate before/after augmentation figures for every policy.

    Saves one PNG per policy showing ``n_samples`` augmented versions of the
    same OCT/OCTA pair alongside the original.

    Parameters
    ----------
    oct_path  : str — path to a single OCT image file.
    octa_path : str — path to the corresponding OCTA image file.
    save_dir  : str — output directory.
    n_samples : int — number of augmented variants to show per policy.
    label     : int — sample label used when testing DR-only policies.
    """
    if not PIL_AVAILABLE:
        warnings.warn("Pillow is required for visualization. Skipping.")
        return
    if not MPL_AVAILABLE:
        warnings.warn("Matplotlib is required for visualization. Skipping.")
        return

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    oct_orig = Image.open(oct_path).convert("RGB")
    octa_orig = Image.open(octa_path).convert("RGB")

    policies = {
        "none": get_augmentation_policy("none"),
        "basic_all": get_augmentation_policy("basic_all"),
        "basic_dr_only": get_augmentation_policy("basic_dr_only"),
        "extended_dr_only": get_augmentation_policy("extended_dr_only"),
    }

    for policy_name, policy in policies.items():
        # columns: original oct | original octa | n_samples * (aug oct | aug octa)
        n_cols = 2 + n_samples * 2
        fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))

        # Row 0 = OCT, Row 1 = OCTA
        axes[0, 0].imshow(oct_orig)
        axes[0, 0].set_title("OCT Original", fontsize=8)
        axes[1, 0].imshow(octa_orig)
        axes[1, 0].set_title("OCTA Original", fontsize=8)

        for col in range(1):
            axes[0, col].axis("off")
            axes[1, col].axis("off")

        for i in range(n_samples):
            aug_oct, aug_octa = policy(oct_orig.copy(), octa_orig.copy(), label=label)
            col = 2 + i * 2
            axes[0, col].imshow(aug_oct)
            axes[0, col].set_title(f"OCT Aug {i + 1}", fontsize=8)
            axes[0, col].axis("off")
            axes[1, col].imshow(aug_octa)
            axes[1, col].set_title(f"OCTA Aug {i + 1}", fontsize=8)
            axes[1, col].axis("off")

            # Blank separator column
            if col + 1 < n_cols:
                axes[0, col + 1].axis("off")
                axes[1, col + 1].axis("off")

        fig.suptitle(f"Augmentation Policy: {policy_name}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        out_file = save_path / f"augmentation_demo_{policy_name}.png"
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[augmentation] Saved: {out_file}")
