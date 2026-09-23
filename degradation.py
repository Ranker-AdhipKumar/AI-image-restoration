"""
degradation.py — Simulate common image degradation types.

All public functions accept and return uint8 RGB numpy arrays [H, W, 3].
For inpainting, a binary mask [H, W] (uint8, 255=missing) is also returned.
"""
from __future__ import annotations

import io
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from utils import get_logger

log = get_logger(__name__)


# ─── Individual degradation functions ────────────────────────────────────────
# Reviewed by Adhip Kumar

def add_gaussian_noise(img: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """Add additive white Gaussian noise (AWGN)."""
    rng = np.random.default_rng()
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img: np.ndarray, prob: float = 0.05) -> np.ndarray:
    """Add salt-and-pepper (impulse) noise."""
    noisy = img.copy()
    rng = np.random.default_rng()
    r = rng.random(img.shape[:2])
    noisy[r < prob / 2] = 0        # pepper
    noisy[r > 1 - prob / 2] = 255  # salt
    return noisy


def apply_gaussian_blur(img: np.ndarray, kernel_size: int = 15, sigma: float = 3.0) -> np.ndarray:
    """Apply Gaussian (out-of-focus) blur."""
    ks = kernel_size | 1  # force odd
    return cv2.GaussianBlur(img, (ks, ks), sigma)


def apply_motion_blur(img: np.ndarray, length: int = 25, angle: float = 45.0) -> np.ndarray:
    """Apply linear motion blur at a given angle (degrees)."""
    k = np.zeros((length, length), dtype=np.float32)
    k[length // 2, :] = 1.0 / length
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    blurred = cv2.filter2D(img, -1, k)
    return blurred


def _brush_stroke_mask(h: int, w: int, n_strokes: int = 5, max_len: int = 120, thickness: int = 30) -> np.ndarray:
    """Generate a random brush-stroke mask (foreground = 255 = missing)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    rng = np.random.default_rng()
    for _ in range(n_strokes):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        n_segs = rng.integers(3, 8)
        for _ in range(n_segs):
            angle = rng.uniform(0, 2 * np.pi)
            length = rng.integers(20, max_len)
            x1 = int(np.clip(x0 + np.cos(angle) * length, 0, w - 1))
            y1 = int(np.clip(y0 + np.sin(angle) * length, 0, h - 1))
            cv2.line(mask, (x0, y0), (x1, y1), 255, thickness=thickness)
            x0, y0 = x1, y1
    return mask


def add_random_mask(
    img: np.ndarray,
    mode: str = "mixed",
    n_patches: int = 3,
    patch_size: int = 80,
    n_strokes: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate missing image regions.

    Parameters
    ----------
    mode : "rect" | "brush" | "mixed"
    n_patches : number of rectangular patches (rect/mixed mode)
    patch_size : approximate size of each rectangular patch (px)
    n_strokes : number of brush strokes (brush/mixed mode)

    Returns
    -------
    (masked_img, mask)  — mask: 255 = missing region
    """
    h, w = img.shape[:2]
    rng = np.random.default_rng()
    mask = np.zeros((h, w), dtype=np.uint8)

    if mode in ("rect", "mixed"):
        for _ in range(n_patches):
            pw = rng.integers(patch_size // 2, patch_size + 1)
            ph = rng.integers(patch_size // 2, patch_size + 1)
            x = rng.integers(0, max(1, w - pw))
            y = rng.integers(0, max(1, h - ph))
            mask[y : y + ph, x : x + pw] = 255

    if mode in ("brush", "mixed"):
        mask = np.maximum(mask, _brush_stroke_mask(h, w, n_strokes=n_strokes))

    masked = img.copy()
    masked[mask == 255] = 0
    return masked, mask


def add_jpeg_artifacts(img: np.ndarray, quality: int = 10) -> np.ndarray:
    """Simulate JPEG compression/transmission artifacts."""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def add_mixed_degradation(img: np.ndarray) -> np.ndarray:
    """Apply a combination of noise + slight blur (simulates transmission loss)."""
    blurred = apply_gaussian_blur(img, kernel_size=5, sigma=1.5)
    noisy = add_gaussian_noise(blurred, sigma=15)
    return noisy


# ─── Unified entry point ──────────────────────────────────────────────────────
# Reviewed by Adhip Kumar

MODES = {
    "noise":        "Gaussian Noise",
    "salt_pepper":  "Salt & Pepper Noise",
    "blur":         "Gaussian Blur",
    "motion_blur":  "Motion Blur",
    "inpaint":      "Missing Regions (Inpainting)",
    "artifact":     "JPEG / Transmission Artifacts",
    "mixed":        "Mixed (Noise + Blur)",
}


def degrade(
    img: np.ndarray,
    mode: str,
    **kwargs,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply a named degradation to *img*.

    Parameters
    ----------
    img   : uint8 RGB [H, W, 3]
    mode  : one of MODES keys
    **kwargs : degradation-specific parameters

    Returns
    -------
    (degraded_img, mask_or_None)
    mask is a uint8 [H, W] binary map (255 = missing) only for 'inpaint' mode.
    """
    log.info("Applying degradation: %s (params=%s)", mode, kwargs)

    mask: Optional[np.ndarray] = None

    if mode == "noise":
        out = add_gaussian_noise(img, sigma=float(kwargs.get("sigma", 25)))

    elif mode == "salt_pepper":
        out = add_salt_pepper_noise(img, prob=float(kwargs.get("prob", 0.05)))

    elif mode == "blur":
        out = apply_gaussian_blur(
            img,
            kernel_size=int(kwargs.get("kernel_size", 15)),
            sigma=float(kwargs.get("sigma", 3.0)),
        )

    elif mode == "motion_blur":
        out = apply_motion_blur(
            img,
            length=int(kwargs.get("length", 25)),
            angle=float(kwargs.get("angle", 45)),
        )

    elif mode == "inpaint":
        out, mask = add_random_mask(
            img,
            mode=kwargs.get("mask_mode", "mixed"),
            n_patches=int(kwargs.get("n_patches", 3)),
            patch_size=int(kwargs.get("patch_size", 80)),
            n_strokes=int(kwargs.get("n_strokes", 5)),
        )

    elif mode == "artifact":
        out = add_jpeg_artifacts(img, quality=int(kwargs.get("quality", 10)))

    elif mode == "mixed":
        out = add_mixed_degradation(img)

    else:
        raise ValueError(
            f"Unknown degradation mode '{mode}'. Valid modes: {list(MODES.keys())}"
        )

    return out, mask
