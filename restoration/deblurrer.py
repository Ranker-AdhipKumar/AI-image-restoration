"""
restoration/deblurrer.py — Image deblurring / sharpening module.

Restoration hierarchy:
  1. NAFNet-GoPro (basicsr, pretrained on motion-blur dataset)
  2. Wiener deconvolution (scikit-image)  ← reliable classical fallback
  3. Richardson-Lucy deconvolution        ← alternative classical method
  4. Unsharp masking                      ← last-resort sharpening
"""
from __future__ import annotations

import functools
from typing import Optional

import cv2
import numpy as np
from scipy.signal import convolve2d
from skimage.restoration import wiener, richardson_lucy

from utils import (
    array_to_tensor, download_weights, get_logger, tensor_to_array,
    to_float32, to_uint8, pad_to_multiple, unpad,
)

log = get_logger(__name__)

# ─── URLs ─────────────────────────────────────────────────────────────────────

_NAFNET_GOPRO_URL  = ("https://github.com/megvii-research/NAFNet/releases/download"
                      "/v0.0.1/NAFNet-GoPro-width64.pth")
_NAFNET_GOPRO_FILE = "NAFNet-GoPro-width64.pth"


# ─── Model loaders ────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _load_nafnet_gopro():
    try:
        import torch
        from restoration.nafnet_arch import NAFNet

        weights_path = download_weights(_NAFNET_GOPRO_URL, _NAFNET_GOPRO_FILE)
        model = NAFNet(
            img_channel=3, width=64, middle_blks_num=12,
            enc_blks=[2, 2, 4, 8], dec_blks=[2, 2, 2, 2],
        )
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("params", state), strict=True)
        model.eval()
        log.info("✓ NAFNet-GoPro loaded for deblurring.")
        return model
    except Exception as exc:
        log.warning("NAFNet-GoPro unavailable: %s", exc)
        return None


# ─── Deblurring methods ───────────────────────────────────────────────────────

def _deblur_nafnet(img: np.ndarray) -> Optional[np.ndarray]:
    model = _load_nafnet_gopro()
    if model is None:
        return None
    try:
        import torch
        x, padding = pad_to_multiple(img, 8)
        tensor_in = array_to_tensor(x)
        with torch.no_grad():
            tensor_out = model(tensor_in).clamp(0, 1)
        out = tensor_to_array(tensor_out)
        return to_uint8(to_float32(unpad(out, padding)))
    except Exception as exc:
        log.warning("NAFNet-GoPro inference failed: %s", exc)
        return None


def _estimate_gaussian_psf(kernel_size: int = 5, sigma: float = 2.0) -> np.ndarray:
    """Create a Gaussian PSF for Wiener / RL deconvolution."""
    ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf /= psf.sum()
    return psf


def _deblur_wiener(img: np.ndarray, balance: float = 0.1) -> np.ndarray:
    """
    Per-channel Wiener deconvolution with a Gaussian PSF estimate.
    *balance* controls regularisation (higher = smoother output).
    """
    psf = _estimate_gaussian_psf(kernel_size=7, sigma=2.5)
    channels = []
    for ch in range(3):
        plane = to_float32(img[:, :, ch])
        restored = wiener(plane, psf, balance)
        channels.append(np.clip(restored, 0, 1))
    return to_uint8(np.stack(channels, axis=2))


def _deblur_richardson_lucy(img: np.ndarray, iterations: int = 15) -> np.ndarray:
    """Per-channel Richardson-Lucy deconvolution."""
    psf = _estimate_gaussian_psf(kernel_size=5, sigma=2.0)
    channels = []
    for ch in range(3):
        plane = to_float32(img[:, :, ch])
        restored = richardson_lucy(plane, psf, num_iter=iterations)
        channels.append(np.clip(restored, 0, 1))
    return to_uint8(np.stack(channels, axis=2))


def _deblur_unsharp(img: np.ndarray, sigma: float = 2.0, amount: float = 1.5) -> np.ndarray:
    """Unsharp masking — fast but artefact-prone; kept as last resort."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharp = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


# ─── Public API ───────────────────────────────────────────────────────────────

def deblur(img: np.ndarray, method: str = "auto", **kwargs) -> np.ndarray:
    """
    Deblur *img*.

    Parameters
    ----------
    img    : uint8 RGB [H, W, 3]
    method : "auto" | "nafnet" | "wiener" | "rl" | "unsharp"
             "auto" tries NAFNet → Wiener → RL
    """
    if method == "nafnet":
        out = _deblur_nafnet(img)
        return out if out is not None else _deblur_wiener(img)

    if method == "wiener":
        balance = float(kwargs.get("balance", 0.1))
        return _deblur_wiener(img, balance=balance)

    if method == "rl":
        iters = int(kwargs.get("iterations", 15))
        return _deblur_richardson_lucy(img, iterations=iters)

    if method == "unsharp":
        return _deblur_unsharp(img)

    # "auto" cascade
    log.info("Deblurrer: trying NAFNet-GoPro …")
    out = _deblur_nafnet(img)
    if out is not None:
        log.info("Deblurrer: used NAFNet-GoPro.")
        return out

    log.info("Deblurrer: falling back to Wiener deconvolution.")
    return _deblur_wiener(img)
