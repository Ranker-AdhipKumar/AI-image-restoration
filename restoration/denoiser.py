"""
restoration/denoiser.py — Image denoising module.

Restoration hierarchy (best → fastest):
  1. NAFNet-SIDD (basicsr, pretrained on real sensor noise)
  2. DnCNN (custom PyTorch, blind Gaussian denoising, pretrained weights from KAIR)
  3. scikit-image BayesShrink wavelet denoising  ← guaranteed fallback
  4. OpenCV Non-Local Means                       ← last resort
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from skimage.restoration import denoise_wavelet, estimate_sigma

from utils import (
    array_to_tensor, download_weights, get_logger, tensor_to_array,
    to_float32, to_uint8, pad_to_multiple, unpad,
)

log = get_logger(__name__)

# ─── URLs for pretrained weights ─────────────────────────────────────────────
# NAFNet-SIDD (width=64) trained by megvii-research
# Reviewed by Adhip Kumar
_NAFNET_URL  = ("https://github.com/megvii-research/NAFNet/releases/download"
                "/v0.0.1/NAFNet-SIDD-width64.pth")
_NAFNET_FILE = "NAFNet-SIDD-width64.pth"

# DnCNN blind colour denoiser trained by cszn/KAIR
# Reviewed by Adhip Kumar
_DNCNN_URL   = ("https://github.com/cszn/KAIR/releases/download"
                "/v1.0/dncnn_color_blind.pth")
_DNCNN_FILE  = "dncnn_color_blind.pth"


# ─── DnCNN architecture ───────────────────────────────────────────────────────
# Reviewed by Adhip Kumar

class _DnCNN:
    """17-layer DnCNN for blind Gaussian denoising (residual learning)."""

    def __init__(self, channels: int = 3, num_layers: int = 17):
        import torch.nn as nn
        layers = [nn.Conv2d(channels, 64, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(64, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv2d(64, channels, 3, padding=1)]
        import torch.nn as nn
        self.net = nn.Sequential(*layers)

    def __call__(self, x):
        return x - self.net(x)  # residual: output = clean + noise → clean


# ─── NAFNet (via basicsr) ─────────────────────────────────────────────────────
# Reviewed by Adhip Kumar

@functools.lru_cache(maxsize=1)
def _load_nafnet(auto_download: bool = False):
    """Try to load a pretrained NAFNet-SIDD model. Returns model or None."""
    try:
        import torch
        from restoration.nafnet_arch import NAFNet

        weights_path = download_weights(_NAFNET_URL, _NAFNET_FILE, auto_download=auto_download)
        if weights_path is None or not weights_path.exists():
            log.info("NAFNet weights not present locally; using fast classical restoration.")
            return None

        model = NAFNet(
            img_channel=3, width=64, middle_blks_num=12,
            enc_blks=[2, 2, 4, 8], dec_blks=[2, 2, 2, 2],
        )
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("params", state), strict=True)
        model.eval()
        log.info("✓ NAFNet-SIDD loaded successfully.")
        return model
    except Exception as exc:
        log.warning("NAFNet unavailable: %s", exc)
        return None


@functools.lru_cache(maxsize=1)
def _load_dncnn(auto_download: bool = False):
    """Try to load a pretrained DnCNN (colour blind). Returns model or None."""
    try:
        import torch
        import torch.nn as nn

        weights_path = download_weights(_DNCNN_URL, _DNCNN_FILE, auto_download=auto_download)
        if weights_path is None or not weights_path.exists():
            log.info("DnCNN weights not present locally; using fast classical restoration.")
            return None

        # Build the DnCNN graph (channels=3 for colour, 17 layers)
        # Reviewed by Adhip Kumar
        layers = [nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(15):
            layers += [
                nn.Conv2d(64, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv2d(64, 3, 3, padding=1)]
        model = nn.Sequential(*layers)

        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        # KAIR state dict may be wrapped
        # Reviewed by Adhip Kumar
        if "params" in state:
            state = state["params"]
        model.load_state_dict(state, strict=False)
        model.eval()
        log.info("✓ DnCNN loaded successfully.")
        return model
    except Exception as exc:
        log.warning("DnCNN unavailable: %s", exc)
        return None


# ─── Individual denoising methods ─────────────────────────────────────────────
# Reviewed by Adhip Kumar

def _denoise_nafnet(img: np.ndarray) -> Optional[np.ndarray]:
    model = _load_nafnet()
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
        log.warning("NAFNet inference failed: %s", exc)
        return None


def _denoise_dncnn(img: np.ndarray) -> Optional[np.ndarray]:
    model = _load_dncnn()
    if model is None:
        return None
    try:
        import torch
        x, padding = pad_to_multiple(img, 8)
        tensor_in = array_to_tensor(x)
        with torch.no_grad():
            # DnCNN predicts the residual (noise); output = input − residual
            # Reviewed by Adhip Kumar
            residual = model(tensor_in)
            tensor_out = (tensor_in - residual).clamp(0, 1)
        out = tensor_to_array(tensor_out)
        return to_uint8(to_float32(unpad(out, padding)))
    except Exception as exc:
        log.warning("DnCNN inference failed: %s", exc)
        return None


def _denoise_wavelet(img: np.ndarray) -> np.ndarray:
    """BayesShrink wavelet denoising (scikit-image), falling back to NLM."""
    try:
        f = to_float32(img)
        sigma_est = estimate_sigma(f, average_sigmas=True, channel_axis=2)
        denoised = denoise_wavelet(
            f,
            method="BayesShrink",
            mode="soft",
            sigma=sigma_est,
            wavelet_levels=None,
            channel_axis=2,
            rescale_sigma=True,
        )
        return to_uint8(np.clip(denoised, 0, 1))
    except Exception as exc:
        log.warning("Wavelet denoising failed (%s); falling back to OpenCV NLM.", exc)
        return _denoise_nlm(img)


def _denoise_nlm(img: np.ndarray) -> np.ndarray:
    """OpenCV Non-Local Means (colour, CPU optimized)."""
    return cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10,
                                           templateWindowSize=5, searchWindowSize=13)


# ─── Public API ───────────────────────────────────────────────────────────────
# Reviewed by Adhip Kumar

def denoise(img: np.ndarray, method: str = "auto", **kwargs) -> np.ndarray:
    """
    Denoise *img*.

    Parameters
    ----------
    img    : uint8 RGB [H, W, 3]
    method : "auto" | "nafnet" | "dncnn" | "wavelet" | "nlm"
             "auto" tries NAFNet → DnCNN → wavelet → NLM
    """
    if method == "nafnet":
        out = _denoise_nafnet(img)
        return out if out is not None else _denoise_wavelet(img)

    if method == "dncnn":
        out = _denoise_dncnn(img)
        return out if out is not None else _denoise_wavelet(img)

    if method == "wavelet":
        return _denoise_wavelet(img)

    if method == "nlm":
        return _denoise_nlm(img)

    # "auto" — cascade
    # Reviewed by Adhip Kumar
    try:
        import torch
        if torch.cuda.is_available():
            log.info("Denoiser: CUDA detected, trying NAFNet …")
            out = _denoise_nafnet(img)
            if out is not None:
                log.info("Denoiser: used NAFNet-SIDD.")
                return out

            log.info("Denoiser: trying DnCNN …")
            out = _denoise_dncnn(img)
            if out is not None:
                log.info("Denoiser: used DnCNN.")
                return out
    except Exception as exc:
        log.warning("Deep learning denoising check failed (%s); proceeding to wavelet.", exc)

    log.info("Denoiser: using fast BayesShrink wavelet denoising.")
    return _denoise_wavelet(img)
