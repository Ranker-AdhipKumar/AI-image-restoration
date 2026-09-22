"""
restoration/artifact_remover.py — JPEG / transmission artifact removal.

Restoration hierarchy:
  1. DnCNN (pretrained, reused from denoiser) — handles block/ringing artifacts
  2. NAFNet-SIDD (if available)
  3. scikit-image total-variation (Chambolle) — preserves edges
  4. OpenCV Non-Local Means with JPEG-tuned parameters
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle

from utils import get_logger, to_float32, to_uint8

log = get_logger(__name__)


def _remove_dncnn(img: np.ndarray):
    """Reuse DnCNN from the denoiser module."""
    try:
        from restoration.denoiser import _denoise_dncnn
        return _denoise_dncnn(img)
    except Exception as exc:
        log.warning("DnCNN artifact removal failed: %s", exc)
        return None


def _remove_nafnet(img: np.ndarray):
    """Reuse NAFNet-SIDD from the denoiser module."""
    try:
        from restoration.denoiser import _denoise_nafnet
        return _denoise_nafnet(img)
    except Exception as exc:
        log.warning("NAFNet artifact removal failed: %s", exc)
        return None


def _remove_tv(img: np.ndarray, weight: float = 0.08) -> np.ndarray:
    """
    Total-variation (Chambolle) denoising — excellent for block artifacts.
    weight: higher → smoother (0.05–0.15 works well for JPEG Q10–Q30).
    """
    f = to_float32(img)
    denoised = denoise_tv_chambolle(f, weight=weight, channel_axis=2)
    return to_uint8(np.clip(denoised, 0, 1))


def _remove_nlm(img: np.ndarray) -> np.ndarray:
    """OpenCV Non-Local Means tuned for JPEG block artifacts."""
    # h=6 hColor=6 are empirically good for JPEG artifacts (less smoothing than
    # for Gaussian noise, to preserve texture detail)
    return cv2.fastNlMeansDenoisingColored(
        img, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21
    )


def remove_artifacts(img: np.ndarray, method: str = "auto", **kwargs) -> np.ndarray:
    """
    Remove JPEG / transmission artifacts from *img*.

    Parameters
    ----------
    img    : uint8 RGB [H, W, 3]
    method : "auto" | "dncnn" | "nafnet" | "tv" | "nlm"
    """
    if method == "dncnn":
        out = _remove_dncnn(img)
        return out if out is not None else _remove_tv(img)

    if method == "nafnet":
        out = _remove_nafnet(img)
        return out if out is not None else _remove_tv(img)

    if method == "tv":
        weight = float(kwargs.get("weight", 0.08))
        return _remove_tv(img, weight=weight)

    if method == "nlm":
        return _remove_nlm(img)

    # "auto" cascade
    log.info("ArtifactRemover: trying DnCNN …")
    out = _remove_dncnn(img)
    if out is not None:
        log.info("ArtifactRemover: used DnCNN.")
        return out

    log.info("ArtifactRemover: trying NAFNet …")
    out = _remove_nafnet(img)
    if out is not None:
        log.info("ArtifactRemover: used NAFNet.")
        return out

    log.info("ArtifactRemover: falling back to Total-Variation (Chambolle).")
    return _remove_tv(img)
