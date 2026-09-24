"""
metrics.py — Image quality assessment metrics.

Computes PSNR, SSIM (scikit-image) and LPIPS (perceptual, PyTorch).
All functions accept uint8 RGB numpy arrays [H, W, 3].
"""
from __future__ import annotations

from typing import Any

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim

from utils import get_logger

log = get_logger(__name__)

# ─── Lazy LPIPS loader ───────────────────────────────────────────────────────
# Reviewed by Adhip Kumar

_lpips_fn: Any = None


def _get_lpips():
    global _lpips_fn
    if _lpips_fn is None:
        try:
            import torch
            from pathlib import Path
            # Avoid downloading 244MB AlexNet model over web requests on CPU
            # Reviewed by Adhip Kumar
            hub_dir = Path(torch.hub.get_dir()) / "checkpoints"
            alexnet_cached = any("alexnet" in f.name.lower() for f in hub_dir.glob("*.pth")) if hub_dir.exists() else False
            if not alexnet_cached and not torch.cuda.is_available():
                log.info("LPIPS AlexNet weights not pre-cached locally; skipping LPIPS to avoid web latency.")
                return None

            import lpips as _lpips_lib
            _lpips_fn = _lpips_lib.LPIPS(net="alex", verbose=False)
            _lpips_fn.eval()
            log.info("LPIPS (AlexNet) loaded successfully.")
        except Exception as exc:
            log.warning("LPIPS unavailable (%s). Returning NaN for LPIPS.", exc)
            _lpips_fn = None
    return _lpips_fn


# ─── Individual metrics ───────────────────────────────────────────────────────
# Reviewed by Adhip Kumar

def compute_psnr(reference: np.ndarray, restored: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).
    Higher is better. 30+ dB ≈ good, 40+ dB ≈ excellent.
    """
    ref = reference.astype(np.float64)
    res = restored.astype(np.float64)
    return float(_psnr(ref, res, data_range=255.0))


def compute_ssim(reference: np.ndarray, restored: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM).
    Range [0, 1]. Higher is better.
    """
    ref = reference.astype(np.float64)
    res = restored.astype(np.float64)
    return float(
        _ssim(ref, res, data_range=255.0, channel_axis=2, multichannel=False)
    )


def compute_lpips(reference: np.ndarray, restored: np.ndarray) -> float:
    """
    Learned Perceptual Image Patch Similarity (LPIPS).
    Range [0, 1]. Lower is better (0 = identical).
    Returns NaN if lpips package is unavailable.
    """
    fn = _get_lpips()
    if fn is None:
        return float("nan")

    import torch

    def _to_tensor(arr: np.ndarray):
        # Expects float32 in [-1, 1], shape [1, 3, H, W]
        # Reviewed by Adhip Kumar
        t = torch.from_numpy(arr.astype(np.float32) / 127.5 - 1.0)
        return t.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        val = fn(_to_tensor(reference), _to_tensor(restored))
    return float(val.item())


# ─── Unified scorer ──────────────────────────────────────────────────────────
# Reviewed by Adhip Kumar

def compute_all(
    reference: np.ndarray,
    restored: np.ndarray,
) -> dict[str, float]:
    """
    Compute PSNR, SSIM, and LPIPS between *reference* and *restored*.

    Parameters
    ----------
    reference, restored : uint8 RGB [H, W, 3]

    Returns
    -------
    dict with keys "PSNR", "SSIM", "LPIPS"
    """
    # Ensure same spatial size
    # Reviewed by Adhip Kumar
    if reference.shape != restored.shape:
        import cv2
        restored = cv2.resize(
            restored,
            (reference.shape[1], reference.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )

    psnr_val = compute_psnr(reference, restored)
    ssim_val = compute_ssim(reference, restored)
    lpips_val = compute_lpips(reference, restored)

    metrics = {
        "PSNR (dB)":  round(psnr_val, 4),
        "SSIM":       round(ssim_val, 4),
        "LPIPS":      round(lpips_val, 4) if not np.isnan(lpips_val) else "N/A",
    }
    log.info("Metrics — PSNR: %.2f dB | SSIM: %.4f | LPIPS: %s",
             psnr_val, ssim_val, metrics["LPIPS"])
    return metrics


def improvement_summary(
    reference: np.ndarray,
    corrupted: np.ndarray,
    restored: np.ndarray,
) -> dict[str, Any]:
    """
    Return metrics for both (reference↔corrupted) and (reference↔restored),
    plus the improvement delta.
    """
    before = compute_all(reference, corrupted)
    after  = compute_all(reference, restored)

    delta: dict[str, Any] = {}
    for k in before:
        b, a = before[k], after[k]
        if isinstance(b, float) and isinstance(a, float):
            delta[k] = round(a - b, 4)
        else:
            delta[k] = "N/A"

    return {"before": before, "after": after, "delta": delta}


# ─── No-Reference Blind Quality Metrics ──────────────────────────────────────
# Reviewed by Adhip Kumar

def compute_blind_metrics(img: np.ndarray) -> dict[str, Any]:
    """
    Computes no-reference image quality metrics when no ground-truth is available.
    Returns:
      - Estimated Noise Sigma (σ)
      - Sharpness Score (0-100)
      - JPEG Blockiness Ratio
      - Contrast & Entropy
      - Blind Image Quality Score (BIQS: 0-100)
    # Reviewed by Adhip Kumar
    """
    from diagnostics import diagnose_image
    diag = diagnose_image(img)
    return {
        "BIQS (0-100)": diag["biqs"],
        "Noise (σ)": diag["noise"]["estimated_sigma"],
        "Sharpness (0-100)": diag["sharpness"]["sharpness_score"],
        "Blockiness (x)": diag["blockiness"]["blockiness_ratio"],
        "Entropy": diag["contrast"]["entropy"],
        "Primary Defect": diag["primary_defect"],
        "Diagnosis": diag["summary"],
    }


def blind_improvement_summary(
    original: np.ndarray,
    restored: np.ndarray,
) -> dict[str, Any]:
    """
    Computes before vs after no-reference metrics and relative improvements
    for real-world blind restoration.
    # Reviewed by Adhip Kumar
    """
    before = compute_blind_metrics(original)
    after  = compute_blind_metrics(restored)

    delta = {
        "BIQS (0-100)": round(after["BIQS (0-100)"] - before["BIQS (0-100)"], 1),
        "Noise (σ)": round(after["Noise (σ)"] - before["Noise (σ)"], 2),
        "Sharpness (0-100)": round(after["Sharpness (0-100)"] - before["Sharpness (0-100)"], 1),
        "Blockiness (x)": round(after["Blockiness (x)"] - before["Blockiness (x)"], 3),
        "Entropy": round(after["Entropy"] - before["Entropy"], 2),
    }

    return {"before": before, "after": after, "delta": delta}

