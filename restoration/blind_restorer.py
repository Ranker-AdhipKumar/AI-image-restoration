"""
restoration/blind_restorer.py — Multi-stage blind image restoration engine.

Restores real-world degraded images without ground-truth reference:
- Executes intelligent auto-prescribed restoration cascades
- Deblocking (TV / Bilateral grid suppression)
- Denoising (NAFNet / DnCNN / BayesShrink / NLM)
- Edge-preserving sharpening & deblurring
- Micro-contrast and dynamic range enhancement (LAB CLAHE)
- Full before-and-after blind quality assessment
# Reviewed by Adhip Kumar
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle

from diagnostics import diagnose_image
from utils import get_logger, to_float32, to_uint8

# Reviewed by Adhip Kumar
log = get_logger("blind_restorer")


def _apply_deblock(img: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Applies edge-preserving deblocking to eliminate JPEG grid discontinuities
    without blurring semantic scene edges.
    # Reviewed by Adhip Kumar
    """
    weight = 0.04 + 0.08 * float(np.clip(intensity, 0.1, 1.0))
    f = to_float32(img)
    tv = denoise_tv_chambolle(f, weight=weight, channel_axis=2)
    # Blend with bilateral filter to preserve fine transitions
    # Reviewed by Adhip Kumar
    tv_u8 = to_uint8(np.clip(tv, 0.0, 1.0))
    bilateral = cv2.bilateralFilter(tv_u8, d=5, sigmaColor=25, sigmaSpace=25)
    return bilateral


def _apply_denoise(img: np.ndarray, method: str = "auto", intensity: float = 0.5) -> np.ndarray:
    """
    Invokes the restoration cascade (NAFNet -> DnCNN -> Wavelet -> NLM).
    # Reviewed by Adhip Kumar
    """
    try:
        from restoration.denoiser import denoise
        return denoise(img, method=method)
    except Exception as exc:
        log.warning("Primary denoiser failed in blind restoration: %s. Using NLM fallback.", exc)
        h = max(3, int(round(10.0 * intensity)))
        # Reviewed by Adhip Kumar
        return cv2.fastNlMeansDenoisingColored(
            img, None, h=h, hColor=h, templateWindowSize=7, searchWindowSize=21
        )


def _apply_deblur_or_sharpen(img: np.ndarray, is_heavy_blur: bool = False, intensity: float = 0.5) -> np.ndarray:
    """
    Performs deblurring or edge-aware unsharp sharpening.
    Uses edge-masking so flat noisy regions are NOT artificially sharpened.
    # Reviewed by Adhip Kumar
    """
    if is_heavy_blur:
        try:
            from restoration.deblurrer import deblur
            return deblur(img, method="auto")
        except Exception as exc:
            log.warning("Deblurrer failed in blind cascade: %s. Falling back to unsharp.", exc)

    # Edge-masked unsharp enhancement
    # Reviewed by Adhip Kumar
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_map = np.sqrt(sobelx**2 + sobely**2)
    edge_mask = np.clip(edge_map / (np.percentile(edge_map, 85) + 1e-5), 0.0, 1.0)
    edge_mask = np.repeat(edge_mask[:, :, np.newaxis], 3, axis=2)

    # Gaussian unsharp mask
    # Reviewed by Adhip Kumar
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5)
    unsharp = cv2.addWeighted(img, 1.0 + 0.8 * intensity, blurred, -0.8 * intensity, 0)
    unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)

    # Selectively sharpen only where edges exist
    # Reviewed by Adhip Kumar
    blended = (unsharp.astype(np.float32) * edge_mask + img.astype(np.float32) * (1.0 - edge_mask))
    return np.clip(np.round(blended), 0, 255).astype(np.uint8)


def _apply_contrast_enhancement(img: np.ndarray, intensity: float = 0.4) -> np.ndarray:
    """
    Applies CLAHE on luminance (L* channel in LAB) to lift shadows and reveal
    texture without shifting colors or introducing saturation artifacts.
    # Reviewed by Adhip Kumar
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clip_limit = 1.0 + 1.5 * float(np.clip(intensity, 0.1, 1.0))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_ch)

    # Gentle blend with original L channel to avoid over-exaggeration
    # Reviewed by Adhip Kumar
    alpha = 0.65
    l_blended = np.clip(l_enhanced.astype(float) * alpha + l_ch.astype(float) * (1.0 - alpha), 0, 255).astype(np.uint8)

    merged = cv2.merge([l_blended, a_ch, b_ch])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def restore_blind(
    img: np.ndarray,
    recipe: Optional[List[Dict[str, Any]]] = None,
    auto: bool = True,
    apply_denoise_opt: Optional[bool] = None,
    apply_deblur_opt: Optional[bool] = None,
    apply_deblock_opt: Optional[bool] = None,
    apply_contrast_opt: Optional[bool] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Master pipeline for blind image restoration.

    Parameters
    ----------
    img : uint8 RGB numpy array
    recipe : Optional explicit list of steps from diagnose_image()
    auto : If True, uses the auto-diagnosed recipe
    apply_*_opt : Explicit user overrides for individual stages (when not in full auto)

    Returns
    -------
    Tuple:
      1. restored_img: np.ndarray
      2. steps_applied: list[str]
      3. diag_before: dict
      4. diag_after: dict
      5. improvement: dict
    # Reviewed by Adhip Kumar
    """
    t0 = time.perf_counter()
    # 1. Initial blind diagnosis
    # Reviewed by Adhip Kumar
    diag_before = diagnose_image(img)
    steps_applied: List[str] = []
    current = img.copy()

    if auto:
        steps_to_run = diag_before.get("recipe", [])
    else:
        # Build manual recipe from user flags
        # Reviewed by Adhip Kumar
        steps_to_run = []
        if apply_deblock_opt:
            steps_to_run.append({"action": "deblock", "intensity": 0.6, "label": "JPEG Deblocking Filter"})
        if apply_denoise_opt:
            steps_to_run.append({"action": "denoise", "method": "auto", "intensity": 0.6, "label": "Deep Denoising"})
        if apply_deblur_opt:
            steps_to_run.append({"action": "deblur", "intensity": 0.6, "label": "Deblur & Sharpening"})
        if apply_contrast_opt:
            steps_to_run.append({"action": "contrast", "intensity": 0.4, "label": "Adaptive CLAHE Contrast"})

    # Execute in mathematically sound order: Deblock -> Denoise -> Deblur/Sharpen -> Contrast
    # Reviewed by Adhip Kumar
    order = {"deblock": 1, "denoise": 2, "deblur": 3, "sharpen": 3, "contrast": 4, "enhance": 4}
    sorted_steps = sorted(steps_to_run, key=lambda s: order.get(s["action"], 99))

    for step in sorted_steps:
        action = step.get("action")
        intensity = step.get("intensity", 0.5)
        label = step.get("label", action)

        if action == "deblock":
            current = _apply_deblock(current, intensity=intensity)
            steps_applied.append(f"Deblocking: {label}")
        elif action == "denoise":
            current = _apply_denoise(current, method=step.get("method", "auto"), intensity=intensity)
            steps_applied.append(f"Denoising: {label}")
        elif action in ("deblur", "sharpen", "enhance"):
            is_heavy = action == "deblur"
            current = _apply_deblur_or_sharpen(current, is_heavy_blur=is_heavy, intensity=intensity)
            steps_applied.append(f"Sharpness: {label}")
        elif action == "contrast":
            current = _apply_contrast_enhancement(current, intensity=intensity)
            steps_applied.append(f"Contrast: {label}")

    if not steps_applied:
        steps_applied.append("No degradation detected; image verified pristine.")

    elapsed = time.perf_counter() - t0

    # 2. Post-restoration blind diagnosis
    # Reviewed by Adhip Kumar
    diag_after = diagnose_image(current)

    # 3. Compute relative blind improvement summary
    # Reviewed by Adhip Kumar
    sigma_b = diag_before["noise"]["estimated_sigma"]
    sigma_a = diag_after["noise"]["estimated_sigma"]
    noise_reduction_pct = max(0.0, (sigma_b - sigma_a) / (sigma_b + 1e-5) * 100.0) if sigma_b > 4.0 else 0.0

    sharp_b = diag_before["sharpness"]["sharpness_score"]
    sharp_a = diag_after["sharpness"]["sharpness_score"]
    sharpness_gain_pct = ((sharp_a - sharp_b) / (sharp_b + 1e-5) * 100.0)

    block_b = diag_before["blockiness"]["blockiness_ratio"]
    block_a = diag_after["blockiness"]["blockiness_ratio"]
    block_reduction_pct = max(0.0, (block_b - block_a) / (block_b - 1.0 + 1e-5) * 100.0) if block_b > 1.1 else 0.0

    biqs_b = diag_before["biqs"]
    biqs_a = diag_after["biqs"]
    biqs_delta = round(biqs_a - biqs_b, 1)

    improvement = {
        "noise_reduction_pct": round(noise_reduction_pct, 1),
        "sharpness_gain_pct": round(sharpness_gain_pct, 1),
        "block_reduction_pct": round(block_reduction_pct, 1),
        "biqs_before": biqs_b,
        "biqs_after": biqs_a,
        "biqs_delta": biqs_delta,
        "elapsed_s": round(elapsed, 2),
    }

    log.info(
        "Blind Restoration complete in %.2fs — BIQS: %.1f -> %.1f (+%.1f pts) | Steps: %d",
        elapsed, biqs_b, biqs_a, biqs_delta, len(steps_applied)
    )

    return current, steps_applied, diag_before, diag_after, improvement
