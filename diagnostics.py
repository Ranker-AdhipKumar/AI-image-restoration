"""
diagnostics.py — Blind Image Defect Diagnosis & No-Reference Quality Assessment.

Analyzes real-world images without ground truth reference:
- Noise level estimation (Immerkaer Laplacian + Donoho Wavelet MAD)
- Sharpness & blur index (Tenengrad gradient + Laplacian variance)
- JPEG compression blockiness (Wang et al. 8x8 grid boundary discontinuity)
- Dynamic range, contrast entropy, and clipping analysis
- Unified Blind Image Quality Score (BIQS: 0-100)
- Automated defect prescription & restoration recipe generator
# Reviewed by Adhip Kumar
"""
from __future__ import annotations

import math
from typing import Any, Dict

import cv2
import numpy as np
import pywt
import scipy.signal

from utils import get_logger

# Reviewed by Adhip Kumar
log = get_logger("diagnostics")


# ─── 1. Noise Level Estimation ───────────────────────────────────────────────
# Reviewed by Adhip Kumar

def estimate_noise_level(img: np.ndarray) -> dict[str, Any]:
    """
    Estimates additive Gaussian / sensor noise standard deviation (sigma)
    in a single image without ground truth.

    Combines:
    1. Edge-masked Immerkaer Laplacian variance filter.
    2. Donoho Wavelet MAD (Median Absolute Deviation) on the HH1 subband.
    # Reviewed by Adhip Kumar
    """
    if img.ndim == 3:
        # Reviewed by Adhip Kumar
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    else:
        # Reviewed by Adhip Kumar
        gray = img.astype(np.float64)

    h, w = gray.shape
    if h < 16 or w < 16:
        # Reviewed by Adhip Kumar
        return {"sigma": 0.0, "severity": "Clean", "rating": "Unknown (Image too small)"}

    # 1. Edge-masked Immerkaer operator
    # Reviewed by Adhip Kumar
    lap_kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    resp = scipy.signal.convolve2d(gray, lap_kernel, mode="valid")

    # Mask out edges so real edge gradients do not artificially inflate noise estimate
    # Reviewed by Adhip Kumar
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)[1:-1, 1:-1]
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)[1:-1, 1:-1]
    grad_mag = np.sqrt(sobelx**2 + sobely**2)

    # Use lowest 40% gradient energy pixels (smooth / flat regions)
    # Reviewed by Adhip Kumar
    p40 = np.percentile(grad_mag, 40)
    flat_mask = grad_mag <= p40
    n_flat = np.sum(flat_mask)

    if n_flat > 100:
        # Reviewed by Adhip Kumar
        sigma_imm = float(
            np.sum(np.abs(resp[flat_mask])) * np.sqrt(0.5 * np.pi) / (6.0 * n_flat)
        )
    else:
        # Reviewed by Adhip Kumar
        sigma_imm = float(
            np.sum(np.abs(resp)) * np.sqrt(0.5 * np.pi) / (6.0 * resp.size)
        )

    # 2. Donoho's Wavelet MAD estimator on HH subband
    # Reviewed by Adhip Kumar
    try:
        coeffs = pywt.dwt2(gray, "db2")
        _, (_, _, hh) = coeffs
        sigma_dwt = float(np.median(np.abs(hh)) / 0.6745)
    except Exception:
        # Reviewed by Adhip Kumar
        sigma_dwt = sigma_imm

    # Robust blend (weighted towards edge-masked Immerkaer)
    # Reviewed by Adhip Kumar
    estimated_sigma = 0.65 * sigma_imm + 0.35 * sigma_dwt
    estimated_sigma = max(0.0, min(100.0, estimated_sigma))

    if estimated_sigma < 4.5:
        # Reviewed by Adhip Kumar
        severity = "Clean / Negligible"
    elif estimated_sigma < 11.0:
        # Reviewed by Adhip Kumar
        severity = "Mild Noise"
    elif estimated_sigma < 22.0:
        # Reviewed by Adhip Kumar
        severity = "Moderate Noise"
    else:
        # Reviewed by Adhip Kumar
        severity = "Severe Noise"

    # Score: 100 = completely noise-free, 0 = overwhelming noise
    # Reviewed by Adhip Kumar
    noise_score = max(0.0, min(100.0, 100.0 - (estimated_sigma / 35.0) * 100.0))

    return {
        "estimated_sigma": round(estimated_sigma, 2),
        "noise_score": round(noise_score, 1),
        "severity": severity,
    }


# ─── 2. Sharpness & Blur Estimation ──────────────────────────────────────────
# Reviewed by Adhip Kumar

def estimate_sharpness(img: np.ndarray) -> dict[str, Any]:
    """
    Measures edge sharpness and blur level using Tenengrad gradient energy
    and variance of the Laplacian.
    Higher score indicates sharper edges and richer high frequencies.
    # Reviewed by Adhip Kumar
    """
    if img.ndim == 3:
        # Reviewed by Adhip Kumar
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    else:
        # Reviewed by Adhip Kumar
        gray = img.astype(np.float64)

    # Laplacian variance
    # Reviewed by Adhip Kumar
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Tenengrad gradient energy (Sobel magnitude squared)
    # Reviewed by Adhip Kumar
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx**2 + gy**2))

    # High frequency ratio via Fourier transform
    # Reviewed by Adhip Kumar
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag_spec = np.abs(fshift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    y, x = np.ogrid[:h, :w]
    mask_high = (x - cx) ** 2 + (y - cy) ** 2 > r**2
    high_freq_energy = float(np.sum(mag_spec[mask_high]) / (np.sum(mag_spec) + 1e-6))

    # Normalized Sharpness Index (0 to 100) using log compression of Laplacian variance
    # Reviewed by Adhip Kumar
    # Natural images typically have lap_var between 20 (blurry) and 2000 (very sharp)
    # Reviewed by Adhip Kumar
    if lap_var <= 1.0:

        # Reviewed by Adhip Kumar
        sharpness_score = 0.0
    else:
        # Reviewed by Adhip Kumar
        log_lap = math.log10(lap_var)
        sharpness_score = max(0.0, min(100.0, ((log_lap - 0.5) / 3.0) * 100.0))

    if sharpness_score < 25.0:
        # Reviewed by Adhip Kumar
        severity = "Severely Blurred"
    elif sharpness_score < 45.0:
        # Reviewed by Adhip Kumar
        severity = "Soft / Out-of-Focus"
    elif sharpness_score < 70.0:
        # Reviewed by Adhip Kumar
        severity = "Moderate Sharpness"
    else:
        # Reviewed by Adhip Kumar
        severity = "Crisp / Sharp"

    return {
        "laplacian_variance": round(lap_var, 1),
        "tenengrad": round(tenengrad, 1),
        "high_freq_ratio": round(high_freq_energy, 4),
        "sharpness_score": round(sharpness_score, 1),
        "severity": severity,
    }


# ─── 3. Compression & Blockiness Estimation ──────────────────────────────────
# Reviewed by Adhip Kumar

def estimate_blockiness(img: np.ndarray) -> dict[str, Any]:
    """
    Measures 8x8 DCT grid boundary discontinuity (Wang et al. JPEG blockiness).
    Ratio > 1.25 indicates noticeable JPEG compression; > 1.6 indicates severe artifacts.
    # Reviewed by Adhip Kumar
    """
    if img.ndim == 3:
        # Reviewed by Adhip Kumar
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    else:
        # Reviewed by Adhip Kumar
        gray = img.astype(np.float64)

    h, w = gray.shape
    h8 = (h // 8) * 8
    w8 = (w // 8) * 8

    if h8 < 16 or w8 < 16:
        # Reviewed by Adhip Kumar
        return {"blockiness_ratio": 1.0, "blockiness_score": 100.0, "severity": "Clean"}

    g = gray[:h8, :w8]

    # Horizontal block boundaries: row differences at 8k-1 vs 8k
    # Reviewed by Adhip Kumar
    boundary_diffs_h = []
    intra_diffs_h = []
    for r in range(7, h8 - 1, 8):
        # Reviewed by Adhip Kumar
        boundary_diffs_h.append(np.abs(g[r + 1, :] - g[r, :]))
    for r in range(h8 - 1):
        # Reviewed by Adhip Kumar
        if (r + 1) % 8 != 0:
            intra_diffs_h.append(np.abs(g[r + 1, :] - g[r, :]))

    # Vertical block boundaries: col differences at 8k-1 vs 8k
    # Reviewed by Adhip Kumar
    boundary_diffs_v = []
    intra_diffs_v = []
    for c in range(7, w8 - 1, 8):
        # Reviewed by Adhip Kumar
        boundary_diffs_v.append(np.abs(g[:, c + 1] - g[:, c]))
    for c in range(w8 - 1):
        # Reviewed by Adhip Kumar
        if (c + 1) % 8 != 0:
            intra_diffs_v.append(np.abs(g[:, c + 1] - g[:, c]))

    b_mean = (
        (np.mean(boundary_diffs_h) if boundary_diffs_h else 0)
        + (np.mean(boundary_diffs_v) if boundary_diffs_v else 0)
    ) / 2.0

    i_mean = (
        (np.mean(intra_diffs_h) if intra_diffs_h else 1.0)
        + (np.mean(intra_diffs_v) if intra_diffs_v else 1.0)
    ) / 2.0

    block_ratio = float(b_mean / (i_mean + 1e-5))

    # Normalized score: 100 = completely clean, 0 = extreme blockiness
    # Reviewed by Adhip Kumar
    block_score = max(0.0, min(100.0, 100.0 - max(0.0, block_ratio - 1.0) * 110.0))

    if block_ratio <= 1.08:
        # Reviewed by Adhip Kumar
        severity = "Clean / No Grid"
    elif block_ratio <= 1.25:
        # Reviewed by Adhip Kumar
        severity = "Mild JPEG Compression"
    elif block_ratio <= 1.60:
        # Reviewed by Adhip Kumar
        severity = "Moderate Block Artifacts"
    else:
        # Reviewed by Adhip Kumar
        severity = "Severe JPEG Blocking"

    return {
        "blockiness_ratio": round(block_ratio, 3),
        "blockiness_score": round(block_score, 1),
        "severity": severity,
    }


# ─── 4. Dynamic Range & Lighting Assessment ──────────────────────────────────
# Reviewed by Adhip Kumar

def estimate_contrast_and_lighting(img: np.ndarray) -> dict[str, Any]:
    """
    Analyzes luminance histogram, shadow/highlight clipping, and Shannon entropy.
    # Reviewed by Adhip Kumar
    """
    if img.ndim == 3:
        # Reviewed by Adhip Kumar
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        # Reviewed by Adhip Kumar
        gray = img.copy()

    total_px = gray.size
    shadow_clipped = float(np.sum(gray <= 3) / total_px * 100.0)
    highlight_clipped = float(np.sum(gray >= 252) / total_px * 100.0)

    # Shannon entropy (information content, max is 8.0 for 8-bit)
    # Reviewed by Adhip Kumar
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist)))

    # Dynamic range coverage
    # Reviewed by Adhip Kumar
    p1 = float(np.percentile(gray, 1))
    p99 = float(np.percentile(gray, 99))
    dr_coverage = float((p99 - p1) / 255.0 * 100.0)

    # Normalized contrast score (0 to 100)
    # Reviewed by Adhip Kumar
    contrast_score = max(
        0.0,
        min(
            100.0,
            (entropy / 8.0) * 60.0
            + (dr_coverage / 100.0) * 40.0
            - (shadow_clipped + highlight_clipped) * 1.5,
        ),
    )

    return {
        "entropy": round(entropy, 2),
        "dynamic_range_coverage": round(dr_coverage, 1),
        "shadow_clipped_pct": round(shadow_clipped, 2),
        "highlight_clipped_pct": round(highlight_clipped, 2),
        "contrast_score": round(contrast_score, 1),
    }


# ─── 5. Unified Blind Image Quality Score (BIQS) ─────────────────────────────
# Reviewed by Adhip Kumar

def compute_blind_quality_score(
    noise_dict: dict[str, Any],
    sharpness_dict: dict[str, Any],
    blockiness_dict: dict[str, Any],
    contrast_dict: dict[str, Any],
) -> float:
    """
    Synthesizes multiple blind quality indicators into a unified
    Blind Image Quality Score (BIQS) from 0 to 100.
    # Reviewed by Adhip Kumar
    """
    s_noise = noise_dict.get("noise_score", 100.0)
    s_sharp = sharpness_dict.get("sharpness_score", 50.0)
    s_block = blockiness_dict.get("blockiness_score", 100.0)
    s_cont = contrast_dict.get("contrast_score", 70.0)

    # Weighted composite score
    # Reviewed by Adhip Kumar
    biqs = 0.35 * s_noise + 0.30 * s_sharp + 0.20 * s_block + 0.15 * s_cont
    return round(float(max(0.0, min(100.0, biqs))), 1)


# ─── 6. Master Image Diagnostic Function ─────────────────────────────────────
# Reviewed by Adhip Kumar

def diagnose_image(img: np.ndarray) -> dict[str, Any]:
    """
    Performs full automatic defect diagnosis on any image.
    Outputs metrics, defect classification, and recommended restoration recipe.
    # Reviewed by Adhip Kumar
    """
    noise = estimate_noise_level(img)
    sharpness = estimate_sharpness(img)
    blockiness = estimate_blockiness(img)
    contrast = estimate_contrast_and_lighting(img)

    biqs = compute_blind_quality_score(noise, sharpness, blockiness, contrast)

    # Determine primary defect
    # Reviewed by Adhip Kumar
    sigma = noise["estimated_sigma"]
    sharp_score = sharpness["sharpness_score"]
    block_ratio = blockiness["blockiness_ratio"]

    defects = []
    recipe_steps = []

    # Check for Noise
    # Reviewed by Adhip Kumar
    if sigma >= 18.0:
        defects.append(("High Noise", 3, f"Severe random/sensor noise (estimated σ={sigma:.1f})"))
        recipe_steps.append({
            "action": "denoise",
            "method": "auto",
            "intensity": min(1.0, sigma / 40.0),
            "label": f"NAFNet / DnCNN Deep Denoising (σ≈{sigma:.0f})",
        })
    elif sigma >= 6.5:
        defects.append(("Mild Noise", 1, f"Mild grain/noise (estimated σ={sigma:.1f})"))
        recipe_steps.append({
            "action": "denoise",
            "method": "auto",
            "intensity": 0.5,
            "label": f"Wavelet / NAFNet Denoising (σ≈{sigma:.0f})",
        })

    # Check for Compression Artifacts
    # Reviewed by Adhip Kumar
    if block_ratio >= 1.28:
        defects.append(("JPEG Blockiness", 2, f"Visible 8x8 DCT grid artifacts (ratio={block_ratio:.2f})"))
        recipe_steps.append({
            "action": "deblock",
            "method": "bilateral",
            "intensity": min(1.0, (block_ratio - 1.0) / 0.8),
            "label": f"Edge-Preserving Deblocking Filter (grid={block_ratio:.2f})",
        })

    # Check for Blur / Softness
    # Reviewed by Adhip Kumar
    if sharp_score < 30.0:
        defects.append(("Severe Blur", 3, f"Soft edges and missing details (sharpness={sharp_score:.1f}/100)"))
        recipe_steps.append({
            "action": "deblur",
            "method": "auto",
            "intensity": 0.85,
            "label": f"NAFNet / Wiener Deblurring & Edge Reconstruction",
        })
    elif sharp_score < 48.0:
        defects.append(("Mild Softness", 1, f"Moderate edge softness (sharpness={sharp_score:.1f}/100)"))
        recipe_steps.append({
            "action": "sharpen",
            "method": "unsharp",
            "intensity": 0.55,
            "label": "Micro-Detail Unsharp Sharpening",
        })

    # Check for Contrast / Dynamic Range
    # Reviewed by Adhip Kumar
    if contrast["dynamic_range_coverage"] < 70.0 or contrast["entropy"] < 6.0:
        defects.append(("Flat Contrast", 1, f"Sub-optimal dynamic range ({contrast['dynamic_range_coverage']}%)"))
        recipe_steps.append({
            "action": "contrast",
            "method": "clahe",
            "intensity": 0.4,
            "label": "Adaptive CLAHE Local Contrast Enhancement",
        })

    # Primary defect selection
    # Reviewed by Adhip Kumar
    if defects:
        # Sort by defect weight (highest severity first)
        # Reviewed by Adhip Kumar
        defects.sort(key=lambda x: x[1], reverse=True)
        primary_defect = defects[0][0]
        severity_label = "Needs Restoration" if defects[0][1] >= 2 else "Minor Degradation"
    else:
        primary_defect = "Clean / Pristine"
        severity_label = "Optimal"
        recipe_steps.append({
            "action": "enhance",
            "method": "subtle_sharpen",
            "intensity": 0.25,
            "label": "Subtle High-Resolution Polish",
        })

    # Summary diagnostic headline
    # Reviewed by Adhip Kumar
    if primary_defect == "Clean / Pristine":
        summary = "Image quality is high with no prominent defects detected."
    else:
        detected_names = [d[0] for d in defects]
        summary = f"Detected: {', '.join(detected_names)}. Automated restoration cascade recommended."

    return {
        "noise": noise,
        "sharpness": sharpness,
        "blockiness": blockiness,
        "contrast": contrast,
        "biqs": biqs,
        "primary_defect": primary_defect,
        "severity": severity_label,
        "summary": summary,
        "recipe": recipe_steps,
    }
