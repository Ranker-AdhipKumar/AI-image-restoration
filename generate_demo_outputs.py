"""
generate_demo_outputs.py — Generate 50+ benchmark output comparison screenshots.

Runs Kodak, CBSD, and Synthetic benchmark images through various restoration
scenarios (Noise, Blur, Inpainting, JPEG Artifacts, Mixed, and Blind Real-World)
and saves the full comparison dashboard figures in assets/demo_outputs/.
# Reviewed by Adhip Kumar
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Reviewed by Adhip Kumar
sys.path.insert(0, str(Path(__file__).parent))

from degradation import degrade
from metrics import compute_all, improvement_summary
from restoration import restore
from restoration.blind_restorer import restore_blind
from utils import load_image, resize_if_larger, get_logger
from visualizer import save_comparison, save_blind_comparison

# Reviewed by Adhip Kumar
log = get_logger("demo_generator")

OUTPUT_DIR = Path("assets/demo_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR = Path("sample_images")


def _get_path(name: str) -> Path:
    # Reviewed by Adhip Kumar
    return SAMPLE_DIR / name


def run_scenario(
    idx: int,
    img_name: str,
    title_suffix: str,
    mode: str,
    params: dict,
    is_blind: bool = False,
    max_dim: int = 384,
) -> None:
    """Run a single restoration scenario and save the output comparison figure."""
    # Reviewed by Adhip Kumar
    t0 = time.perf_counter()
    src_path = _get_path(img_name)
    if not src_path.exists():
        # Fallback if image missing
        # Reviewed by Adhip Kumar
        log.warning("Image %s not found; skipping scenario %d", img_name, idx)
        return

    out_name = f"demo_{idx:02d}_{src_path.stem}_{title_suffix}.png"
    out_path = OUTPUT_DIR / out_name

    orig = load_image(src_path)
    orig = resize_if_larger(orig, max_dim)

    if is_blind:
        # 1. Blind Real-World Restoration & Diagnosis
        # Reviewed by Adhip Kumar
        # Add light realistic camera/sensor degradation first
        # Reviewed by Adhip Kumar
        degraded = orig.copy()
        if "noise_sigma" in params:
            # Reviewed by Adhip Kumar
            noise = np.random.normal(0, params["noise_sigma"], orig.shape)
            degraded = np.clip(degraded.astype(float) + noise, 0, 255).astype(np.uint8)
        if "blur_k" in params:
            # Reviewed by Adhip Kumar
            k = params["blur_k"]
            degraded = cv2.GaussianBlur(degraded, (k, k), params.get("blur_s", 2.0))
        if "jpeg_q" in params:
            # Reviewed by Adhip Kumar
            _, enc = cv2.imencode(".jpg", degraded, [int(cv2.IMWRITE_JPEG_QUALITY), params["jpeg_q"]])
            degraded = cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1]

        restored, steps, diag_b, diag_a, imp = restore_blind(degraded, auto=True)
        title = f"AI Blind Diagnosis & Real-World Restoration — {title_suffix}"
        save_blind_comparison(degraded, restored, diag_b, diag_a, out_path, title=title)
        log.info(
            "[%02d/52] Saved Blind: %s in %.2fs (BIQS: %.1f -> %.1f)",
            idx, out_name, time.perf_counter() - t0, imp["biqs_before"], imp["biqs_after"]
        )

    else:
        # 2. Reference-based Degradation & Restoration
        # Reviewed by Adhip Kumar
        corr, mask = degrade(orig, mode, **params)
        rest = restore(corr, mode=mode, mask=mask, method="auto")
        summary = improvement_summary(orig, corr, rest)
        title = f"Restoration Benchmark — {mode.upper()}: {title_suffix}"
        save_comparison(orig, corr, rest, summary["before"], summary["after"], out_path, title=title, mask=mask)
        psnr_b = summary["before"]["PSNR (dB)"]
        psnr_a = summary["after"]["PSNR (dB)"]
        log.info(
            "[%02d/52] Saved: %s in %.2fs (PSNR: %.1f -> %.1f dB)",
            idx, out_name, time.perf_counter() - t0, psnr_b, psnr_a
        )


def main():
    log.info("🚀 Starting generation of 52 benchmark output comparison figures in %s ...", OUTPUT_DIR)
    t_start = time.perf_counter()

    # Define the 52 scenarios
    # Reviewed by Adhip Kumar
    scenarios = [
        # Kodak 01-06: Noise
        # Reviewed by Adhip Kumar
        (1,  "kodak_kodim01.png", "gaussian_noise_s25", "noise", {"sigma": 25.0}, False),
        (2,  "kodak_kodim02.png", "gaussian_noise_s35", "noise", {"sigma": 35.0}, False),
        (3,  "kodak_kodim03.png", "gaussian_noise_s50", "noise", {"sigma": 50.0}, False),
        (4,  "kodak_kodim04.png", "salt_pepper_p05",   "salt_pepper", {"prob": 0.05}, False),
        (5,  "kodak_kodim05.png", "salt_pepper_p08",   "salt_pepper", {"prob": 0.08}, False),
        (6,  "kodak_kodim06.png", "speckle_noise",     "noise", {"sigma": 30.0}, False),

        # Kodak 07-12: Blur
        # Reviewed by Adhip Kumar
        (7,  "kodak_kodim07.png", "gaussian_blur_k11",  "blur", {"kernel_size": 11, "sigma_blur": 2.5}, False),
        (8,  "kodak_kodim08.png", "gaussian_blur_k15",  "blur", {"kernel_size": 15, "sigma_blur": 3.0}, False),
        (9,  "kodak_kodim09.png", "motion_blur_l25_a45","motion_blur", {"length": 25, "angle": 45.0}, False),
        (10, "kodak_kodim10.png", "motion_blur_l35_a30","motion_blur", {"length": 35, "angle": 30.0}, False),
        (11, "kodak_kodim11.png", "motion_blur_l20_a90","motion_blur", {"length": 20, "angle": 90.0}, False),
        (12, "kodak_kodim12.png", "gaussian_blur_k19",  "blur", {"kernel_size": 19, "sigma_blur": 3.5}, False),

        # Kodak 13-18: Inpainting
        # Reviewed by Adhip Kumar
        (13, "kodak_kodim13.png", "inpaint_2patches_s70", "inpaint", {"n_patches": 2, "patch_size": 70}, False),
        (14, "kodak_kodim14.png", "inpaint_3patches_s60", "inpaint", {"n_patches": 3, "patch_size": 60}, False),
        (15, "kodak_kodim15.png", "inpaint_4patches_s50", "inpaint", {"n_patches": 4, "patch_size": 50}, False),
        (16, "kodak_kodim16.png", "inpaint_3patches_s75", "inpaint", {"n_patches": 3, "patch_size": 75}, False),
        (17, "kodak_kodim17.png", "inpaint_2patches_s90", "inpaint", {"n_patches": 2, "patch_size": 90}, False),
        (18, "kodak_kodim18.png", "inpaint_4patches_s65", "inpaint", {"n_patches": 4, "patch_size": 65}, False),

        # Kodak 19-24: JPEG Artifacts & Mixed
        # Reviewed by Adhip Kumar
        (19, "kodak_kodim19.png", "jpeg_artifact_q05", "artifact", {"quality": 5}, False),
        (20, "kodak_kodim20.png", "jpeg_artifact_q10", "artifact", {"quality": 10}, False),
        (21, "kodak_kodim21.png", "jpeg_artifact_q15", "artifact", {"quality": 15}, False),
        (22, "kodak_kodim22.png", "jpeg_artifact_q20", "artifact", {"quality": 20}, False),
        (23, "kodak_kodim23.png", "mixed_noise_blur",  "mixed", {"sigma": 20.0, "kernel_size": 9}, False),
        (24, "kodak_kodim24.png", "mixed_noise_jpeg",  "mixed", {"sigma": 25.0, "quality": 12}, False),

        # CBSD 01-04: Classical Benchmarks
        # Reviewed by Adhip Kumar
        (25, "cbsd_0001.png", "gaussian_noise_s30", "noise", {"sigma": 30.0}, False),
        (26, "cbsd_0002.png", "motion_blur_l25",    "motion_blur", {"length": 25, "angle": 45.0}, False),
        (27, "cbsd_0003.png", "inpaint_3patches",   "inpaint", {"n_patches": 3, "patch_size": 60}, False),
        (28, "cbsd_0004.png", "jpeg_artifact_q10",  "artifact", {"quality": 10}, False),

        # Blind Real-World Diagnostics & Restoration on CBSD & Kodak
        # Reviewed by Adhip Kumar
        (29, "cbsd_0001.png", "blind_camera_noise", "", {"noise_sigma": 18.0}, True),
        (30, "cbsd_0002.png", "blind_lens_softness", "", {"blur_k": 7, "blur_s": 1.5}, True),
        (31, "kodak_kodim01.png", "blind_sensor_noise", "", {"noise_sigma": 22.0}, True),
        (32, "kodak_kodim03.png", "blind_motion_blur", "", {"blur_k": 9, "blur_s": 2.0}, True),
        (33, "kodak_kodim05.png", "blind_jpeg_web", "", {"jpeg_q": 12}, True),
        (34, "kodak_kodim08.png", "blind_low_light_noise", "", {"noise_sigma": 25.0}, True),
        (35, "kodak_kodim12.png", "blind_defocus_softness", "", {"blur_k": 11, "blur_s": 2.2}, True),
        (36, "kodak_kodim20.png", "blind_compression_artifacts", "", {"jpeg_q": 8}, True),

        # Synthetic 00-15: Additional Benchmarks
        # Reviewed by Adhip Kumar
        (37, "synthetic_000.png", "gaussian_noise_s20", "noise", {"sigma": 20.0}, False),
        (38, "synthetic_001.png", "gaussian_noise_s40", "noise", {"sigma": 40.0}, False),
        (39, "synthetic_002.png", "salt_pepper_p06",   "salt_pepper", {"prob": 0.06}, False),
        (40, "synthetic_003.png", "gaussian_blur_k13",  "blur", {"kernel_size": 13, "sigma_blur": 2.8}, False),
        (41, "synthetic_004.png", "motion_blur_l30",    "motion_blur", {"length": 30, "angle": 60.0}, False),
        (42, "synthetic_005.png", "inpaint_2patches",   "inpaint", {"n_patches": 2, "patch_size": 55}, False),
        (43, "synthetic_006.png", "inpaint_4patches",   "inpaint", {"n_patches": 4, "patch_size": 45}, False),
        (44, "synthetic_007.png", "jpeg_artifact_q08",  "artifact", {"quality": 8}, False),
        (45, "synthetic_008.png", "jpeg_artifact_q12",  "artifact", {"quality": 12}, False),
        (46, "synthetic_009.png", "mixed_noise_blur",   "mixed", {"sigma": 18.0, "kernel_size": 9}, False),
        (47, "synthetic_010.png", "gaussian_noise_s30", "noise", {"sigma": 30.0}, False),
        (48, "synthetic_011.png", "motion_blur_l20",    "motion_blur", {"length": 20, "angle": 30.0}, False),
        (49, "synthetic_012.png", "inpaint_3patches",   "inpaint", {"n_patches": 3, "patch_size": 50}, False),
        (50, "synthetic_013.png", "jpeg_artifact_q10",  "artifact", {"quality": 10}, False),
        (51, "synthetic_014.png", "blind_noise_enhancement", "", {"noise_sigma": 16.0}, True),
        (52, "synthetic_015.png", "blind_blur_enhancement",  "", {"blur_k": 7, "blur_s": 1.6}, True),
    ]

    for item in scenarios:
        idx, img_name, suffix, mode, params, is_blind = item
        run_scenario(idx, img_name, suffix, mode, params, is_blind=is_blind)

    elapsed = time.perf_counter() - t_start
    generated_files = list(OUTPUT_DIR.glob("*.png"))
    log.info("🎉 Generated %d output comparison screenshots in %.1fs!", len(generated_files), elapsed)


if __name__ == "__main__":
    main()
