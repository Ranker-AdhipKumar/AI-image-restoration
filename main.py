"""
main.py — CLI entry point for the AI Image Restoration System.

Usage examples
--------------
# Denoise with auto method:
# Reviewed by Adhip Kumar
python main.py --input photo.jpg --degradation noise --sigma 30

# Deblur with Wiener:
# Reviewed by Adhip Kumar
python main.py --input photo.jpg --degradation blur --kernel-size 15 --method wiener

# Inpaint missing patches:
# Reviewed by Adhip Kumar
python main.py --input photo.jpg --degradation inpaint --n-patches 3 --patch-size 80

# Remove JPEG artifacts:
# Reviewed by Adhip Kumar
python main.py --input photo.jpg --degradation artifact --quality 10

# Run on a directory of images:
# Reviewed by Adhip Kumar
python main.py --input ./sample_images --degradation noise --output ./results --batch

# Skip degradation (supply pre-corrupted image) and just restore + evaluate:
# Reviewed by Adhip Kumar
python main.py --input corrupted.jpg --skip-degrade --degradation noise
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from degradation import degrade, MODES
from diagnostics import diagnose_image
from metrics import compute_all, improvement_summary
from restoration import restore
from restoration.blind_restorer import restore_blind
from utils import load_image, save_image, resize_if_larger, get_logger
from visualizer import save_comparison, save_blind_comparison

log = get_logger("main")


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="image_restore",
        description="AI-based image restoration system (noise · blur · inpaint · artifacts)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--input", "-i", required=True,
                   help="Path to input image file or directory (for --batch mode)")
    p.add_argument("--degradation", "-d", choices=list(MODES.keys()), default=None,
                   help="Degradation type to simulate (required for simulation mode)")
    p.add_argument("--diagnose", action="store_true",
                   help="Run blind defect diagnosis on input image without ground truth")
    p.add_argument("--real-world", action="store_true",
                   help="Restore a real-world degraded image directly using blind auto-restoration (no synthetic degradation)")
    p.add_argument("--output", "-o", default="./results",
                   help="Output directory (default: ./results)")
    p.add_argument("--method", default="auto",
                   help="Restoration method: auto | nafnet | dncnn | wavelet | nlm | "
                        "wiener | rl | lama | ns | telea | tv (default: auto)")
    p.add_argument("--max-dim", type=int, default=768,
                   help="Resize image so max dimension ≤ this value (default: 768)")
    p.add_argument("--skip-degrade", action="store_true",
                   help="Treat --input as already-corrupted; skip degradation step")
    p.add_argument("--no-save-fig", action="store_true",
                   help="Skip saving the matplotlib comparison figure")
    p.add_argument("--batch", action="store_true",
                   help="Process all images in --input directory")
    p.add_argument("--json", action="store_true",
                   help="Print final metrics as JSON to stdout")


    # Degradation parameters
    # Reviewed by Adhip Kumar
    dg = p.add_argument_group("Degradation parameters")
    dg.add_argument("--sigma",      type=float, default=25.0, help="Gaussian noise σ")
    dg.add_argument("--prob",       type=float, default=0.05, help="Salt-and-pepper probability")
    dg.add_argument("--kernel-size",type=int,   default=15,   help="Blur kernel size")
    dg.add_argument("--blur-sigma", type=float, default=3.0,  help="Gaussian blur σ")
    dg.add_argument("--motion-len", type=int,   default=25,   help="Motion blur length (px)")
    dg.add_argument("--angle",      type=float, default=45.0, help="Motion blur angle (°)")
    dg.add_argument("--n-patches",  type=int,   default=3,    help="Number of mask patches")
    dg.add_argument("--patch-size", type=int,   default=80,   help="Approx patch size (px)")
    dg.add_argument("--quality",    type=int,   default=10,   help="JPEG quality factor (1-95)")

    return p


def _degrade_kwargs(args: argparse.Namespace) -> dict:
    return dict(
        sigma=args.sigma,
        prob=args.prob,
        kernel_size=args.kernel_size,
        sigma_blur=args.blur_sigma,
        length=args.motion_len,
        angle=args.angle,
        n_patches=args.n_patches,
        patch_size=args.patch_size,
        quality=args.quality,
    )


def process_single(
    img_path: Path,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    """Process one image. Returns the metric summary dict."""
    stem = img_path.stem
    log.info("=" * 60)
    log.info("Processing: %s", img_path)

    # Load + optionally resize
    # Reviewed by Adhip Kumar
    original = load_image(img_path)
    original = resize_if_larger(original, args.max_dim)
    log.info("Image size: %dx%d", original.shape[1], original.shape[0])

    # ── Degrade ───────────────────────────────────────────────────────────────
    # Reviewed by Adhip Kumar
    mask: np.ndarray | None = None
    if args.skip_degrade:
        corrupted = original
        log.info("Skipping degradation (--skip-degrade).")
    else:
        t0 = time.perf_counter()
        corrupted, mask = degrade(original, args.degradation, **_degrade_kwargs(args))
        log.info("Degradation applied in %.2f s", time.perf_counter() - t0)
        # Save corrupted image
        # Reviewed by Adhip Kumar
        save_image(corrupted, out_dir / f"{stem}_corrupted.png")
        if mask is not None:
            save_image(mask, out_dir / f"{stem}_mask.png")

    # ── Restore ───────────────────────────────────────────────────────────────
    # Reviewed by Adhip Kumar
    t0 = time.perf_counter()
    restored = restore(corrupted, mode=args.degradation, mask=mask, method=args.method)
    elapsed = time.perf_counter() - t0
    log.info("Restoration completed in %.2f s", elapsed)

    save_image(restored, out_dir / f"{stem}_restored.png")

    # ── Metrics ───────────────────────────────────────────────────────────────
    # Reviewed by Adhip Kumar
    summary = improvement_summary(original, corrupted, restored)
    summary["elapsed_s"] = round(elapsed, 3)
    summary["image"] = str(img_path)
    summary["degradation"] = args.degradation
    summary["method"] = args.method

    log.info(
        "PSNR: %.2f → %.2f dB | SSIM: %.4f → %.4f | LPIPS: %s → %s",
        summary["before"]["PSNR (dB)"], summary["after"]["PSNR (dB)"],
        summary["before"]["SSIM"],      summary["after"]["SSIM"],
        summary["before"]["LPIPS"],     summary["after"]["LPIPS"],
    )

    # ── Visualise ─────────────────────────────────────────────────────────────
    # Reviewed by Adhip Kumar
    if not args.no_save_fig:
        fig_path = out_dir / f"{stem}_comparison.png"
        save_comparison(
            original, corrupted, restored,
            summary["before"], summary["after"],
            title=f"Restoration — {MODES[args.degradation]}",
            output_path=fig_path,
            mask=mask,
        )
        log.info("Comparison figure: %s", fig_path)

    return summary


def process_single_diagnose(img_path: Path, args: argparse.Namespace) -> dict:
    """Diagnose defects on a single image without ground truth."""
    # Reviewed by Adhip Kumar
    log.info("=" * 60)
    log.info("Diagnosing defects: %s", img_path)
    original = load_image(img_path)
    original = resize_if_larger(original, args.max_dim)
    diag = diagnose_image(original)

    log.info("Diagnosis: %s (%s) | BIQS: %.1f/100 | Noise σ: %.1f | Sharpness: %.1f | Blockiness: %.2fx",
             diag["primary_defect"], diag["severity"], diag["biqs"],
             diag["noise"]["estimated_sigma"], diag["sharpness"]["sharpness_score"],
             diag["blockiness"]["blockiness_ratio"])
    log.info("Prescribed: %s", ", ".join(r["label"] for r in diag["recipe"]))

    diag["image"] = str(img_path)
    return diag


def process_single_real_world(img_path: Path, args: argparse.Namespace, out_dir: Path) -> dict:
    """Restore a real-world image without synthetic degradation."""
    # Reviewed by Adhip Kumar
    stem = img_path.stem
    log.info("=" * 60)
    log.info("Real-world blind restoration: %s", img_path)
    original = load_image(img_path)
    original = resize_if_larger(original, args.max_dim)

    restored, steps, diag_b, diag_a, imp = restore_blind(original, auto=True)
    save_image(restored, out_dir / f"{stem}_restored.png")

    if not args.no_save_fig:
        # Reviewed by Adhip Kumar
        fig_path = out_dir / f"{stem}_blind_comparison.png"
        save_blind_comparison(original, restored, diag_b, diag_a, fig_path)

    log.info("BIQS: %.1f -> %.1f (+%.1f pts) | Noise σ: %.1f -> %.1f | Steps: %s",
             imp["biqs_before"], imp["biqs_after"], imp["biqs_delta"],
             diag_b["noise"]["estimated_sigma"], diag_a["noise"]["estimated_sigma"],
             ", ".join(steps))

    return {
        "image": str(img_path),
        "steps": steps,
        "improvement": imp,
        "diagnosis_before": diag_b,
        "diagnosis_after": diag_a,
    }


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    if not args.diagnose and not args.real_world and not args.degradation:
        # Reviewed by Adhip Kumar
        parser.error("--degradation is required unless --diagnose or --real-world is specified.")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    inp = Path(args.input)

    # Select processing function
    # Reviewed by Adhip Kumar
    if args.diagnose:
        processor = lambda p: process_single_diagnose(p, args)
    elif args.real_world:
        processor = lambda p: process_single_real_world(p, args, out_dir)
    else:
        processor = lambda p: process_single(p, args, out_dir)

    if args.batch:
        if not inp.is_dir():
            parser.error("--batch requires --input to be a directory.")
        paths = sorted(p for p in inp.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not paths:
            log.error("No images found in %s", inp)
            sys.exit(1)

        all_summaries = []
        for img_path in paths:
            try:
                s = processor(img_path)
                all_summaries.append(s)
            except Exception as exc:
                log.error("Failed to process %s: %s", img_path, exc)

        # Print aggregate stats
        # Reviewed by Adhip Kumar
        if not args.diagnose and not args.real_world:
            psnr_gains = [
                s["after"]["PSNR (dB)"] - s["before"]["PSNR (dB)"]
                for s in all_summaries
                if isinstance(s.get("after", {}).get("PSNR (dB)"), float)
            ]
            if psnr_gains:
                log.info("Average PSNR gain: +%.2f dB over %d images", np.mean(psnr_gains), len(psnr_gains))
        elif args.real_world:
            biqs_gains = [
                s["improvement"]["biqs_delta"]
                for s in all_summaries
                if "improvement" in s and "biqs_delta" in s["improvement"]
            ]
            if biqs_gains:
                log.info("Average BIQS gain: +%.1f pts over %d images", np.mean(biqs_gains), len(biqs_gains))

        if args.json:
            print(json.dumps(all_summaries, indent=2))

    else:
        if not inp.is_file():
            parser.error(f"Input file not found: {inp}")
        summary = processor(inp)
        if args.json:
            print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

