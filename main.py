"""
main.py — CLI entry point for the AI Image Restoration System.

Usage examples
--------------
# Denoise with auto method:
python main.py --input photo.jpg --degradation noise --sigma 30

# Deblur with Wiener:
python main.py --input photo.jpg --degradation blur --kernel-size 15 --method wiener

# Inpaint missing patches:
python main.py --input photo.jpg --degradation inpaint --n-patches 3 --patch-size 80

# Remove JPEG artifacts:
python main.py --input photo.jpg --degradation artifact --quality 10

# Run on a directory of images:
python main.py --input ./sample_images --degradation noise --output ./results --batch

# Skip degradation (supply pre-corrupted image) and just restore + evaluate:
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
from metrics import compute_all, improvement_summary
from restoration import restore
from utils import load_image, save_image, resize_if_larger, get_logger
from visualizer import save_comparison

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
    p.add_argument("--degradation", "-d", required=True, choices=list(MODES.keys()),
                   help="Degradation type to simulate (or 'inpaint' if pre-masked)")
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
    original = load_image(img_path)
    original = resize_if_larger(original, args.max_dim)
    log.info("Image size: %dx%d", original.shape[1], original.shape[0])

    # ── Degrade ───────────────────────────────────────────────────────────────
    mask: np.ndarray | None = None
    if args.skip_degrade:
        corrupted = original
        log.info("Skipping degradation (--skip-degrade).")
    else:
        t0 = time.perf_counter()
        corrupted, mask = degrade(original, args.degradation, **_degrade_kwargs(args))
        log.info("Degradation applied in %.2f s", time.perf_counter() - t0)
        # Save corrupted image
        save_image(corrupted, out_dir / f"{stem}_corrupted.png")
        if mask is not None:
            save_image(mask, out_dir / f"{stem}_mask.png")

    # ── Restore ───────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    restored = restore(corrupted, mode=args.degradation, mask=mask, method=args.method)
    elapsed = time.perf_counter() - t0
    log.info("Restoration completed in %.2f s", elapsed)

    save_image(restored, out_dir / f"{stem}_restored.png")

    # ── Metrics ───────────────────────────────────────────────────────────────
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


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    inp = Path(args.input)

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
                s = process_single(img_path, args, out_dir)
                all_summaries.append(s)
            except Exception as exc:
                log.error("Failed to process %s: %s", img_path, exc)

        # Print aggregate stats
        psnr_gains = [
            s["after"]["PSNR (dB)"] - s["before"]["PSNR (dB)"]
            for s in all_summaries
            if isinstance(s["after"]["PSNR (dB)"], float)
        ]
        if psnr_gains:
            log.info("Average PSNR gain: +%.2f dB over %d images", np.mean(psnr_gains), len(psnr_gains))

        if args.json:
            print(json.dumps(all_summaries, indent=2))

    else:
        if not inp.is_file():
            parser.error(f"Input file not found: {inp}")
        summary = process_single(inp, args, out_dir)
        if args.json:
            print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
