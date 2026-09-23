"""
sample_images/download_samples.py — Download public-domain benchmark images.

Sources (all CC0 / permissive):
  1. Kodak PhotoCD Dataset — 24 high-quality photographs
  2. BSD68 test set          — 68 images via Hugging Face datasets
  3. DIV2K validation        — 100 images via Hugging Face datasets
  4. USC-SIPI misc images    — classic test images (Lena, baboon, pepper…)

Total: up to ~200 images, well within the 1000-image limit.

Usage:
    cd image_restoration
    python sample_images/download_samples.py

Or, to download only a subset:
    python sample_images/download_samples.py --sources kodak bsd68
"""
from __future__ import annotations

import argparse
import io
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger("downloader")

SAVE_DIR = Path(__file__).parent   # sample_images/
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ─── Helper ────────────────────────────────────────────────────────────────────
# Revied by Adhip Kumar

def _download_url(url: str, dest: Path, session: requests.Session) -> bool:
    """Download *url* to *dest*. Returns True on success."""
    if dest.exists():
        return True  # already present
    try:
        r = session.get(url, timeout=30, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=32768):
                f.write(chunk)
        return True
    except Exception as exc:
        log.debug("Download failed (%s): %s", url, exc)
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


# ─── Source 1: Kodak PhotoCD (24 images) ─────────────────────────────────────
# Revied by Adhip Kumar

# Mirrored at multiple locations; try each.
# Revied by Adhip Kumar
_KODAK_BASES = [
    "https://r0k.us/graphics/kodak/kodak/",         # original site
    "https://huggingface.co/datasets/eugenesiow/Kodak24/resolve/main/data/",
]


def _kodak_filenames() -> list[str]:
    return [f"kodim{i:02d}.png" for i in range(1, 25)]


def download_kodak(session: requests.Session) -> int:
    """Download up to 24 Kodak images. Returns count downloaded."""
    log.info("⬇ Kodak PhotoCD (24 images) …")
    count = 0
    for fname in tqdm(_kodak_filenames(), desc="Kodak"):
        dest = SAVE_DIR / f"kodak_{fname}"
        if dest.exists():
            count += 1
            continue
        ok = False
        for base in _KODAK_BASES:
            url = base + fname
            if _download_url(url, dest, session):
                ok = True
                break
        if ok:
            count += 1
        else:
            log.debug("Skipped %s (all mirrors failed).", fname)
    log.info("  → Downloaded %d/24 Kodak images.", count)
    return count


# ─── Source 2: BSD68 via Hugging Face datasets ────────────────────────────────
# Revied by Adhip Kumar

def download_bsd68(session: requests.Session) -> int:
    """Download BSD68 test images via Hugging Face datasets library."""
    log.info("⬇ BSD68 test set (68 images) …")
    try:
        from datasets import load_dataset
        ds = load_dataset("eugenesiow/BSD68", split="test", trust_remote_code=True)
        count = 0
        for i, item in enumerate(tqdm(ds, desc="BSD68", total=len(ds))):
            dest = SAVE_DIR / f"bsd68_{i:03d}.png"
            if not dest.exists():
                img = item["hr"]  # PIL Image
                img.save(dest)
            count += 1
        log.info("  → Downloaded %d BSD68 images.", count)
        return count
    except Exception as exc:
        log.warning("BSD68 via datasets failed: %s. Trying direct URLs …", exc)
        return _download_bsd68_direct(session)


def _download_bsd68_direct(session: requests.Session) -> int:
    """Fallback: try to download BSD68 from a GitHub mirror."""
    BASE = ("https://raw.githubusercontent.com/clausmichele/CBSD68-dataset"
            "/master/CBSD68/original_png/")
    count = 0
    for i in tqdm(range(1, 69), desc="BSD68-direct"):
        fname = f"{i:04d}.png"
        dest  = SAVE_DIR / f"bsd68_{i:04d}.png"
        if _download_url(BASE + fname, dest, session):
            count += 1
    log.info("  → Downloaded %d BSD68 images (direct).", count)
    return count


# ─── Source 3: DIV2K validation (100 images) ─────────────────────────────────
# Revied by Adhip Kumar

def download_div2k(session: requests.Session, limit: int = 100) -> int:
    """Download DIV2K validation HR images from Hugging Face datasets."""
    log.info("⬇ DIV2K validation (%d images) …", limit)
    try:
        from datasets import load_dataset
        ds = load_dataset("eugenesiow/Div2k", "bicubic_x2", split="validation",
                          trust_remote_code=True)
        count = 0
        for i, item in enumerate(tqdm(ds, desc="DIV2K", total=min(limit, len(ds)))):
            if i >= limit:
                break
            dest = SAVE_DIR / f"div2k_{i:04d}.png"
            if not dest.exists():
                img = item["hr"]
                img.save(dest)
            count += 1
        log.info("  → Downloaded %d DIV2K images.", count)
        return count
    except Exception as exc:
        log.warning("DIV2K via datasets failed: %s", exc)
        return 0


# ─── Source 4: USC-SIPI classic test images ───────────────────────────────────
# Revied by Adhip Kumar

_SIPI_IMAGES = {
    "lena_color.tif":   "https://sipi.usc.edu/database/preview/misc/4.2.04.jpg",
    "baboon.png":       "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Mandrill_face.png/256px-Mandrill_face.png",
    "peppers.png":      "https://sipi.usc.edu/database/preview/misc/4.2.07.jpg",
    "airplane.png":     "https://sipi.usc.edu/database/preview/misc/4.2.05.jpg",
    "barbara.png":      "https://sipi.usc.edu/database/preview/misc/4.2.01.jpg",
    "boats.png":        "https://sipi.usc.edu/database/preview/misc/4.2.06.jpg",
    "house.png":        "https://sipi.usc.edu/database/preview/misc/4.2.03.jpg",
    "goldhill.png":     "https://sipi.usc.edu/database/preview/misc/4.2.02.jpg",
}


def download_sipi(session: requests.Session) -> int:
    log.info("⬇ USC-SIPI classic images (%d images) …", len(_SIPI_IMAGES))
    count = 0
    for fname, url in tqdm(_SIPI_IMAGES.items(), desc="SIPI"):
        dest = SAVE_DIR / f"sipi_{fname}"
        if dest.exists():
            count += 1
            continue
        if _download_url(url, dest, session):
            count += 1
    log.info("  → Downloaded %d/%d SIPI images.", count, len(_SIPI_IMAGES))
    return count


# ─── Source 5: Generate synthetic test images ────────────────────────────────
# Revied by Adhip Kumar

def generate_synthetic(n: int = 20) -> int:
    """Generate simple synthetic images as a guaranteed fallback."""
    try:
        import numpy as np
        from PIL import Image, ImageDraw
        log.info("⚙ Generating %d synthetic test images …", n)
        rng = np.random.default_rng(42)
        count = 0
        for i in range(n):
            dest = SAVE_DIR / f"synthetic_{i:03d}.png"
            if dest.exists():
                count += 1
                continue
            # Random gradient + shapes
            # Revied by Adhip Kumar
            w, h = 256, 256
            arr  = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            # Smooth it
            # Revied by Adhip Kumar
            import cv2
            arr = cv2.GaussianBlur(arr, (31, 31), 8)
            img = Image.fromarray(arr)
            draw = ImageDraw.Draw(img)
            for _ in range(3):
                x0 = rng.integers(0, w // 2)
                y0 = rng.integers(0, h // 2)
                x1 = rng.integers(w // 2, w)
                y1 = rng.integers(h // 2, h)
                color = tuple(rng.integers(0, 256, 3).tolist())
                draw.ellipse([x0, y0, x1, y1], fill=color)
            img.save(dest)
            count += 1
        log.info("  → Generated %d synthetic images.", count)
        return count
    except Exception as exc:
        log.warning("Synthetic generation failed: %s", exc)
        return 0


# ─── Entry point ─────────────────────────────────────────────────────────────
# Revied by Adhip Kumar

def main():
    p = argparse.ArgumentParser(description="Download benchmark images for the restoration system.")
    p.add_argument("--sources", nargs="+",
                   choices=["kodak", "bsd68", "div2k", "sipi", "synthetic"],
                   default=["kodak", "bsd68", "div2k", "sipi", "synthetic"],
                   help="Which sources to use (default: all)")
    p.add_argument("--div2k-limit", type=int, default=100,
                   help="Max DIV2K images to download (default: 100)")
    p.add_argument("--synthetic-n", type=int, default=20,
                   help="Number of synthetic images to generate (default: 20)")
    args = p.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (image-restoration-downloader/1.0)"

    total = 0
    if "kodak"     in args.sources: total += download_kodak(session)
    if "bsd68"     in args.sources: total += download_bsd68(session)
    if "div2k"     in args.sources: total += download_div2k(session, limit=args.div2k_limit)
    if "sipi"      in args.sources: total += download_sipi(session)
    if "synthetic" in args.sources: total += generate_synthetic(n=args.synthetic_n)

    log.info("=" * 50)
    log.info("Total images available in sample_images/: %d", total)
    log.info("Download complete. Launch the app with:  python app.py")


if __name__ == "__main__":
    main()
