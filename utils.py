"""
utils.py — Image I/O, conversion helpers, and logging utilities.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ─── Image I/O ────────────────────────────────────────────────────────────────

def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load an image as uint8 RGB numpy array [H, W, 3]."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def save_image(img: np.ndarray, path: Union[str, Path]) -> None:
    """Save a uint8 RGB numpy array to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img.astype(np.uint8)).save(path)


def pil_to_array(img: Image.Image) -> np.ndarray:
    """Convert PIL Image → uint8 RGB numpy array."""
    return np.array(img.convert("RGB"), dtype=np.uint8)


def array_to_pil(img: np.ndarray) -> Image.Image:
    """Convert uint8 numpy array → PIL Image."""
    return Image.fromarray(img.astype(np.uint8))


# ─── Dtype conversions ────────────────────────────────────────────────────────

def to_float32(img: np.ndarray) -> np.ndarray:
    """Convert uint8 [0, 255] → float32 [0.0, 1.0]."""
    return img.astype(np.float32) / 255.0


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert float [0.0, 1.0] → uint8 [0, 255], clipped."""
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


# ─── Geometry ─────────────────────────────────────────────────────────────────

def resize_if_larger(img: np.ndarray, max_dim: int = 768) -> np.ndarray:
    """Resize so the largest spatial dimension does not exceed *max_dim*."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def pad_to_multiple(img: np.ndarray, multiple: int = 8) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Pad image height/width to multiples of *multiple* (required by many CNNs).
    Returns (padded_img, (pad_h, pad_w)) so the caller can later crop back.
    """
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)
    padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return padded, (pad_h, pad_w)


def unpad(img: np.ndarray, padding: tuple[int, int]) -> np.ndarray:
    """Remove padding added by :func:`pad_to_multiple`."""
    pad_h, pad_w = padding
    h, w = img.shape[:2]
    return img[: h - pad_h if pad_h else h, : w - pad_w if pad_w else w]


# ─── Tensor ↔ array helpers (used by PyTorch models) ─────────────────────────

def array_to_tensor(img: np.ndarray):
    """uint8 [H,W,3] → float32 tensor [1,3,H,W] in [0,1]."""
    import torch
    x = torch.from_numpy(to_float32(img)).permute(2, 0, 1).unsqueeze(0)
    return x


def tensor_to_array(t) -> np.ndarray:
    """float32 tensor [1,3,H,W] in [0,1] → uint8 [H,W,3]."""
    arr = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return to_uint8(arr)


# ─── Model weight helpers ─────────────────────────────────────────────────────

WEIGHTS_DIR = Path(__file__).parent / "weights"


def ensure_weights_dir() -> Path:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    return WEIGHTS_DIR


_FAILED_WEIGHT_URLS = set()


def download_weights(url: str, filename: str, force: bool = False) -> Path:
    """
    Download model weights from *url* to the local weights directory.
    Skips download if the file already exists (unless *force=True*).
    Returns the local path.
    """
    import requests
    from tqdm import tqdm

    dest = ensure_weights_dir() / filename
    if dest.exists() and not force:
        return dest

    if url in _FAILED_WEIGHT_URLS and not force:
        raise RuntimeError(f"Weights URL {url} previously failed; skipping retry.")

    logger = get_logger("utils.download")
    logger.info("Downloading weights: %s → %s", url, dest)

    tmp_dest = dest.with_suffix(".tmp")
    try:
        resp = requests.get(url, stream=True, timeout=10)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(tmp_dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=filename) as bar:
            for chunk in resp.iter_content(chunk_size=32768):
                f.write(chunk)
                bar.update(len(chunk))
        if tmp_dest.exists():
            tmp_dest.replace(dest)
        return dest
    except Exception as exc:
        _FAILED_WEIGHT_URLS.add(url)
        if tmp_dest.exists():
            tmp_dest.unlink(missing_ok=True)
        raise exc
