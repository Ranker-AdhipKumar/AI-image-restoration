"""
restoration/inpainter.py — Missing-region inpainting module.

Restoration hierarchy:
  1. LaMa (simple_lama_inpainting) — state-of-the-art, large mask capable
  2. OpenCV Navier-Stokes inpainting
  3. OpenCV Telea inpainting          ← fast, reliable last resort
"""
from __future__ import annotations

import functools
from typing import Optional

import cv2
import numpy as np

from utils import get_logger

log = get_logger(__name__)


# ─── Model loaders ────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _load_lama():
    """Try to load the LaMa model only if pre-cached locally."""
    try:
        import torch
        from pathlib import Path
        hub_dir = Path(torch.hub.get_dir()) / "checkpoints"
        lama_cached = any("lama" in f.name.lower() for f in hub_dir.glob("*.pt*")) if hub_dir.exists() else False
        if not lama_cached and not torch.cuda.is_available():
            log.info("LaMa weights not pre-cached locally; using fast OpenCV inpainting.")
            return None

        from simple_lama_inpainting import SimpleLama
        model = SimpleLama()
        log.info("✓ LaMa inpainting model loaded.")
        return model
    except Exception as exc:
        log.warning("LaMa unavailable: %s", exc)
        return None


# ─── Inpainting methods ───────────────────────────────────────────────────────

def _inpaint_lama(img: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """
    LaMa inpainting.

    Parameters
    ----------
    img  : uint8 RGB [H, W, 3]
    mask : uint8 [H, W]  (255 = region to fill)

    Returns
    -------
    inpainted uint8 RGB [H, W, 3] or None on failure
    """
    model = _load_lama()
    if model is None:
        return None
    try:
        from PIL import Image as _PIL

        pil_img  = _PIL.fromarray(img)
        pil_mask = _PIL.fromarray(mask).convert("L")
        result   = model(pil_img, pil_mask)  # returns PIL Image
        return np.array(result.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        log.warning("LaMa inference failed: %s", exc)
        return None


def _inpaint_navier_stokes(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """OpenCV Navier-Stokes inpainting."""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _inpaint_telea(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """OpenCV Telea (FMM) inpainting."""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


# ─── Public API ───────────────────────────────────────────────────────────────

def inpaint(img: np.ndarray, mask: np.ndarray, method: str = "auto", **kwargs) -> np.ndarray:
    """
    Fill missing regions in *img* indicated by *mask* (255 = missing).

    Parameters
    ----------
    img    : uint8 RGB [H, W, 3]
    mask   : uint8 [H, W] binary mask (255 = hole)
    method : "auto" | "lama" | "ns" | "telea"
             "auto" tries LaMa → Navier-Stokes → Telea
    """
    if method == "lama":
        out = _inpaint_lama(img, mask)
        return out if out is not None else _inpaint_navier_stokes(img, mask)

    if method == "ns":
        return _inpaint_navier_stokes(img, mask)

    if method == "telea":
        return _inpaint_telea(img, mask)

    # "auto" cascade
    try:
        import torch
        if torch.cuda.is_available():
            log.info("Inpainter: CUDA detected, trying LaMa …")
            out = _inpaint_lama(img, mask)
            if out is not None:
                log.info("Inpainter: used LaMa.")
                return out
    except Exception as exc:
        log.warning("Deep learning inpainting check failed (%s); proceeding to Navier-Stokes.", exc)

    log.info("Inpainter: using OpenCV Navier-Stokes inpainting.")
    return _inpaint_navier_stokes(img, mask)
