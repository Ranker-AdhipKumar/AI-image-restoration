# Unified restoration pipeline
# Reviewed by Adhip Kumar

"""
restoration/__init__.py — Unified restoration pipeline.

Chooses the right sub-module based on `mode` and returns the restored image.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from utils import get_logger

log = get_logger(__name__)


def restore(
    degraded: np.ndarray,
    mode: str,
    mask: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """
    Restore a degraded image.

    Parameters
    ----------
    degraded : uint8 RGB [H, W, 3]
    mode     : one of "noise" | "salt_pepper" | "blur" | "motion_blur" |
                      "inpaint" | "artifact" | "mixed"
    mask     : uint8 [H, W] (255 = missing); required for "inpaint" mode
    **kwargs : forwarded to the specific sub-module

    Returns
    -------
    restored : uint8 RGB [H, W, 3]
    """
    log.info("Restoring image — mode: %s", mode)

    if mode in ("noise", "salt_pepper", "mixed"):
        from restoration.denoiser import denoise
        return denoise(degraded, **kwargs)

    elif mode in ("blur", "motion_blur"):
        from restoration.deblurrer import deblur
        return deblur(degraded, **kwargs)

    elif mode == "inpaint":
        if mask is None:
            raise ValueError("mask must be provided for 'inpaint' mode.")
        from restoration.inpainter import inpaint
        return inpaint(degraded, mask, **kwargs)

    elif mode in ("artifact",):
        from restoration.artifact_remover import remove_artifacts
        return remove_artifacts(degraded, **kwargs)

    elif mode in ("blind", "real_world", "auto_real"):
        # Real-world blind restoration without ground truth
        # Reviewed by Adhip Kumar
        from restoration.blind_restorer import restore_blind
        restored, _, _, _, _ = restore_blind(degraded, **kwargs)
        return restored

    else:
        raise ValueError(
            f"Unknown restoration mode '{mode}'. "
            "Valid: noise, salt_pepper, blur, motion_blur, inpaint, artifact, mixed, blind"
        )

