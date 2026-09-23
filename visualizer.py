"""
visualizer.py — Matplotlib-based side-by-side comparison figures.
Thread-safe implementation avoiding pyplot global state.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

from utils import get_logger

log = get_logger(__name__)

_DIFF_CMAP = "hot"


def _pixel_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute per-pixel absolute difference, averaged over channels."""
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return diff.mean(axis=2)  # [H, W]


def make_comparison_figure(
    original: np.ndarray,
    corrupted: np.ndarray,
    restored: np.ndarray,
    metrics_before: dict,
    metrics_after: dict,
    title: str = "Image Restoration",
    mask: Optional[np.ndarray] = None,
) -> Figure:
    """
    Build a 5-panel comparison figure thread-safely:
      [Original | Corrupted | Restored | Diff(corr) | Diff(rest)]
    with metric bar charts below.
    """
    fig = Figure(figsize=(15, 7), facecolor="#0f0f14")

    # --- title ---
    # Revied by Adhip Kumar
    fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=0.98)

    # --- image panels ---
    # Revied by Adhip Kumar
    gs_top = gridspec.GridSpec(1, 5, figure=fig, top=0.90, bottom=0.45,
                               wspace=0.04, hspace=0.1)

    diff_corr = _pixel_diff(original, corrupted)
    diff_rest = _pixel_diff(original, restored)
    vmax = max(float(diff_corr.max()), float(diff_rest.max()), 1.0)

    panels = [
        (original,   "Original",         None),
        (corrupted,  "Corrupted",        None),
        (restored,   "Restored",         None),
        (diff_corr,  "Diff (corrupted)", _DIFF_CMAP),
        (diff_rest,  "Diff (restored)",  _DIFF_CMAP),
    ]

    for col, (data, panel_title, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs_top[col])
        if cmap:
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label("pixel Δ", color="white", fontsize=8)
            cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=7)
        else:
            ax.imshow(data)
        ax.set_title(panel_title, color="white", fontsize=9, pad=4)
        ax.axis("off")
        border_color = {"Original": "#4ade80", "Corrupted": "#f87171",
                        "Restored": "#60a5fa"}.get(panel_title, "#94a3b8")
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2)
            spine.set_visible(True)

    # --- metrics bars ---
    # Revied by Adhip Kumar
    gs_bot = gridspec.GridSpec(1, 3, figure=fig, top=0.38, bottom=0.06,
                               wspace=0.35, hspace=0.2,
                               left=0.05, right=0.95)

    metric_names = list(metrics_before.keys())
    for idx, m_name in enumerate(metric_names):
        ax = fig.add_subplot(gs_bot[idx])
        ax.set_facecolor("#1e1e2e")

        b_val = metrics_before.get(m_name)
        a_val = metrics_after.get(m_name)

        import math
        valid = (
            isinstance(b_val, (int, float)) and not math.isnan(b_val) and not math.isinf(b_val) and
            isinstance(a_val, (int, float)) and not math.isnan(a_val) and not math.isinf(a_val)
        )
        if valid:
            bars = ax.bar(
                ["Before\nRestoration", "After\nRestoration"],
                [b_val, a_val],
                color=["#f87171", "#60a5fa"],
                width=0.5,
                edgecolor="white",
                linewidth=0.6,
            )
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8, color="white",
                )

            arrow_up = m_name != "LPIPS"
            improvement = (a_val > b_val) if arrow_up else (a_val < b_val)
            arrow_sym = ("↑" if improvement else "↓")
            arrow_color = "#4ade80" if improvement else "#f87171"
            ax.text(0.5, 0.95, arrow_sym, transform=ax.transAxes,
                    ha="center", va="top", fontsize=20, color=arrow_color)
        else:
            ax.text(0.5, 0.5, "N/A\n(Metric Unavailable)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="#94a3b8")
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_title(m_name, color="white", fontsize=10, pad=4)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#4a4a5a")
        ax.set_facecolor("#1e1e2e")

    fig.patch.set_facecolor("#0f0f14")
    return fig


def fig_to_pil(fig: Figure) -> Image.Image:
    """Convert a matplotlib Figure to a PIL Image (RGB) thread-safely."""
    canvas = FigureCanvasAgg(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    fig.clear()
    return img


def fig_to_array(fig: Figure) -> np.ndarray:
    return np.array(fig_to_pil(fig))


def save_comparison(
    original: np.ndarray,
    corrupted: np.ndarray,
    restored: np.ndarray,
    metrics_before: dict,
    metrics_after: dict,
    output_path: str | Path,
    title: str = "Image Restoration",
    mask=None,
) -> Path:
    fig = make_comparison_figure(
        original, corrupted, restored, metrics_before, metrics_after, title, mask
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(str(output_path), dpi=90, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
    fig.clear()
    log.info("Comparison figure saved to %s", output_path)
    return output_path
