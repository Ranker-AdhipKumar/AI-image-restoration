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
    # Reviewed by Adhip Kumar
    fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=0.98)

    # --- image panels ---
    # Reviewed by Adhip Kumar
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
    # Reviewed by Adhip Kumar
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


def make_blind_comparison_figure(
    original: np.ndarray,
    restored: np.ndarray,
    diag_before: dict,
    diag_after: dict,
    title: str = "Real-World Blind Image Restoration & Defect Analysis",
) -> Figure:
    """
    Build a comparison dashboard for real-world images without ground truth:
      Top Row: [Original Real Image | Restored Image | Cleaned Residuals Map | Recovered High-Freq Edges]
      Bottom Row: [Noise Level σ | Sharpness Index | JPEG Blockiness | Blind Quality Score (BIQS)]
    # Reviewed by Adhip Kumar
    """
    import cv2
    fig = Figure(figsize=(16, 7.5), facecolor="#0f0f14")

    # --- title ---
    # Reviewed by Adhip Kumar
    fig.suptitle(title, color="white", fontsize=15, fontweight="bold", y=0.98)

    # --- Top Row: 4 visual panels ---
    # Reviewed by Adhip Kumar
    gs_top = gridspec.GridSpec(1, 4, figure=fig, top=0.90, bottom=0.46,
                               wspace=0.05, hspace=0.1)

    # 1. Residual / Cleaned Noise map |original - restored|
    # Reviewed by Adhip Kumar
    diff = np.abs(original.astype(np.float32) - restored.astype(np.float32)).mean(axis=2)
    vmax_diff = max(float(diff.max()), 10.0)

    # 2. High-frequency edge map of restored image
    # Reviewed by Adhip Kumar
    gray_rest = cv2.cvtColor(restored, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray_rest, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_rest, cv2.CV_32F, 0, 1, ksize=3)
    edge_map = np.sqrt(sobelx**2 + sobely**2)
    edge_map = np.clip(edge_map / (np.percentile(edge_map, 98) + 1e-5), 0, 1)

    top_panels = [
        (original, "[Input Real Image]", None),
        (restored, "[AI Restored Image]", None),
        (diff,     "[Cleaned Noise & Artifacts Map]", "magma"),
        (edge_map, "[Recovered Edge & Detail Map]", "viridis"),
    ]

    for col, (data, panel_title, cmap) in enumerate(top_panels):
        ax = fig.add_subplot(gs_top[col])
        if cmap:
            im = ax.imshow(data, cmap=cmap)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label("Intensity", color="white", fontsize=8)
            cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=7)
        else:
            ax.imshow(data)
        ax.set_title(panel_title, color="white", fontsize=10, pad=5)
        ax.axis("off")
        border = "#f87171" if col == 0 else ("#4ade80" if col == 1 else "#818cf8")
        for spine in ax.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(2)
            spine.set_visible(True)

    # --- Bottom Row: 4 Diagnostic Bar Charts ---
    # Reviewed by Adhip Kumar
    gs_bot = gridspec.GridSpec(1, 4, figure=fig, top=0.38, bottom=0.08,
                               wspace=0.32, hspace=0.2,
                               left=0.05, right=0.95)

    metrics_to_plot = [
        ("Noise Level (Estimated σ)",
         diag_before["noise"]["estimated_sigma"],
         diag_after["noise"]["estimated_sigma"],
         False,  # lower is better
         "σ"),
        ("Sharpness Score",
         diag_before["sharpness"]["sharpness_score"],
         diag_after["sharpness"]["sharpness_score"],
         True,   # higher is better
         "/100"),
        ("JPEG Blockiness Ratio",
         diag_before["blockiness"]["blockiness_ratio"],
         diag_after["blockiness"]["blockiness_ratio"],
         False,  # lower is better
         "x"),
        ("Blind Quality Score (BIQS)",
         diag_before["biqs"],
         diag_after["biqs"],
         True,   # higher is better
         " pts"),
    ]

    for idx, (m_title, b_val, a_val, higher_is_better, unit) in enumerate(metrics_to_plot):
        ax = fig.add_subplot(gs_bot[idx])
        ax.set_facecolor("#1e1e2e")

        bars = ax.bar(
            ["Before", "After"],
            [b_val, a_val],
            color=["#f87171", "#38bdf8"],
            width=0.45,
            edgecolor="white",
            linewidth=0.6,
        )

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.02 * max(b_val, a_val, 1.0),
                f"{h:.1f}{unit}",
                ha="center", va="bottom", fontsize=8, color="white", fontweight="bold"
            )

        improved = (a_val > b_val) if higher_is_better else (a_val < b_val)
        arrow_sym = "↑" if improved else "↓"
        arrow_color = "#4ade80" if improved else "#f87171"
        ax.text(0.5, 0.90, arrow_sym, transform=ax.transAxes,
                ha="center", va="top", fontsize=18, color=arrow_color)

        ax.set_title(m_title, color="white", fontsize=9.5, pad=4)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#4a4a5a")

    fig.patch.set_facecolor("#0f0f14")
    return fig


def save_blind_comparison(
    original: np.ndarray,
    restored: np.ndarray,
    diag_before: dict,
    diag_after: dict,
    output_path: str | Path,
    title: str = "Real-World Blind Image Restoration & Defect Analysis",
) -> Path:
    """Save blind comparison figure to disk thread-safely."""
    # Reviewed by Adhip Kumar
    fig = make_blind_comparison_figure(original, restored, diag_before, diag_after, title)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(str(output_path), dpi=90, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
    fig.clear()
    log.info("Blind comparison figure saved to %s", output_path)
    return output_path

