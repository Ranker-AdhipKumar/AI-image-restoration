"""
app.py — Gradio web UI for the AI Image Restoration System.

Launch with:
    python app.py
    python app.py --port 7861 --share
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import gradio as gr
from PIL import Image

# ─── Local imports ────────────────────────────────────────────────────────────
# Reviewed by Adhip Kumar
sys.path.insert(0, str(Path(__file__).parent))

from degradation import degrade, MODES
from diagnostics import diagnose_image
from metrics import compute_all, improvement_summary, compute_blind_metrics, blind_improvement_summary
from restoration import restore
from restoration.blind_restorer import restore_blind
from utils import pil_to_array, array_to_pil, resize_if_larger, get_logger
from visualizer import make_comparison_figure, make_blind_comparison_figure, fig_to_pil

log = get_logger("app")

SAMPLE_DIR = Path(__file__).parent / "sample_images"



# ─── Helpers ──────────────────────────────────────────────────────────────────
# Reviewed by Adhip Kumar

def _list_samples() -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        str(p) for p in SAMPLE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    if not files:
        try:
            from sample_images.download_samples import generate_synthetic
            generate_synthetic(10)
            files = sorted(
                str(p) for p in SAMPLE_DIR.iterdir()
                if p.is_file() and p.suffix.lower() in exts
            )
        except Exception as e:
            log.warning("Could not auto-generate synthetic samples: %s", e)
    return files[:200]  # show up to 200 samples in the gallery


def _degrade_params(mode: str, sigma: float, prob: float, kernel_size: int,
                    motion_len: int, angle: float, n_patches: int,
                    patch_size: int, quality: int) -> dict:
    return dict(
        sigma=sigma, prob=prob, kernel_size=int(kernel_size),
        length=int(motion_len), angle=angle,
        n_patches=int(n_patches), patch_size=int(patch_size),
        quality=int(quality),
    )


# ─── Core pipeline (shared by all tabs) ──────────────────────────────────────
# Reviewed by Adhip Kumar

def run_pipeline(
    original_pil: Image.Image | None,
    mode: str,
    method: str,
    max_dim: int,
    sigma: float, prob: float, kernel_size: int,
    motion_len: int, angle: float,
    n_patches: int, patch_size: int, quality: int,
    progress=gr.Progress(track_tqdm=True),
) -> tuple:
    """
    Full pipeline: degrade → restore → metrics → comparison figure.
    Returns (corrupted_pil, restored_pil, comparison_pil, metrics_df, status_str)
    """
    if original_pil is None:
        return None, None, None, None, "⚠️ Please upload or select an image first."

    progress(0, desc="Loading image …")
    original = pil_to_array(original_pil)
    original = resize_if_larger(original, int(max_dim))

    # ── Degrade ───────────────────────────────────────────────────────────────
    # Reviewed by Adhip Kumar
    progress(0.15, desc=f"Applying degradation: {MODES[mode]} …")
    params = _degrade_params(mode, sigma, prob, kernel_size, motion_len,
                              angle, n_patches, patch_size, quality)
    corrupted, mask = degrade(original, mode, **params)

    # ── Restore ───────────────────────────────────────────────────────────────
    # Reviewed by Adhip Kumar
    progress(0.35, desc="Restoring image …")
    t0 = time.perf_counter()
    try:
        restored = restore(corrupted, mode=mode, mask=mask, method=method.lower())
    except Exception as exc:
        return (
            array_to_pil(corrupted), None, None, None,
            f"❌ Restoration failed: {exc}"
        )
    elapsed = time.perf_counter() - t0

    # ── Metrics ───────────────────────────────────────────────────────────────
    # Reviewed by Adhip Kumar
    progress(0.70, desc="Computing quality metrics …")

    import math
    import pandas as pd

    try:
        summary = improvement_summary(original, corrupted, restored)
    except Exception as exc:
        log.warning("Metric computation failed: %s", exc)
        summary = {"before": {}, "after": {}}

    metrics_data = []
    for k in summary.get("before", {}):
        b = summary["before"].get(k)
        a = summary["after"].get(k)
        valid = (
            isinstance(b, (int, float)) and not math.isnan(b) and not math.isinf(b) and
            isinstance(a, (int, float)) and not math.isnan(a) and not math.isinf(a)
        )
        if valid:
            delta = a - b
            better = (delta > 0) if k != "LPIPS" else (delta < 0)
            sign   = "+" if better else "-"
            arrow  = f"{sign} {abs(delta):.4f}"
            metrics_data.append([k, f"{b:.4f}", f"{a:.4f}", arrow])
        else:
            metrics_data.append([k, "N/A", "N/A", "-"])

    metrics_df = pd.DataFrame(
        metrics_data,
        columns=["Metric", "Before Restoration", "After Restoration", "Improvement"],
    )

    # ── Comparison figure ─────────────────────────────────────────────────────
    # Reviewed by Adhip Kumar
    progress(0.85, desc="Building comparison report …")
    comparison_pil = None
    try:
        fig = make_comparison_figure(
            original, corrupted, restored,
            summary.get("before", {}), summary.get("after", {}),
            title=f"Restoration — {MODES[mode]}",
            mask=mask,
        )
        comparison_pil = fig_to_pil(fig)
    except Exception as exc:
        log.warning("Comparison figure creation failed: %s", exc)

    progress(1.0, desc="Done!")
    psnr_b = summary.get("before", {}).get("PSNR (dB)")
    psnr_a = summary.get("after", {}).get("PSNR (dB)")
    ssim_b = summary.get("before", {}).get("SSIM")
    ssim_a = summary.get("after", {}).get("SSIM")

    status_parts = [f"Completed in {elapsed:.1f} s"]
    if isinstance(psnr_b, (int, float)) and not math.isnan(psnr_b) and isinstance(psnr_a, (int, float)) and not math.isnan(psnr_a):
        status_parts.append(f"PSNR: {psnr_b:.2f} -> {psnr_a:.2f} dB")
    if isinstance(ssim_b, (int, float)) and not math.isnan(ssim_b) and isinstance(ssim_a, (int, float)) and not math.isnan(ssim_a):
        status_parts.append(f"SSIM: {ssim_b:.4f} -> {ssim_a:.4f}")

    status = " | ".join(status_parts)
    return array_to_pil(corrupted), array_to_pil(restored), comparison_pil, metrics_df, status


# ─── Blind Real-World Pipeline ───────────────────────────────────────────────
# Reviewed by Adhip Kumar

def format_diagnostic_html(diag: dict) -> str:
    """Renders a modern, visually stunning diagnostic summary card."""
    # Reviewed by Adhip Kumar
    primary = diag.get("primary_defect", "Clean")
    severity = diag.get("severity", "Optimal")
    biqs = diag.get("biqs", 0.0)
    sigma = diag.get("noise", {}).get("estimated_sigma", 0.0)
    sharp = diag.get("sharpness", {}).get("sharpness_score", 0.0)
    block = diag.get("blockiness", {}).get("blockiness_ratio", 1.0)
    recipe = diag.get("recipe", [])

    color_map = {
        "Needs Restoration": "#ef4444",
        "Minor Degradation": "#f59e0b",
        "Optimal": "#10b981",
    }
    badge_color = color_map.get(severity, "#60a5fa")

    recipe_items = "".join([f"<li style='margin-bottom:4px;'><b>{r.get('action', '').upper()}:</b> {r.get('label', '')}</li>" for r in recipe])
    if not recipe_items:
        # Reviewed by Adhip Kumar
        recipe_items = "<li>No active restoration steps required.</li>"

    html = f"""
    <div style="background:rgba(15, 23, 42, 0.75); border:1px solid rgba(148, 163, 184, 0.25); border-radius:12px; padding:16px; margin-bottom:14px; box-shadow:0 4px 20px rgba(0,0,0,0.3);">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; border-bottom:1px solid rgba(148, 163, 184, 0.15); padding-bottom:8px;">
        <span style="font-weight:700; font-size:1.05rem; color:#f8fafc; display:flex; align-items:center; gap:8px;">
          <span>🩺</span> AI Defect Diagnosis Report
        </span>
        <span style="background:{badge_color}; color:#fff; font-size:0.75rem; font-weight:700; padding:4px 12px; border-radius:999px;">
          {primary} · {severity}
        </span>
      </div>
      
      <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; margin-bottom:12px; text-align:center;">
        <div style="background:rgba(30, 41, 59, 0.65); padding:10px 6px; border-radius:8px; border:1px solid rgba(56, 189, 248, 0.2);">
          <div style="color:#94a3b8; font-size:0.74rem;">Quality Score (BIQS)</div>
          <div style="color:#38bdf8; font-size:1.2rem; font-weight:700;">{biqs:.1f}/100</div>
        </div>
        <div style="background:rgba(30, 41, 59, 0.65); padding:10px 6px; border-radius:8px; border:1px solid rgba(248, 113, 113, 0.2);">
          <div style="color:#94a3b8; font-size:0.74rem;">Estimated Noise (σ)</div>
          <div style="color:#f87171; font-size:1.2rem; font-weight:700;">{sigma:.1f}</div>
        </div>
        <div style="background:rgba(30, 41, 59, 0.65); padding:10px 6px; border-radius:8px; border:1px solid rgba(74, 222, 128, 0.2);">
          <div style="color:#94a3b8; font-size:0.74rem;">Sharpness Index</div>
          <div style="color:#4ade80; font-size:1.2rem; font-weight:700;">{sharp:.1f}</div>
        </div>
        <div style="background:rgba(30, 41, 59, 0.65); padding:10px 6px; border-radius:8px; border:1px solid rgba(192, 132, 252, 0.2);">
          <div style="color:#94a3b8; font-size:0.74rem;">JPEG Blockiness</div>
          <div style="color:#c084fc; font-size:1.2rem; font-weight:700;">{block:.2f}x</div>
        </div>
      </div>

      <div style="font-size:0.85rem; color:#cbd5e1; background:rgba(2, 6, 23, 0.45); padding:10px 14px; border-radius:8px; border:1px solid rgba(148, 163, 184, 0.15);">
        <div style="font-weight:600; color:#60a5fa; margin-bottom:4px;">💡 AI Prescribed Restoration Recipe:</div>
        <ul style="margin:0; padding-left:18px; color:#e2e8f0;">
          {recipe_items}
        </ul>
      </div>
    </div>
    """
    return html


def run_blind_pipeline(
    image_pil: Image.Image | None,
    max_dim: int,
    auto_recipe: bool,
    apply_denoise: bool,
    apply_deblur: bool,
    apply_deblock: bool,
    apply_contrast: bool,
    progress=gr.Progress(track_tqdm=True),
) -> tuple:
    """
    Blind real-world restoration without ground truth reference.
    # Reviewed by Adhip Kumar
    """
    if image_pil is None:
        # Reviewed by Adhip Kumar
        return None, None, None, None, "<p style='color:#f87171;'>⚠️ Please upload an image first.</p>", "⚠️ Please upload an image first."

    # Load and resize
    # Reviewed by Adhip Kumar
    progress(0.1, desc="Analyzing image & diagnosing defects …")
    img_arr = pil_to_array(image_pil)
    img_arr = resize_if_larger(img_arr, int(max_dim))

    # Execute blind restoration
    # Reviewed by Adhip Kumar
    progress(0.35, desc="Executing blind restoration cascade …")
    restored, steps, diag_b, diag_a, imp = restore_blind(
        img_arr,
        auto=auto_recipe,
        apply_denoise_opt=apply_denoise,
        apply_deblur_opt=apply_deblur,
        apply_deblock_opt=apply_deblock,
        apply_contrast_opt=apply_contrast,
    )

    # Comparison figure
    # Reviewed by Adhip Kumar
    progress(0.75, desc="Building blind comparison figure …")
    fig = make_blind_comparison_figure(img_arr, restored, diag_b, diag_a)
    fig_pil = fig_to_pil(fig)

    # Metrics table
    # Reviewed by Adhip Kumar
    progress(0.9, desc="Generating quality metric breakdown …")
    import pandas as pd
    rows = [
        ["Blind Quality Score (BIQS)", f"{imp['biqs_before']:.1f}/100", f"{imp['biqs_after']:.1f}/100", f"+{imp['biqs_delta']} pts" if imp['biqs_delta'] >= 0 else f"{imp['biqs_delta']} pts"],
        ["Estimated Noise (σ)", f"{diag_b['noise']['estimated_sigma']:.1f}", f"{diag_a['noise']['estimated_sigma']:.1f}", f"-{imp['noise_reduction_pct']:.1f}%"],
        ["Edge Sharpness Index", f"{diag_b['sharpness']['sharpness_score']:.1f}", f"{diag_a['sharpness']['sharpness_score']:.1f}", f"{'+' if imp['sharpness_gain_pct']>=0 else ''}{imp['sharpness_gain_pct']:.1f}%"],
        ["JPEG Blockiness Ratio", f"{diag_b['blockiness']['blockiness_ratio']:.3f}x", f"{diag_a['blockiness']['blockiness_ratio']:.3f}x", f"-{imp['block_reduction_pct']:.1f}%"],
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Degraded Input", "Restored Output", "Relative Improvement"])

    diag_html = format_diagnostic_html(diag_b)
    status_str = f"Completed in {imp['elapsed_s']:.1f} s | BIQS: {imp['biqs_before']:.1f} -> {imp['biqs_after']:.1f} (+{imp['biqs_delta']:.1f} pts) | {len(steps)} steps applied"

    progress(1.0, desc="Done!")
    return array_to_pil(img_arr), array_to_pil(restored), fig_pil, df, diag_html, status_str


def run_blind_diagnosis_only(image_pil: Image.Image | None, max_dim: int) -> tuple:
    """Instant defect diagnosis without restoration."""
    # Reviewed by Adhip Kumar
    if image_pil is None:
        # Reviewed by Adhip Kumar
        return "<p style='color:#f87171;'>⚠️ Please upload an image first.</p>", "⚠️ Please upload an image first."
    img_arr = pil_to_array(image_pil)
    img_arr = resize_if_larger(img_arr, int(max_dim))
    diag = diagnose_image(img_arr)
    html = format_diagnostic_html(diag)
    status = f"Diagnosis complete | Detected: {diag['primary_defect']} ({diag['severity']}) | BIQS: {diag['biqs']:.1f}/100"
    return html, status


# ─── Build UI ────────────────────────────────────────────────────────────────
# Reviewed by Adhip Kumar


def build_app() -> gr.Blocks:
    sample_files = _list_samples()

    css = """
    /* ── Before / After Split Gradient Background ── */
    /* Fades from muted, faded desaturated grey-slate on the left to vibrant, vivid deep blue/indigo/violet on the right */
    body, .gradio-container, gradio-app {
        background: 
            radial-gradient(ellipse at 90% 25%, rgba(56, 189, 248, 0.20) 0%, transparent 55%),
            radial-gradient(ellipse at 92% 80%, rgba(139, 92, 246, 0.22) 0%, transparent 60%),
            radial-gradient(ellipse at 12% 40%, rgba(71, 85, 105, 0.40) 0%, transparent 60%),
            linear-gradient(108deg, #111215 0%, #1a1c22 30%, #162032 55%, #0d1e40 76%, #1a143b 100%) !important;
        background-attachment: fixed !important;
        min-height: 100vh !important;
        color: #e2e8f0 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif !important;
    }

    /* Left control column: desaturated subtle slate glass */
    .controls-panel {
        background: rgba(22, 24, 30, 0.75) !important;
        border: 1px solid rgba(148, 163, 184, 0.20) !important;
        border-radius: 14px !important;
        padding: 16px !important;
        backdrop-filter: blur(12px) !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3) !important;
    }

    /* Right workspace column: rich vivid deep glass with subtle glowing border */
    .workspace-panel {
        background: rgba(14, 20, 38, 0.70) !important;
        border: 1px solid rgba(96, 165, 250, 0.28) !important;
        border-radius: 14px !important;
        padding: 18px !important;
        backdrop-filter: blur(14px) !important;
        box-shadow: 0 14px 40px -10px rgba(96, 165, 250, 0.16) !important;
    }

    /* Primary Action button with vibrant restoration gradient */
    button.primary {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
        border: none !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 18px rgba(37, 99, 235, 0.45) !important;
        transition: all 0.25s ease !important;
    }
    button.primary:hover {
        box-shadow: 0 6px 24px rgba(124, 58, 237, 0.55) !important;
        transform: translateY(-1px) !important;
    }

    /* Try Another Image button styling */
    .retry-btn {
        margin-top: 16px !important;
        background: rgba(30, 41, 59, 0.75) !important;
        border: 1px solid rgba(148, 163, 184, 0.35) !important;
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
        transition: all 0.25s ease !important;
    }
    .retry-btn:hover {
        background: rgba(51, 65, 85, 0.95) !important;
        border-color: #60a5fa !important;
        color: #ffffff !important;
        box-shadow: 0 6px 20px rgba(96, 165, 250, 0.25) !important;
        transform: translateY(-1px) !important;
    }

    footer { display: none !important; }
    """

    with gr.Blocks(title="🖼️ AI Image Restoration") as demo:

        # ── Header ────────────────────────────────────────────────────────────
        # Reviewed by Adhip Kumar
        gr.HTML(f"""
        <style>{css}</style>
        <div style="text-align:center; padding:22px 0 10px;">
          <div style="margin-bottom:12px;">
            <span style="display:inline-flex; align-items:center; gap:8px; padding:5px 16px; border-radius:999px;
                        background:linear-gradient(90deg, rgba(71,85,105,0.45) 0%, rgba(56,189,248,0.22) 100%);
                        border:1px solid rgba(148,163,184,0.30); font-size:0.84rem; color:#cbd5e1; box-shadow:0 2px 10px rgba(0,0,0,0.2);">
              <span>🟣 Degraded & Desaturated</span>
              <span style="opacity:0.6;">➔</span>
              <span style="color:#60a5fa; font-weight:600;">✨ Reconstructed & Vivid</span>
            </span>
          </div>
          <h1 style="font-size:2.2rem; font-weight:800; margin:0; display:flex; align-items:center; justify-content:center; gap:12px;">
            <svg style="flex-shrink:0;" width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
              <circle cx="8.5" cy="8.5" r="1.5"/>
              <polyline points="21 15 16 10 5 21"/>
              <path d="M15 5l1.5-1.5L18 5"/>
              <path d="M19.5 9.5l1-1"/>
            </svg>
            <span style="background:linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
              AI Image Restoration System
            </span>
          </h1>
          <p style="color:#94a3b8; font-size:1rem; margin-top:8px;">
            Reconstruct images degraded by noise · blur · missing regions · compression artifacts
          </p>
        </div>
        """)

        # ── Shared degradation parameter state ────────────────────────────────
        # Reviewed by Adhip Kumar
        with gr.Row(equal_height=False):

            # ── Left column: controls ─────────────────────────────────────────
            # Reviewed by Adhip Kumar
            with gr.Column(scale=1, min_width=300, elem_classes=["controls-panel"]):
                gr.Markdown("### ⚙️ Configuration")

                mode_dd = gr.Dropdown(
                    choices=list(MODES.keys()),
                    label="Degradation Type",
                    value="noise",
                    info="Type of damage to simulate",
                )

                method_dd = gr.Dropdown(
                    choices=["auto", "nafnet", "dncnn", "wavelet", "nlm",
                             "wiener", "rl", "lama", "ns", "telea", "tv"],
                    label="Restoration Method",
                    value="auto",
                    info="'auto' uses the best available model",
                )

                max_dim_sl = gr.Slider(256, 1024, value=512, step=64,
                                       label="Max Image Dimension (px)",
                                       info="Larger = slower on CPU")

                gr.Markdown("#### Degradation Parameters")

                with gr.Accordion("Noise Parameters", open=True) as noise_acc:
                    sigma_sl  = gr.Slider(5, 100, value=25, step=5, label="Gaussian σ (noise level)")
                    prob_sl   = gr.Slider(0.01, 0.2, value=0.05, step=0.01, label="Salt-Pepper Probability")

                with gr.Accordion("Blur Parameters", open=False) as blur_acc:
                    kernel_sl = gr.Slider(3, 51, value=15, step=2, label="Kernel Size")
                    mlen_sl   = gr.Slider(5, 60, value=25, step=5, label="Motion Length (px)")
                    angle_sl  = gr.Slider(0, 180, value=45, step=5, label="Motion Angle (°)")

                with gr.Accordion("Inpainting Parameters", open=False) as inp_acc:
                    npatch_sl = gr.Slider(1, 8, value=3, step=1, label="Number of Missing Patches")
                    psize_sl  = gr.Slider(20, 200, value=80, step=10, label="Patch Size (px)")

                with gr.Accordion("Artifact Parameters", open=False) as art_acc:
                    qual_sl   = gr.Slider(1, 50, value=10, step=1, label="JPEG Quality (lower = worse)")

            # ── Right column: tabs ───────────────────────────────────────────
            # Reviewed by Adhip Kumar
            with gr.Column(scale=3, elem_classes=["workspace-panel"]):

                with gr.Tabs():

                    # ── Tab 1: Real-World AI Diagnosis & Restoration ──────
                    # Reviewed by Adhip Kumar
                    with gr.TabItem("🩺 Real-World AI Diagnosis & Restoration"):
                        gr.Markdown(
                            "### 🔍 Blind Defect Diagnosis & Real-World Restoration\n"
                            "Upload an already-degraded real image (scanned vintage photo, noisy low-light shot, blurry picture, compressed JPEG). "
                            "Our AI inspects the image without ground-truth reference, detects defects, prescribes the optimal restoration cascade, and enhances quality."
                        )

                        with gr.Row():
                            blind_input_img = gr.Image(
                                label="Upload Real Degraded Image", type="pil",
                                height=320, sources=["upload", "clipboard"],
                            )

                        with gr.Row():
                            auto_recipe_chk = gr.Checkbox(label="🤖 Full Auto-Pilot Mode (Recommended: AI detects and restores all defects)", value=True)

                        with gr.Accordion("⚙️ Manual Custom Cascade Toggles (when Auto-Pilot is unchecked)", open=False):
                            with gr.Row():
                                blind_chk_deblock  = gr.Checkbox(label="Deblocking (JPEG Artifacts)", value=True)
                                blind_chk_denoise  = gr.Checkbox(label="Deep Denoising", value=True)
                                blind_chk_deblur   = gr.Checkbox(label="Deblur & Edge Sharpening", value=True)
                                blind_chk_contrast = gr.Checkbox(label="CLAHE Dynamic Range", value=True)

                        with gr.Row():
                            blind_diag_btn    = gr.Button("🩺 1. Diagnose Defects Only", variant="secondary", size="lg")
                            blind_restore_btn = gr.Button("✨ 2. Auto-Restore Real Image", variant="primary", size="lg")

                        blind_status_box = gr.Textbox(label="Status", interactive=False, lines=2)
                        blind_diag_html  = gr.HTML()

                        with gr.Row():
                            blind_orig_out     = gr.Image(label="🔴 Input Degraded Image", type="pil", height=280)
                            blind_restored_out = gr.Image(label="🟢 AI Restored Image",    type="pil", height=280)

                        blind_comparison_out = gr.Image(label="📊 No-Reference Quality & Residual Analysis", type="pil", height=420)
                        blind_metrics_table  = gr.DataFrame(label="No-Reference Quality Metrics & Improvement", row_count=4)
                        retry_blind_btn      = gr.Button("🔄  Try Another Image", variant="secondary", size="lg", elem_classes=["retry-btn"])

                        def _reset_blind():
                            # Reviewed by Adhip Kumar
                            import pandas as pd
                            empty_df = pd.DataFrame(columns=["Metric", "Degraded Input", "Restored Output", "Relative Improvement"])
                            return None, None, None, None, empty_df, "", ""

                        blind_diag_btn.click(
                            fn=run_blind_diagnosis_only,
                            inputs=[blind_input_img, max_dim_sl],
                            outputs=[blind_diag_html, blind_status_box],
                        )

                        blind_restore_btn.click(
                            fn=run_blind_pipeline,
                            inputs=[
                                blind_input_img, max_dim_sl, auto_recipe_chk,
                                blind_chk_denoise, blind_chk_deblur,
                                blind_chk_deblock, blind_chk_contrast,
                            ],
                            outputs=[
                                blind_orig_out, blind_restored_out, blind_comparison_out,
                                blind_metrics_table, blind_diag_html, blind_status_box,
                            ],
                            concurrency_limit=5,
                        )

                        retry_blind_btn.click(
                            fn=_reset_blind,
                            inputs=[],
                            outputs=[
                                blind_input_img, blind_orig_out, blind_restored_out, blind_comparison_out,
                                blind_metrics_table, blind_diag_html, blind_status_box,
                            ],
                            js="() => { window.scrollTo({top: 0, behavior: 'smooth'}); }",
                        )

                    # ── Tab 2: Single Image Simulation ─────────────────────
                    # Reviewed by Adhip Kumar
                    with gr.TabItem("🖼️  Simulation: Single Image"):
                        with gr.Row():
                            upload_img = gr.Image(
                                label="Upload Image", type="pil",
                                height=320, sources=["upload", "clipboard"],
                            )

                        run_btn = gr.Button("🚀  Apply & Restore", variant="primary", size="lg")
                        status_box = gr.Textbox(label="Status", interactive=False, lines=2)

                        with gr.Row():
                            corrupted_out = gr.Image(label="🔴 Corrupted", type="pil", height=280)
                            restored_out  = gr.Image(label="🟢 Restored",  type="pil", height=280)

                        comparison_out = gr.Image(label="📊 Comparison Report", type="pil", height=420)
                        metrics_table  = gr.DataFrame(label="Quality Metrics", row_count=3)
                        retry_single_btn = gr.Button("🔄  Try Another Image", variant="secondary", size="lg", elem_classes=["retry-btn"])

                        def _reset_single():
                            import pandas as pd
                            empty_df = pd.DataFrame(columns=["Metric", "Before Restoration", "After Restoration", "Improvement"])
                            return None, None, None, None, empty_df, ""

                        run_btn.click(
                            fn=run_pipeline,
                            inputs=[
                                upload_img, mode_dd, method_dd, max_dim_sl,
                                sigma_sl, prob_sl, kernel_sl,
                                mlen_sl, angle_sl,
                                npatch_sl, psize_sl, qual_sl,
                            ],
                            outputs=[corrupted_out, restored_out, comparison_out,
                                     metrics_table, status_box],
                            concurrency_limit=5,
                        )

                        retry_single_btn.click(
                            fn=_reset_single,
                            inputs=[],
                            outputs=[upload_img, corrupted_out, restored_out, comparison_out,
                                     metrics_table, status_box],
                            js="() => { window.scrollTo({top: 0, behavior: 'smooth'}); }",
                        )

                    # ── Tab 2: Sample Gallery ─────────────────────────────
                    # Reviewed by Adhip Kumar
                    with gr.TabItem("🗂️  Sample Gallery"):
                        gr.Markdown(
                            f"**{len(sample_files)} sample images available.** "
                            "Click an image to select it, then press *Restore Selected*."
                        )

                        gallery = gr.Gallery(
                            value=sample_files[:50] if sample_files else [],
                            label="Sample Images",
                            columns=6,
                            height=360,
                            allow_preview=True,
                            object_fit="cover",
                        )

                        selected_img = gr.Image(label="Selected Sample", type="pil", height=240,
                                                visible=True)
                        restore_sample_btn = gr.Button("🚀  Restore Selected Sample",
                                                        variant="primary")
                        sample_status = gr.Textbox(label="Status", interactive=False, lines=2)

                        with gr.Row():
                            sample_corrupted = gr.Image(label="🔴 Corrupted", type="pil", height=260)
                            sample_restored  = gr.Image(label="🟢 Restored",  type="pil", height=260)

                        sample_comparison = gr.Image(label="📊 Comparison Report", type="pil", height=400)
                        sample_metrics    = gr.DataFrame(label="Quality Metrics", row_count=3)
                        retry_sample_btn  = gr.Button("🔄  Try Another Sample", variant="secondary", size="lg", elem_classes=["retry-btn"])

                        def _select_gallery(evt: gr.SelectData):
                            if sample_files and evt.index < len(sample_files):
                                return Image.open(sample_files[evt.index]).convert("RGB")
                            return None

                        gallery.select(_select_gallery, outputs=selected_img)

                        def _reset_sample():
                            import pandas as pd
                            empty_df = pd.DataFrame(columns=["Metric", "Before Restoration", "After Restoration", "Improvement"])
                            return None, None, None, None, empty_df, ""

                        restore_sample_btn.click(
                            fn=run_pipeline,
                            inputs=[
                                selected_img, mode_dd, method_dd, max_dim_sl,
                                sigma_sl, prob_sl, kernel_sl,
                                mlen_sl, angle_sl,
                                npatch_sl, psize_sl, qual_sl,
                            ],
                            outputs=[sample_corrupted, sample_restored, sample_comparison,
                                     sample_metrics, sample_status],
                            concurrency_limit=5,
                        )

                        retry_sample_btn.click(
                            fn=_reset_sample,
                            inputs=[],
                            outputs=[selected_img, sample_corrupted, sample_restored, sample_comparison,
                                     sample_metrics, sample_status],
                            js="() => { window.scrollTo({top: 0, behavior: 'smooth'}); }",
                        )

                    # ── Tab 3: Batch Evaluate ─────────────────────────────
                    # Reviewed by Adhip Kumar
                    with gr.TabItem("📈  Batch Evaluate"):
                        gr.Markdown(
                            "Run all sample images through the selected pipeline and compute "
                            "average metrics."
                        )
                        n_samples = len(sample_files)
                        max_batch = max(5, min(100, n_samples)) if n_samples > 1 else 10
                        n_images_sl = gr.Slider(
                            minimum=1,
                            maximum=max_batch,
                            value=min(5, max_batch),
                            step=1,
                            label="Number of Sample Images to Process",
                        )
                        batch_btn   = gr.Button("▶️  Run Batch Evaluation", variant="primary")
                        batch_status = gr.Textbox(label="Progress", interactive=False, lines=3)
                        batch_results = gr.DataFrame(
                            label="Per-Image Results",
                            headers=["Image", "PSNR Before", "PSNR After", "SSIM Before",
                                     "SSIM After", "LPIPS Before", "LPIPS After"],
                        )
                        avg_results = gr.DataFrame(label="Average Metrics", row_count=3)

                        def run_batch(
                            n_images, mode, method, max_dim,
                            sigma, prob, kernel_size, motion_len, angle,
                            n_patches, patch_size, quality,
                            progress=gr.Progress(track_tqdm=True),
                        ):
                            if not sample_files:
                                return "No samples found. Run download_samples.py first.", None, None

                            files = sample_files[:int(n_images)]
                            rows = []
                            params = _degrade_params(mode, sigma, prob, kernel_size, motion_len,
                                                      angle, n_patches, patch_size, quality)

                            for i, fp in enumerate(files):
                                progress(i / len(files), desc=f"[{i+1}/{len(files)}] {Path(fp).name}")
                                try:
                                    from utils import load_image
                                    orig = load_image(fp)
                                    orig = resize_if_larger(orig, int(max_dim))
                                    corr, msk = degrade(orig, mode, **params)
                                    rest = restore(corr, mode=mode, mask=msk, method=method.lower())
                                    m_b = compute_all(orig, corr)
                                    m_a = compute_all(orig, rest)
                                    rows.append([
                                        Path(fp).name,
                                        round(m_b["PSNR (dB)"], 2), round(m_a["PSNR (dB)"], 2),
                                        round(m_b["SSIM"], 4),     round(m_a["SSIM"], 4),
                                        str(m_b["LPIPS"]),         str(m_a["LPIPS"]),
                                    ])
                                except Exception as exc:
                                    log.warning("Batch: error on %s — %s", fp, exc)

                            if not rows:
                                return "No images processed successfully.", None, None

                            import pandas as pd
                            per_img_df = pd.DataFrame(rows, columns=[
                                "Image", "PSNR Before", "PSNR After",
                                "SSIM Before", "SSIM After",
                                "LPIPS Before", "LPIPS After",
                            ])

                            # Averages
                            # Reviewed by Adhip Kumar
                            avg_data = []
                            for metric, b_col, a_col in [
                                ("PSNR (dB)", "PSNR Before", "PSNR After"),
                                ("SSIM",      "SSIM Before",  "SSIM After"),
                            ]:
                                b_mean = per_img_df[b_col].mean()
                                a_mean = per_img_df[a_col].mean()
                                avg_data.append([metric, f"{b_mean:.4f}", f"{a_mean:.4f}",
                                                  f"+{a_mean - b_mean:.4f}"])
                            avg_df = pd.DataFrame(avg_data,
                                                   columns=["Metric", "Avg Before", "Avg After", "Avg Δ"])

                            status_msg = (
                                f"✅ Processed {len(rows)}/{len(files)} images. "
                                f"Avg PSNR gain: +{per_img_df['PSNR After'].mean() - per_img_df['PSNR Before'].mean():.2f} dB"
                            )
                            return status_msg, per_img_df, avg_df

                        batch_btn.click(
                            fn=run_batch,
                            inputs=[n_images_sl, mode_dd, method_dd, max_dim_sl,
                                    sigma_sl, prob_sl, kernel_sl,
                                    mlen_sl, angle_sl,
                                    npatch_sl, psize_sl, qual_sl],
                            outputs=[batch_status, batch_results, avg_results],
                        )

        # ── Footer ────────────────────────────────────────────────────────────
        # Reviewed by Adhip Kumar
        gr.HTML("""
        <div style="text-align:center; padding:16px 0 4px; color:#64748b; font-size:0.85rem;">
          Models: NAFNet (megvii-research) · DnCNN (cszn/KAIR) · LaMa (saic-mdal) ·
          Wiener / RL / BayesShrink / TV classical methods<br>
          Metrics: PSNR · SSIM (scikit-image) · LPIPS (lpips)
        </div>
        """)

    return demo


# ─── Entry point ─────────────────────────────────────────────────────────────
# Reviewed by Adhip Kumar

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port",  type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    p.add_argument("--host",  default="127.0.0.1")
    args = p.parse_args()
    host = os.environ.get("HOST", args.host)
    port = int(os.environ.get("PORT", args.port))

    demo = build_app()
    app, local_url, share_url = demo.launch(
        server_name=host,
        server_port=port,
        share=args.share,
        max_file_size="50mb",
        inbrowser=False if "PORT" in os.environ else True,
        prevent_thread_lock=True,
        theme=gr.themes.Base(primary_hue="blue", secondary_hue="slate", neutral_hue="slate"),
    )
    if share_url:
        Path("live_url.txt").write_text(share_url, encoding="utf-8")
    print(f"\n========================================", flush=True)
    print(f"  LOCAL URL:  {local_url}", flush=True)
    print(f"  PUBLIC URL: {share_url}", flush=True)
    print(f"========================================\n", flush=True)
    log.info("LOCAL URL: %s", local_url)
    log.info("PUBLIC SHARE URL: %s", share_url)
    demo.block_thread()


if __name__ == "__main__":
    main()
