"""
generate_demo_recording.py
Generates an animated screen recording video (MP4) and animated GIF (assets/demo_recording.gif)
showcasing the full interactive restoration web app.
# Reviewed by Adhip Kumar
"""
import io
import math
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
FPS = 12

# Fonts
font_title = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 22)
font_h2 = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 15)
font_body = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 13)
font_bold = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 13)
font_small = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 11)
font_badge = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 11)
font_metrics = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 14)

# Load Real Assets
cbsd_orig = Image.open("sample_images/cbsd_0001.png").convert("RGB")
cbsd_corr = Image.open("assets/demo_corrupted.png").convert("RGB")
cbsd_rest = Image.open("assets/demo_restored.png").convert("RGB")
cbsd_comp = Image.open("assets/demo_comparison_real.png").convert("RGB")

def draw_browser_chrome(img):
    draw = ImageDraw.Draw(img)
    # Browser top bar
    draw.rectangle([0, 0, W, 42], fill="#16181d")
    draw.line([0, 42, W, 42], fill="#272a34", width=1)
    
    # Window controls (macOS / clean dots)
    draw.ellipse([16, 15, 27, 26], fill="#ef4444")
    draw.ellipse([33, 15, 44, 26], fill="#f59e0b")
    draw.ellipse([50, 15, 61, 26], fill="#10b981")
    
    # Tab
    draw.rounded_rectangle([75, 8, 300, 42], radius=6, fill="#1c202a")
    draw.text((90, 17), "🖼️  AI Image Restoration System", fill="#e2e8f0", font=font_small)
    
    # Address bar
    draw.rounded_rectangle([320, 10, 960, 34], radius=12, fill="#111317", outline="#2e3340")
    draw.text((335, 15), "🔒  https://ai-image-restoration-zane.onrender.com", fill="#94a3b8", font=font_small)

def draw_split_background():
    bg = np.zeros((H, W, 3), dtype=np.float32)
    for x in range(W):
        t = x / W
        r = 17 * (1 - t) + 26 * t
        g = 19 * (1 - t) + 22 * t
        b = 24 * (1 - t) + 65 * t
        glow = math.exp(-((x - W * 0.85)**2) / (2 * (W * 0.35)**2)) * 25
        bg[:, x, 0] = np.clip(r + glow * 0.5, 0, 255)
        bg[:, x, 1] = np.clip(g + glow * 0.6, 0, 255)
        bg[:, x, 2] = np.clip(b + glow * 1.4, 0, 255)
    return Image.fromarray(bg.astype(np.uint8))

base_bg = draw_split_background()

def draw_cursor(draw, x, y, click=False):
    poly = [
        (x, y), (x, y + 17), (x + 4, y + 13),
        (x + 9, y + 21), (x + 12, y + 20),
        (x + 7, y + 12), (x + 13, y + 12)
    ]
    draw.polygon(poly, fill="#ffffff" if not click else "#38bdf8", outline="#000000")

def render_frame(state):
    frame = base_bg.copy()
    draw_browser_chrome(frame)
    draw = ImageDraw.Draw(frame)
    
    scroll_y = state.get("scroll_y", 0)
    
    # ── Header ──
    hdr_y = 52 - scroll_y
    badge_text = "🟣 Degraded & Desaturated   ➔   ✨ Reconstructed & Vivid"
    draw.rounded_rectangle([W//2 - 170, hdr_y, W//2 + 170, hdr_y + 22], radius=11,
                           fill="#1e2433", outline="#3b4861", width=1)
    draw.text((W//2 - 150, hdr_y + 4), badge_text, fill="#cbd5e1", font=font_badge)
    
    draw.text((W//2 - 145, hdr_y + 27), "AI Image Restoration System", fill="#60a5fa", font=font_title)
    draw.text((W//2 - 210, hdr_y + 57), "Reconstruct images degraded by noise · blur · missing regions · compression artifacts",
              fill="#94a3b8", font=font_small)
              
    content_top = hdr_y + 80
    
    # ── Left Column: Controls (Width: 280) ──
    ctrl_x, ctrl_y = 35, content_top
    ctrl_w, ctrl_h = 280, 520
    draw.rounded_rectangle([ctrl_x, ctrl_y, ctrl_x + ctrl_w, ctrl_y + ctrl_h], radius=12,
                           fill="#16181f", outline="#2b3140", width=1)
    draw.text((ctrl_x + 16, ctrl_y + 14), "⚙️ Configuration", fill="#f8fafc", font=font_h2)
    
    draw.text((ctrl_x + 16, ctrl_y + 45), "Degradation Type", fill="#cbd5e1", font=font_small)
    draw.rounded_rectangle([ctrl_x + 16, ctrl_y + 63, ctrl_x + ctrl_w - 16, ctrl_y + 95], radius=6,
                           fill="#1f242d", outline="#3b4252")
    draw.text((ctrl_x + 26, ctrl_y + 71), "noise (Gaussian)", fill="#ffffff", font=font_body)
    
    draw.text((ctrl_x + 16, ctrl_y + 107), "Restoration Method", fill="#cbd5e1", font=font_small)
    draw.rounded_rectangle([ctrl_x + 16, ctrl_y + 125, ctrl_x + ctrl_w - 16, ctrl_y + 157], radius=6,
                           fill="#1f242d", outline="#3b4252")
    draw.text((ctrl_x + 26, ctrl_y + 133), "auto (Wavelet / DL)", fill="#ffffff", font=font_body)
    
    draw.text((ctrl_x + 16, ctrl_y + 169), "Max Image Dimension: 512px", fill="#cbd5e1", font=font_small)
    draw.line([ctrl_x + 16, ctrl_y + 195, ctrl_x + ctrl_w - 16, ctrl_y + 195], fill="#334155", width=4)
    draw.line([ctrl_x + 16, ctrl_y + 195, ctrl_x + 140, ctrl_y + 195], fill="#3b82f6", width=4)
    draw.ellipse([ctrl_x + 136, ctrl_y + 190, ctrl_x + 146, ctrl_y + 200], fill="#60a5fa")
    
    draw.text((ctrl_x + 16, ctrl_y + 215), "Gaussian σ (noise): 25", fill="#94a3b8", font=font_small)
    draw.line([ctrl_x + 16, ctrl_y + 238, ctrl_x + ctrl_w - 16, ctrl_y + 238], fill="#334155", width=4)
    draw.line([ctrl_x + 16, ctrl_y + 238, ctrl_x + 95, ctrl_y + 238], fill="#3b82f6", width=4)
    draw.ellipse([ctrl_x + 91, ctrl_y + 233, ctrl_x + 101, ctrl_y + 243], fill="#60a5fa")
    
    # ── Right Column: Workspace (Width: 900) ──
    work_x = 330
    work_w = W - work_x - 35
    work_y = content_top
    work_h = 920
    
    draw.rounded_rectangle([work_x, work_y, work_x + work_w, work_y + work_h], radius=12,
                           fill="#0f1629", outline="#253556", width=1)
                           
    draw.text((work_x + 20, work_y + 14), "🖼️  Single Image", fill="#60a5fa", font=font_bold)
    draw.line([work_x + 20, work_y + 34, work_x + 135, work_y + 34], fill="#60a5fa", width=2)
    draw.text((work_x + 155, work_y + 14), "🗂️  Sample Gallery", fill="#64748b", font=font_body)
    draw.text((work_x + 285, work_y + 14), "📈  Batch Evaluate", fill="#64748b", font=font_body)
    
    box_y = work_y + 45
    box_h = 160
    has_image = state.get("has_image", False)
    
    if not has_image:
        draw.rounded_rectangle([work_x + 20, box_y, work_x + work_w - 20, box_y + box_h],
                               radius=8, fill="#131c33", outline="#2c3c63", width=1)
        draw.text((work_x + work_w//2 - 110, box_y + 55), "📁 Drop Image Here or Click to Upload",
                  fill="#94a3b8", font=font_body)
        draw.text((work_x + work_w//2 - 70, box_y + 78), "Supports PNG, JPG, WEBP",
                  fill="#64748b", font=font_small)
    else:
        draw.rounded_rectangle([work_x + 20, box_y, work_x + work_w - 20, box_y + box_h],
                               radius=8, fill="#131c33", outline="#3b82f6", width=1)
        preview = cbsd_orig.copy()
        preview.thumbnail((work_w - 60, box_h - 16))
        px = work_x + (work_w - preview.width) // 2
        py = box_y + (box_h - preview.height) // 2
        frame.paste(preview, (px, py))
        draw.rounded_rectangle([work_x + 28, box_y + 8, work_x + 165, box_y + 28], radius=4,
                               fill="#000000cc")
        draw.text((work_x + 34, box_y + 12), "cbsd_0001.png (Loaded)", fill="#38bdf8", font=font_small)

    btn_y = box_y + box_h + 12
    btn_w = work_w - 40
    btn_pressed = state.get("btn_pressed", False)
    btn_fill = "#1d4ed8" if not btn_pressed else "#2563eb"
    draw.rounded_rectangle([work_x + 20, btn_y, work_x + 20 + btn_w, btn_y + 38], radius=8,
                           fill=btn_fill, outline="#3b82f6" if btn_pressed else None)
    btn_text = "🚀  Apply & Restore"
    draw.text((work_x + work_w//2 - 60, btn_y + 10), btn_text, fill="#ffffff", font=font_bold)
    
    prog_y = btn_y + 48
    progress = state.get("progress", 0.0)
    prog_desc = state.get("prog_desc", "")
    
    if progress > 0:
        draw.rounded_rectangle([work_x + 20, prog_y, work_x + work_w - 20, prog_y + 44],
                               radius=6, fill="#111827", outline="#1f2937")
        draw.text((work_x + 32, prog_y + 6), f"{prog_desc} ({int(progress*100)}%)",
                  fill="#e2e8f0", font=font_small)
        draw.line([work_x + 32, prog_y + 28, work_x + work_w - 32, prog_y + 28], fill="#1f2937", width=6)
        bar_len = int((work_w - 64) * progress)
        draw.line([work_x + 32, prog_y + 28, work_x + 32 + bar_len, prog_y + 28], fill="#3b82f6", width=6)
        
    out_y = prog_y + 54
    is_done = state.get("is_done", False)
    
    if is_done:
        img_w, img_h = 415, 230
        
        draw.rounded_rectangle([work_x + 20, out_y, work_x + 20 + img_w, out_y + img_h],
                               radius=8, fill="#13192a", outline="#ef4444", width=2)
        c_prev = cbsd_corr.copy()
        c_prev.thumbnail((img_w - 8, img_h - 28))
        cx = work_x + 20 + (img_w - c_prev.width) // 2
        cy = out_y + 24 + (img_h - 28 - c_prev.height) // 2
        frame.paste(c_prev, (cx, cy))
        draw.text((work_x + 30, out_y + 6), "🔴 Corrupted (Gaussian Noise σ=25)", fill="#fca5a5", font=font_bold)
        
        draw.rounded_rectangle([work_x + 40 + img_w, out_y, work_x + 40 + img_w * 2, out_y + img_h],
                               radius=8, fill="#13192a", outline="#22c55e", width=2)
        r_prev = cbsd_rest.copy()
        r_prev.thumbnail((img_w - 8, img_h - 28))
        rx = work_x + 40 + img_w + (img_w - r_prev.width) // 2
        ry = out_y + 24 + (img_h - 28 - r_prev.height) // 2
        frame.paste(r_prev, (rx, ry))
        draw.text((work_x + 50 + img_w, out_y + 6), "🟢 Restored (BayesShrink Wavelet)", fill="#86efac", font=font_bold)
        
        tbl_y = out_y + img_h + 14
        tbl_w = work_w - 40
        draw.rounded_rectangle([work_x + 20, tbl_y, work_x + 20 + tbl_w, tbl_y + 85],
                               radius=8, fill="#111827", outline="#1f2937")
        draw.text((work_x + 32, tbl_y + 8), "Quality Metrics Improvement", fill="#94a3b8", font=font_small)
        
        col_w = tbl_w // 3
        draw.text((work_x + 32, tbl_y + 28), "PSNR (Peak Signal-to-Noise)", fill="#cbd5e1", font=font_small)
        draw.text((work_x + 32, tbl_y + 46), "20.26 dB  ➔  25.48 dB", fill="#f1f5f9", font=font_bold)
        draw.text((work_x + 32, tbl_y + 64), "+ 5.22 dB  (Greatly Enhanced)", fill="#4ade80", font=font_small)
        
        draw.text((work_x + 32 + col_w, tbl_y + 28), "SSIM (Structural Similarity)", fill="#cbd5e1", font=font_small)
        draw.text((work_x + 32 + col_w, tbl_y + 46), "0.4682  ➔  0.6852", fill="#f1f5f9", font=font_bold)
        draw.text((work_x + 32 + col_w, tbl_y + 64), "+ 0.2170  (Sharp Edge Recovery)", fill="#4ade80", font=font_small)
        
        draw.text((work_x + 32 + col_w*2, tbl_y + 28), "LPIPS (Perceptual Error)", fill="#cbd5e1", font=font_small)
        draw.text((work_x + 32 + col_w*2, tbl_y + 46), "0.3582  ➔  0.2455", fill="#f1f5f9", font=font_bold)
        draw.text((work_x + 32 + col_w*2, tbl_y + 64), "- 0.1127  (Closer to Real Photo)", fill="#4ade80", font=font_small)
        
        dash_y = tbl_y + 98
        draw.rounded_rectangle([work_x + 20, dash_y, work_x + 20 + tbl_w, dash_y + 240],
                               radius=8, fill="#0b0f19", outline="#253556")
        draw.text((work_x + 32, dash_y + 8), "📊 Detailed 5-Panel Comparison & Error Heatmaps", fill="#93c5fd", font=font_bold)
        comp_prev = cbsd_comp.copy()
        comp_prev.thumbnail((tbl_w - 20, 200))
        cpx = work_x + 20 + (tbl_w - comp_prev.width) // 2
        cpy = dash_y + 30
        frame.paste(comp_prev, (cpx, cpy))
        
        retry_y = dash_y + 250
        retry_click = state.get("retry_click", False)
        draw.rounded_rectangle([work_x + 20, retry_y, work_x + 20 + tbl_w, retry_y + 38], radius=8,
                               fill="#1e293b" if not retry_click else "#334155",
                               outline="#60a5fa" if retry_click else "#475569")
        draw.text((work_x + tbl_w//2 - 65, retry_y + 10), "🔄  Try Another Image", fill="#ffffff", font=font_bold)

    cx = state.get("cursor_x", -100)
    cy = state.get("cursor_y", -100)
    if cx > 0 and cy > 0:
        draw_cursor(draw, cx, cy, click=state.get("cursor_click", False))

    return frame

print("Generating demo frames...")
frames = []

# Phase 1: Initial empty UI (15 frames)
for i in range(15):
    t = i / 14
    cx = int(900 - t * 150)
    cy = int(500 - t * 290)
    frames.append(render_frame({"cursor_x": cx, "cursor_y": cy, "has_image": False}))

# Phase 2: Click to upload image (8 frames)
for i in range(8):
    click = i < 4
    frames.append(render_frame({"cursor_x": 750, "cursor_y": 210, "cursor_click": click, "has_image": True}))

# Phase 3: Move cursor to 'Apply & Restore' button (12 frames)
for i in range(12):
    t = i / 11
    cx = int(750 - t * 50)
    cy = int(210 + t * 90)
    frames.append(render_frame({"cursor_x": cx, "cursor_y": cy, "has_image": True}))

# Phase 4: Click 'Apply & Restore' button (8 frames)
for i in range(8):
    click = i < 4
    frames.append(render_frame({"cursor_x": 700, "cursor_y": 300, "cursor_click": click, "btn_pressed": click, "has_image": True}))

# Phase 5: Fast Smooth Restoration Progress (20 frames)
prog_steps = [
    (0.15, "Applying degradation: Gaussian Noise (σ=25) …"),
    (0.35, "Restoring image with BayesShrink Wavelet …"),
    (0.70, "Computing PSNR & SSIM metrics …"),
    (0.85, "Building 5-panel comparison report …"),
    (1.00, "Completed in 0.3s!"),
]
for p_idx, (p_val, p_desc) in enumerate(prog_steps):
    for f in range(4):
        frames.append(render_frame({
            "has_image": True,
            "progress": p_val,
            "prog_desc": p_desc,
            "cursor_x": 800,
            "cursor_y": 340,
        }))

# Phase 6: Showcase Restored Output & Quality Metrics (25 frames)
for i in range(25):
    t = i / 24
    cx = int(500 + t * 300)
    cy = int(590 + math.sin(t * 3.14) * 20)
    frames.append(render_frame({
        "has_image": True,
        "is_done": True,
        "progress": 1.0,
        "prog_desc": "Completed in 0.3s!",
        "cursor_x": cx,
        "cursor_y": cy,
        "scroll_y": 0,
    }))

# Phase 7: Smooth Scroll down to Detailed Matplotlib Dashboard (20 frames)
for i in range(20):
    t = i / 19
    smooth_t = (1 - math.cos(t * math.pi)) / 2
    sy = int(smooth_t * 220)
    frames.append(render_frame({
        "has_image": True,
        "is_done": True,
        "progress": 1.0,
        "prog_desc": "Completed in 0.3s!",
        "cursor_x": 750,
        "cursor_y": 550,
        "scroll_y": sy,
    }))

# Phase 8: Move to 'Try Another Image' button & click (18 frames)
for i in range(18):
    t = i / 17
    cx = 750
    cy = int(580 + t * 40)
    click = i >= 10 and i <= 14
    frames.append(render_frame({
        "has_image": True,
        "is_done": True,
        "progress": 1.0,
        "prog_desc": "Completed in 0.3s!",
        "cursor_x": cx,
        "cursor_y": cy,
        "scroll_y": 220,
        "retry_click": click,
    }))

# Phase 9: Reset back to top and clean state (10 frames)
for i in range(10):
    t = i / 9
    sy = int((1 - t) * 220)
    frames.append(render_frame({
        "has_image": False,
        "cursor_x": 750,
        "cursor_y": 300,
        "scroll_y": sy,
    }))

print(f"Total frames generated: {len(frames)}")

# Export as MP4
mp4_path = "assets/demo_recording.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(mp4_path, fourcc, FPS, (W, H))
for fr in frames:
    bgr = cv2.cvtColor(np.array(fr), cv2.COLOR_RGB2BGR)
    writer.write(bgr)
writer.release()
print(f"[OK] Saved MP4: {mp4_path} ({os.path.getsize(mp4_path):,} bytes)")

# Export as Optimized Animated GIF (for inline GitHub README playback)
# Scale to 960x540 for fast loading on GitHub
gif_path = "assets/demo_recording.gif"
print("Optimizing and saving GIF...")
gif_frames = []
for fr in frames:
    small = fr.resize((960, 540), Image.Resampling.BILINEAR)
    gif_frames.append(small.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))

gif_frames[0].save(
    gif_path,
    save_all=True,
    append_images=gif_frames[1:],
    duration=int(1000 / FPS),
    loop=0,
    optimize=True,
)
print(f"[OK] Saved GIF: {gif_path} ({os.path.getsize(gif_path):,} bytes)")
