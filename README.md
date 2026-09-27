---
title: AI Image Restoration System
emoji: 🖼️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
license: mit
---

# 🖼️ AI-Based Image Restoration System

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://ai-image-restoration-zane.onrender.com/)
[![Output Gallery](https://img.shields.io/badge/📊_Output_Gallery-52_Benchmark_Screenshots-7c3aed?style=for-the-badge)](#-sample-results--extensive-52-image-benchmark-gallery)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ranker-AdhipKumar/AI-image-restoration)


A modular deep learning + classical CV pipeline that reconstructs images degraded by **noise**, **blur**, **missing regions**, and **compression artifacts** — with a full **Gradio web UI**, **CLI**, and quantitative evaluation via **PSNR**, **SSIM**, and **LPIPS**.

![Interactive Web UI](assets/web_ui.png)

---

## ✨ Features

- **🩺 AI Auto-Diagnosis Engine**: Referenceless defect detection (noise level σ, edge sharpness index, JPEG 8x8 blockiness ratio, dynamic range) without needing ground truth.
- **✨ Real-World Blind Restoration**: One-click auto-pilot restoration for real vintage scans, blurry photos, noisy low-light shots, and compressed web images.
- **4 synthetic degradation modes** with tunable parameters for benchmarking and algorithmic experimentation.
- **Cascading restoration architecture** — state-of-the-art DL models (NAFNet, DnCNN, LaMa) with automatic fallback to robust classical methods.
- **Full Quality Metrics Suite**: Both Full-Reference (PSNR, SSIM, LPIPS) and No-Reference (BIQS 0-100, estimated noise σ, sharpness, blockiness ratio).
- **Gradio web UI** with dedicated Real-World Restoration, Single-Image simulation, Sample Gallery, and Batch Evaluation tabs.
- **CLI** supporting blind diagnosis, real-world enhancement, and synthetic batch benchmarking.
- **Matplotlib comparison dashboards**: 4-panel blind diagnostic residual maps + 5-panel full-reference heatmaps and metric charts.
- **44 benchmark sample images** downloadable out of the box (Kodak, CBSD68, synthetic).


---

## 🌐 Live Demo & Screen Recording

You can try the interactive restoration web app live in your browser (no local setup required):

[![Live Demo](https://img.shields.io/badge/🚀_Launch_Live_Demo-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://ai-image-restoration-zane.onrender.com/)

👉 **Live URL:** [https://ai-image-restoration-zane.onrender.com/](https://ai-image-restoration-zane.onrender.com/)

> [!NOTE]
> Due to cloud container CPU and memory constraints on free-tier hosting, resource-heavy operations (especially single-image simulation pipelines with large dimensions or complex deconvolution filters) may occasionally experience latency or unresponsive states. For the fastest, fully unconstrained, and most dependable experience, running the application locally is recommended.

> [!IMPORTANT]
> **A Note on Design Intent & Compute Scope**:  
> This project was intentionally engineered from the ground up as a **completely standalone, self-contained system capable of running smoothly on consumer laptops and standard CPUs without requiring a dedicated high-end GPU**.  
> Unlike commercial enterprise AI tools whose processing pipelines run on massive multi-GPU clusters in hyperscale cloud data centers, this project prioritizes **instant edge accessibility, local privacy, zero API costs, and lightweight resource efficiency**. Please evaluate and benchmark this system with this edge-first, CPU-friendly design philosophy in mind!

### 🎥 System Walkthrough & Demo Recording



The screen recording below demonstrates the complete workflow in action: image upload, real-time degradation simulation, instant restoration, quantitative metrics calculation (PSNR/SSIM/LPIPS), detailed 5-panel error heatmap comparison, and single-click reset:

![AI Image Restoration System Demo](assets/demo_recording.gif)

> 📹 *A high-definition MP4 recording is also available at [`assets/demo_recording.mp4`](assets/demo_recording.mp4).*

---

## 💡 Methodology: Our Approach & Why We Chose It

Image restoration is a mathematically ill-posed inverse problem: an observed image $y$ is formed by $y = \mathcal{H}(x) + n$, where infinitely many plausible clean images $x$ could produce $y$. Rather than relying on heavyweight, black-box cloud APIs, our system is engineered around **four core methodological principles**:

1. **Hybrid Cascading Architecture (Deep Learning + Deterministic Classical Fallback)**:
   - *Why*: Deep neural models (**NAFNet**, **DnCNN**, **LaMa**) achieve peak structural and perceptual fidelity, but can fail in resource-constrained environments (missing GPU drivers, high memory consumption, weight download timeouts).
   - *Our Solution*: A multi-tier cascade where neural models run when hardware permits, and automatically fall back to mathematical classical algorithms (**BayesShrink wavelets**, **Wiener deconvolution**, **Total Variation**, **Navier-Stokes**) without crashing.
   - *The Advantage*: **100% operational reliability** on any laptop CPU with zero setup hurdles.

2. **Dual-Mode Operation: Laboratory Simulation vs. Real-World Blind Restoration**:
   - *Why*: Academic pipelines evaluate models by synthesizing artificial noise onto clean images with known parameters ($\sigma$, blur kernel). Real users, however, bring images that are *already* damaged with unknown defects and no ground truth.
   - *Our Solution*: We provide both a **Simulation Laboratory** (for parameter-controlled algorithmic benchmarking) and an **AI Blind Real-World Restoration Engine** (which automatically diagnoses unknown defects and applies tailored multi-stage restoration).

3. **Referenceless Defect Diagnostics & No-Reference Quality (NR-IQA)**:
   - *Why*: Full-reference metrics (PSNR, SSIM, LPIPS) cannot be computed on real-world photos without a reference.
   - *Our Solution*: Implemented mathematically grounded natural scene statistics (NSS):
     - **Noise Level ($\sigma$)**: Edge-masked 3×3 Immerkaer Laplacian variance combined with Donoho's Wavelet MAD on $HH_1$ subbands.
     - **Sharpness**: Tenengrad gradient energy and variance of the Laplacian $\text{Var}(\Delta I)$.
     - **JPEG Blockiness**: Wang et al. 8×8 DCT grid boundary discontinuity ratio.
     - **Blind Image Quality Score (BIQS)**: A unified 0–100 index giving verifiable before/after improvement tracking.

4. **Self-Contained, Framework-Agnostic Implementation**:
   - *Why*: Popular restoration packages often rely on brittle toolkits (e.g. `basicsr`) requiring specific CUDA compiler setups and outdated dependencies.
   - *Our Solution*: NAFNet is implemented as a clean, self-contained PyTorch module with zero external framework dependencies, and matplotlib visualizations are rendered using headless FigureCanvas backends for thread-safe UI execution.

> 📖 *For a comprehensive mathematical deep dive into our algorithmic formulation and architectural choices, see [`APPROACH.md`](APPROACH.md).*

---

## 🧠 Model Architecture


Each degradation type has a **primary DL model** with graceful fallbacks:

| Degradation | Primary Model | Fallback |
|---|---|---|
| Noise | NAFNet-SIDD / DnCNN (pretrained) | BayesShrink Wavelet → OpenCV NLM |
| Blur | NAFNet-GoPro (pretrained) | Wiener Deconvolution → Richardson-Lucy |
| Missing Regions | LaMa Inpainting | Navier-Stokes → Telea FMM |
| JPEG Artifacts | DnCNN / NAFNet | TV Chambolle → OpenCV NLM |

**NAFNet** (Nonlinear Activation Free Network) — ECCV 2022, megvii-research — is implemented as a self-contained PyTorch module (`restoration/nafnet_arch.py`), with no dependency on `basicsr`.

---

## 📊 Sample Results & Extensive 52-Image Benchmark Gallery

To demonstrate the full reconstruction capability, numerical fidelity, and quantitative metrics tracking of our restoration cascade across standard computer vision test suites, we have generated an archive of **52 real output comparison screenshots** in [`assets/demo_outputs/`](assets/demo_outputs/).

Each figure captures the **complete 5-panel or 4-panel visualizer output** generated directly by the system:
1. **Side-by-Side Visual Comparison**: Pristine Reference vs. Degraded Input vs. AI Restored Output
2. **Pixel Difference Heatmaps**: Channel-averaged residual maps visualizing exactly what was cleaned away
3. **Quantitative Metrics Bar Charts**: Side-by-side **PSNR (dB)**, **SSIM**, and **LPIPS** comparisons with directional indicators ($\uparrow$ / $\downarrow$)
4. **AI Blind Diagnostics (for real-world cases)**: Edge maps, artifact maps, and before/after **Blind Image Quality Scores (BIQS)**

---

### 🌟 Featured Benchmark Comparisons (Preview from Gallery)

| 📸 Kodak PhotoCD Denoising (Gaussian $\sigma=25$) | 🌪️ Kodak PhotoCD Motion Deblurring ($l=25$px, $45^\circ$) |
|:---:|:---:|
| [![Kodak Denoising](assets/demo_outputs/demo_01_kodak_kodim01_gaussian_noise_s25.png)](assets/demo_outputs/demo_01_kodak_kodim01_gaussian_noise_s25.png) | [![Kodak Deblurring](assets/demo_outputs/demo_09_kodak_kodim09_motion_blur_l25_a45.png)](assets/demo_outputs/demo_09_kodak_kodim09_motion_blur_l25_a45.png) |
| **PSNR: 20.3 dB → 22.7 dB (+2.4 dB)** · *Noise residual heatmap* | **PSNR: 23.4 dB → 24.1 dB (+0.7 dB)** · *Wiener deconvolution* |

| 🩹 Kodak PhotoCD Inpainting (Missing Regions) | 🩺 AI Blind Real-World Diagnosis & Auto-Restoration |
|:---:|:---:|
| [![Kodak Inpainting](assets/demo_outputs/demo_13_kodak_kodim13_inpaint_2patches_s70.png)](assets/demo_outputs/demo_13_kodak_kodim13_inpaint_2patches_s70.png) | [![Blind Real-World Diagnosis](assets/demo_outputs/demo_29_cbsd_0001_blind_camera_noise.png)](assets/demo_outputs/demo_29_cbsd_0001_blind_camera_noise.png) |
| **PSNR: 11.2 dB → 24.5 dB (+13.3 dB)** · *Navier-Stokes FMM* | **BIQS: 75.8 → 88.2 (+12.4 pts)** · *Referenceless defect cascade* |

---

### 📂 Comprehensive 52-Image Output Catalog

The complete collection of **52 real output comparison screenshots** is organized directly in the repository at [`assets/demo_outputs/`](assets/demo_outputs/):

| Benchmark Category | Screenshots | Image Source | Degradation Scenarios & Models Evaluated |
|---|:---:|---|---|
| **📸 Kodak PhotoCD Suite** | **24 Images** | Kodak `kodim01` – `kodim24` | Gaussian Noise ($\sigma = 25, 35, 50$), Salt & Pepper ($p = 0.05, 0.08$), Gaussian Blur ($k=11, 15, 19$), Motion Blur ($l=20–35$px, $\theta=30^\circ–90^\circ$), Multi-Patch Inpainting, JPEG Artifacts ($Q=5, 10, 15, 20$), Mixed noise+blur |
| **🩺 AI Blind Real-World Diagnosis** | **10 Images** | Kodak, CBSD68, Synthetic | Sensor noise estimation ($\sigma$), lens softness, severe web JPEG compression, low-light grain, multi-stage blind restoration without ground truth |
| **🔬 CBSD Classical Benchmark** | **4 Images** | CBSD68 (`0001` – `0004`) | Classical color image denoising, motion deblurring, structured inpainting, and JPEG deblocking |
| **🎨 Synthetic Benchmark Suite** | **14 Images** | Synthetic (`000` – `015`) | Parametric stress testing across geometric gradients, high-frequency contours, and edge topologies |

> 📁 *Browse all 52 high-resolution comparison screenshots directly in [`assets/demo_outputs/`](assets/demo_outputs/) or regenerate them anytime using `python generate_demo_outputs.py`.*

---

## 📚 Benchmark Datasets & Official Links

The models, evaluations, and sample galleries in this project are grounded in standard, peer-reviewed computer vision datasets. You can inspect, download, and cite the official sources below:

| Dataset | Sample Count | Primary Purpose | Official Links & Repositories |
|---|:---:|---|---|
| **Kodak PhotoCD (Kodak24)** | 24 images | Standard benchmark for denoising, deblurring, compression, and perceptual fidelity | [Official Archive (r0k.us)](https://r0k.us/graphics/kodak/) · [Hugging Face Dataset (eugenesiow/Kodak24)](https://huggingface.co/datasets/eugenesiow/Kodak24) |
| **CBSD68 / BSD68 (Berkeley Segmentation)** | 68 images | Color benchmark for classical & deep denoising, edge preservation, and structural integrity | [Berkeley BSDS Portal](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) · [CBSD68 GitHub Mirror (clausmichele)](https://github.com/clausmichele/CBSD68-dataset) · [Hugging Face Dataset](https://huggingface.co/datasets/eugenesiow/BSD68) |
| **DIV2K Validation Set** | 100 images | High-resolution 2K photographic benchmark for super-resolution and image restoration | [Official DIV2K Portal](https://data.vision.ee.ethz.ch/cvl/DIV2K/) · [Hugging Face Dataset (eugenesiow/Div2k)](https://huggingface.co/datasets/eugenesiow/Div2k) |
| **USC-SIPI Miscellaneous** | 16 images | Classic historical test images (*Lena*, *Mandrill/Baboon*, *Peppers*, *House*) | [USC-SIPI Database](https://sipi.usc.edu/database/database.php?volume=misc) |
| **SIDD (Smartphone Image Denoising Dataset)** | 30,000 pairs | Real raw/sRGB smartphone camera noise dataset (used in NAFNet-SIDD pre-training) | [Official SIDD Project Page](https://www.eecs.yorku.ca/~kamyar/datasets/sidd/) |
| **GoPro Motion Blur Dataset** | 3,214 pairs | High-speed dynamic motion blur captures (used in NAFNet-GoPro pre-training) | [DeepDeblur GitHub](https://github.com/SeungjunNah/DeepDeblur_release) |

> 📥 *You can download and populate the benchmark samples locally with a single command:*
> ```bash
> python sample_images/download_samples.py
> ```

---




## 🚀 Quick Start

### 1. Clone & set up environment
```bash
git clone https://github.com/Ranker-AdhipKumar/AI-image-restoration.git
cd AI-image-restoration

python -m venv venv
# Windows:
.\venv\Scripts\pip.exe install -r requirements.txt
# Linux/Mac:
pip install -r requirements.txt
```

### 2. Download sample images
```bash
python sample_images/download_samples.py
```

### 3. Launch the Web UI
```bash
# Windows:
.\venv\Scripts\python.exe app.py

# Linux/Mac:
python app.py
```
Open **http://127.0.0.1:7860** in your browser.

---

## 💻 CLI Usage

```bash
# 🩺 1. Blindly diagnose defects on a real photo (no ground truth needed)
python main.py --input old_photo.jpg --diagnose

# ✨ 2. Restore a real-world degraded image directly using AI auto-restoration
python main.py --input old_photo.jpg --real-world --output results/

# 🔬 3. Simulate degradation & restore with benchmark evaluation
python main.py --input photo.jpg --degradation noise --sigma 30 --output results/

# Deblur with Wiener deconvolution
python main.py --input photo.jpg --degradation blur --kernel-size 15 --method wiener

# Fill missing regions
python main.py --input photo.jpg --degradation inpaint --n-patches 3 --patch-size 80

# Remove JPEG artifacts
python main.py --input photo.jpg --degradation artifact --quality 10

# Batch process a folder
python main.py --input ./sample_images --degradation noise --batch --output results/

# Print metrics as JSON
python main.py --input photo.jpg --degradation noise --json
```

### CLI Parameters

| Flag | Description | Default |
|---|---|---|
| `--diagnose` | Run blind defect diagnosis without ground truth | `False` |
| `--real-world` | Restore a real-world degraded image using blind auto-restoration | `False` |
| `--degradation` | `noise`, `salt_pepper`, `blur`, `motion_blur`, `inpaint`, `artifact`, `mixed` | required (for simulation) |
| `--method` | `auto`, `nafnet`, `dncnn`, `wavelet`, `nlm`, `wiener`, `rl`, `lama`, `ns`, `telea`, `tv` | `auto` |
| `--max-dim` | Resize image so max dimension ≤ this value | `768` |
| `--sigma` | Gaussian noise level | `25` |

| `--kernel-size` | Blur kernel size | `15` |
| `--quality` | JPEG quality (1=worst, 50=moderate) | `10` |
| `--n-patches` | Number of inpainting mask patches | `3` |
| `--batch` | Process all images in input directory | `false` |
| `--json` | Output metrics as JSON to stdout | `false` |

---

## 🏗️ Project Structure

```
├── app.py                     # Gradio web UI (Real-world restoration + simulation tabs)
├── main.py                    # CLI entry point (--diagnose, --real-world, simulation)
├── diagnostics.py             # Blind defect diagnosis & referenceless quality assessment
├── degradation.py             # 7 degradation simulators
├── metrics.py                 # PSNR · SSIM · LPIPS · No-Reference BIQS
├── visualizer.py              # Matplotlib comparison figures & blind diagnostic dashboards
├── utils.py                   # I/O helpers, atomic weight downloader
├── generate_demo_outputs.py   # Script generating 52 benchmark comparison figures
├── APPROACH.md                # Comprehensive engineering methodology & architectural deep-dive
├── restoration/
│   ├── __init__.py            # Unified restoration dispatcher
│   ├── blind_restorer.py      # Multi-stage auto-pilot blind restoration cascade
│   ├── nafnet_arch.py         # Self-contained NAFNet (no basicsr needed)
│   ├── denoiser.py            # Denoising cascade (NAFNet / DnCNN / Wavelet / NLM)
│   ├── deblurrer.py           # Deblurring cascade (NAFNet / Wiener / Richardson-Lucy)
│   ├── inpainter.py           # Inpainting cascade (LaMa / Navier-Stokes / Telea)
│   └── artifact_remover.py   # Artifact removal cascade (DnCNN / TV / NLM)
├── assets/
│   ├── web_ui.png             # UI screenshot
│   ├── demo_recording.gif     # Walkthrough animated recording
│   └── demo_outputs/          # 52 real output benchmark comparison figures
└── sample_images/
    └── download_samples.py    # Downloads Kodak, CBSD68, DIV2K, synthetic images
```


---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `torch` + `torchvision` | Deep learning inference |
| `opencv-python` | Image I/O, classical restoration |
| `scikit-image` | PSNR, SSIM, wavelet denoising |
| `scipy` | Wiener / Richardson-Lucy deconvolution |
| `lpips` | Perceptual image quality metric |
| `gradio` | Web interface |
| `matplotlib` | Comparison figure generation |
| `Pillow`, `numpy` | Image arrays and I/O |

---

## 📖 References

- **NAFNet**: [Simple Baselines for Image Restoration](https://arxiv.org/abs/2204.04676), Chen et al., ECCV 2022
- **DnCNN**: [Beyond a Gaussian Denoiser](https://arxiv.org/abs/1608.03981), Zhang et al., TIP 2017
- **LaMa**: [Resolution-robust Large Mask Inpainting](https://arxiv.org/abs/2109.07161), Suvorov et al., WACV 2022
- **LPIPS**: [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924), Zhang et al., CVPR 2018
- **SSIM**: [Image Quality Assessment](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf), Wang et al., TIP 2004

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
