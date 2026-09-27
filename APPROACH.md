# 💡 Engineering Methodology & Architectural Approach

> **Project:** AI-Based Image Restoration System  
> **Author:** Adhip Kumar  
> **Repository:** [AI-image-restoration](https://github.com/Ranker-AdhipKumar/AI-image-restoration)

---

## 📌 Executive Summary

Image restoration is a classical, severely ill-posed inverse problem. Given an observed image $y$ degraded by an unknown or parameterized degradation operator $\mathcal{H}$ and corrupted by noise $n$:

$$y = \mathcal{H}(x) + n$$

The objective is to reconstruct an estimate $\hat{x}$ that maximizes structural similarity, perceptual realism, and signal fidelity to the latent ground truth $x$.

Modern computer vision offers two extreme approaches:
1. **Hyperscale Cloud Models (Diffusion / Large GANs)**: Millions or billions of parameters, heavy server-side GPU requirements, high inference latency (5–30s per image), and costly cloud API dependencies.
2. **Pure Classical Filters**: Fast and deterministic, but rigid and unable to adapt to complex semantic distributions or deep sensor noise.

This project was built around a third, pragmatic design philosophy: **a lightweight, hybrid cascading system engineered to run reliably on consumer laptops without requiring a dedicated GPU**, while providing state-of-the-art neural restoration when compute is available and instant, graceful fallbacks when it is not.

---

## 💻 Design Philosophy: Laptop-First Execution vs. Hyperscale Data Centers

> [!IMPORTANT]
> **A Note on Design Intent & Benchmark Scope**:  
> This project was intentionally designed from the ground up as a **completely standalone, self-contained system capable of running on consumer laptops and standard CPUs without dedicated GPU hardware**.  
> 
> Unlike commercial AI image tools (e.g. Midjourney, Magnific AI, Adobe Firefly) whose processing pipelines run on massive multi-GPU clusters in enterprise hyperscale data centers, this project prioritizes:
> - **Zero cloud dependency & complete privacy**: All image processing executes strictly within the local environment.
> - **Instant CPU inference**: Near-instant execution (<0.2s for classical filters, ~1–4s for neural cascades) on standard quad-core laptop processors.
> - **Zero API fees or operational costs**: Fully open-source and free to run indefinitely.
> - **Deterministic & auditable metrics**: Quantitative verification using mathematical metrics (PSNR, SSIM, LPIPS, BIQS) rather than hallucinated generative pixels.
> 
> Please evaluate and benchmark this system with this lightweight, edge-first architecture in mind.

---

## 🔬 Why We Chose This Approach: The Four Pillars

### Pillar 1: Hybrid Cascading Architecture (Neural + Classical Fallback)

* **The Problem**: Standard PyTorch restoration models often fail in production environments due to missing CUDA drivers, memory overflow (OOM), or network latency when downloading gigabyte-sized weights.
* **Our Solution**: Every restoration mode operates as a **hierarchical cascade**:
  * **Tier 1 (Neural Primary)**: Self-contained **NAFNet** (Nonlinear Activation Free Network, ECCV 2022) or **DnCNN** (IEEE TIP 2017) when weights and resources are available.
  * **Tier 2 (Wavelet / Frequency Deconvolution)**: BayesShrink adaptive wavelet soft-thresholding (for noise) or Wiener deconvolution with estimated point spread functions (for blur).
  * **Tier 3 (Spatial Classical)**: OpenCV Non-Local Means (NLM), Total Variation Chambolle, or Navier-Stokes Fast Marching inpainting.
* **Why We Chose It**: Guarantees **100% operational uptime**. The application never crashes or hangs with CUDA errors; it delivers the best possible reconstruction within the hardware envelope of the host machine.

---

### Pillar 2: Dual-Mode Operation (Simulation Lab vs. Real-World Restoration)

* **The Problem**: Academic papers test models by adding synthetic Gaussian noise to clean images and computing PSNR. Real-world users, however, bring images that are *already* damaged (vintage family photos, blurry action shots, noisy low-light smartphone snaps, blocky web JPEGs) where no clean reference image exists.
* **Our Solution**:
  1. **Simulation Lab (Tabs 2–4 & CLI)**: Parameterized synthetic corruption (Gaussian/Poisson/Salt-and-Pepper noise, Gaussian/Motion blur, block masking, JPEG compression) for controlled research benchmarking.
  2. **Blind Real-World Engine (Tab 1 & `--real-world`)**: Automated defect diagnostics without ground truth, executing an intelligent sequential cascade tailored to the diagnosed damage.
* **Why We Chose It**: Provides academic rigor for controlled benchmarking while delivering practical everyday utility for real photographs.

---

### Pillar 3: Referenceless Defect Diagnostics & No-Reference Quality (NR-IQA)

* **The Problem**: Full-reference metrics like PSNR and SSIM require a pristine ground-truth image. When restoring real photos, these metrics are mathematically undefined.
* **Our Solution**: We implemented mathematical estimators derived from natural scene statistics (NSS):
  * **Noise Standard Deviation ($\sigma$)**: Edge-masked Immerkaer 3×3 Laplacian variance filter blended with Donoho's Wavelet Median Absolute Deviation (MAD) on high-frequency $HH_1$ subbands:
    $$\hat{\sigma} = \frac{\text{median}(|HH_1|)}{0.6745}$$
  * **Edge Sharpness**: Tenengrad gradient energy $\frac{1}{HW} \sum (\nabla_x^2 + \nabla_y^2)$ combined with log-compressed Laplacian variance $\text{Var}(\Delta I)$.
  * **JPEG Blockiness**: Wang, Sheikh & Bovik 8×8 DCT grid boundary discontinuity ratio across horizontal and vertical block interfaces:
    $$B = \frac{D_{\text{boundary}}}{Z_{\text{intra}} + \epsilon}$$
  * **Blind Image Quality Score (BIQS, 0–100)**: A composite score synthesizing noise floor, sharpness index, blockiness penalty, and luminance entropy.
* **Why We Chose It**: Allows users to inspect objective, quantifiable before-and-after improvements (e.g., "Noise reduced by 64%, Sharpness improved by 18%, BIQS gained +11.2 points") without needing a ground truth.

---

### Pillar 4: Framework-Agnostic & Self-Contained Engineering

* **The Problem**: Upstream repositories like `basicsr` introduce fragile C++/CUDA extensions, strict PyTorch version locks, and deprecated `torchvision.transforms` APIs that frequently break modern Python environments.
* **Our Solution**: We extracted and refactored the core **NAFNet** architecture into a single self-contained PyTorch module (`restoration/nafnet_arch.py`) requiring only standard `torch` and `numpy`. All matplotlib visualizers use headless FigureCanvas instances to guarantee thread-safe execution without GUI lockups.
* **Why We Chose It**: Ensures immediate cross-platform portability across Windows, macOS, Linux, and cloud containers.

---

## 📚 Benchmark Datasets Used in the Project

The evaluation, training weights, and benchmark suites in this project are grounded in established, open computer vision datasets:

| Dataset | Sample Count | Primary Role | Reference & Access Links |
|---|:---:|---|---|
| **Kodak PhotoCD (Kodak24)** | 24 images | Standard benchmark for denoising, deblurring, compression, and perceptual fidelity | [Official Archive (r0k.us)](https://r0k.us/graphics/kodak/)<br>[Hugging Face Dataset (eugenesiow/Kodak24)](https://huggingface.co/datasets/eugenesiow/Kodak24) |
| **CBSD68 / BSD68 (Berkeley Segmentation)** | 68 images | Color benchmark for classical & deep denoising, edge preservation, and structural integrity | [Berkeley Vision BSDS Portal](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)<br>[CBSD68 GitHub Mirror (clausmichele)](https://github.com/clausmichele/CBSD68-dataset)<br>[Hugging Face Dataset (eugenesiow/BSD68)](https://huggingface.co/datasets/eugenesiow/BSD68) |
| **DIV2K Validation Set** | 100 images | High-resolution 2K photographic benchmark for super-resolution and image restoration | [Official DIV2K Portal](https://data.vision.ee.ethz.ch/cvl/DIV2K/)<br>[Hugging Face Dataset (eugenesiow/Div2k)](https://huggingface.co/datasets/eugenesiow/Div2k) |
| **USC-SIPI Miscellaneous** | 16 images | Classic historical test images (*Lena*, *Mandrill*, *Peppers*, *House*) | [USC-SIPI Database](https://sipi.usc.edu/database/database.php?volume=misc) |
| **SIDD (Smartphone Image Denoising Dataset)** | 30,000 pairs | Real raw/sRGB smartphone camera noise dataset (used in NAFNet-SIDD pre-training) | [Official SIDD Project](https://www.eecs.yorku.ca/~kamyar/datasets/sidd/) |
| **GoPro Motion Blur Dataset** | 3,214 pairs | High-speed dynamic motion blur captures (used in NAFNet-GoPro pre-training) | [DeepDeblur GitHub](https://github.com/SeungjunNah/DeepDeblur_release) |

> *To download and verify all sample benchmark images into the local repository, execute:*
> ```bash
> python sample_images/download_samples.py
> ```
