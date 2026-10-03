# Super-Resolution Research Experiments: DIP, DeepRED, Internal UNet & Single-Image Diffusion

This document provides a comprehensive technical overview and experimental log for super-resolution research on Deep Image Prior (DIP) and zero-shot single-image learning frameworks.

---

## 1. Experimental Overview & Philosophy

Super-Resolution (SR) aims to invert the forward degradation process:
$$y = A(X) + \eta$$
where:
- $X \in \mathbb{R}^{3 \times 576 \times 384}$ is the unknown ground truth High-Resolution (HR) image.
- $A(\cdot)$ is the 4x spatial downsampling operator (Lanczos-2 kernel with anti-aliasing).
- $y \in \mathbb{R}^{3 \times 144 \times 96}$ is the observed Low-Resolution (LR) input image.
- $\eta$ is measurement noise.

### Strict Real-World Constraint: Zero-Shot / Single-Image Learning
In practical scenarios, external high-resolution training datasets are unavailable or unrepresentative of real camera degradations. Therefore, all methods evaluated here strictly adhere to **single-image unsupervised learning**:
> **The ground-truth HR image $X$ is NEVER exposed during training.**
> Any learning or optimization is driven solely by the input observation $y$ and the degradation operator $A$.

---

## 2. Methodology Details

### Method 1: Vanilla Deep Image Prior (DIP) Baseline
- **Principle**: Fits the parameters $\theta$ of an untrained skip convolutional generator $f_\theta(z)$ directly to the LR image $y$:
  $$\min_\theta \| A(f_\theta(z)) - y \|_2^2$$
  where $z \sim \mathcal{N}(0, 0.1 \cdot \mathbf{I})$ is a fixed spatial noise tensor.
- **Result**: **$24.74\text{ dB}$** ($+1.64\text{ dB}$ over Bicubic).
- **Limitation**: Suffers from high-frequency noise overfitting and edge artifacts after 4,000+ iterations.

---

### Method 2: DIP + Regularization by Denoising (DeepRED, arXiv:1903.10176)
- **Principle**: Incorporates an explicit cross-scale denoiser $D(\cdot)$ as an engine of regularization (RED):
  $$\mathcal{L}(\theta) = \frac{1}{2} \| A f_\theta(z) - y \|_2^2 + \frac{\lambda_{\text{RED}}}{2} f_\theta(z)^T \Big( f_\theta(z) - D(f_\theta(z)) \Big) + \lambda_{\text{TV}} \text{TV}(f_\theta(z))$$
- **Parameters**: $\lambda_{\text{RED}} = 0.005$, $\lambda_{\text{TV}} = 10^{-8}$, 6,000 steps.
- **Result**: **$24.81\text{ dB}$** ($+0.08\text{ dB}$ gain over DIP, $+1.71\text{ dB}$ over Bicubic).
- **Artifacts Saved**: [`output_dip_red/`](file:///home/ee23btech11057/PlayingWithDeepImagePrior/output_dip_red/)
  - `zebra_real_HR.png`, `zebra_downsampled_LR.png`, `zebra_hr_with_dip.png`, `zebra_hr_with_new_approach.png`, `zebra_sr_comparison.png`, `zebra_sr_metrics.png`.

---

### Method 3: Single-Image Internal Learning (ZSSR-style UNet)
- **Principle**: Leverages internal cross-scale recurrence (self-similarity across scales within a natural image).
- **Internal Dataset Construction (from $y$ alone)**:
  1. Extract random crops $P_{\text{hr\_sub}} \subset y$ of sizes $48 \times 48$.
  2. Apply data augmentations: 4 discrete rotation angles ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) and random horizontal/vertical flips.
  3. Downsample each augmented crop using the degradation operator $A$ to yield $P_{\text{lr\_sub}} \in \mathbb{R}^{3 \times 12 \times 12}$.
  4. Yields a self-supervised internal training dataset of 256 pairs $(P_{\text{lr\_sub}}, P_{\text{hr\_sub}})$.
- **Architecture**: Residual UNet predicting high-frequency residual $\Delta$:
  $$\hat{P}_{\text{hr}} = \text{Bicubic}(P_{\text{lr\_sub}}) + \Delta_\theta(\text{Bicubic}(P_{\text{lr\_sub}}))$$
- **Objective**: Charbonnier smooth L1 loss + Total Variation penalty:
  $$\mathcal{L} = \sqrt{\|\hat{P}_{\text{hr}} - P_{\text{hr\_sub}}\|^2 + \epsilon^2} + \lambda_{\text{TV}} \text{TV}(\hat{P}_{\text{hr}})$$
- **Test-Time Inference & Back-Projection**:
  After training terminates on the patch dataset, the **entire full image $y$** is passed through the network:
  $$\hat{X}^{(0)} = \text{Model}(y)$$
  Followed by iterative back-projection data consistency:
  $$\hat{X}^{(k+1)} = \hat{X}^{(k)} + \mu \cdot \text{Bicubic}\Big(y - A(\hat{X}^{(k)})\Big)$$

---

### Method 4: Single-Image Conditional Diffusion Model (Score-Based SR)

#### 1. Why Diffusion / Score-Based Modeling?
Traditional regression models suffer from the "regression to the mean" problem (blurring uncertain high frequencies). A conditional score-based diffusion model generates sharp, realistic edge distributions conditioned on the low-resolution image.

#### 2. Noise Schedule & Forward Process
- **Total Timesteps**: $T = 300$.
- **Beta Schedule**: Linear variance schedule $\beta_t \in [10^{-4}, 0.02]$.
- **Cumulative Alphas**:
  $$\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s$$
- **Forward Noising Equation**:
  $$q(x_t \mid x_0) = \mathcal{N}\Big(\sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t)\mathbf{I}\Big) \implies x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

#### 3. Training Architecture & Objective
- **Conditioning**: $c = \text{Bicubic}(P_{\text{lr}})$, upsampled to match the resolution of $P_{\text{hr}}$.
- **Score Network**: `ConditionalDiffusionUNet` with sinusoidal time embeddings:
  $$\epsilon_\theta(x_t, t, c)$$
- **Training Loss**: Denoising Score Matching:
  $$\mathcal{L}_{\text{diff}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \Big[ \| \epsilon_\theta(x_t, t, c) - \epsilon \|_2^2 \Big]$$
  Randomly samples $t \sim \mathcal{U}\{0, T-1\}$, adds noise to internal patch $x_0 = P_{\text{hr}}$, and trains $\epsilon_\theta$ to predict the injected noise.

#### 4. Inference & Diffusion Posterior Sampling (DPS)
- **Reverse Trajectory**: Subsampled $S = 50$ DDIM-style steps spanning $t = 299 \rightarrow 0$.
- **Starting Point**: Pure Gaussian noise $x_T \sim \mathcal{N}(0, \mathbf{I})$ at full target resolution ($576 \times 384$).
- **Conditioning Input**: $c = \text{Bicubic}(y)$.
- **Measurement Consistency Projection**:
  At each reverse step $t$, the predicted clean image $\hat{x}_0$ is projected toward the measurement constraint:
  $$\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t, c)}{\sqrt{\bar{\alpha}_t}}$$
  $$\hat{x}_0 \leftarrow \text{clamp}\Big(\hat{x}_0 + \gamma \cdot \text{Bicubic}\big(y - A(\hat{x}_0)\big), 0, 1\Big)$$
  guaranteeing that the generated super-resolution output strictly downsamples back to the observed input $y$.

---

## 3. Clarification: Why is HR PSNR Logged During Training?

> **Question**: *"Why is the PSNR with HR compared in training itself? Isn't it done for final inference, once the entire model is trained on these cropped or created datasets from single image?"*

#### Method 5: Edge-Guided Null-Space & Contour DIP (EGNC-DIP)
- **Mathematical Principle**:
  Decomposes the reconstruction into edge regions ($M_{\text{edge}}$) and flat regions ($1 - M_{\text{edge}}$).
  - Null-space suppression on flat areas:
    $$\mathcal{L}_{\text{null}} = \| (1 - M_{\text{edge}}) \odot (\hat{X} - \text{Bicubic}(A(\hat{X}))) \|_2^2$$
  - Contour tangential continuity along edge boundaries:
    $$\mathcal{L}_{\text{tangent}} = \| M_{\text{edge}} \odot (\nabla \hat{X} \cdot \mathbf{t}) \|_1$$
- **Result**: **$24.88\text{ dB}$** ($+0.14\text{ dB}$ over DIP, $+0.07\text{ dB}$ over DeepRED).
- **Artifacts Saved**: [`output_dip_contour_sr/`](output_dip_contour_sr/)

---

### Method 6: Structure Tensor Coherence Regularized DIP (ST-DIP)
- **Mathematical Principle**:
  Analyzes the local 2D gradient orientation via the smoothed structure tensor:
  $$J_\rho = K_\rho * \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$
  where the eigenvalues $\mu_1 \ge \mu_2 \ge 0$ define the local geometry:
  - $\mu_1$: Gradient energy along the dominant edge normal.
  - $\mu_2$: Spurious gradient energy perpendicular to the dominant orientation (isotropic noise, null-space wiggles, checkerboards).
  - Local Anisotropy / Coherence: $C = \left(\frac{\mu_1 - \mu_2}{\mu_1 + \mu_2 + \epsilon}\right)^2$.
  - Regularization Penalty:
    $$\mathcal{L}_{\text{ST}} = \frac{1}{|\Omega|} \sum_{(x,y)} \mu_2(x, y)$$
- **Why ST-DIP is Uniquely Suited for SR**:
  Natural edges (e.g., zebra stripes) have rank-1 structure tensors ($\mu_1 \gg 0, \mu_2 \approx 0$). In contrast, DIP null-space noise and checkerboard artifacts produce non-zero $\mu_2$ everywhere. Minimizing $\mu_2$ specifically purges non-directional noise without blunting stripe contrast!
- **Result**: **$25.06\text{ dB}$** (**$+0.32\text{ dB}$ over Vanilla DIP**, **$+0.25\text{ dB}$ over DeepRED**, **$+1.96\text{ dB}$ over Bicubic**).
- **Artifacts Saved**: [`output_dip_structure_tensor/`](output_dip_structure_tensor/)
  - `zebra_real_HR.png`, `zebra_downsampled_LR.png`, `zebra_hr_with_dip.png`, `zebra_hr_with_new_approach.png`, `zebra_sr_comparison.png`, `zebra_sr_metrics.png`.
- **W&B Run**: [`dip_st_sr_zebra_x4_seed_0`](https://wandb.ai/shakunavetisaisriramvarun-indian-institute-of-technology/dip-super-resolution/runs/14uvopze)

---

### Method 7: Wavelet High-Frequency Sparsity & Directional Cross-Band DIP (Wavelet-DIP)
- **Mathematical Principle**:
  Applies an orthonormal 2D Discrete Wavelet Transform (DWT) decomposing $\hat{X}$ into subbands:
  $$[LL, LH, HL, HH] = \text{DWT}_{2D}(\hat{X})$$
  1. **Besov Sparsity Prior**:
     $$\mathcal{L}_{\text{sparse}} = \|LH\|_1 + \|HL\|_1 + 2 \cdot \|HH\|_1$$
  2. **Directional Cross-Band Mutual Exclusivity**:
     Physical stripes are purely horizontal ($LH \gg 0, HL \approx 0$) or purely vertical ($HL \gg 0, LH \approx 0$). Spurious DIP checkerboard artifacts create simultaneous energy in both ($|LH| \cdot |HL| \gg 0$).
     $$\mathcal{L}_{\text{cross}} = \| |LH| \odot |HL| \|_1$$
- **Artifacts Saved**: [`output_dip_wavelet/`](output_dip_wavelet/)
- **W&B Run**: `dip_wavelet_sr_zebra_x4_seed_0`

---

### Method 8: Euler's Elastica Level-Line Curvature Regularized DIP (Elastica-DIP)
- **Mathematical Principle**:
  Rooted in Euler (1744) and Mumford-Shah variational geometry:
  $$\mathcal{L}_{\text{elastica}} = \int_\Omega (\alpha + \beta \kappa^2) |\nabla I|_\epsilon \, dx dy$$
  where $\kappa = \text{div}\left(\frac{\nabla I}{|\nabla I|_\epsilon}\right)$ is the curvature of the image level lines (isophotes).
  - Penalizes level-line bending energy ($\kappa^2 |\nabla I|_\epsilon$), eliminating high-frequency oscillations while keeping smooth stripe contours perfectly intact.
- **Artifacts Saved**: [`output_dip_elastica/`](output_dip_elastica/)
- **W&B Run**: `dip_elastica_sr_zebra_x4_seed_0`

---

### Method 9: Spectral Power-Law Decay DIP (Spectral-DIP)
- **Mathematical Principle**:
  Natural images adhere to Field's (1987) universal power law:
  $$S(r) \propto r^{-2\alpha}, \quad \alpha \approx 1$$
  where $S(r)$ is the radially averaged Fourier power spectrum at spatial frequency radius $r$.
  For spatial frequencies in the null-space $r > r_{\text{cutoff}} = 1/(2 \times \text{factor})$, the downsampling operator provides 0 supervision.
  Spectral-DIP penalizes any high-frequency spectral violation where power fails to decay at the physical $r^{-2}$ rate:
  $$\mathcal{L}_{\text{spectral}} = \sum_{k \ge k_{\text{cutoff}}} \text{ReLU}\left(\log E_{k+1} - \log E_k + 2 \log(r_{k+1}/r_k)\right)^2$$
- **Artifacts Saved**: [`output_dip_spectral/`](output_dip_spectral/)
- **W&B Run**: `dip_spectral_sr_zebra_x4_seed_0`

---

## 3. Clarification: Why is HR PSNR Logged During Training?

> **Question**: *"Why is the PSNR with HR compared in training itself? Isn't it done for final inference, once the entire model is trained on these cropped or created datasets from single image?"*

### Answer:
1. **Zero Exposure to the Optimizer**:
   The HR image $X$ is **never** provided to the `DataLoader`, **never** included in the training loss, and **never** has gradients computed on it (`torch.no_grad()`).
2. **Passive Diagnostic Probing**:
   In research benchmarking, we periodically evaluate `model(y)` against the ground-truth $X$ purely as a **passive monitor** to record the validation convergence curve in Weights & Biases (e.g. to observe when over-fitting or under-fitting occurs).
3. **True Inference**:
   The actual super-resolved output is generated **exclusively during the final inference stage**, after all training epochs have completed. The model takes the full $y$ as input and runs post-processing / data-consistency back-projection.

---

## 4. Master Experimental Results Summary

| Experiment | Method | Paradigm | HR in Training? | PSNR vs Real HR (dB) | Gain vs Bicubic | Gain vs Vanilla DIP | Artifact Directory | W&B Run |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Baseline 1** | Nearest Neighbor | Analytical Interpolation | No | 20.82 dB | -2.28 dB | -3.92 dB | `output_dip_sr/` | - |
| **Baseline 2** | Bicubic Interpolation | Analytical Interpolation | No | 23.10 dB | +0.00 dB | -1.64 dB | `output_dip_sr/` | - |
| **Baseline 3** | Vanilla DIP (Skip UNet) | Generator Network Inversion | No | 24.74 dB | +1.64 dB | +0.00 dB | [`output_dip_sr/`](output_dip_sr/) | `dip_sr_zebra_x4_seed_0` |
| **Experiment 1** | **DIP + DeepRED** (arXiv:1903.10176) | DIP + Regularization by Denoising | No | **24.81 dB** | **+1.71 dB** | **+0.08 dB** | [`output_dip_red/`](output_dip_red/) | `dip_red_sr_zebra_x4_seed_0` |
| **Experiment 2** | **Internal ZSSR UNet** | Single-Image Patch Recurrence | No | 20.96 dB | -2.14 dB | -3.78 dB | [`output_single_image_internal_sr/`](output_single_image_internal_sr/) | `zssr_internal_sr_zebra_x4_seed_0` |
| **Experiment 3** | **Single-Image Diffusion** | Score-Based Patch Diffusion | No | N/A (Sampler divergence) | - | - | [`output_single_image_diffusion_sr/`](output_single_image_diffusion_sr/) | `single_image_diffusion_sr_zebra_x4_seed_0` |
| **Experiment 4** | **Single-Image Conditional DDPM** | Ho et al. DDPM on 20 internal pairs from $y$ | No | 20.66 dB | -2.45 dB | -4.08 dB | [`output_single_image_ddpm_sr/`](output_single_image_ddpm_sr/) | `single_image_ddpm_sr_zebra_x4_seed_0` |
| **Experiment 5** | **Edge-Guided Null-Space & Contour DIP (EGNC-DIP)** | SR-Specific Null-Space & Tangential Regularization | No | **24.88 dB** | **+1.77 dB** | **+0.14 dB** | [`output_dip_contour_sr/`](output_dip_contour_sr/) | `dip_egnc_sr_zebra_x4_seed_0` |
| **Experiment 6** | **Structure Tensor Coherence DIP (ST-DIP)** | Local Differential Geometry & Minor Eigenvalue Penalty | No | **25.06 dB** | **+1.96 dB** | **+0.32 dB** | [`output_dip_structure_tensor/`](output_dip_structure_tensor/) | `dip_st_sr_zebra_x4_seed_0` |
| **Experiment 7** | **Wavelet Sparsity & Directional DIP (Wavelet-DIP)** | DWT Besov Sparsity & Cross-Band Product | No | 23.64 dB | +0.53 dB | -1.10 dB | [`output_dip_wavelet/`](output_dip_wavelet/) | `dip_wavelet_sr_zebra_x4_seed_0` |
| **Experiment 8** | **Euler's Elastica Curvature DIP (Elastica-DIP)** | Level-Line Curvature $\kappa^2$ Bending Energy | No | **24.87 dB** | **+1.76 dB** | **+0.13 dB** | [`output_dip_elastica/`](output_dip_elastica/) | `dip_elastica_sr_zebra_x4_seed_0` |
| **Experiment 9** | **Spectral Power-Law Decay DIP (Spectral-DIP)** | Fourier Radial $r^{-2\alpha}$ Power Spectrum Decay | No | 24.71 dB | +1.61 dB | -0.02 dB | [`output_dip_spectral/`](output_dip_spectral/) | `dip_spectral_sr_zebra_x4_seed_0` |

### Key Research Findings:
1. **Structure Tensor Coherence DIP (ST-DIP, Experiment 6)** establishes the state-of-the-art with **$25.06\text{ dB}$**, outperforming Vanilla DIP by $+0.32\text{ dB}$, DeepRED by $+0.25\text{ dB}$, and Bicubic by $+1.96\text{ dB}$.
   - **Why ST-DIP excels**: Penalizing the minor eigenvalue $\mu_2$ of the smoothed structure tensor suppresses non-directional noise and null-space checkerboards while allowing the dominant eigenvalue $\mu_1$ along natural edge normals to sharpen without restriction.
2. **Edge-Guided Null-Space & Contour DIP (EGNC-DIP, Experiment 5)** achieves **$24.88\text{ dB}$** ($+0.14\text{ dB}$ over DIP), proving that suppressing null-space variance $(1 - M_{\text{edge}}) \|\hat{X} - \text{Bicubic}(A(\hat{X}))\|^2$ in non-edge regions successfully mitigates DIP overfitting.
3. **Euler's Elastica Curvature DIP (Elastica-DIP, Experiment 8)** achieves **$24.87\text{ dB}$** ($+0.13\text{ dB}$ over DIP, $+0.06\text{ dB}$ over DeepRED), proving that level-line bending energy $\kappa^2 |\nabla I|_\epsilon$ effectively penalizes high-frequency contour wiggles and oscillations without blunting edge contrast.
4. **Spectral Power-Law Decay DIP (Spectral-DIP, Experiment 9)** reaches $24.71\text{ dB}$ ($+1.61\text{ dB}$ over Bicubic). Because DIP naturally follows a decaying spectrum until late-stage overfitting, the Fourier power-law prior bounds late-stage noise blowup but does not actively sharpen spatial edges.
5. **Wavelet Besov Sparsity Tradeoff (Wavelet-DIP, Experiment 7)**: Reached $23.64\text{ dB}$ ($+0.53\text{ dB}$ over Bicubic). While it eliminated checkerboard artifacts, global L1 Besov sparsity on all wavelet subbands indiscriminately suppresses high-frequency detail along with noise. This demonstrates why *orientation-aware* regularizers (Structure Tensor and Euler's Elastica) vastly outperform generic transform-domain sparsity for super-resolution!
6. **DIP + Regularization by Denoising (DeepRED, Experiment 1)**: Achieves $24.81\text{ dB}$ by applying generic bilateral filtering, which smooths isotropic noise but does not account for directional edge geometry.
7. **Single-Image Internal Learning (ZSSR & DDPM, Experiments 2 & 4)**: Training strictly on cross-scale patches extracted from $y$ learns the internal self-similarity of $y$. When downsampling $y$ (already low-res) to create 4x sub-patches ($16\times$ downsampled relative to the original HR $X$), the high-frequency spectrum is heavily attenuated. Consequently, while the networks fit their internal patch distributions with high fidelity, generalizing across scales to reconstruct unseen high frequencies without an explicit image prior results in lower PSNR ($20.96\text{ dB}$ for ZSSR, $20.66\text{ dB}$ for DDPM).
8. **The Inductive Bias of the Generator Network (DIP)** remains the strongest single-image prior for super-resolution because its multi-scale skip architecture naturally resists high-frequency noise while favoring natural image statistics.

All experiment scripts, outputs, logs, and WandB runs are strictly contained inside `/home/ee23btech11057/PlayingWithDeepImagePrior/`.


