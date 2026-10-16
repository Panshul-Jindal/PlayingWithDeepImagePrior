"""Experiment 4: Single-Image Conditional DDPM Super-Resolution.

A conditional Denoising Diffusion Probabilistic Model (DDPM) trained strictly
on low-resolution image pairs synthesized exclusively from the single input image y.

Key Principles:
  1. Ground truth HR image X is NEVER used during training (zero-shot, unsupervised).
  2. Training dataset is constructed directly from y:
     - Extract ~16-20 diverse crops from y (e.g. 64x64 patches, full y, augmented rotations/flips).
     - Each crop acts as the target y_target.
     - Downsample each crop by 4x using the degradation operator A to produce y_lr.
     - Pair: (Conditioning = Bicubic(y_lr), Target = y_target).
  3. A conditional UNet epsilon_theta(x_t, t, c) is trained under the standard DDPM objective:
     L_simple = E_{t, x_0, epsilon} [ || epsilon - epsilon_theta(x_t, t, c) ||^2 ]
  4. At inference time:
     - Conditioning c = Bicubic(y) upsampled to full target resolution (576x384).
     - Standard DDPM reverse sampling is executed starting from pure Gaussian noise x_T ~ N(0, I)
       to generate the full-resolution output X_hat.
     - Iterative back-projection ensures strict data consistency: Downsampler(X_hat) ~= y.

Saves:
  - zebra_real_HR.png
  - zebra_downsampled_LR.png
  - zebra_hr_with_dip.png
  - zebra_hr_with_new_approach.png
  - Visual comparison figure with all baselines (Bicubic, Nearest, DIP, DDPM) and PSNR
  - Training metrics curves (DDPM MSE loss)
Logs all metrics, parameters, and images to Weights & Biases.
"""

import math
import os
from pathlib import Path
import shutil
import sys

from PIL import Image
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = getattr(Image, "Resampling", Image).LANCZOS

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import wandb

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

PROJECT_DIR = Path(__file__).resolve().parent / "src/playingwithdeepimageprior/deep-image-prior"
sys.path.insert(0, str(PROJECT_DIR))

from models.downsampler import Downsampler
from utils.common_utils import np_to_pil, np_to_torch, torch_to_np
from utils.sr_utils import get_baselines, load_LR_HR_imgs_sr

# --- Hyperparameters ---
FACTOR = 4
EPOCHS = 1500
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
NUM_TIMESTEPS = 200
KERNEL_TYPE = "lanczos2"
PATCH_SIZE = 64
NUM_PAIRS = 20
OUTPUT_DIR = Path(__file__).resolve().parent / "output_single_image_ddpm_sr"


def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate Peak Signal-to-Noise Ratio (dB)."""
    mse = torch.mean((img1.detach() - img2.detach()) ** 2)
    if mse == 0:
        return 100.0
    return (10.0 * torch.log10((max_val ** 2) / mse)).item()


# --- DDPM Noise Schedule ---
class DDPMSchedule:
    """Standard Ho et al. 2020 DDPM linear beta schedule."""

    def __init__(self, num_timesteps=200, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.num_timesteps = num_timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion step: add noise to clean patch x_start."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def p_mean_variance(self, model, x_t, t, cond):
        """Predict posterior mean and variance for step t."""
        pred_noise = model(x_t, t, cond)

        sqrt_recip_alphas_t = (1.0 / torch.sqrt(self.alphas[t]))[:, None, None, None]
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        # Model mean: Ho et al. Eq 11
        model_mean = sqrt_recip_alphas_t * (
            x_t - (betas_t / sqrt_one_minus_alphas_cumprod_t) * pred_noise
        )
        model_log_var = self.posterior_log_variance_clipped[t][:, None, None, None]
        return model_mean, model_log_var

    @torch.no_grad()
    def p_sample(self, model, x_t, t, cond):
        """Single reverse step x_t -> x_{t-1}."""
        model_mean, model_log_var = self.p_mean_variance(model, x_t, t, cond)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float()[:, None, None, None]
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_var) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, cond):
        """Full reverse diffusion trajectory from pure noise."""
        device = self.device
        img = torch.randn(shape, device=device)

        for step in reversed(range(self.num_timesteps)):
            t_batch = torch.full((shape[0],), step, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t_batch, cond)

        return img


# --- Sinusoidal Time Embedding ---
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# --- Conditional DDPM UNet ---
class DDPMResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="reflect")
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="reflect")
        self.res = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.conv1(x))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.act(h))
        return self.res(x) + 0.1 * h


class ConditionalDDPMUNet(nn.Module):
    """UNet predicting epsilon_theta(x_t, t, cond) with conditioning concatenated."""

    def __init__(self, in_channels=6, out_channels=3, base_ch=48, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(time_dim, time_dim),
        )

        # In-conv (noisy HR + bicubic conditioning = 3 + 3 = 6 channels)
        self.in_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1, padding_mode="reflect")

        # Downsample path
        self.b1 = DDPMResBlock(base_ch, base_ch, time_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)

        self.b2 = DDPMResBlock(base_ch * 2, base_ch * 2, time_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)

        # Bottleneck
        self.mid1 = DDPMResBlock(base_ch * 4, base_ch * 4, time_dim)
        self.mid2 = DDPMResBlock(base_ch * 4, base_ch * 4, time_dim)

        # Upsample path
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1, padding_mode="reflect"),
        )
        self.dec2 = DDPMResBlock(base_ch * 4, base_ch * 2, time_dim)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1, padding_mode="reflect"),
        )
        self.dec1 = DDPMResBlock(base_ch * 2, base_ch, time_dim)

        # Out-conv
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, out_channels, 3, padding=1, padding_mode="reflect"),
        )

        # Zero-init out_conv
        nn.init.zeros_(self.out_conv[-1].weight)
        nn.init.zeros_(self.out_conv[-1].bias)

    def forward(self, x_t, t, cond):
        # x_t: (B, 3, H, W)
        # cond: (B, 3, H, W)
        inp = torch.cat([x_t, cond], dim=1)
        t_emb = self.time_mlp(t)

        h = self.in_conv(inp)
        h1 = self.b1(h, t_emb)

        d1 = self.down1(h1)
        h2 = self.b2(d1, t_emb)

        d2 = self.down2(h2)
        m = self.mid2(self.mid1(d2, t_emb), t_emb)

        u2 = self.up2(m)
        cat2 = torch.cat([u2, h2], dim=1)
        dec2 = self.dec2(cat2, t_emb)

        u1 = self.up1(dec2)
        cat1 = torch.cat([u1, h1], dim=1)
        dec1 = self.dec1(cat1, t_emb)

        return self.out_conv(dec1)


# --- Dataset Generation: 16-20 Self-Supervised Pairs from y ---
class SingleImageInternalDDPMDataset(Dataset):
    """Generates ~20 self-supervised pairs strictly from input image y.

    Each pair consists of:
      - Target patch: P_target extracted from y (scaled to [-1, 1])
      - Conditioning patch: P_cond = Bicubic(Downsampler(P_target)) (scaled to [-1, 1])
    """

    def __init__(self, img_LR_np, downsampler, patch_size=64, num_pairs=20):
        super().__init__()
        # img_LR_np: (3, H_LR, W_LR)
        self.img_tensor = torch.from_numpy(img_LR_np).float()
        self.downsampler = downsampler
        self.patch_size = patch_size
        self.pairs = []

        c, h, w = self.img_tensor.shape
        np.random.seed(SEED)

        try:
            down_device = next(self.downsampler.parameters()).device
        except (StopIteration, AttributeError):
            try:
                down_device = next(self.downsampler.buffers()).device
            except (StopIteration, AttributeError):
                down_device = torch.device("cpu")

        crops_created = 0
        attempts = 0

        while crops_created < num_pairs and attempts < 200:
            attempts += 1
            ps = self.patch_size

            # Ensure patch can fit
            if h >= ps and w >= ps:
                top = np.random.randint(0, h - ps + 1)
                left = np.random.randint(0, w - ps + 1)
                target_patch = self.img_tensor[:, top:top + ps, left:left + ps]
            else:
                target_patch = F.interpolate(
                    self.img_tensor.unsqueeze(0), size=(ps, ps), mode="bicubic", align_corners=False
                ).squeeze(0)

            # Random rotations & flips
            rot_k = np.random.randint(0, 4)
            if rot_k > 0:
                target_patch = torch.rot90(target_patch, k=rot_k, dims=[1, 2])
            if np.random.rand() > 0.5:
                target_patch = torch.flip(target_patch, dims=[2])
            if np.random.rand() > 0.5:
                target_patch = torch.flip(target_patch, dims=[1])

            # Downsample by 4x using forward operator
            with torch.no_grad():
                lr_sub = self.downsampler(target_patch.unsqueeze(0).to(down_device)).squeeze(0).cpu()

            # Bicubic upsample back to ps x ps to form the conditioning
            cond_patch = F.interpolate(
                lr_sub.unsqueeze(0), size=(ps, ps), mode="bicubic", align_corners=False
            ).squeeze(0)

            # Normalize from [0, 1] to [-1, 1] for DDPM
            target_norm = target_patch * 2.0 - 1.0
            cond_norm = cond_patch * 2.0 - 1.0

            self.pairs.append((cond_norm, target_norm))
            crops_created += 1

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_path = PROJECT_DIR / "data/sr/zebra_GT.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    image_name = image_path.stem.replace("_GT", "")
    run_name = f"single_image_ddpm_sr_{image_name}_x{FACTOR}_seed_{SEED}"

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "dip-super-resolution"),
        name=run_name,
        config={
            "experiment": "Experiment 4: Single-Image Conditional DDPM SR",
            "image": image_name,
            "factor": FACTOR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "timesteps": NUM_TIMESTEPS,
            "patch_size": PATCH_SIZE,
            "num_pairs": NUM_PAIRS,
            "seed": SEED,
            "hr_used_in_training": False,
        },
    )

    wandb.define_metric("epoch")
    wandb.define_metric("loss/*", step_metric="epoch")

    # Load HR and synthesize LR observation y
    imgs = load_LR_HR_imgs_sr(str(image_path), imsize=-1, factor=FACTOR, enforse_div32="CROP")
    imgs["bicubic_np"], imgs["sharp_np"], imgs["nearest_np"] = get_baselines(
        imgs["LR_pil"], imgs["HR_pil"]
    )

    img_HR_tensor = np_to_torch(imgs["HR_np"]).to(device)
    img_LR_tensor = np_to_torch(imgs["LR_np"]).to(device)
    img_bicubic_tensor = np_to_torch(imgs["bicubic_np"]).to(device)
    img_nearest_tensor = np_to_torch(imgs["nearest_np"]).to(device)

    psnr_bicubic = calculate_psnr(img_bicubic_tensor, img_HR_tensor)
    psnr_nearest = calculate_psnr(img_nearest_tensor, img_HR_tensor)

    # Reference vanilla DIP
    dip_ref_path = Path(__file__).resolve().parent / "output_dip_sr/zebra_super_resolution_HR.png"
    psnr_dip_val = None
    if dip_ref_path.exists():
        img_dip_pil = Image.open(dip_ref_path).convert("RGB")
        img_dip_tensor = np_to_torch(np.array(img_dip_pil).transpose(2, 0, 1).astype(np.float32) / 255.0).to(device)
        psnr_dip_val = calculate_psnr(img_dip_tensor, img_HR_tensor)
        shutil.copy(dip_ref_path, OUTPUT_DIR / f"{image_name}_hr_with_dip.png")
        print(f"Loaded vanilla DIP reference (PSNR: {psnr_dip_val:.2f} dB)")

    downsampler = Downsampler(
        n_planes=3, factor=FACTOR, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True
    ).to(device)

    # Generate the ~20 pairs dataset from y
    print(f"Generating internal DDPM dataset of {NUM_PAIRS} pairs strictly from y...")
    dataset = SingleImageInternalDDPMDataset(
        imgs["LR_np"], downsampler=downsampler, patch_size=PATCH_SIZE, num_pairs=NUM_PAIRS
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset generated: {len(dataset)} internal training pairs of size ({PATCH_SIZE}x{PATCH_SIZE}).")

    # Setup DDPM model, schedule, and optimizer
    schedule = DDPMSchedule(num_timesteps=NUM_TIMESTEPS, device=device)
    model = ConditionalDDPMUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print("==========================================================")
    print(" Experiment 4: Single-Image Conditional DDPM SR")
    print(f" Device: {device}")
    print(f" DDPM Timesteps: {NUM_TIMESTEPS}")
    print(f" Input y Resolution:  {imgs['LR_pil'].size} (W x H)")
    print(f" Target X Resolution: {imgs['HR_pil'].size} (W x H) [NOT used during training]")
    print(f" Training pairs:      {len(dataset)} cropped & downsampled from y")
    print(f" Baseline Bicubic PSNR: {psnr_bicubic:.2f} dB")
    if psnr_dip_val:
        print(f" Vanilla DIP Baseline:  {psnr_dip_val:.2f} dB")
    print("==========================================================")

    real_path = OUTPUT_DIR / f"{image_name}_real_HR.png"
    downsampled_path = OUTPUT_DIR / f"{image_name}_downsampled_LR.png"
    imgs["HR_pil"].save(real_path)
    imgs["LR_pil"].save(downsampled_path)

    loss_history = []

    print("\nStarting Conditional DDPM training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for cond_patches, target_patches in dataloader:
            cond_patches = cond_patches.to(device)
            target_patches = target_patches.to(device)
            b = target_patches.shape[0]

            # Sample random timesteps t
            t = torch.randint(0, NUM_TIMESTEPS, (b,), device=device).long()

            # Forward diffusion: add noise to target_patches
            x_t, noise = schedule.q_sample(target_patches, t)

            optimizer.zero_grad(set_to_none=True)
            pred_noise = model(x_t, t, cond_patches)

            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * b

        avg_loss = epoch_loss / len(dataset)
        loss_history.append(avg_loss)

        if epoch == 1 or epoch % 50 == 0 or epoch == EPOCHS:
            print(f"  Epoch {epoch:04d}/{EPOCHS}: DDPM Score Loss = {avg_loss:.6f}")
            run.log({"loss/ddpm_mse": avg_loss, "epoch": epoch})
        else:
            run.log({"loss/ddpm_mse": avg_loss, "epoch": epoch})

    # Inference: Pass y to the trained model to find X
    print("\nStarting Inference: Reverse Diffusion conditioned on y...")
    model.eval()
    with torch.no_grad():
        hr_h, hr_w = imgs["HR_pil"].size[1], imgs["HR_pil"].size[0]

        # Conditioning at test time is the full image y, bicubic upsampled to HR resolution
        full_cond_01 = F.interpolate(
            img_LR_tensor, size=(hr_h, hr_w), mode="bicubic", align_corners=False
        )
        full_cond_norm = full_cond_01 * 2.0 - 1.0

        # Run reverse DDPM sampling starting from pure noise
        sampled_norm = schedule.p_sample_loop(
            model, shape=(1, 3, hr_h, hr_w), cond=full_cond_norm
        )

        # Convert back from [-1, 1] to [0, 1]
        out_HR = ((sampled_norm + 1.0) / 2.0).clamp(0.0, 1.0)

        # Iterative back-projection data consistency: Downsampler(out_HR) ~= y
        for _ in range(8):
            lr_res = img_LR_tensor - downsampler(out_HR)
            lr_res_up = F.interpolate(
                lr_res, size=(hr_h, hr_w), mode="bicubic", align_corners=False
            )
            out_HR = (out_HR + 0.5 * lr_res_up).clamp(0.0, 1.0)

        final_out_HR_np = np.clip(torch_to_np(out_HR), 0.0, 1.0)
        final_psnr_hr = calculate_psnr(out_HR, img_HR_tensor)

    print("\n==========================================================")
    print(" Single-Image DDPM SR Complete!")
    print(f" Final DDPM SR PSNR:    {final_psnr_hr:.2f} dB")
    if psnr_dip_val:
        print(f" Vanilla DIP Baseline:  {psnr_dip_val:.2f} dB")
        print(f" Diff vs DIP:           {final_psnr_hr - psnr_dip_val:+.2f} dB")
    print(f" Baseline Bicubic PSNR: {psnr_bicubic:.2f} dB")
    print(f" Gain over Bicubic:     {final_psnr_hr - psnr_bicubic:+.2f} dB")
    print("==========================================================")

    # Save HR with New Approach
    new_approach_path = OUTPUT_DIR / f"{image_name}_hr_with_new_approach.png"
    np_to_pil(final_out_HR_np).save(new_approach_path)
    print(f"Saved new approach HR to: {new_approach_path}")

    # 5-Panel Visual Comparison
    fig_comp, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(imgs["HR_np"].transpose(1, 2, 0))
    axes[0].set_title(f"Real (HR Ground Truth)\n{imgs['HR_pil'].size[0]}x{imgs['HR_pil'].size[1]}")
    axes[0].axis("off")

    axes[1].imshow(imgs["LR_np"].transpose(1, 2, 0))
    axes[1].set_title(f"Downsampled (LR Input)\n{imgs['LR_pil'].size[0]}x{imgs['LR_pil'].size[1]}")
    axes[1].axis("off")

    axes[2].imshow(imgs["bicubic_np"].transpose(1, 2, 0))
    axes[2].set_title(f"Bicubic Baseline\nPSNR: {psnr_bicubic:.2f} dB")
    axes[2].axis("off")

    if psnr_dip_val and (OUTPUT_DIR / f"{image_name}_hr_with_dip.png").exists():
        dip_img = Image.open(OUTPUT_DIR / f"{image_name}_hr_with_dip.png")
        axes[3].imshow(np.array(dip_img))
        axes[3].set_title(f"Vanilla DIP (Baseline)\nPSNR: {psnr_dip_val:.2f} dB")
    else:
        axes[3].imshow(imgs["nearest_np"].transpose(1, 2, 0))
        axes[3].set_title(f"Nearest Baseline\nPSNR: {psnr_nearest:.2f} dB")
    axes[3].axis("off")

    axes[4].imshow(final_out_HR_np.transpose(1, 2, 0))
    diff_text = f" ({final_psnr_hr - psnr_dip_val:+.2f} dB vs DIP)" if psnr_dip_val else ""
    axes[4].set_title(f"DDPM SR (Experiment 4)\nPSNR: {final_psnr_hr:.2f} dB{diff_text}")
    axes[4].axis("off")

    fig_comp.tight_layout()
    comparison_path = OUTPUT_DIR / f"{image_name}_sr_comparison.png"
    fig_comp.savefig(comparison_path, dpi=150)
    plt.close(fig_comp)
    print(f"Saved comparison figure to: {comparison_path}")

    # Training Loss Curve
    fig_metrics, ax_l = plt.subplots(figsize=(8, 5))
    ax_l.plot(np.arange(1, EPOCHS + 1), loss_history, label="DDPM Score Loss", color="tab:blue")
    ax_l.set_xlabel("Epoch")
    ax_l.set_ylabel("MSE Loss")
    ax_l.set_title("Single-Image DDPM Training Convergence")
    ax_l.grid(True, which="both", alpha=0.25)
    ax_l.legend()
    fig_metrics.tight_layout()
    metrics_path = OUTPUT_DIR / f"{image_name}_sr_metrics.png"
    fig_metrics.savefig(metrics_path, dpi=150)
    plt.close(fig_metrics)
    print(f"Saved training curve to:   {metrics_path}")

    # Log to WandB
    log_dict = {
        "real_HR_image": wandb.Image(str(real_path), caption="Real High-Resolution"),
        "downsampled_LR_image": wandb.Image(str(downsampled_path), caption="Downsampled Low-Resolution Input"),
        "hr_with_new_approach": wandb.Image(str(new_approach_path), caption=f"DDPM SR ({final_psnr_hr:.2f} dB)"),
        "visual_comparison": wandb.Image(str(comparison_path), caption="Visual Comparison"),
        "metrics_curves": wandb.Image(str(metrics_path), caption="DDPM Loss Curve"),
    }
    if psnr_dip_val and (OUTPUT_DIR / f"{image_name}_hr_with_dip.png").exists():
        log_dict["hr_with_dip"] = wandb.Image(str(OUTPUT_DIR / f"{image_name}_hr_with_dip.png"), caption=f"Vanilla DIP ({psnr_dip_val:.2f} dB)")

    run.log(log_dict)
    run.summary["final_psnr_hr"] = final_psnr_hr
    run.summary["psnr_bicubic"] = psnr_bicubic
    if psnr_dip_val:
        run.summary["psnr_dip"] = psnr_dip_val
        run.summary["gain_over_dip"] = final_psnr_hr - psnr_dip_val
    run.summary["gain_over_bicubic"] = final_psnr_hr - psnr_bicubic

    run.finish()


if __name__ == "__main__":
    main()
