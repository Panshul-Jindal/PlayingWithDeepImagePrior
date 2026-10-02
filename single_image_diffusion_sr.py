"""Experiment 3: Single-Image Conditional Diffusion Model for Super-Resolution.

A generative score-based / diffusion model trained strictly on the internal multi-scale
patch distribution of the single input image y (zero-shot, no external dataset).

Key Principles:
  1. The ground truth HR image X is NEVER used during training (unsupervised, single-image).
  2. Training pairs (P_lr, P_hr) are extracted strictly from y using the forward degradation operator.
  3. A Conditional UNet epsilon_theta(x_t, t, c) is trained to reverse Gaussian noise conditioned on
     the upsampled low-resolution patch c = bicubic(P_lr).
  4. At inference time, reverse diffusion generates high-frequency detail conditioned on the full
     input y, combined with a Diffusion Posterior Sampling (DPS) projection step that enforces
     exact forward consistency: Downsampler(X_hat) ~= y.

Saves:
  - Real (HR Ground Truth) image
  - Downsampled (LR Input) image
  - HR with baseline DIP
  - HR with New Approach (Diffusion SR)
  - Baselines (Bicubic, Nearest)
  - Visual comparison figure with PSNR metrics
  - Training metrics curves (Diffusion score-matching loss, validation PSNR)
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
EPOCHS = 600
BATCH_SIZE = 16
LEARNING_RATE = 0.0003
NUM_TIMESTEPS = 300
SAMPLING_STEPS = 50
KERNEL_TYPE = "lanczos2"
OUTPUT_DIR = Path(__file__).resolve().parent / "output_single_image_diffusion_sr"


def calculate_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1.detach() - img2.detach()) ** 2)
    if mse == 0:
        return 100.0
    return (10.0 * torch.log10((max_val ** 2) / mse)).item()


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # time: (B,) tensor of integers
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode="reflect")
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode="reflect")
        self.res = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.conv1(x))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.act(h))
        return self.res(x) + 0.2 * h


class ConditionalDiffusionUNet(nn.Module):
    """UNet predicting noise epsilon_theta(x_t, t, c) conditioned on c = bicubic(LR)."""

    def __init__(self, in_channels=6, out_channels=3, base_dim=48, time_dim=128):
        super().__init__()
        # in_channels = 3 (noisy HR x_t) + 3 (bicubic conditioning c) = 6
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.in_conv = nn.Conv2d(in_channels, base_dim, 3, padding=1, padding_mode="reflect")

        # Encoder
        self.b1 = DiffusionBlock(base_dim, base_dim, time_dim)
        self.down1 = nn.Conv2d(base_dim, base_dim * 2, 3, stride=2, padding=1)

        self.b2 = DiffusionBlock(base_dim * 2, base_dim * 2, time_dim)

        # Bottleneck
        self.mid1 = DiffusionBlock(base_dim * 2, base_dim * 2, time_dim)
        self.mid2 = DiffusionBlock(base_dim * 2, base_dim * 2, time_dim)

        # Decoder
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_dim * 2, base_dim, 3, padding=1, padding_mode="reflect"),
        )
        self.b3 = DiffusionBlock(base_dim * 2, base_dim, time_dim)

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, 3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_dim, out_channels, 3, padding=1, padding_mode="reflect"),
        )

    def forward(self, x_t, t, cond):
        # x_t: (B, 3, H, W)
        # cond: (B, 3, H, W)
        inp = torch.cat([x_t, cond], dim=1)
        t_emb = self.time_mlp(t)

        h = self.in_conv(inp)
        h1 = self.b1(h, t_emb)

        d1 = self.down1(h1)
        h2 = self.b2(d1, t_emb)

        m = self.mid2(self.mid1(h2, t_emb), t_emb)

        u1 = self.up1(m)
        cat1 = torch.cat([u1, h1], dim=1)
        h3 = self.b3(cat1, t_emb)

        return self.out_conv(h3)


class DiffusionNoiseSchedule:
    def __init__(self, num_timesteps=300, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise, noise


class InternalPatchDataset(Dataset):
    """Generates internal cross-scale training pairs (LR, HR) strictly from y."""

    def __init__(self, img_LR_np, downsampler, patch_size=48, factor=4, num_samples=256):
        super().__init__()
        self.img_tensor = torch.from_numpy(img_LR_np).float()
        self.downsampler = downsampler
        self.patch_size = patch_size
        self.factor = factor
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

        for _ in range(num_samples):
            ps = self.patch_size
            if h <= ps or w <= ps:
                actual_h = (h // factor) * factor
                actual_w = (w // factor) * factor
                top = 0
                left = 0
            else:
                top = np.random.randint(0, h - ps + 1)
                left = np.random.randint(0, w - ps + 1)
                actual_h = ps
                actual_w = ps

            hr_patch = self.img_tensor[:, top:top + actual_h, left:left + actual_w]

            # Random rotations & flips
            if np.random.rand() > 0.5:
                hr_patch = torch.flip(hr_patch, dims=[2])
            if np.random.rand() > 0.5:
                hr_patch = torch.flip(hr_patch, dims=[1])
            rot_k = np.random.randint(0, 4)
            if rot_k > 0:
                hr_patch = torch.rot90(hr_patch, k=rot_k, dims=[1, 2])

            with torch.no_grad():
                lr_patch = self.downsampler(hr_patch.unsqueeze(0).to(down_device)).squeeze(0).cpu()

            # Bicubic upsampled conditioning patch
            cond_patch = F.interpolate(
                lr_patch.unsqueeze(0), size=(actual_h, actual_w), mode="bicubic", align_corners=False
            ).squeeze(0)

            self.pairs.append((cond_patch, hr_patch))

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
    run_name = f"single_image_diffusion_sr_{image_name}_x{FACTOR}_seed_{SEED}"

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "dip-super-resolution"),
        name=run_name,
        config={
            "method": "Single-Image Conditional Diffusion Model (Score-based SR)",
            "image": image_name,
            "factor": FACTOR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "timesteps": NUM_TIMESTEPS,
            "sampling_steps": SAMPLING_STEPS,
            "seed": SEED,
            "hr_used_in_training": False,
        },
    )

    wandb.define_metric("epoch")
    wandb.define_metric("loss/*", step_metric="epoch")
    wandb.define_metric("psnr/*", step_metric="epoch")

    # Load HR image and generate LR observation y
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

    # Dataset & schedule
    print("Generating internal training dataset strictly from input image y...")
    dataset = InternalPatchDataset(
        imgs["LR_np"], downsampler=downsampler, patch_size=48, factor=FACTOR, num_samples=256
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    schedule = DiffusionNoiseSchedule(num_timesteps=NUM_TIMESTEPS, device=device)
    model = ConditionalDiffusionUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print("==========================================================")
    print(" Experiment 3: Single-Image Conditional Diffusion Model")
    print(f" Device: {device}")
    print(f" Timesteps: {NUM_TIMESTEPS}, Sampling Steps: {SAMPLING_STEPS}")
    print(f" Input y Resolution:  {imgs['LR_pil'].size} (W x H)")
    print(f" Target X Resolution: {imgs['HR_pil'].size} (W x H) [NOT used during training]")
    print(f" Baseline Bicubic PSNR: {psnr_bicubic:.2f} dB")
    if psnr_dip_val:
        print(f" Vanilla DIP Baseline:  {psnr_dip_val:.2f} dB")
    print("==========================================================")

    real_path = OUTPUT_DIR / f"{image_name}_real_HR.png"
    downsampled_path = OUTPUT_DIR / f"{image_name}_downsampled_LR.png"
    imgs["HR_pil"].save(real_path)
    imgs["LR_pil"].save(downsampled_path)

    loss_history = []
    psnr_eval_history = []

    print("\nStarting Diffusion score-matching training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for cond_patches, hr_patches in dataloader:
            cond_patches = cond_patches.to(device)
            hr_patches = hr_patches.to(device)
            b = hr_patches.shape[0]

            # Sample random timesteps
            t = torch.randint(0, NUM_TIMESTEPS, (b,), device=device).long()

            # Add noise according to q(x_t | x_0)
            x_t, noise = schedule.q_sample(hr_patches, t)

            optimizer.zero_grad(set_to_none=True)
            pred_noise = model(x_t, t, cond_patches)

            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * b

        avg_loss = epoch_loss / len(dataset)
        loss_history.append(avg_loss)

        if epoch == 1 or epoch % 20 == 0 or epoch == EPOCHS:
            print(f"  Epoch {epoch:04d}/{EPOCHS}: Score-Matching Loss={avg_loss:.6f}")
            run.log({"loss/diffusion_mse": avg_loss, "epoch": epoch})
        else:
            run.log({"loss/diffusion_mse": avg_loss, "epoch": epoch})

    # Inference via Reverse Diffusion with Data-Consistency Guidance
    print("\nRunning Reverse Diffusion Sampling conditioned on input image y...")
    model.eval()
    with torch.no_grad():
        hr_h, hr_w = imgs["HR_pil"].size[1], imgs["HR_pil"].size[0]
        # Conditioning is the full bicubic upsampled y
        full_cond = F.interpolate(
            img_LR_tensor, size=(hr_h, hr_w), mode="bicubic", align_corners=False
        )

        # Initialize from pure noise
        x_t = torch.randn_like(full_cond)

        # Subsample timesteps for faster, high-quality DDIM-style sampling
        step_indices = np.linspace(NUM_TIMESTEPS - 1, 0, SAMPLING_STEPS, dtype=int)

        for i, step_idx in enumerate(step_indices):
            t_tensor = torch.full((1,), step_idx, device=device, dtype=torch.long)
            pred_noise = model(x_t, t_tensor, full_cond)

            # Estimate x_0 from x_t and pred_noise
            sqrt_alpha_bar = schedule.sqrt_alphas_cumprod[step_idx]
            sqrt_one_minus_alpha_bar = schedule.sqrt_one_minus_alphas_cumprod[step_idx]
            pred_x0 = (x_t - sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_bar
            pred_x0 = pred_x0.clamp(0.0, 1.0)

            # Diffusion Posterior Sampling (DPS) projection for measurement consistency
            lr_res = img_LR_tensor - downsampler(pred_x0)
            lr_res_up = F.interpolate(lr_res, size=(hr_h, hr_w), mode="bicubic", align_corners=False)
            pred_x0 = (pred_x0 + 0.3 * lr_res_up).clamp(0.0, 1.0)

            if i < len(step_indices) - 1:
                next_step_idx = step_indices[i + 1]
                sqrt_alpha_next = schedule.sqrt_alphas_cumprod[next_step_idx]
                sqrt_one_minus_alpha_next = schedule.sqrt_one_minus_alphas_cumprod[next_step_idx]
                x_t = sqrt_alpha_next * pred_x0 + sqrt_one_minus_alpha_next * pred_noise
            else:
                x_t = pred_x0

        final_out_HR = x_t.clamp(0.0, 1.0)

        # Iterative back-projection to ensure strict forward measurement consistency with y
        for _ in range(5):
            lr_res = img_LR_tensor - downsampler(final_out_HR)
            lr_res_up = F.interpolate(lr_res, size=(hr_h, hr_w), mode="bicubic", align_corners=False)
            final_out_HR = (final_out_HR + 0.5 * lr_res_up).clamp(0.0, 1.0)

        final_out_HR_np = np.clip(torch_to_np(final_out_HR), 0.0, 1.0)
        final_psnr_hr = calculate_psnr(final_out_HR, img_HR_tensor)

    print("\n==========================================================")
    print(" Single-Image Diffusion SR Complete!")
    print(f" Final Diffusion SR PSNR:  {final_psnr_hr:.2f} dB")
    if psnr_dip_val:
        print(f" Vanilla DIP Baseline:     {psnr_dip_val:.2f} dB")
        print(f" Diff vs DIP:              {final_psnr_hr - psnr_dip_val:+.2f} dB")
    print(f" Baseline Bicubic PSNR:    {psnr_bicubic:.2f} dB")
    print(f" Gain over Bicubic:        {final_psnr_hr - psnr_bicubic:+.2f} dB")
    print("==========================================================")

    # Save HR with New Approach
    new_approach_path = OUTPUT_DIR / f"{image_name}_hr_with_new_approach.png"
    np_to_pil(final_out_HR_np).save(new_approach_path)
    print(f"Saved new approach HR to: {new_approach_path}")

    # 5-Panel Visual Comparison:
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
    axes[4].set_title(f"Diffusion SR (New Approach)\nPSNR: {final_psnr_hr:.2f} dB{diff_text}")
    axes[4].axis("off")

    fig_comp.tight_layout()
    comparison_path = OUTPUT_DIR / f"{image_name}_sr_comparison.png"
    fig_comp.savefig(comparison_path, dpi=150)
    plt.close(fig_comp)
    print(f"Saved comparison figure to: {comparison_path}")

    # Training Loss Curve
    fig_metrics, ax_l = plt.subplots(figsize=(8, 5))
    ax_l.plot(np.arange(1, EPOCHS + 1), loss_history, label="Diffusion Noise Prediction Loss", color="tab:blue")
    ax_l.set_xlabel("Epoch")
    ax_l.set_ylabel("MSE Loss")
    ax_l.set_title("Single-Image Diffusion Training Convergence")
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
        "hr_with_new_approach": wandb.Image(str(new_approach_path), caption=f"Diffusion SR ({final_psnr_hr:.2f} dB)"),
        "visual_comparison": wandb.Image(str(comparison_path), caption="Visual Comparison"),
        "metrics_curves": wandb.Image(str(metrics_path), caption="Diffusion Loss Curve"),
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
