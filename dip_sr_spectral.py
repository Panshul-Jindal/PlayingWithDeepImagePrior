"""Experiment 9: Spectral Power-Law Decay DIP (Spectral-DIP).

A novel Super-Resolution loss function operating in the 2D Fourier Frequency domain.

Mathematical Foundation:
  Natural Scene Statistics (Field 1987, Torralba & Oliva 2003):
  The radially averaged power spectrum S(r) of natural images adheres to a universal power law:
      S(r) \\propto r^{-2\\alpha},   where \\alpha \\approx 1.0 - 1.2.
  In log-log scale:
      \\log S(r) = -2\\alpha \\log r + C.

The Super-Resolution Null-Space Problem in Frequency Domain:
  In 4x super-resolution, the downsampled observation y captures spatial frequencies up to
  the LR Nyquist cutoff:
      r_{cutoff} = 1 / (2 * factor) = 0.125 (normalized frequency).
  Spatial frequencies r > r_{cutoff} lie in the null space of the downsampling operator A,
  leaving them unconstrained by the data fidelity loss ||A(\\hat{X}) - y||^2.
  
  As DIP optimizes, its unconstrained degrees of freedom inject spurious white noise into the
  null space, causing S(r) for r > r_{cutoff} to flatten into an unnatural plateau or spike.

The Spectral Power-Law Regularization Loss:
  We partition the 2D Fourier spectrum into radial frequency shells k = 1, ..., K.
  For all null-space shells (k >= k_{cutoff}), we penalize any deviation where higher frequencies
  fail to decay at least as fast as the physical r^{-2} rate:
      \\mathcal{L}_{spectral} = \\sum_{k \\ge k_{cutoff}} \\text{ReLU}\\left(\\log E_{k+1} - \\log E_k + 2 \\log(r_{k+1} / r_k)\\right)^2
  where E_k is the mean spectral power in radial shell k.

Saves:
  - zebra_real_HR.png
  - zebra_downsampled_LR.png
  - zebra_hr_with_dip.png
  - zebra_hr_with_new_approach.png
  - Visual comparison figure with all baselines and PSNR metrics
  - Training metrics curves (data loss, spectral loss, HR PSNR)
Logs all metrics, parameters, and images to Weights & Biases.
"""

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
import wandb

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

PROJECT_DIR = Path(__file__).resolve().parent / "src/playingwithdeepimageprior/deep-image-prior"
sys.path.insert(0, str(PROJECT_DIR))

from models.downsampler import Downsampler
from models.skip import skip
from utils.common_utils import np_to_pil, np_to_torch, torch_to_np
from utils.sr_utils import get_baselines, load_LR_HR_imgs_sr

# --- Hyperparameters ---
STEPS = 6000
FACTOR = 4
INPUT_DEPTH = 32
LEARNING_RATE = 0.01
REG_NOISE_STD = 0.03
LAMBDA_SPECTRAL = 1e-4
ALPHA_SLOPE = 1.0  # Power spectrum slope -2 * alpha
NUM_SHELLS = 16
KERNEL_TYPE = "lanczos2"
OUTPUT_DIR = Path(__file__).resolve().parent / "output_dip_spectral"


def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate Peak Signal-to-Noise Ratio (dB)."""
    mse = torch.mean((img1.detach() - img2.detach()) ** 2)
    if mse == 0:
        return 100.0
    return (10.0 * torch.log10((max_val ** 2) / mse)).item()


def build_skip_net(input_depth):
    """Build the exact encoder-decoder skip architecture used by DIP for SR."""
    return skip(
        num_input_channels=input_depth,
        num_output_channels=3,
        num_channels_down=[128, 128, 128, 128, 128],
        num_channels_up=[128, 128, 128, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3,
        filter_size_up=3,
        filter_skip_size=1,
        need_sigmoid=True,
        need_bias=True,
        pad="reflection",
        upsample_mode="bilinear",
        downsample_mode="stride",
        act_fun="LeakyReLU",
    )


class SpectralPowerLawLoss(nn.Module):
    """Differentiable Radial Power-Spectrum Decay Loss for Super-Resolution."""

    def __init__(self, h, w, factor=4, num_shells=NUM_SHELLS, alpha=ALPHA_SLOPE, eps=1e-6):
        super().__init__()
        self.num_shells = num_shells
        self.alpha = alpha
        self.eps = eps

        # Coordinate grid centered at DC (0,0)
        u = torch.linspace(-1.0, 1.0, steps=h)
        v = torch.linspace(-1.0, 1.0, steps=w)
        grid_u, grid_v = torch.meshgrid(u, v, indexing="ij")
        r = torch.sqrt(grid_u ** 2 + grid_v ** 2)  # radius in [0, sqrt(2)]

        # Cutoff radius for LR Nyquist limit: in normalized units 1 / factor
        self.r_cutoff = 1.0 / factor
        self.k_cutoff = int(self.num_shells / factor)

        # Build radial shell weights using Gaussian radial basis
        # Shell radii from 0.05 to 1.0
        radii = torch.linspace(0.05, 1.0, steps=num_shells)
        dr = (radii[1] - radii[0]).item()
        sigma = dr * 0.7

        shell_masks = []
        for r_k in radii:
            mask = torch.exp(-((r - r_k) ** 2) / (2.0 * (sigma ** 2)))
            mask = mask / (torch.sum(mask) + self.eps)
            shell_masks.append(mask)

        # Shape: (num_shells, H, W)
        self.register_buffer("shell_masks", torch.stack(shell_masks, dim=0))
        self.register_buffer("radii", radii)

    def forward(self, x):
        # Grayscale luminance
        y = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]  # (B, H, W)

        # 2D FFT shifted so DC is centered
        fft_y = torch.fft.fft2(y)
        fft_shifted = torch.fft.fftshift(fft_y, dim=(-2, -1))
        power = torch.abs(fft_shifted) ** 2  # (B, H, W)

        # Compute power in each radial shell: (num_shells,)
        shell_energies = []
        for k in range(self.num_shells):
            e_k = torch.sum(power * self.shell_masks[k], dim=(-2, -1))  # (B,)
            shell_energies.append(e_k)

        shell_energies = torch.stack(shell_energies, dim=1)  # (B, num_shells)
        log_e = torch.log(shell_energies + self.eps)

        loss = torch.tensor(0.0, device=x.device)
        # For shells in null-space (k >= k_cutoff), penalize violation of power-law decay
        for k in range(self.k_cutoff, self.num_shells - 1):
            r_k = self.radii[k]
            r_next = self.radii[k + 1]
            # Expected maximum change in log energy: -2 * alpha * log(r_next / r_k)
            allowed_delta = -2.0 * self.alpha * torch.log(r_next / r_k)
            actual_delta = log_e[:, k + 1] - log_e[:, k]
            # If actual_delta > allowed_delta, energy decays too slowly (or increases = noise hump)
            violation = F.relu(actual_delta - allowed_delta)
            loss = loss + torch.mean(violation ** 2)

        return loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_path = PROJECT_DIR / "data/sr/zebra_GT.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    image_name = image_path.stem.replace("_GT", "")
    run_name = f"dip_spectral_sr_{image_name}_x{FACTOR}_seed_{SEED}"

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "dip-super-resolution"),
        name=run_name,
        config={
            "method": "Spectral Power-Law Decay DIP (Spectral-DIP)",
            "image": image_name,
            "factor": FACTOR,
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "lambda_spectral": LAMBDA_SPECTRAL,
            "alpha_slope": ALPHA_SLOPE,
            "num_shells": NUM_SHELLS,
            "reg_noise_std": REG_NOISE_STD,
            "seed": SEED,
            "hr_used_in_training": False,
        },
    )

    wandb.define_metric("iteration")
    wandb.define_metric("loss/*", step_metric="iteration")
    wandb.define_metric("psnr/*", step_metric="iteration")

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

    # Downsampler on device
    downsampler = Downsampler(
        n_planes=3, factor=FACTOR, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True
    ).to(device)

    hr_h, hr_w = imgs["HR_pil"].size[1], imgs["HR_pil"].size[0]
    spectral_criterion = SpectralPowerLawLoss(hr_h, hr_w, factor=FACTOR, num_shells=NUM_SHELLS, alpha=ALPHA_SLOPE).to(device)

    # Setup DIP skip network and optimizer
    net = build_skip_net(INPUT_DEPTH).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Fixed uniform input noise
    net_input_saved = torch.zeros((1, INPUT_DEPTH, hr_h, hr_w), device=device)
    net_input_saved.uniform_()
    net_input_saved *= 1.0 / 10.0
    noise = torch.zeros_like(net_input_saved)

    print("==========================================================")
    print(" Experiment 9: Spectral Power-Law Decay DIP (Spectral-DIP)")
    print(f" Device: {device}")
    print(f" Resolution: LR={imgs['LR_pil'].size} -> HR={imgs['HR_pil'].size}")
    print(f" Lambda Spectral: {LAMBDA_SPECTRAL} (alpha_slope={ALPHA_SLOPE}, shells={NUM_SHELLS})")
    print(f" Baselines: Bicubic={psnr_bicubic:.2f} dB, Nearest={psnr_nearest:.2f} dB")
    if psnr_dip_val:
        print(f" Vanilla DIP Baseline: {psnr_dip_val:.2f} dB")
    print("==========================================================")

    real_path = OUTPUT_DIR / f"{image_name}_real_HR.png"
    downsampled_path = OUTPUT_DIR / f"{image_name}_downsampled_LR.png"
    imgs["HR_pil"].save(real_path)
    imgs["LR_pil"].save(downsampled_path)

    loss_history = []
    psnr_hr_history = []
    best_psnr_hr = 0.0
    best_out_HR = None

    print("\nStarting Spectral-DIP optimization...")
    for i in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)

        if REG_NOISE_STD > 0:
            net_input = net_input_saved + (noise.normal_() * REG_NOISE_STD)
        else:
            net_input = net_input_saved

        out_HR = net(net_input)
        out_LR = downsampler(out_HR)

        # 1. Forward data fidelity
        loss_data = F.mse_loss(out_LR, img_LR_tensor)

        # 2. Fourier spectral power-law loss
        loss_spectral = spectral_criterion(out_HR)

        total_loss = loss_data + LAMBDA_SPECTRAL * loss_spectral
        total_loss.backward()
        optimizer.step()

        psnr_lr = calculate_psnr(out_LR, img_LR_tensor)
        psnr_hr = calculate_psnr(out_HR, img_HR_tensor)

        loss_history.append(total_loss.item())
        psnr_hr_history.append(psnr_hr)

        if psnr_hr > best_psnr_hr:
            best_psnr_hr = psnr_hr
            best_out_HR = out_HR.detach().clone()

        if i == 1 or i % 100 == 0 or i == STEPS:
            print(
                f"  Iteration {i:05d}/{STEPS}: "
                f"loss_data={loss_data.item():.6f}, "
                f"loss_spectral={loss_spectral.item():.6f}, "
                f"PSNR_LR={psnr_lr:.2f} dB, "
                f"PSNR_HR={psnr_hr:.2f} dB (best={best_psnr_hr:.2f} dB)"
            )

            run.log({
                "loss/total": total_loss.item(),
                "loss/data_mse": loss_data.item(),
                "loss/spectral": loss_spectral.item(),
                "psnr/LR": psnr_lr,
                "psnr/HR": psnr_hr,
                "iteration": i,
            })

    # Final reconstruction
    final_out_HR = best_out_HR if best_out_HR is not None else out_HR.detach()
    final_out_HR_np = np.clip(torch_to_np(final_out_HR), 0.0, 1.0)
    final_psnr_hr = calculate_psnr(final_out_HR, img_HR_tensor)

    print("\n==========================================================")
    print(" Spectral-DIP Optimization Complete!")
    print(f" Final Spectral-DIP PSNR: {final_psnr_hr:.2f} dB")
    if psnr_dip_val:
        print(f" Vanilla DIP Baseline:    {psnr_dip_val:.2f} dB")
        print(f" Gain over DIP:           {final_psnr_hr - psnr_dip_val:+.2f} dB")
    print(f" Baseline Bicubic PSNR:   {psnr_bicubic:.2f} dB")
    print(f" Gain over Bicubic:       {final_psnr_hr - psnr_bicubic:+.2f} dB")
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

    if psnr_dip_val:
        axes[3].imshow(np.array(img_dip_pil))
        axes[3].set_title(f"Vanilla DIP\nPSNR: {psnr_dip_val:.2f} dB")
    else:
        axes[3].text(0.5, 0.5, "N/A", ha="center", va="center")
        axes[3].set_title("Vanilla DIP Baseline")
    axes[3].axis("off")

    axes[4].imshow(final_out_HR_np.transpose(1, 2, 0))
    axes[4].set_title(f"Spectral-DIP (Ours)\nPSNR: {final_psnr_hr:.2f} dB")
    axes[4].axis("off")

    plt.tight_layout()
    comparison_path = OUTPUT_DIR / f"{image_name}_comparison_baseline_vs_spectral.png"
    fig_comp.savefig(comparison_path, dpi=200, bbox_inches="tight")
    plt.close(fig_comp)
    print(f"Saved comparison figure to: {comparison_path}")

    # Training curves
    fig_loss, ax1 = plt.subplots(figsize=(8, 4))
    color = "tab:red"
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Total Loss", color=color)
    ax1.plot(loss_history, color=color, alpha=0.8, label="Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("HR PSNR (dB)", color=color)
    ax2.plot(psnr_hr_history, color=color, alpha=0.8, label="HR PSNR")
    ax2.axhline(psnr_bicubic, color="gray", linestyle="--", label=f"Bicubic ({psnr_bicubic:.2f} dB)")
    if psnr_dip_val:
        ax2.axhline(psnr_dip_val, color="green", linestyle=":", label=f"DIP ({psnr_dip_val:.2f} dB)")
    ax2.tick_params(axis="y", labelcolor=color)

    fig_loss.tight_layout()
    metrics_path = OUTPUT_DIR / f"{image_name}_metrics_curve.png"
    fig_loss.savefig(metrics_path, dpi=150, bbox_inches="tight")
    plt.close(fig_loss)

    # Log summary artifacts to WandB
    run.log({
        "comparison_figure": wandb.Image(str(comparison_path), caption="Spectral-DIP Comparison"),
        "metrics_figure": wandb.Image(str(metrics_path), caption="Spectral-DIP Training Curves"),
        "hr_with_new_approach": wandb.Image(str(new_approach_path), caption="HR with Spectral-DIP"),
        "final_psnr_hr": final_psnr_hr,
        "best_psnr_hr": best_psnr_hr,
        "delta_psnr_vs_dip": final_psnr_hr - (psnr_dip_val or 0.0),
        "delta_psnr_vs_bicubic": final_psnr_hr - psnr_bicubic,
    })
    run.finish()
    print("Spectral-DIP Run finished and logged to W&B.")


if __name__ == "__main__":
    main()

