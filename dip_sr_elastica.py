"""Experiment 8: Euler's Elastica Curvature Regularized DIP (Elastica-DIP).

A novel Super-Resolution loss function rooted in differential geometry and variational mechanics.

Mathematical Foundation:
  Standard Total Variation (TV) penalizes gradient magnitude:
      E_TV = \\int |\\nabla I| dx dy
  A critical limitation of TV is "staircasing" (transforming smooth transitions into blocky plateaus)
  and blunting sharp high-frequency corners.
  
  Euler's Elastica (Euler 1744, Mumford 1994, Chan-Kang-Shen 2002) minimizes:
      E_Elastica = \\int (\\alpha + \\beta \\kappa^2) |\\nabla I| dx dy
  where:
      \\kappa = div(\\nabla I / |\\nabla I|) is the curvature of the level lines (isophotes).
      \\alpha |\\nabla I| penalizes the total perimeter length.
      \\beta \\kappa^2 |\\nabla I| penalizes the total bending energy of the level lines.

Physical Justification for Super-Resolution:
  1. High-frequency null-space noise and checkerboard artifacts cause rapid spatial oscillations
     in the isophote orientation, causing \\kappa^2 to explode.
  2. True physical edges (e.g. zebra stripes, boundaries) have smooth, coherent level-line trajectories
     with near-zero curvature (\\kappa \\approx 0).
  3. By penalizing level-line bending energy (\\kappa^2 |\\nabla I|), Elastica selectively extinguishes
     chaotic null-space fluctuations while completely preserving sharp, low-curvature stripe edges!

Saves:
  - zebra_real_HR.png
  - zebra_downsampled_LR.png
  - zebra_hr_with_dip.png
  - zebra_hr_with_new_approach.png
  - Visual comparison figure with all baselines and PSNR metrics
  - Training metrics curves (data loss, elastica loss, HR PSNR)
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
LAMBDA_ELASTICA = 1e-4
ALPHA_LENGTH = 0.1
BETA_BENDING = 1.0
EPSILON = 1e-4
KERNEL_TYPE = "lanczos2"
OUTPUT_DIR = Path(__file__).resolve().parent / "output_dip_elastica"


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


class EulerElasticaLoss(nn.Module):
    """Differentiable Euler's Elastica Regularization Loss for Image Super-Resolution."""

    def __init__(self, alpha=ALPHA_LENGTH, beta=BETA_BENDING, eps=EPSILON):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        # Central difference kernels for gradient and divergence
        kx = torch.tensor([[[[0.0, 0.0, 0.0],
                             [-0.5, 0.0, 0.5],
                             [0.0, 0.0, 0.0]]]], dtype=torch.float32)
        ky = torch.tensor([[[[0.0, -0.5, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.5, 0.0]]]], dtype=torch.float32)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x):
        # Compute on grayscale luminance Y for rotational and photometric invariance
        y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        # First derivatives with replication padding
        y_pad = F.pad(y, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(y_pad, self.kx)
        gy = F.conv2d(y_pad, self.ky)

        # Gradient norm |grad y|_eps
        norm_grad = torch.sqrt(gx ** 2 + gy ** 2 + self.eps ** 2)

        # Unit normal vector field along level lines: n = grad y / |grad y|
        nx = gx / norm_grad
        ny = gy / norm_grad

        # Curvature kappa = div(n) = d(nx)/dx + d(ny)/dy
        nx_pad = F.pad(nx, (1, 1, 1, 1), mode="replicate")
        ny_pad = F.pad(ny, (1, 1, 1, 1), mode="replicate")
        kappa_x = F.conv2d(nx_pad, self.kx)
        kappa_y = F.conv2d(ny_pad, self.ky)
        kappa = kappa_x + kappa_y

        # Clamp curvature to prevent extreme gradient spikes at inflection points
        kappa = torch.clamp(kappa, min=-50.0, max=50.0)

        # Euler's Elastica energy density: (alpha + beta * kappa^2) * |grad y|
        energy = (self.alpha + self.beta * (kappa ** 2)) * norm_grad
        return torch.mean(energy)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_path = PROJECT_DIR / "data/sr/zebra_GT.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    image_name = image_path.stem.replace("_GT", "")
    run_name = f"dip_elastica_sr_{image_name}_x{FACTOR}_seed_{SEED}"

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "dip-super-resolution"),
        name=run_name,
        config={
            "method": "Euler Elastica Curvature DIP (Elastica-DIP)",
            "image": image_name,
            "factor": FACTOR,
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "lambda_elastica": LAMBDA_ELASTICA,
            "alpha_length": ALPHA_LENGTH,
            "beta_bending": BETA_BENDING,
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

    elastica_criterion = EulerElasticaLoss().to(device)

    hr_h, hr_w = imgs["HR_pil"].size[1], imgs["HR_pil"].size[0]

    # Setup DIP skip network and optimizer
    net = build_skip_net(INPUT_DEPTH).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Fixed uniform input noise
    net_input_saved = torch.zeros((1, INPUT_DEPTH, hr_h, hr_w), device=device)
    net_input_saved.uniform_()
    net_input_saved *= 1.0 / 10.0
    noise = torch.zeros_like(net_input_saved)

    print("==========================================================")
    print(" Experiment 8: Euler's Elastica DIP (Elastica-DIP)")
    print(f" Device: {device}")
    print(f" Resolution: LR={imgs['LR_pil'].size} -> HR={imgs['HR_pil'].size}")
    print(f" Lambda Elastica: {LAMBDA_ELASTICA} (alpha={ALPHA_LENGTH}, beta={BETA_BENDING})")
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

    print("\nStarting Elastica-DIP optimization...")
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

        # 2. Level-line curvature elastica loss
        loss_elastica = elastica_criterion(out_HR)

        total_loss = loss_data + LAMBDA_ELASTICA * loss_elastica
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
                f"loss_elastica={loss_elastica.item():.6f}, "
                f"PSNR_LR={psnr_lr:.2f} dB, "
                f"PSNR_HR={psnr_hr:.2f} dB (best={best_psnr_hr:.2f} dB)"
            )

            run.log({
                "loss/total": total_loss.item(),
                "loss/data_mse": loss_data.item(),
                "loss/elastica": loss_elastica.item(),
                "psnr/LR": psnr_lr,
                "psnr/HR": psnr_hr,
                "iteration": i,
            })

    # Final reconstruction
    final_out_HR = best_out_HR if best_out_HR is not None else out_HR.detach()
    final_out_HR_np = np.clip(torch_to_np(final_out_HR), 0.0, 1.0)
    final_psnr_hr = calculate_psnr(final_out_HR, img_HR_tensor)

    print("\n==========================================================")
    print(" Elastica-DIP Optimization Complete!")
    print(f" Final Elastica-DIP PSNR: {final_psnr_hr:.2f} dB")
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
    axes[4].set_title(f"Elastica-DIP (Ours)\nPSNR: {final_psnr_hr:.2f} dB")
    axes[4].axis("off")

    plt.tight_layout()
    comparison_path = OUTPUT_DIR / f"{image_name}_comparison_baseline_vs_elastica.png"
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
        "comparison_figure": wandb.Image(str(comparison_path), caption="Elastica-DIP Comparison"),
        "metrics_figure": wandb.Image(str(metrics_path), caption="Elastica-DIP Training Curves"),
        "hr_with_new_approach": wandb.Image(str(new_approach_path), caption="HR with Elastica-DIP"),
        "final_psnr_hr": final_psnr_hr,
        "best_psnr_hr": best_psnr_hr,
        "delta_psnr_vs_dip": final_psnr_hr - (psnr_dip_val or 0.0),
        "delta_psnr_vs_bicubic": final_psnr_hr - psnr_bicubic,
    })
    run.finish()
    print("Elastica-DIP Run finished and logged to W&B.")


if __name__ == "__main__":
    main()

