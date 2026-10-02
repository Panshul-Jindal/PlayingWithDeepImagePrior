"""Reproduce Deep Image Prior (DIP) Super-Resolution.

This script reproduces the super-resolution experiment from the Deep Image Prior
paper (Ulyanov et al.), fitting a skip-architecture convolutional prior to a
low-resolution image such that Downsampler(output) reconstructs the low-resolution
input.

Saves:
  - Real (ground truth HR) image
  - Downsampled (LR) image
  - Baseline images (Bicubic, Nearest)
  - Super-resolved (SR) output after 5k-6k steps
  - Comparison figures and loss/PSNR curves
Tracks and logs training progress and artifacts to Weights & Biases (wandb).
"""

import os
from pathlib import Path
import sys

# Ensure PIL compatibility with older code expecting ANTIALIAS
from PIL import Image
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = getattr(Image, "Resampling", Image).LANCZOS

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

PROJECT_DIR = Path(__file__).resolve().parent / "src/playingwithdeepimageprior/deep-image-prior"
sys.path.insert(0, str(PROJECT_DIR))

from models.downsampler import Downsampler
from models.skip import skip
from utils.common_utils import np_to_pil, np_to_torch, pil_to_np, torch_to_np
from utils.sr_utils import get_baselines, load_LR_HR_imgs_sr

# --- Hyperparameters from DIP paper & super-resolution.ipynb ---
STEPS = 6000
FACTOR = 4  # 4x super-resolution
INPUT_DEPTH = 32
LEARNING_RATE = 0.01
REG_NOISE_STD = 0.03
KERNEL_TYPE = "lanczos2"
PAD = "reflection"
UPSAMPLE_MODE = "bilinear"
OUTPUT_DIR = Path(__file__).resolve().parent / "output_dip_sr"


def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate Peak Signal-to-Noise Ratio (dB) between two image tensors."""
    mse = torch.mean((img1.detach() - img2.detach()) ** 2)
    if mse == 0:
        return 100.0
    return (10.0 * torch.log10((max_val ** 2) / mse)).item()


def build_skip_net(input_depth, pad=PAD, upsample_mode=UPSAMPLE_MODE):
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
        pad=pad,
        upsample_mode=upsample_mode,
        downsample_mode="stride",
        act_fun="LeakyReLU",
    )


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_path = PROJECT_DIR / "data/sr/zebra_GT.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    image_name = image_path.stem.replace("_GT", "")
    run_name = f"dip_sr_{image_name}_x{FACTOR}_seed_{SEED}"

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "dip-super-resolution"),
        name=run_name,
        config={
            "architecture": "dip-skip",
            "image": image_name,
            "factor": FACTOR,
            "steps": STEPS,
            "input_depth": INPUT_DEPTH,
            "learning_rate": LEARNING_RATE,
            "reg_noise_std": REG_NOISE_STD,
            "kernel_type": KERNEL_TYPE,
            "upsample_mode": UPSAMPLE_MODE,
            "seed": SEED,
        },
    )

    wandb.define_metric("iteration")
    wandb.define_metric("loss/*", step_metric="iteration")
    wandb.define_metric("psnr/*", step_metric="iteration")

    # Load HR image (crop to divisible by 32) and synthesize LR observation
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

    print("==========================================================")
    print(" Deep Image Prior: Super-Resolution Reproduction")
    print(f" Device: {device}")
    print(f" Ground Truth HR Resolution: {imgs['HR_pil'].size} (W x H)")
    print(f" Downsampled LR Resolution:  {imgs['LR_pil'].size} (W x H)")
    print(f" Factor: {FACTOR}x")
    print(f" Baseline PSNR (Bicubic): {psnr_bicubic:.2f} dB")
    print(f" Baseline PSNR (Nearest): {psnr_nearest:.2f} dB")
    print("==========================================================")

    # Save initial ground truth and baseline images
    real_path = OUTPUT_DIR / f"{image_name}_real_HR.png"
    downsampled_path = OUTPUT_DIR / f"{image_name}_downsampled_LR.png"
    bicubic_path = OUTPUT_DIR / f"{image_name}_bicubic_HR.png"
    nearest_path = OUTPUT_DIR / f"{image_name}_nearest_HR.png"

    imgs["HR_pil"].save(real_path)
    imgs["LR_pil"].save(downsampled_path)
    np_to_pil(imgs["bicubic_np"]).save(bicubic_path)
    np_to_pil(imgs["nearest_np"]).save(nearest_path)

    # Setup DIP model, downsampler, and optimizer
    net = build_skip_net(INPUT_DEPTH).to(device)
    downsampler = Downsampler(
        n_planes=3, factor=FACTOR, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True
    ).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    print(f"Network parameters: {parameter_count(net):,}")

    # Generate fixed uniform input noise, scaled by 0.1 as in DIP
    hr_h, hr_w = imgs["HR_pil"].size[1], imgs["HR_pil"].size[0]
    net_input_saved = torch.zeros((1, INPUT_DEPTH, hr_h, hr_w), device=device)
    net_input_saved.uniform_()
    net_input_saved *= 1.0 / 10.0
    noise = torch.zeros_like(net_input_saved)

    losses = []
    psnr_hr_history = []
    psnr_lr_history = []

    print("\nStarting optimization...")
    for iteration in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)

        # Input jitter regularization
        if REG_NOISE_STD > 0:
            net_input = net_input_saved + (noise.normal_() * REG_NOISE_STD)
        else:
            net_input = net_input_saved

        out_HR = net(net_input)
        out_LR = downsampler(out_HR)

        loss = criterion(out_LR, img_LR_tensor)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        with torch.no_grad():
            psnr_hr = calculate_psnr(out_HR, img_HR_tensor)
            psnr_lr = calculate_psnr(out_LR, img_LR_tensor)
        psnr_hr_history.append(psnr_hr)
        psnr_lr_history.append(psnr_lr)

        run.log({
            "loss/total": loss_val,
            "psnr/HR": psnr_hr,
            "psnr/LR": psnr_lr,
            "iteration": iteration,
        })

        if iteration == 1 or iteration % 100 == 0 or iteration == STEPS:
            print(
                f"  Iteration {iteration:05d}/{STEPS}: "
                f"loss={loss_val:.6f}, "
                f"PSNR_LR={psnr_lr:.2f} dB, "
                f"PSNR_HR={psnr_hr:.2f} dB"
            )

    # Final inference without jitter
    with torch.no_grad():
        final_out_HR = net(net_input_saved)
        final_out_HR_np = np.clip(torch_to_np(final_out_HR), 0.0, 1.0)
        final_psnr_hr = calculate_psnr(final_out_HR, img_HR_tensor)

    print("\n==========================================================")
    print(f" Training Complete after {STEPS} steps!")
    print(f" Final DIP SR PSNR:    {final_psnr_hr:.2f} dB")
    print(f" Baseline Bicubic PSNR: {psnr_bicubic:.2f} dB")
    print(f" Improvement:           +{final_psnr_hr - psnr_bicubic:.2f} dB")
    print("==========================================================")

    # Save final super-resolved image
    sr_path = OUTPUT_DIR / f"{image_name}_super_resolution_HR.png"
    np_to_pil(final_out_HR_np).save(sr_path)
    print(f"Saved real HR image to:             {real_path}")
    print(f"Saved downsampled LR image to:      {downsampled_path}")
    print(f"Saved super-resolved image to:      {sr_path}")

    # Generate 4-panel visual comparison (Real, Downsampled, Bicubic, DIP SR)
    fig_comp, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(imgs["HR_np"].transpose(1, 2, 0))
    axes[0].set_title(f"Real (HR Ground Truth)\n{imgs['HR_pil'].size[0]}x{imgs['HR_pil'].size[1]}")
    axes[0].axis("off")

    axes[1].imshow(imgs["LR_np"].transpose(1, 2, 0))
    axes[1].set_title(f"Downsampled (LR Input)\n{imgs['LR_pil'].size[0]}x{imgs['LR_pil'].size[1]}")
    axes[1].axis("off")

    axes[2].imshow(imgs["bicubic_np"].transpose(1, 2, 0))
    axes[2].set_title(f"Bicubic Baseline\nPSNR: {psnr_bicubic:.2f} dB")
    axes[2].axis("off")

    axes[3].imshow(final_out_HR_np.transpose(1, 2, 0))
    axes[3].set_title(f"DIP Super-Resolution ({STEPS} steps)\nPSNR: {final_psnr_hr:.2f} dB")
    axes[3].axis("off")

    fig_comp.tight_layout()
    comparison_path = OUTPUT_DIR / f"{image_name}_sr_comparison.png"
    fig_comp.savefig(comparison_path, dpi=150)
    plt.close(fig_comp)
    print(f"Saved visual comparison figure to:  {comparison_path}")

    # Generate training curves plot (Loss & PSNR vs iteration)
    fig_metrics, (ax_l, ax_p) = plt.subplots(1, 2, figsize=(14, 5))
    iters = np.arange(1, STEPS + 1)

    ax_l.plot(iters, losses, label="Reconstruction MSE Loss", color="tab:blue")
    ax_l.set_xscale("log")
    ax_l.set_yscale("log")
    ax_l.set_xlabel("Iteration (log scale)")
    ax_l.set_ylabel("MSE Loss")
    ax_l.set_title("Training Loss (Downsampled Reconstruction)")
    ax_l.grid(True, which="both", alpha=0.25)
    ax_l.legend()

    ax_p.plot(iters, psnr_hr_history, label="DIP Output vs Real HR", color="tab:green")
    ax_p.plot(iters, psnr_lr_history, label="DIP Output vs Downsampled LR", color="tab:orange", alpha=0.6)
    ax_p.axhline(psnr_bicubic, color="tab:red", linestyle="--", label=f"Bicubic Baseline ({psnr_bicubic:.2f} dB)")
    ax_p.set_xscale("log")
    ax_p.set_xlabel("Iteration (log scale)")
    ax_p.set_ylabel("PSNR (dB)")
    ax_p.set_title("PSNR Progress Across Iterations")
    ax_p.grid(True, which="both", alpha=0.25)
    ax_p.legend()

    fig_metrics.tight_layout()
    metrics_path = OUTPUT_DIR / f"{image_name}_sr_metrics.png"
    fig_metrics.savefig(metrics_path, dpi=150)
    plt.close(fig_metrics)
    print(f"Saved training curves to:           {metrics_path}")

    # Log all artifacts and final images to WandB
    run.log({
        "real_HR_image": wandb.Image(str(real_path), caption="Real High-Resolution"),
        "downsampled_LR_image": wandb.Image(str(downsampled_path), caption="Downsampled Low-Resolution Input"),
        "bicubic_baseline": wandb.Image(str(bicubic_path), caption=f"Bicubic ({psnr_bicubic:.2f} dB)"),
        "super_resolved_HR": wandb.Image(str(sr_path), caption=f"DIP SR ({final_psnr_hr:.2f} dB)"),
        "visual_comparison": wandb.Image(str(comparison_path), caption="Visual Comparison"),
        "metrics_curves": wandb.Image(str(metrics_path), caption="Loss & PSNR Curves"),
    })
    run.summary["final_psnr_hr"] = final_psnr_hr
    run.summary["psnr_bicubic"] = psnr_bicubic
    run.summary["psnr_nearest"] = psnr_nearest
    run.summary["psnr_gain"] = final_psnr_hr - psnr_bicubic

    run.finish()


if __name__ == "__main__":
    main()
