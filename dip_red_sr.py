"""Experiment 1: DeepRED (Deep Image Prior with Regularization by Denoising).

Based on arXiv:1903.10176 ("DeepRED: Deep Image Prior Powered by RED", Mataev et al.).
Enhances Deep Image Prior for Super-Resolution by combining the architectural
prior of DIP with an explicit Regularization by Denoising (RED) objective:
    min_theta 0.5 * ||A f_theta(z) - y||_2^2 + lambda_red * 0.5 * ||f_theta(z) - D(f_theta(z))||_2^2 + lambda_tv * TV(f_theta(z))

Where D(x) is an edge-preserving denoiser that suppresses high-frequency artifacts
and noise overfitting while preserving structural edges.

Saves:
  - Real (HR Ground Truth) image
  - Downsampled (LR Input) image
  - HR with baseline DIP
  - HR with New Approach (DeepRED)
  - Baselines (Bicubic, Nearest)
  - Comprehensive comparison figure with PSNR metrics
  - Training metrics curves (Loss, PSNR_HR, PSNR_LR)
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
LAMBDA_RED = 0.005      # Weight for RED regularization ||x - D(x)||^2
LAMBDA_TV = 1e-8        # Weight for Total Variation regularizer
KERNEL_TYPE = "lanczos2"
PAD = "reflection"
UPSAMPLE_MODE = "bilinear"
OUTPUT_DIR = Path(__file__).resolve().parent / "output_dip_red"


def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate Peak Signal-to-Noise Ratio (dB)."""
    mse = torch.mean((img1.detach() - img2.detach()) ** 2)
    if mse == 0:
        return 100.0
    return (10.0 * torch.log10((max_val ** 2) / mse)).item()


def total_variation_loss(img):
    """Anisotropic Total Variation loss for edge preservation."""
    dh = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    dw = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    return torch.mean(dh[:, :, :-1, :]) + torch.mean(dw[:, :, :, :-1])


def edge_preserving_denoiser(img, radius=3, eps=1e-3):
    """Fast GPU-native Guided / Edge-preserving Filter as denoiser D(x).

    Smooths flat and noisy regions while strictly maintaining sharp edges.
    """
    kernel_size = 2 * radius + 1
    # Average filter for local statistics
    pad = radius
    img_pad = F.pad(img, (pad, pad, pad, pad), mode="reflect")

    # Mean filtering per channel
    channels = img.shape[1]
    weight = torch.ones((channels, 1, kernel_size, kernel_size), device=img.device) / (kernel_size * kernel_size)

    mean_I = F.conv2d(img_pad, weight, groups=channels)
    mean_II = F.conv2d(img_pad ** 2, weight, groups=channels)
    var_I = mean_II - mean_I ** 2

    a = var_I / (var_I + eps)
    b = mean_I - a * mean_I

    mean_a = F.conv2d(F.pad(a, (pad, pad, pad, pad), mode="reflect"), weight, groups=channels)
    mean_b = F.conv2d(F.pad(b, (pad, pad, pad, pad), mode="reflect"), weight, groups=channels)

    q = mean_a * img + mean_b
    return q.clamp(0.0, 1.0)


def build_skip_net(input_depth):
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
        pad=PAD,
        upsample_mode=UPSAMPLE_MODE,
        downsample_mode="stride",
        act_fun="LeakyReLU",
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_path = PROJECT_DIR / "data/sr/zebra_GT.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    image_name = image_path.stem.replace("_GT", "")
    run_name = f"dip_red_sr_{image_name}_x{FACTOR}_seed_{SEED}"

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "dip-super-resolution"),
        name=run_name,
        config={
            "method": "DeepRED (DIP + Regularization by Denoising)",
            "paper": "arXiv:1903.10176",
            "image": image_name,
            "factor": FACTOR,
            "steps": STEPS,
            "input_depth": INPUT_DEPTH,
            "learning_rate": LEARNING_RATE,
            "lambda_red": LAMBDA_RED,
            "lambda_tv": LAMBDA_TV,
            "reg_noise_std": REG_NOISE_STD,
            "seed": SEED,
        },
    )

    wandb.define_metric("iteration")
    wandb.define_metric("loss/*", step_metric="iteration")
    wandb.define_metric("psnr/*", step_metric="iteration")

    # Load HR image and generate LR input observation
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

    # Check for existing vanilla DIP reference result
    dip_ref_path = Path(__file__).resolve().parent / "output_dip_sr/zebra_super_resolution_HR.png"
    psnr_dip_val = None
    if dip_ref_path.exists():
        img_dip_pil = Image.open(dip_ref_path).convert("RGB")
        img_dip_tensor = np_to_torch(np.array(img_dip_pil).transpose(2, 0, 1).astype(np.float32) / 255.0).to(device)
        psnr_dip_val = calculate_psnr(img_dip_tensor, img_HR_tensor)
        shutil.copy(dip_ref_path, OUTPUT_DIR / f"{image_name}_hr_with_dip.png")
        print(f"Loaded vanilla DIP reference (PSNR: {psnr_dip_val:.2f} dB)")
    else:
        print("Vanilla DIP reference not found locally; will compare with baseline.")

    print("==========================================================")
    print(" Experiment 1: DeepRED Super-Resolution (arXiv:1903.10176)")
    print(f" Device: {device}")
    print(f" Resolution: LR={imgs['LR_pil'].size} -> HR={imgs['HR_pil'].size}")
    print(f" Lambda RED: {LAMBDA_RED}, Lambda TV: {LAMBDA_TV}")
    print(f" Baselines: Bicubic={psnr_bicubic:.2f} dB, Nearest={psnr_nearest:.2f} dB")
    if psnr_dip_val:
        print(f" Vanilla DIP Baseline: {psnr_dip_val:.2f} dB")
    print("==========================================================")

    # Save Ground Truth HR and Downsampled LR
    real_path = OUTPUT_DIR / f"{image_name}_real_HR.png"
    downsampled_path = OUTPUT_DIR / f"{image_name}_downsampled_LR.png"
    imgs["HR_pil"].save(real_path)
    imgs["LR_pil"].save(downsampled_path)

    # Network & downsampler setup
    net = build_skip_net(INPUT_DEPTH).to(device)
    downsampler = Downsampler(
        n_planes=3, factor=FACTOR, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True
    ).to(device)
    mse_criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Fixed uniform noise input
    hr_h, hr_w = imgs["HR_pil"].size[1], imgs["HR_pil"].size[0]
    net_input_saved = torch.zeros((1, INPUT_DEPTH, hr_h, hr_w), device=device)
    net_input_saved.uniform_()
    net_input_saved *= 1.0 / 10.0
    noise = torch.zeros_like(net_input_saved)

    losses_total = []
    losses_data = []
    losses_red = []
    psnr_hr_history = []
    psnr_lr_history = []

    print("\nStarting DeepRED optimization...")
    for iteration in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)

        if REG_NOISE_STD > 0:
            net_input = net_input_saved + (noise.normal_() * REG_NOISE_STD)
        else:
            net_input = net_input_saved

        out_HR = net(net_input)
        out_LR = downsampler(out_HR)

        # 1. Data fidelity loss
        loss_data = mse_criterion(out_LR, img_LR_tensor)

        # 2. RED regularization: ||x - D(x)||^2 with detached denoiser target
        with torch.no_grad():
            denoised_target = edge_preserving_denoiser(out_HR)
        loss_red = 0.5 * torch.mean((out_HR - denoised_target) ** 2)

        # 3. TV regularization
        loss_tv = total_variation_loss(out_HR)

        total_loss = loss_data + LAMBDA_RED * loss_red + LAMBDA_TV * loss_tv
        total_loss.backward()
        optimizer.step()

        loss_total_val = total_loss.item()
        loss_data_val = loss_data.item()
        loss_red_val = loss_red.item()

        losses_total.append(loss_total_val)
        losses_data.append(loss_data_val)
        losses_red.append(loss_red_val)

        with torch.no_grad():
            psnr_hr = calculate_psnr(out_HR, img_HR_tensor)
            psnr_lr = calculate_psnr(out_LR, img_LR_tensor)
        psnr_hr_history.append(psnr_hr)
        psnr_lr_history.append(psnr_lr)

        run.log({
            "loss/total": loss_total_val,
            "loss/data_mse": loss_data_val,
            "loss/red_penalty": loss_red_val,
            "psnr/HR": psnr_hr,
            "psnr/LR": psnr_lr,
            "iteration": iteration,
        })

        if iteration == 1 or iteration % 100 == 0 or iteration == STEPS:
            print(
                f"  Iteration {iteration:05d}/{STEPS}: "
                f"loss_data={loss_data_val:.6f}, "
                f"loss_red={loss_red_val:.6f}, "
                f"PSNR_LR={psnr_lr:.2f} dB, "
                f"PSNR_HR={psnr_hr:.2f} dB"
            )

    # Final inference without jitter
    with torch.no_grad():
        final_out_HR = net(net_input_saved)
        final_out_HR_np = np.clip(torch_to_np(final_out_HR), 0.0, 1.0)
        final_psnr_hr = calculate_psnr(final_out_HR, img_HR_tensor)

    print("\n==========================================================")
    print(" DeepRED Training Complete!")
    print(f" Final DeepRED PSNR:    {final_psnr_hr:.2f} dB")
    if psnr_dip_val:
        print(f" Vanilla DIP PSNR:      {psnr_dip_val:.2f} dB")
        print(f" Gain over DIP:         {final_psnr_hr - psnr_dip_val:+.2f} dB")
    print(f" Baseline Bicubic PSNR: {psnr_bicubic:.2f} dB")
    print(f" Gain over Bicubic:     {final_psnr_hr - psnr_bicubic:+.2f} dB")
    print("==========================================================")

    # Save HR with New Approach
    new_approach_path = OUTPUT_DIR / f"{image_name}_hr_with_new_approach.png"
    np_to_pil(final_out_HR_np).save(new_approach_path)
    print(f"Saved new approach HR to: {new_approach_path}")

    # Generate 5-Panel Visual Comparison:
    # 1: Real HR, 2: Downsampled LR, 3: Bicubic, 4: Vanilla DIP, 5: DeepRED (New Approach)
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
    axes[4].set_title(f"DeepRED (New Approach)\nPSNR: {final_psnr_hr:.2f} dB{diff_text}")
    axes[4].axis("off")

    fig_comp.tight_layout()
    comparison_path = OUTPUT_DIR / f"{image_name}_sr_comparison.png"
    fig_comp.savefig(comparison_path, dpi=150)
    plt.close(fig_comp)
    print(f"Saved comparison figure to: {comparison_path}")

    # Generate training curves plot
    fig_metrics, (ax_l, ax_p) = plt.subplots(1, 2, figsize=(14, 5))
    iters = np.arange(1, STEPS + 1)

    ax_l.plot(iters, losses_data, label="Data Fidelity Loss (MSE)", color="tab:blue")
    ax_l.plot(iters, losses_red, label="RED Regularization Loss", color="tab:purple", alpha=0.7)
    ax_l.set_xscale("log")
    ax_l.set_yscale("log")
    ax_l.set_xlabel("Iteration (log scale)")
    ax_l.set_ylabel("Loss")
    ax_l.set_title("DeepRED Optimization Losses")
    ax_l.grid(True, which="both", alpha=0.25)
    ax_l.legend()

    ax_p.plot(iters, psnr_hr_history, label="DeepRED Output vs Real HR", color="tab:green", linewidth=1.5)
    ax_p.plot(iters, psnr_lr_history, label="DeepRED Output vs Downsampled LR", color="tab:orange", alpha=0.5)
    if psnr_dip_val:
        ax_p.axhline(psnr_dip_val, color="tab:gray", linestyle=":", label=f"Vanilla DIP Final ({psnr_dip_val:.2f} dB)")
    ax_p.axhline(psnr_bicubic, color="tab:red", linestyle="--", label=f"Bicubic Baseline ({psnr_bicubic:.2f} dB)")
    ax_p.set_xscale("log")
    ax_p.set_xlabel("Iteration (log scale)")
    ax_p.set_ylabel("PSNR (dB)")
    ax_p.set_title("DeepRED PSNR Progress Across Iterations")
    ax_p.grid(True, which="both", alpha=0.25)
    ax_p.legend()

    fig_metrics.tight_layout()
    metrics_path = OUTPUT_DIR / f"{image_name}_sr_metrics.png"
    fig_metrics.savefig(metrics_path, dpi=150)
    plt.close(fig_metrics)
    print(f"Saved training curves to: {metrics_path}")

    # Log all images and comparison plots to WandB
    log_dict = {
        "real_HR_image": wandb.Image(str(real_path), caption="Real High-Resolution"),
        "downsampled_LR_image": wandb.Image(str(downsampled_path), caption="Downsampled Low-Resolution Input"),
        "hr_with_new_approach": wandb.Image(str(new_approach_path), caption=f"DeepRED ({final_psnr_hr:.2f} dB)"),
        "visual_comparison": wandb.Image(str(comparison_path), caption="Visual Comparison"),
        "metrics_curves": wandb.Image(str(metrics_path), caption="Loss & PSNR Curves"),
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
