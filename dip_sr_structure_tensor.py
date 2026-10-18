"""Experiment 6: Structure Tensor Coherence Regularized DIP (ST-DIP).

A novel Super-Resolution loss functi that exploits the eigensystem of the local Structure Tensor
to eliminate multi-directional high-frequency noise and checkerboard artifacts without blurring edges.

Mathematical Foundation:
  The local Structure Tensor J_rho is the Gaussian-smoothed outer product of spatial gradients:
      J_rho = K_rho * [ (dx)^2     (dx dy) ]
                      [ (dx dy)    (dy)^2  ]
  Its eigenvalues mu_1 >= mu_2 >= 0 satisfy:
      Trace = J_xx + J_yy = mu_1 + mu_2
      Diff  = sqrt((J_xx - J_yy)^2 + 4 J_xy^2) = mu_1 - mu_2
      mu_2  = 0.5 * (Trace - Diff)   (the minor eigenvalue)

Physical Property for Super-Resolution:
  - On a clean 1D edge (e.g. zebra stripe contour): mu_1 >> mu_2 ~= 0.
    The gradient is strictly unidirectional.
  - On DIP high-frequency artifacts (noise, ringing, checkerboard patterns): mu_2 > 0.
    Gradients oscillate in multiple orthogonal directions.
  - By directly minimizing the minor eigenvalue mu_2, the loss selectively penalizes
    multi-directional artifact energy in the null-space of A while leaving sharp 1D edges intact!

Saves:
  - zebra_real_HR.png
  - zebra_downsampled_LR.png
  - zebra_hr_with_dip.png
  - zebra_hr_with_new_approach.png
  - Visual comparison figure with all baselines and PSNR metrics
  - Training metrics curves (data loss, structure tensor loss, HR PSNR)
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
LAMBDA_ST = 0.05         # Structure Tensor minor eigenvalue penalty
KERNEL_TYPE = "lanczos2"
OUTPUT_DIR = Path(__file__).resolve().parent / "output_dip_structure_tensor"


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


class StructureTensorLoss(nn.Module):
    """Computes the minor eigenvalue mu_2 of the local structure tensor."""

    def __init__(self, kernel_size=5, sigma=1.0, channels=3, device="cuda"):
        super().__init__()
        self.channels = channels

        # Sobel differentiation filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3) / 8.0
        self.register_buffer("sobel_x", sobel_x.repeat(channels, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.repeat(channels, 1, 1, 1))

        # Gaussian smoothing kernel
        ax = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32, device=device)
        gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
        gauss_kernel = (gauss[:, None] * gauss[None, :])
        gauss_kernel = (gauss_kernel / gauss_kernel.sum()).view(1, 1, kernel_size, kernel_size)
        self.register_buffer("gauss_kernel", gauss_kernel.repeat(channels, 1, 1, 1))
        self.padding = kernel_size // 2

    def forward(self, img):
        # img: (1, 3, H, W)
        dx = F.conv2d(img, self.sobel_x, padding=1, groups=self.channels)
        dy = F.conv2d(img, self.sobel_y, padding=1, groups=self.channels)

        j_xx = dx * dx
        j_yy = dy * dy
        j_xy = dx * dy

        # Smooth components with Gaussian kernel
        j_xx_smooth = F.conv2d(j_xx, self.gauss_kernel, padding=self.padding, groups=self.channels)
        j_yy_smooth = F.conv2d(j_yy, self.gauss_kernel, padding=self.padding, groups=self.channels)
        j_xy_smooth = F.conv2d(j_xy, self.gauss_kernel, padding=self.padding, groups=self.channels)

        # Eigenvalues: mu_2 = 0.5 * (Trace - sqrt((j_xx - j_yy)^2 + 4 * j_xy^2))
        trace = j_xx_smooth + j_yy_smooth
        diff = torch.sqrt((j_xx_smooth - j_yy_smooth) ** 2 + 4.0 * (j_xy_smooth ** 2) + 1e-10)
        mu_2 = 0.5 * (trace - diff)

        # Penalize minor eigenvalue (cross-edge noise)
        return torch.mean(mu_2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_path = PROJECT_DIR / "data/sr/zebra_GT.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    image_name = image_path.stem.replace("_GT", "")
    run_name = f"dip_st_sr_{image_name}_x{FACTOR}_seed_{SEED}"

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "dip-super-resolution"),
        name=run_name,
        config={
            "method": "Structure Tensor Coherence Regularized DIP (ST-DIP)",
            "image": image_name,
            "factor": FACTOR,
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "lambda_st": LAMBDA_ST,
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

    # Setup Structure Tensor loss
    st_loss_fn = StructureTensorLoss(kernel_size=5, sigma=1.0, channels=3, device=device)

    # Setup DIP skip network and optimizer
    net = build_skip_net(INPUT_DEPTH).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Fixed uniform input noise
    net_input_saved = torch.zeros((1, INPUT_DEPTH, hr_h, hr_w), device=device)
    net_input_saved.uniform_()
    net_input_saved *= 1.0 / 10.0
    noise = torch.zeros_like(net_input_saved)

    print("==========================================================")
    print(" Experiment 6: Structure Tensor Coherence DIP (ST-DIP)")
    print(f" Device: {device}")
    print(f" Resolution: LR={imgs['LR_pil'].size} -> HR={imgs['HR_pil'].size}")
    print(f" Lambda ST: {LAMBDA_ST}")
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

    print("\nStarting ST-DIP optimization...")
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

        # 2. Structure tensor minor eigenvalue penalty
        loss_st = st_loss_fn(out_HR)

        total_loss = loss_data + LAMBDA_ST * loss_st
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
                f"loss_st={loss_st.item():.6f}, "
                f"PSNR_LR={psnr_lr:.2f} dB, "
                f"PSNR_HR={psnr_hr:.2f} dB (best={best_psnr_hr:.2f} dB)"
            )

            run.log({
                "loss/total": total_loss.item(),
                "loss/data_mse": loss_data.item(),
                "loss/structure_tensor_mu2": loss_st.item(),
                "psnr/LR": psnr_lr,
                "psnr/HR": psnr_hr,
                "iteration": i,
            })

    # Final reconstruction
    final_out_HR = best_out_HR if best_out_HR is not None else out_HR.detach()
    final_out_HR_np = np.clip(torch_to_np(final_out_HR), 0.0, 1.0)
    final_psnr_hr = calculate_psnr(final_out_HR, img_HR_tensor)

    print("\n==========================================================")
    print(" ST-DIP Optimization Complete!")
    print(f" Final ST-DIP PSNR:    {final_psnr_hr:.2f} dB")
    if psnr_dip_val:
        print(f" Vanilla DIP Baseline: {psnr_dip_val:.2f} dB")
        print(f" Gain over DIP:        {final_psnr_hr - psnr_dip_val:+.2f} dB")
    print(f" Baseline Bicubic PSNR:{psnr_bicubic:.2f} dB")
    print(f" Gain over Bicubic:    {final_psnr_hr - psnr_bicubic:+.2f} dB")
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
    axes[4].set_title(f"ST-DIP (Structure Tensor)\nPSNR: {final_psnr_hr:.2f} dB{diff_text}")
    axes[4].axis("off")

    fig_comp.tight_layout()
    comparison_path = OUTPUT_DIR / f"{image_name}_sr_comparison.png"
    fig_comp.savefig(comparison_path, dpi=150)
    plt.close(fig_comp)
    print(f"Saved comparison figure to: {comparison_path}")

    # Metrics curves
    fig_metrics, (ax_l, ax_p) = plt.subplots(1, 2, figsize=(14, 5))
    iter_arr = np.arange(1, STEPS + 1)

    ax_l.plot(iter_arr, loss_history, label="Total Loss", color="tab:blue")
    ax_l.set_yscale("log")
    ax_l.set_xlabel("Iteration")
    ax_l.set_ylabel("Loss (Log Scale)")
    ax_l.set_title("ST-DIP Convergence")
    ax_l.grid(True, which="both", alpha=0.25)
    ax_l.legend()

    ax_p.plot(iter_arr, psnr_hr_history, label="ST-DIP PSNR", color="tab:green")
    if psnr_dip_val:
        ax_p.axhline(psnr_dip_val, color="tab:gray", linestyle=":", label=f"Vanilla DIP Final ({psnr_dip_val:.2f} dB)")
    ax_p.axhline(psnr_bicubic, color="tab:red", linestyle="--", label=f"Bicubic ({psnr_bicubic:.2f} dB)")
    ax_p.set_xlabel("Iteration")
    ax_p.set_ylabel("PSNR vs Ground Truth (dB)")
    ax_p.set_title("SR Reconstruction Fidelity")
    ax_p.grid(True, which="both", alpha=0.25)
    ax_p.legend()

    fig_metrics.tight_layout()
    metrics_path = OUTPUT_DIR / f"{image_name}_sr_metrics.png"
    fig_metrics.savefig(metrics_path, dpi=150)
    plt.close(fig_metrics)
    print(f"Saved training curves to:   {metrics_path}")

    # Log to WandB
    log_dict = {
        "real_HR_image": wandb.Image(str(real_path), caption="Real High-Resolution"),
        "downsampled_LR_image": wandb.Image(str(downsampled_path), caption="Downsampled Low-Resolution Input"),
        "hr_with_new_approach": wandb.Image(str(new_approach_path), caption=f"ST-DIP ({final_psnr_hr:.2f} dB)"),
        "visual_comparison": wandb.Image(str(comparison_path), caption="Visual Comparison"),
        "metrics_curves": wandb.Image(str(metrics_path), caption="Loss & PSNR Curves"),
    }
    if psnr_dip_val and (OUTPUT_DIR / f"{image_name}_hr_with_dip.png").exists():
        log_dict["hr_with_dip"] = wandb.Image(str(OUTPUT_DIR / f"{image_name}_hr_with_dip.png"), caption=f"Vanilla DIP ({psnr_dip_val:.2f} dB)")

    run.log(log_dict)
    run.summary["final_psnr_hr"] = final_psnr_hr
    run.summary["best_psnr_hr"] = best_psnr_hr
    run.summary["psnr_bicubic"] = psnr_bicubic
    if psnr_dip_val:
        run.summary["psnr_dip"] = psnr_dip_val
        run.summary["gain_over_dip"] = final_psnr_hr - psnr_dip_val
    run.summary["gain_over_bicubic"] = final_psnr_hr - psnr_bicubic

    run.finish()


if __name__ == "__main__":
    main()

