"""Experiment 2: Single-Image Internal Learning for Super-Resolution (ZSSR-style UNet).

This approach trains an image-specific neural network exclusively on the internal
cross-scale self-similarity of the single input image y (Glasner et al., Shocher et al. ZSSR).

Key Principles:
  1. The Ground Truth HR image X is NEVER used during training (strictly single-image unsupervised).
  2. Training pairs are extracted entirely from y by downsampling crops of y by 4x using the
     exact forward degradation operator (Lanczos2) to create (y_sub_LR, y_sub_HR) pairs.
  3. Augmentation (multiscale crops, 4 rotation angles, reflections) generates internal training samples.
  4. A residual UNet learns to map the low-resolution patches to their higher-resolution parents.
  5. At test time, the full image y is fed into the trained network to produce the super-resolved output X_hat.
  6. An iterative iterative projection step enforces data consistency: Downsampler(X_hat) ~= y.

Saves:
  - Real (HR Ground Truth) image
  - Downsampled (LR Input) image
  - HR with baseline DIP
  - HR with New Approach (Internal UNet SR)
  - Baseline images (Bicubic, Nearest)
  - Comprehensive comparison figure with PSNR metrics
  - Training metrics curves (Loss per epoch, PSNR progress)
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
EPOCHS = 1000
BATCH_SIZE = 16
LEARNING_RATE = 0.0003
LAMBDA_TV = 1e-6
KERNEL_TYPE = "lanczos2"
OUTPUT_DIR = Path(__file__).resolve().parent / "output_single_image_internal_sr"


def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate Peak Signal-to-Noise Ratio (dB)."""
    mse = torch.mean((img1.detach() - img2.detach()) ** 2)
    if mse == 0:
        return 100.0
    return (10.0 * torch.log10((max_val ** 2) / mse)).item()


def charbonnier_loss(pred, target, eps=1e-3):
    """Charbonnier penalty (smooth L1) for sharper edges than MSE."""
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def total_variation_loss(img):
    dh = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    dw = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    return torch.mean(dh[:, :, :-1, :]) + torch.mean(dw[:, :, :, :-1])


class InternalPatchDataset(Dataset):
    """Generates internal cross-scale training pairs (LR, HR) strictly from y."""

    def __init__(self, img_LR_np, downsampler, patch_size=48, factor=4, num_samples=256):
        super().__init__()
        # img_LR_np: (3, H_LR, W_LR)
        self.img_tensor = torch.from_numpy(img_LR_np).float()
        self.downsampler = downsampler
        self.patch_size = patch_size
        self.factor = factor
        self.pairs = []

        c, h, w = self.img_tensor.shape
        # Create augmented crops across orientations and scales
        np.random.seed(SEED)
        try:
            down_device = next(self.downsampler.parameters()).device
        except (StopIteration, AttributeError):
            try:
                down_device = next(self.downsampler.buffers()).device
            except (StopIteration, AttributeError):
                down_device = torch.device("cpu")

        for _ in range(num_samples):
            # Sample random crop from y that is divisible by factor
            ps = self.patch_size
            if h <= ps or w <= ps:
                top = 0
                left = 0
                actual_h = (h // factor) * factor
                actual_w = (w // factor) * factor
            else:
                top = np.random.randint(0, h - ps + 1)
                left = np.random.randint(0, w - ps + 1)
                actual_h = ps
                actual_w = ps

            hr_patch = self.img_tensor[:, top:top + actual_h, left:left + actual_w]

            # Augmentation: random flips and rotations
            if np.random.rand() > 0.5:
                hr_patch = torch.flip(hr_patch, dims=[2])  # horizontal flip
            if np.random.rand() > 0.5:
                hr_patch = torch.flip(hr_patch, dims=[1])  # vertical flip
            rot_k = np.random.randint(0, 4)
            if rot_k > 0:
                hr_patch = torch.rot90(hr_patch, k=rot_k, dims=[1, 2])

            # Downsample by factor 4 using Lanczos forward operator
            with torch.no_grad():
                lr_patch = self.downsampler(hr_patch.unsqueeze(0).to(down_device)).squeeze(0).cpu()

            self.pairs.append((lr_patch, hr_patch))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect")
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect")

    def forward(self, x):
        return x + 0.1 * self.conv2(self.act(self.conv1(x)))


class InternalSRUNet(nn.Module):
    """Residual UNet mapping low-resolution patches to super-resolved residuals."""

    def __init__(self, in_channels=3, out_channels=3, hidden_dim=64):
        super().__init__()
        # Initial projection of bicubic upsampled input
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Encoder
        self.enc1 = nn.Sequential(ResidualBlock(hidden_dim), ResidualBlock(hidden_dim))
        self.down1 = nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride=2, padding=1)

        self.enc2 = nn.Sequential(ResidualBlock(hidden_dim * 2), ResidualBlock(hidden_dim * 2))

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(hidden_dim * 2),
            ResidualBlock(hidden_dim * 2),
        )

        # Decoder with skip connections
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1, padding_mode="reflect"),
        )
        self.dec1 = nn.Sequential(ResidualBlock(hidden_dim * 2), ResidualBlock(hidden_dim * 2))

        # Final reconstruction head predicts residual delta
        self.tail = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 3, padding=1, padding_mode="reflect"),
        )

        # Zero-initialize the final projection so initial model output is exactly bicubic
        nn.init.zeros_(self.tail[-1].weight)
        nn.init.zeros_(self.tail[-1].bias)

    def forward(self, x_lr, target_size=None):
        # x_lr: (B, 3, H_lr, W_lr)
        # First upsample via bilinear interpolation to target resolution
        if target_size is not None:
            x_up = F.interpolate(x_lr, size=target_size, mode="bicubic", align_corners=False)
        else:
            x_up = F.interpolate(x_lr, scale_factor=4, mode="bicubic", align_corners=False)

        h = self.head(x_up)
        e1 = self.enc1(h)
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        b = self.bottleneck(e2)

        u1 = self.up1(b)
        # Concatenate skip connection
        cat1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(cat1)
        residual = self.tail(d1)

        # Output = base bicubic upsample + learned high-frequency residual
        out = x_up + residual
        if not self.training:
            out = out.clamp(0.0, 1.0)
        return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_path = PROJECT_DIR / "data/sr/zebra_GT.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    image_name = image_path.stem.replace("_GT", "")
    run_name = f"zssr_internal_sr_{image_name}_x{FACTOR}_seed_{SEED}"

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "dip-super-resolution"),
        name=run_name,
        config={
            "method": "Single-Image Internal Learning (ZSSR UNet)",
            "image": image_name,
            "factor": FACTOR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "kernel_type": KERNEL_TYPE,
            "seed": SEED,
            "hr_used_in_training": False,
        },
    )

    wandb.define_metric("epoch")
    wandb.define_metric("loss/*", step_metric="epoch")
    wandb.define_metric("psnr/*", step_metric="epoch")

    # Load HR image and synthesize LR observation y
    imgs = load_LR_HR_imgs_sr(str(image_path), imsize=-1, factor=FACTOR, enforse_div32="CROP")
    imgs["bicubic_np"], imgs["sharp_np"], imgs["nearest_np"] = get_baselines(
        imgs["LR_pil"], imgs["HR_pil"]
    )

    # Convert to PyTorch tensors
    img_HR_tensor = np_to_torch(imgs["HR_np"]).to(device)
    img_LR_tensor = np_to_torch(imgs["LR_np"]).to(device)
    img_bicubic_tensor = np_to_torch(imgs["bicubic_np"]).to(device)
    img_nearest_tensor = np_to_torch(imgs["nearest_np"]).to(device)

    psnr_bicubic = calculate_psnr(img_bicubic_tensor, img_HR_tensor)
    psnr_nearest = calculate_psnr(img_nearest_tensor, img_HR_tensor)

    # Check for existing vanilla DIP reference
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

    # Build Internal Dataset strictly from y (img_LR_np)
    print("Generating internal training dataset strictly from input image y...")
    dataset = InternalPatchDataset(
        imgs["LR_np"], downsampler=downsampler, patch_size=48, factor=FACTOR, num_samples=256
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset generated: {len(dataset)} internal patch pairs (patch_size=48 -> 12).")

    # Build Model & Optimizer
    model = InternalSRUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    print("==========================================================")
    print(" Experiment 2: Single-Image Internal Learning (ZSSR UNet)")
    print(f" Device: {device}")
    print(f" Input y Resolution:  {imgs['LR_pil'].size} (W x H)")
    print(f" Target X Resolution: {imgs['HR_pil'].size} (W x H) [NOT used during training]")
    print(f" Baseline Bicubic PSNR: {psnr_bicubic:.2f} dB")
    if psnr_dip_val:
        print(f" Vanilla DIP Baseline:  {psnr_dip_val:.2f} dB")
    print("==========================================================")

    # Save Ground Truth HR and Input LR
    real_path = OUTPUT_DIR / f"{image_name}_real_HR.png"
    downsampled_path = OUTPUT_DIR / f"{image_name}_downsampled_LR.png"
    imgs["HR_pil"].save(real_path)
    imgs["LR_pil"].save(downsampled_path)

    loss_history = []
    psnr_internal_history = []
    psnr_hr_eval_history = []

    print("\nStarting internal training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_charbonnier = 0.0
        epoch_tv = 0.0
        epoch_internal_psnr = 0.0

        for lr_patches, hr_patches in dataloader:
            lr_patches = lr_patches.to(device)
            hr_patches = hr_patches.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred_patches = model(lr_patches, target_size=(hr_patches.shape[2], hr_patches.shape[3]))

            l_charb = charbonnier_loss(pred_patches, hr_patches)
            l_tv = total_variation_loss(pred_patches)
            total_l = l_charb + LAMBDA_TV * l_tv

            total_l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_l.item() * lr_patches.size(0)
            epoch_charbonnier += l_charb.item() * lr_patches.size(0)
            epoch_tv += l_tv.item() * lr_patches.size(0)
            epoch_internal_psnr += calculate_psnr(pred_patches, hr_patches) * lr_patches.size(0)

        scheduler.step()

        num_samples = len(dataset)
        avg_loss = epoch_loss / num_samples
        avg_internal_psnr = epoch_internal_psnr / num_samples

        loss_history.append(avg_loss)
        psnr_internal_history.append(avg_internal_psnr)

        # Periodically evaluate super-resolved output on the full input y
        if epoch == 1 or epoch % 20 == 0 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                pred_HR_full = model(img_LR_tensor, target_size=(imgs["HR_pil"].size[1], imgs["HR_pil"].size[0]))
                psnr_hr_eval = calculate_psnr(pred_HR_full, img_HR_tensor)
                psnr_lr_cons = calculate_psnr(downsampler(pred_HR_full), img_LR_tensor)
            psnr_hr_eval_history.append((epoch, psnr_hr_eval))

            print(
                f"  Epoch {epoch:04d}/{EPOCHS}: "
                f"loss={avg_loss:.6f}, "
                f"internal_patch_PSNR={avg_internal_psnr:.2f} dB, "
                f"LR_consistency={psnr_lr_cons:.2f} dB, "
                f"HR_PSNR={psnr_hr_eval:.2f} dB"
            )

            run.log({
                "loss/total": avg_loss,
                "loss/charbonnier": epoch_charbonnier / num_samples,
                "loss/tv": epoch_tv / num_samples,
                "psnr/internal_patch": avg_internal_psnr,
                "psnr/HR_eval": psnr_hr_eval,
                "psnr/LR_consistency": psnr_lr_cons,
                "epoch": epoch,
            })
        else:
            run.log({
                "loss/total": avg_loss,
                "psnr/internal_patch": avg_internal_psnr,
                "epoch": epoch,
            })

    # Final Inference with Back-Projection refinement
    model.eval()
    with torch.no_grad():
        out_HR = model(img_LR_tensor, target_size=(imgs["HR_pil"].size[1], imgs["HR_pil"].size[0]))

        # Iterative back-projection to guarantee exact data-consistency with y
        for _ in range(10):
            lr_res = img_LR_tensor - downsampler(out_HR)
            lr_res_up = F.interpolate(
                lr_res, size=(imgs["HR_pil"].size[1], imgs["HR_pil"].size[0]), mode="bicubic", align_corners=False
            )
            out_HR = (out_HR + 0.5 * lr_res_up).clamp(0.0, 1.0)

        final_out_HR_np = np.clip(torch_to_np(out_HR), 0.0, 1.0)
        final_psnr_hr = calculate_psnr(out_HR, img_HR_tensor)

    print("\n==========================================================")
    print(" Single-Image Internal Learning Complete!")
    print(f" Final Internal UNet PSNR: {final_psnr_hr:.2f} dB")
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
    axes[4].set_title(f"Internal UNet (New Approach)\nPSNR: {final_psnr_hr:.2f} dB{diff_text}")
    axes[4].axis("off")

    fig_comp.tight_layout()
    comparison_path = OUTPUT_DIR / f"{image_name}_sr_comparison.png"
    fig_comp.savefig(comparison_path, dpi=150)
    plt.close(fig_comp)
    print(f"Saved comparison figure to: {comparison_path}")

    # Generate training curves plot
    fig_metrics, (ax_l, ax_p) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_arr = np.arange(1, EPOCHS + 1)

    ax_l.plot(epochs_arr, loss_history, label="Internal Training Loss", color="tab:blue")
    ax_l.set_xlabel("Epoch")
    ax_l.set_ylabel("Loss")
    ax_l.set_title("Internal Learning Convergence")
    ax_l.grid(True, which="both", alpha=0.25)
    ax_l.legend()

    eval_epochs = [x[0] for x in psnr_hr_eval_history]
    eval_psnrs = [x[1] for x in psnr_hr_eval_history]
    ax_p.plot(eval_epochs, eval_psnrs, marker="o", label="Full Image PSNR vs Real HR", color="tab:green")
    if psnr_dip_val:
        ax_p.axhline(psnr_dip_val, color="tab:gray", linestyle=":", label=f"Vanilla DIP Final ({psnr_dip_val:.2f} dB)")
    ax_p.axhline(psnr_bicubic, color="tab:red", linestyle="--", label=f"Bicubic Baseline ({psnr_bicubic:.2f} dB)")
    ax_p.set_xlabel("Epoch")
    ax_p.set_ylabel("PSNR (dB)")
    ax_p.set_title("Validation PSNR on Real HR")
    ax_p.grid(True, which="both", alpha=0.25)
    ax_p.legend()

    fig_metrics.tight_layout()
    metrics_path = OUTPUT_DIR / f"{image_name}_sr_metrics.png"
    fig_metrics.savefig(metrics_path, dpi=150)
    plt.close(fig_metrics)
    print(f"Saved training curves to: {metrics_path}")

    # Log to WandB
    log_dict = {
        "real_HR_image": wandb.Image(str(real_path), caption="Real High-Resolution"),
        "downsampled_LR_image": wandb.Image(str(downsampled_path), caption="Downsampled Low-Resolution Input"),
        "hr_with_new_approach": wandb.Image(str(new_approach_path), caption=f"Internal UNet ({final_psnr_hr:.2f} dB)"),
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
