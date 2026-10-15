"""Reproduce Deep Image Prior (DIP) Super-Resolution.

Fits a skip-architecture convolutional prior to a low-resolution image
such that Downsampler(output) reconstructs the low-resolution input.
"""

import os
from pathlib import Path
import sys
import time

from PIL import Image
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = getattr(Image, "Resampling", Image).LANCZOS

import numpy as np
import torch
import torch.nn as nn

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

PROJECT_DIR = Path(__file__).resolve().parent / "src/playingwithdeepimageprior/deep-image-prior"
sys.path.insert(0, str(PROJECT_DIR))

from models.downsampler import Downsampler
from models.skip import skip
from utils.common_utils import np_to_pil, np_to_torch, torch_to_np
from utils.sr_utils import get_baselines, load_LR_HR_imgs_sr

# --- Hyperparameters ---
STEPS = 4000
FACTOR = 4
INPUT_DEPTH = 32
LEARNING_RATE = 0.0005
REG_NOISE_STD = 0.03
SEED = 0
IMAGE_PATH = PROJECT_DIR / "data/sr/zebra_GT.png"
OUTPUT_DIR = Path(__file__).resolve().parent / "output_dip_sr"


def rgb_to_y(img):
    """Convert RGB tensor in [0, 1] to ITU-R BT.601 Y (luminance) channel."""
    return 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]


def calc_psnr(x, y, max_val=1.0):
    mse = torch.mean((x.detach() - y.detach()) ** 2)
    return (10.0 * torch.log10((max_val ** 2) / mse)).item() if mse > 0 else 100.0


def calc_psnr_y(x, y, shave=4, max_val=1.0):
    """Calculate standard benchmark PSNR on Y channel with boundary shaved."""
    if shave > 0:
        x = x[:, :, shave:-shave, shave:-shave]
        y = y[:, :, shave:-shave, shave:-shave]
    xy = rgb_to_y(x)
    yy = rgb_to_y(y)
    mse = torch.mean((xy.detach() - yy.detach()) ** 2)
    return (10.0 * torch.log10((max_val ** 2) / mse)).item() if mse > 0 else 100.0


def build_skip_net(input_depth=INPUT_DEPTH):
    return skip(
        num_input_channels=input_depth,
        num_output_channels=3,
        num_channels_down=[128] * 5,
        num_channels_up=[128] * 5,
        num_channels_skip=[4] * 5,
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


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_name = IMAGE_PATH.stem.replace("_GT", "")
    log_file = open(OUTPUT_DIR / f"{image_name}_dip_sr.log", "w", buffering=1)

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")

    log("=" * 80)
    log(" Deep Image Prior: Super-Resolution Reproduction")
    log(f" Target: {IMAGE_PATH.name} | Steps: {STEPS} | LR: {LEARNING_RATE} | Seed: {SEED}")
    log("=" * 80)

    run = None
    if HAS_WANDB:
        try:
            run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "dip-super-resolution"),
                name=f"dip_sr_{image_name}_x{FACTOR}",
                reinit=True,
            )
        except Exception:
            run = None

    imgs = load_LR_HR_imgs_sr(str(IMAGE_PATH), imsize=-1, factor=FACTOR, enforse_div32="CROP")
    imgs["bicubic_np"], _, _ = get_baselines(imgs["LR_pil"], imgs["HR_pil"])

    img_HR = np_to_torch(imgs["HR_np"]).to(device)
    img_LR = np_to_torch(imgs["LR_np"]).to(device)
    img_bic = np_to_torch(imgs["bicubic_np"]).to(device)

    psnr_bic_rgb = calc_psnr(img_bic, img_HR)
    psnr_bic_y = calc_psnr_y(img_bic, img_HR, shave=4)

    hr_w, hr_h = imgs["HR_pil"].size
    lr_w, lr_h = imgs["LR_pil"].size
    log(f" HR: {hr_w}x{hr_h} | LR: {lr_w}x{lr_h}")
    log(f" Baseline Bicubic: RGB={psnr_bic_rgb:.2f} dB, Y-shaved={psnr_bic_y:.2f} dB\n")

    imgs["HR_pil"].save(OUTPUT_DIR / f"{image_name}_real_HR.png")
    imgs["LR_pil"].save(OUTPUT_DIR / f"{image_name}_downsampled_LR.png")
    np_to_pil(imgs["bicubic_np"]).save(OUTPUT_DIR / f"{image_name}_bicubic_HR.png")

    net = build_skip_net().to(device)
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log(f"Network Parameters: {params:,}")

    downsampler = Downsampler(n_planes=3, factor=FACTOR, kernel_type="lanczos2", phase=0.5, preserve_size=True).to(device)
    for p in downsampler.parameters():
        p.requires_grad = False

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    net_input_saved = torch.rand((1, INPUT_DEPTH, hr_h, hr_w), device=device) * 0.1
    noise = torch.zeros_like(net_input_saved)

    best_rgb, best_y = -1.0, -1.0
    t0 = time.time()

    for step in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        inp = net_input_saved + (noise.normal_() * REG_NOISE_STD)

        out_HR = net(inp)
        out_LR = downsampler(out_HR)

        loss = criterion(out_LR, img_LR)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            psnr_rgb = calc_psnr(out_HR, img_HR)
            psnr_y = calc_psnr_y(out_HR, img_HR, shave=4)

        best_rgb = max(best_rgb, psnr_rgb)
        best_y = max(best_y, psnr_y)

        if run is not None:
            try:
                run.log({
                    "loss": loss.item(),
                    "psnr_rgb": psnr_rgb,
                    "psnr_y": psnr_y,
                    "step": step,
                })
            except Exception:
                pass

        if step == 1 or step % 100 == 0 or step == STEPS:
            log(
                f"  step {step:04d}/{STEPS} | "
                f"Loss: {loss.item():.6f} | "
                f"RGB: {psnr_rgb:.2f} dB (best: {best_rgb:.2f}) | "
                f"Y: {psnr_y:.2f} dB (best: {best_y:.2f}) | "
                f"Elapsed: {time.time()-t0:.1f}s"
            )

    # Save final output directly from step 4000
    with torch.no_grad():
        final_out_np = np.clip(torch_to_np(out_HR), 0.0, 1.0)
        np_to_pil(final_out_np).save(OUTPUT_DIR / f"{image_name}_super_resolution_HR.png")

    log("\n" + "=" * 80)
    log(" SUMMARY TABLE")
    log("=" * 80)
    log(f"{'Method':<16} | {'Final RGB':<12} | {'Best RGB':<12} | {'Final Y':<12} | {'Best Y':<12} | {'Y Gain vs Bic':<15}")
    log("-" * 80)
    log(f"{'Bicubic':<16} | {psnr_bic_rgb:<12.2f} | {psnr_bic_rgb:<12.2f} | {psnr_bic_y:<12.2f} | {psnr_bic_y:<12.2f} | {'Baseline':<15}")
    log(f"{'DIP Skip':<16} | {psnr_rgb:<12.2f} | {best_rgb:<12.2f} | {psnr_y:<12.2f} | {best_y:<12.2f} | {psnr_y - psnr_bic_y:+12.2f} dB")
    log("=" * 80)

    if run is not None:
        try:
            run.finish()
        except Exception:
            pass

    log_file.close()
    print(f"\nDone! DIP SR results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
