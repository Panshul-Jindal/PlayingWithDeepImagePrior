"""DIP Super-Resolution with Loss & Regularizer Variants (Skip Architecture)."""

import os
from pathlib import Path
import sys
import time

from PIL import Image
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = getattr(Image, "Resampling", Image).LANCZOS

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

PROJECT_DIR = Path(__file__).resolve().parent / "src/playingwithdeepimageprior/deep-image-prior"
sys.path.insert(0, str(PROJECT_DIR))

from models.downsampler import Downsampler
from models.skip import skip
from utils.common_utils import np_to_pil, np_to_torch, pil_to_np, torch_to_np
from utils.sr_utils import get_baselines, load_LR_HR_imgs_sr


def rgb_to_y(img):
    """Convert RGB tensor in [0, 1] to ITU-R BT.601 Y (luminance) channel."""
    return 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]


def calc_psnr(x, y):
    mse = torch.mean((x.detach() - y.detach()) ** 2)
    return (10.0 * torch.log10(1.0 / mse)).item() if mse > 0 else 100.0


def calc_psnr_y(x, y, shave=4):
    """Calculate standard benchmark PSNR on Y channel with boundary shaved."""
    if shave > 0:
        x = x[:, :, shave:-shave, shave:-shave]
        y = y[:, :, shave:-shave, shave:-shave]
    xy = rgb_to_y(x)
    yy = rgb_to_y(y)
    mse = torch.mean((xy.detach() - yy.detach()) ** 2)
    return (10.0 * torch.log10(1.0 / mse)).item() if mse > 0 else 100.0


def tv_penalty(x):
    return torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])) + torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))


def edge_preserving_denoiser(img, radius=3, eps=1e-3):
    kernel_size = 2 * radius + 1
    pad = radius
    img_pad = F.pad(img, (pad, pad, pad, pad), mode="reflect")
    channels = img.shape[1]
    weight = torch.ones((channels, 1, kernel_size, kernel_size), device=img.device) / (kernel_size * kernel_size)
    mean_I = F.conv2d(img_pad, weight, groups=channels)
    mean_II = F.conv2d(img_pad ** 2, weight, groups=channels)
    var_I = torch.clamp(mean_II - mean_I ** 2, min=0.0)
    a = var_I / (var_I + eps)
    b = mean_I - a * mean_I
    mean_a = F.conv2d(F.pad(a, (pad, pad, pad, pad), mode="reflect"), weight, groups=channels)
    mean_b = F.conv2d(F.pad(b, (pad, pad, pad, pad), mode="reflect"), weight, groups=channels)
    return (mean_a * img + mean_b).clamp(0.0, 1.0)


def red_penalty(x):
    with torch.no_grad():
        denoised = edge_preserving_denoiser(x)
    return 0.5 * torch.mean((x - denoised) ** 2)


def build_skip(in_depth=32):
    return skip(
        num_input_channels=in_depth, num_output_channels=3,
        num_channels_down=[128] * 5, num_channels_up=[128] * 5, num_channels_skip=[4] * 5,
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, pad="reflection", upsample_mode="bilinear",
        downsample_mode="stride", act_fun="LeakyReLU",
    )


def main():
    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(__file__).resolve().parent / "output_dip_losses_sr"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_path = PROJECT_DIR / "data/sr/zebra_GT.png"
    name = img_path.stem.replace("_GT", "")
    log_file = open(out_dir / f"{name}_losses.log", "w", buffering=1)

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")

    # Load data & baselines
    imgs = load_LR_HR_imgs_sr(str(img_path), imsize=-1, factor=4, enforse_div32="CROP")
    imgs["bicubic_np"], _, imgs["nearest_np"] = get_baselines(imgs["LR_pil"], imgs["HR_pil"])
    imgs["bilinear_pil"] = imgs["LR_pil"].resize(imgs["HR_pil"].size, Image.BILINEAR)
    imgs["bilinear_np"] = pil_to_np(imgs["bilinear_pil"])

    hr_t = np_to_torch(imgs["HR_np"]).to(device)
    lr_t = np_to_torch(imgs["LR_np"]).to(device)
    bic_t = np_to_torch(imgs["bicubic_np"]).to(device)
    w, h = imgs["HR_pil"].size

    # Save reference images
    imgs["HR_pil"].save(out_dir / f"{name}_original_HR.png")
    imgs["LR_pil"].save(out_dir / f"{name}_observed_LR.png")
    np_to_pil(imgs["bicubic_np"]).save(out_dir / f"{name}_bicubic_HR.png")
    imgs["bilinear_pil"].save(out_dir / f"{name}_bilinear_HR.png")

    psnr_bic_rgb = calc_psnr(bic_t, hr_t)
    psnr_bic_y = calc_psnr_y(bic_t, hr_t, shave=4)
    psnr_bil_rgb = calc_psnr(np_to_torch(imgs["bilinear_np"]).to(device), hr_t)

    run = None
    if HAS_WANDB:
        try:
            run = wandb.init(project=os.environ.get("WANDB_PROJECT", "dip-losses-sr"), name=f"skip_losses_{name}", reinit=True)
        except Exception:
            run = None

    downsampler = Downsampler(n_planes=3, factor=4, kernel_type="lanczos2", phase=0.5, preserve_size=True).to(device)
    for p in downsampler.parameters():
        p.requires_grad = False

    net_input_saved = torch.rand((1, 32, h, w), device=device) * 0.1

    # Loss dictionary: l2, l1, tv, deepred_l2, deepred_l1
    loss_dict = {
        "l2":         lambda d, hr: F.mse_loss(d, lr_t),
        "l1":         lambda d, hr: F.l1_loss(d, lr_t),
        "tv":         lambda d, hr: F.mse_loss(d, lr_t) + 1e-4 * tv_penalty(hr),
        "deepred_l2": lambda d, hr: F.mse_loss(d, lr_t) + 1e-3 * red_penalty(hr),
        "deepred_l1": lambda d, hr: F.l1_loss(d, lr_t) + 1e-3 * red_penalty(hr),
    }

    log("=" * 90)
    log(f" DIP Loss & Regularizer Comparison | Device: {device} | Resolution: {w}x{h} ({w*h:,} px) | Seed: {SEED}")
    log(f" Baselines: Bicubic RGB={psnr_bic_rgb:.2f} dB, Y-shaved={psnr_bic_y:.2f} dB | Bilinear RGB={psnr_bil_rgb:.2f} dB")
    log("=" * 90)

    results = {}
    for loss_name, loss_fn in loss_dict.items():
        net = build_skip(32).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=0.0005)
        params = sum(p.numel() for p in net.parameters() if p.requires_grad)

        log(f"\n[{loss_name.upper()}] Starting 4000 steps | Params: {params:,} | Input Pixels: {w*h:,}")
        best_psnr, t0 = -1.0, time.time()
        noise = torch.zeros_like(net_input_saved)

        for step in range(1, 4001):
            opt.zero_grad(set_to_none=True)
            inp = net_input_saved + (noise.normal_() * 0.03)
            out_hr = net(inp)
            out_lr = downsampler(out_hr)

            loss = loss_fn(out_lr, out_hr)
            loss.backward()
            opt.step()

            if step == 1 or step % 100 == 0 or step == 4000:
                with torch.no_grad():
                    psnr_hr = calc_psnr(out_hr, hr_t)
                    psnr_y = calc_psnr_y(out_hr, hr_t, shave=4)
                best_psnr = max(best_psnr, psnr_hr)
                elapsed = time.time() - t0
                log(f"  [{loss_name:>10}] step {step:04d}/4000 | Loss: {loss.item():.6f} | RGB: {psnr_hr:.2f} dB | Y: {psnr_y:.2f} dB | {elapsed:.1f}s")

                if run is not None:
                    run.log({f"loss/{loss_name}": loss.item(), f"psnr/{loss_name}": psnr_hr, f"psnr_y/{loss_name}": psnr_y, "step": step})

        with torch.no_grad():
            final_hr = out_hr
            final_psnr = calc_psnr(final_hr, hr_t)
            final_psnr_y = calc_psnr_y(final_hr, hr_t, shave=4)
            np_to_pil(np.clip(torch_to_np(final_hr), 0, 1)).save(out_dir / f"{name}_HR_{loss_name}.png")

        results[loss_name] = {"final_psnr": final_psnr, "final_psnr_y": final_psnr_y, "best_psnr": best_psnr, "loss": loss.item(), "params": params}
        del net
        torch.cuda.empty_cache()

    # Final summary table
    log("\n" + "=" * 95)
    log(" FINAL SUMMARY TABLE: LOSS COMPARISON (4000 STEPS)")
    log("=" * 95)
    log(f"{'Approach':<14} | {'Params':<12} | {'RGB PSNR':<12} | {'RGB Gain':<10} | {'Y-shaved PSNR':<14} | {'Y Gain':<10}")
    log("-" * 95)
    log(f"{'Bicubic':<14} | {'N/A':<12} | {psnr_bic_rgb:<12.2f} | {'Baseline':<10} | {psnr_bic_y:<14.2f} | {'Baseline':<10}")
    for k, v in results.items():
        gain_rgb = v['final_psnr'] - psnr_bic_rgb
        gain_y = v['final_psnr_y'] - psnr_bic_y
        log(f"{k:<14} | {v['params']:<12,} | {v['final_psnr']:<12.2f} | {gain_rgb:+.2f} dB  | {v['final_psnr_y']:<14.2f} | {gain_y:+.2f} dB")
    log("=" * 95)

    log_file.close()
    if run is not None:
        run.finish()
    print(f"\nCompleted! Outputs and log saved in: {out_dir}")


if __name__ == "__main__":
    main()
