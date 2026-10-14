"""DIP Super-Resolution with Depthwise-Separable Skip Architecture across Losses."""

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
from models.dpwise_skip import dpwise_skip
from utils.common_utils import np_to_pil, np_to_torch, pil_to_np, torch_to_np
from utils.sr_utils import get_baselines, load_LR_HR_imgs_sr


def calc_psnr(x, y):
    mse = torch.mean((x.detach() - y.detach()) ** 2)
    return (10.0 * torch.log10(1.0 / mse)).item() if mse > 0 else 100.0


def tv_penalty(x):
    return torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])) + torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))


def red_penalty(x):
    kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32, device=x.device) / 16.0
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)
    with torch.no_grad():
        denoised = F.conv2d(x, kernel, padding=1, groups=x.shape[1])
    return 0.5 * torch.mean(x * (x - denoised))


def build_dpwise_skip(input_depth=32, pad="reflection", upsample_mode="bilinear"):
    """Build exact DIP SR configuration using depthwise-separable convolutions."""
    return dpwise_skip(
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(__file__).resolve().parent / "output_dip_dpwise_sr"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_path = PROJECT_DIR / "data/sr/zebra_GT.png"
    name = img_path.stem.replace("_GT", "")
    log_file = open(out_dir / f"{name}_dpwise_sr.log", "w", buffering=1)

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")

    # Load data & compute baselines
    imgs = load_LR_HR_imgs_sr(str(img_path), imsize=-1, factor=4, enforse_div32="CROP")
    imgs["bicubic_np"], _, imgs["nearest_np"] = get_baselines(imgs["LR_pil"], imgs["HR_pil"])
    imgs["bilinear_pil"] = imgs["LR_pil"].resize(imgs["HR_pil"].size, Image.BILINEAR)
    imgs["bilinear_np"] = pil_to_np(imgs["bilinear_pil"])

    hr_t = np_to_torch(imgs["HR_np"]).to(device)
    lr_t = np_to_torch(imgs["LR_np"]).to(device)
    w, h = imgs["HR_pil"].size

    # Save baseline images
    imgs["HR_pil"].save(out_dir / f"{name}_original_HR.png")
    imgs["LR_pil"].save(out_dir / f"{name}_observed_LR.png")
    np_to_pil(imgs["bicubic_np"]).save(out_dir / f"{name}_bicubic_HR.png")
    imgs["bilinear_pil"].save(out_dir / f"{name}_bilinear_HR.png")

    psnr_bic = calc_psnr(np_to_torch(imgs["bicubic_np"]).to(device), hr_t)
    psnr_bil = calc_psnr(np_to_torch(imgs["bilinear_np"]).to(device), hr_t)

    run = None
    if HAS_WANDB:
        try:
            run = wandb.init(project=os.environ.get("WANDB_PROJECT", "dip-dpwise-sr"), name=f"dpwise_{name}", reinit=True)
        except Exception:
            run = None

    downsampler = Downsampler(n_planes=3, factor=4, kernel_type="lanczos2", phase=0.5, preserve_size=True).to(device)
    net_input_saved = torch.rand((1, 32, h, w), device=device) * 0.1

    # Loss configurations: DIP L2, Total Variation for DIP L2, Deep RED L1, Deep RED L2
    loss_dict = {
        "l2":         lambda d, hr: F.mse_loss(d, lr_t),
        "tv":         lambda d, hr: F.mse_loss(d, lr_t) + 1e-4 * tv_penalty(hr),
        "deepred_l1": lambda d, hr: F.l1_loss(d, lr_t) + 1e-3 * red_penalty(hr),
        "deepred_l2": lambda d, hr: F.mse_loss(d, lr_t) + 1e-3 * red_penalty(hr),
    }

    log("=" * 80)
    log(f" DIP Depthwise-Separable Skip Super-Resolution | Device: {device} | {w}x{h} ({w*h:,} px)")
    log(f" Baselines: Bicubic={psnr_bic:.2f} dB | Bilinear={psnr_bil:.2f} dB")
    log("=" * 80)

    results = {}
    for loss_name, loss_fn in loss_dict.items():
        net = build_dpwise_skip(32).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=0.0005)
        params = sum(p.numel() for p in net.parameters() if p.requires_grad)

        log(f"\n[DEPTHWISE-SKIP | {loss_name.upper()}] 6000 steps | Params: {params:,} | Input Pixels: {w*h:,}")
        best_psnr, t0 = -1.0, time.time()

        for step in range(1, 6001):
            opt.zero_grad(set_to_none=True)
            inp = net_input_saved + (torch.randn_like(net_input_saved) * 0.03)
            out_hr = net(inp)
            out_lr = downsampler(out_hr)

            loss = loss_fn(out_lr, out_hr)
            loss.backward()
            opt.step()

            with torch.no_grad():
                psnr_hr = calc_psnr(out_hr, hr_t)
            best_psnr = max(best_psnr, psnr_hr)

            if step == 1 or step % 500 == 0 or step == 6000:
                elapsed = time.time() - t0
                log(f"  [{loss_name:>10}] step {step:04d}/6000 | Loss: {loss.item():.6f} | PSNR_HR: {psnr_hr:.2f} dB (Best: {best_psnr:.2f}) | {elapsed:.1f}s")

            if run is not None:
                run.log({f"loss/{loss_name}": loss.item(), f"psnr/{loss_name}": psnr_hr, "step": step})

        # Final prediction is directly the output of the final trained step (step 6000)
        with torch.no_grad():
            final_hr = out_hr
            final_psnr = psnr_hr
            np_to_pil(np.clip(torch_to_np(final_hr), 0, 1)).save(out_dir / f"{name}_HR_{loss_name}.png")

        results[loss_name] = {"final_psnr": final_psnr, "best_psnr": best_psnr, "loss": loss.item(), "params": params}
        del net
        torch.cuda.empty_cache()

    # Final summary table
    log("\n" + "=" * 80)
    log(f"{'Approach':<14} | {'Params':<12} | {'Input Pixels':<14} | {'Loss':<12} | {'Final PSNR':<12} | {'Gain vs Bicubic':<15}")
    log("-" * 80)
    log(f"{'Bicubic':<14} | {'N/A':<12} | {f'{w*h:,}':<14} | {'N/A':<12} | {psnr_bic:<12.2f} | Baseline")
    log(f"{'Bilinear':<14} | {'N/A':<12} | {f'{w*h:,}':<14} | {'N/A':<12} | {psnr_bil:<12.2f} | {psnr_bil-psnr_bic:+.2f} dB")
    for k, v in results.items():
        log(f"{k:<14} | {v['params']:<12,} | {f'{w*h:,}':<14} | {v['loss']:<12.6f} | {v['final_psnr']:<12.2f} | {v['final_psnr']-psnr_bic:+.2f} dB")
    log("=" * 80)

    log_file.close()
    if run is not None:
        run.finish()
    print(f"\nExperiment complete! Outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()
