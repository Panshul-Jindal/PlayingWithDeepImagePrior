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
from models.dpwise_skip import dpwise_skip
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


def build_model(arch="dpwise", width=128):
    builder = skip if arch == "skip" else dpwise_skip
    return builder(
        num_input_channels=32,
        num_output_channels=3,
        num_channels_down=[width] * 5,
        num_channels_up=[width] * 5,
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
    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(__file__).resolve().parent / "output_zebra_scaling"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "zebra_scaling.log", "w", buffering=1)

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")

    img_path = PROJECT_DIR / "data/sr/zebra_GT.png"
    imgs = load_LR_HR_imgs_sr(str(img_path), imsize=-1, factor=4, enforse_div32="CROP")
    imgs["bicubic_np"], _, _ = get_baselines(imgs["LR_pil"], imgs["HR_pil"])

    hr_t = np_to_torch(imgs["HR_np"]).to(device)
    lr_t = np_to_torch(imgs["LR_np"]).to(device)
    bic_t = np_to_torch(imgs["bicubic_np"]).to(device)
    w, h = imgs["HR_pil"].size

    imgs["HR_pil"].save(out_dir / "zebra_original_HR.png")
    imgs["LR_pil"].save(out_dir / "zebra_observed_LR.png")
    np_to_pil(imgs["bicubic_np"]).save(out_dir / "zebra_bicubic_HR.png")

    psnr_bic_rgb = calc_psnr(bic_t, hr_t)
    psnr_bic_y = calc_psnr_y(bic_t, hr_t, shave=4)
    log("=" * 90)
    log(f" ZEBRA DEPTHWISE SCALING TEST (346K vs 1M vs Classic Skip 2.2M)")
    log(f" Resolution: HR={w}x{h} | LR={imgs['LR_pil'].size[0]}x{imgs['LR_pil'].size[1]} | Steps: 4000 | Seed: {SEED}")
    log(f" Baselines:  Bicubic RGB = {psnr_bic_rgb:.2f} dB | Bicubic Y-shaved = {psnr_bic_y:.2f} dB")
    log("=" * 90)

    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "dip-zebra-scaling"), name="zebra_dpwise_1M_test", reinit=True) if HAS_WANDB else None
    downsampler = Downsampler(n_planes=3, factor=4, kernel_type="lanczos2", phase=0.5, preserve_size=True).to(device)
    for p in downsampler.parameters():
        p.requires_grad = False

    eval_configs = [
        ("dpwise_346k_l2",         "dpwise", 128, "l2"),
        ("dpwise_346k_deepred_l2",  "dpwise", 128, "deepred_l2"),
        ("dpwise_1M_l2",           "dpwise", 224, "l2"),
        ("dpwise_1M_deepred_l2",    "dpwise", 224, "deepred_l2"),
    ]

    net_input_saved = torch.rand((1, 32, h, w), device=device) * 0.1
    results = {}

    for tag, arch, width, loss_type in eval_configs:
        net = build_model(arch, width=width).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=0.0005)
        params = sum(p.numel() for p in net.parameters() if p.requires_grad)

        log(f"\n---> [{tag:>26}] Params: {params:,} (width={width}) | Steps: 4000")
        t0 = time.time()
        noise = torch.zeros_like(net_input_saved)

        for step in range(1, 4001):
            opt.zero_grad(set_to_none=True)
            inp = net_input_saved + (noise.normal_() * 0.03)
            out_hr = net(inp)
            out_lr = downsampler(out_hr)

            loss = F.mse_loss(out_lr, lr_t)
            if loss_type == "deepred_l2":
                loss = loss + 1e-3 * red_penalty(out_hr)

            loss.backward()
            opt.step()

            if step % 100 == 0 or step == 4000:
                with torch.no_grad():
                    psnr_val = calc_psnr(out_hr, hr_t)
                    psnr_y_val = calc_psnr_y(out_hr, hr_t, shave=4)
                log(f"  [{tag:>26}] step {step:04d}/4000 | Loss: {loss.item():.6f} | RGB: {psnr_val:.2f} dB | Y: {psnr_y_val:.2f} dB | {time.time()-t0:.1f}s")
                if run:
                    run.log({f"loss/{tag}": loss.item(), f"psnr/{tag}": psnr_val, f"psnr_y/{tag}": psnr_y_val, "step": step})

        # Save step 4000 output directly
        with torch.no_grad():
            np_to_pil(np.clip(torch_to_np(out_hr), 0, 1)).save(out_dir / f"zebra_HR_{tag}.png")
            results[tag] = {"psnr_rgb": psnr_val, "psnr_y": psnr_y_val, "params": params}

        del net
        torch.cuda.empty_cache()

    # Comparison summary table
    log("\n" + "=" * 95)
    log(f" FINAL SUMMARY TABLE: ZEBRA SCALING (4000 STEPS)")
    log("=" * 95)
    log(f"{'Configuration':<28} | {'Params':<12} | {'RGB PSNR':<12} | {'RGB Gain':<10} | {'Y-shaved PSNR':<14} | {'Y Gain':<10}")
    log("-" * 95)
    log(f"{'Bicubic':<28} | {'N/A':<12} | {psnr_bic_rgb:<12.2f} | {'Baseline':<10} | {psnr_bic_y:<14.2f} | {'Baseline':<10}")
    for tag, res in results.items():
        gain_rgb = res["psnr_rgb"] - psnr_bic_rgb
        gain_y = res["psnr_y"] - psnr_bic_y
        log(f"{tag:<28} | {res['params']:<12,} | {res['psnr_rgb']:<12.2f} | {gain_rgb:+.2f} dB  | {res['psnr_y']:<14.2f} | {gain_y:+.2f} dB")
    log("=" * 95)

    log_f.close()
    if run:
        run.finish()


if __name__ == "__main__":
    main()

