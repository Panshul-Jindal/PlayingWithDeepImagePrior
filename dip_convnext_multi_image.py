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
from models.convnext_skip import convnext_skip
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


def build_model(arch="convnext_skip"):
    if arch == "convnext_skip":
        return convnext_skip(
            num_input_channels=32,
            num_output_channels=3,
            num_channels_down=[128] * 5,
            num_channels_up=[128] * 5,
            num_channels_skip=[4] * 5,
            kernel_size=7,
            expansion=2,
            pad="reflection",
            upsample_mode="bilinear",
        )
    elif arch == "skip":
        return skip(
            num_input_channels=32,
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
    else:
        raise ValueError(f"Unknown arch: {arch}")


def main():
    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_out = Path(__file__).resolve().parent / "output_convnext_multi_image"
    base_out.mkdir(parents=True, exist_ok=True)
    log_f = open(base_out / "convnext_multi_image.log", "w", buffering=1)

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")

    set14_dir = PROJECT_DIR / "data/sr/Set14"
    image_paths = sorted(set14_dir.glob("*.png"))
    log("=" * 95)
    log(" MULTI-IMAGE DIP SUPER-RESOLUTION: ConvNeXt-Skip vs Classic Skip on Set14")
    log(f" Found {len(image_paths)} images in {set14_dir}")
    log(f" Device: {device} | Factor: 4x | Steps: 4000 | LR: 0.0005 | Noise: 0.03 | Seed: {SEED}")
    log("=" * 95)

    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "dip-convnext-multi"), name="set14_convnext_vs_skip", reinit=True) if HAS_WANDB else None
    downsampler = Downsampler(n_planes=3, factor=4, kernel_type="lanczos2", phase=0.5, preserve_size=True).to(device)
    for p in downsampler.parameters():
        p.requires_grad = False

    eval_configs = [
        ("convnext_skip", "l2"),
        ("convnext_skip", "deepred_l2"),
        ("skip", "l2"),
        ("skip", "deepred_l2"),
    ]
    results = {}

    for idx, img_path in enumerate(image_paths, 1):
        name = img_path.stem
        img_dir = base_out / name
        img_dir.mkdir(parents=True, exist_ok=True)

        imgs = load_LR_HR_imgs_sr(str(img_path), imsize=-1, factor=4, enforse_div32="CROP")
        if imgs["HR_np"].shape[0] == 1:
            imgs["HR_np"] = np.repeat(imgs["HR_np"], 3, axis=0)
            imgs["LR_np"] = np.repeat(imgs["LR_np"], 3, axis=0)
            imgs["HR_pil"] = imgs["HR_pil"].convert("RGB")
            imgs["LR_pil"] = imgs["LR_pil"].convert("RGB")

        imgs["bicubic_np"], _, _ = get_baselines(imgs["LR_pil"], imgs["HR_pil"])
        hr_t = np_to_torch(imgs["HR_np"]).to(device)
        lr_t = np_to_torch(imgs["LR_np"]).to(device)
        bic_t = np_to_torch(imgs["bicubic_np"]).to(device)
        w, h = imgs["HR_pil"].size

        # Save baseline images into this image's folder
        imgs["HR_pil"].save(img_dir / f"{name}_original_HR.png")
        imgs["LR_pil"].save(img_dir / f"{name}_observed_LR.png")
        np_to_pil(imgs["bicubic_np"]).save(img_dir / f"{name}_bicubic_HR.png")

        psnr_bic_rgb = calc_psnr(bic_t, hr_t)
        psnr_bic_y = calc_psnr_y(bic_t, hr_t, shave=4)
        log(f"\n[{idx}/{len(image_paths)}] {name} ({w}x{h}) | Bicubic: RGB={psnr_bic_rgb:.2f} dB, Y-shaved={psnr_bic_y:.2f} dB")
        results[name] = {"bicubic": psnr_bic_rgb, "bicubic_y": psnr_bic_y}

        net_input_saved = torch.rand((1, 32, h, w), device=device) * 0.1

        for arch, loss_type in eval_configs:
            tag = f"{arch}_{loss_type}"
            net = build_model(arch).to(device)
            opt = torch.optim.Adam(net.parameters(), lr=0.0005)
            params = sum(p.numel() for p in net.parameters() if p.requires_grad)

            log(f"  ---> [{tag:>26}] Params: {params:,} | Image: {name} | 4000 steps")
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
                    log(f"       [{tag:>26}] step {step:04d}/4000 | Loss: {loss.item():.6f} | RGB: {psnr_val:.2f} dB | Y: {psnr_y_val:.2f} dB | {time.time()-t0:.1f}s")
                    if run:
                        run.log({f"loss/{name}_{tag}": loss.item(), f"psnr/{name}_{tag}": psnr_val, f"psnr_y/{name}_{tag}": psnr_y_val, "step": step})

            # Save final trained step output directly into this image's folder
            with torch.no_grad():
                np_to_pil(np.clip(torch_to_np(out_hr), 0, 1)).save(img_dir / f"{name}_HR_{tag}.png")
                results[name][tag] = psnr_val
                results[name][f"{tag}_y"] = psnr_y_val
            del net
            torch.cuda.empty_cache()

    # Final summary table - RGB
    log("\n" + "=" * 95)
    log(" SUMMARY TABLE (FULL RGB PSNR, 4000 STEPS)")
    log("=" * 95)
    log(f"{'Image':<22} | {'Bicubic':<8} | {'convnext_l2':<14} | {'convnext_red':<14} | {'skip_l2':<14} | {'skip_red':<14}")
    log("-" * 95)
    tags = ["convnext_skip_l2", "convnext_skip_deepred_l2", "skip_l2", "skip_deepred_l2"]
    for img, res in results.items():
        row = f"{img:<22} | {res['bicubic']:<8.2f} | " + " | ".join(f"{res.get(t, 0.0):.2f} ({res.get(t, 0.0)-res['bicubic']:+.2f})" for t in tags)
        log(row)
    log("-" * 95)
    avg_row = f"{'AVERAGE':<22} | {np.mean([r['bicubic'] for r in results.values()]):<8.2f} | " + " | ".join(
        f"{np.mean([r.get(t, 0.0) for r in results.values()]):.2f}" for t in tags
    )
    log(avg_row)
    log("=" * 95)

    # Final summary table - Standard SR Benchmark (Y-channel, shaved border)
    log("\n" + "=" * 95)
    log(" SUMMARY TABLE (STANDARD SR BENCHMARK: Y-CHANNEL, SHAVED 4px, 4000 STEPS)")
    log("=" * 95)
    log(f"{'Image':<22} | {'Bicubic_Y':<9} | {'convnext_l2_y':<15} | {'convnext_red_y':<15} | {'skip_l2_y':<15} | {'skip_red_y':<15}")
    log("-" * 95)
    tags_y = [f"{t}_y" for t in tags]
    for img, res in results.items():
        row = f"{img:<22} | {res['bicubic_y']:<9.2f} | " + " | ".join(f"{res.get(ty, 0.0):.2f} ({res.get(ty, 0.0)-res['bicubic_y']:+.2f})" for ty in tags_y)
        log(row)
    log("-" * 95)
    avg_y_row = f"{'AVERAGE':<22} | {np.mean([r['bicubic_y'] for r in results.values()]):<9.2f} | " + " | ".join(
        f"{np.mean([r.get(ty, 0.0) for r in results.values()]):.2f}" for ty in tags_y
    )
    log(avg_y_row)
    log("=" * 95)

    log_f.close()
    if run:
        run.finish()


if __name__ == "__main__":
    main()

