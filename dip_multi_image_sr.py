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


def calc_psnr(x, y):
    mse = torch.mean((x.detach() - y.detach()) ** 2)
    return (10.0 * torch.log10(1.0 / mse)).item() if mse > 0 else 100.0


def red_penalty(x):
    k = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32, device=x.device) / 16.0
    k = k.repeat(x.shape[1], 1, 1, 1)
    with torch.no_grad():
        denoised = F.conv2d(x, k, padding=1, groups=x.shape[1])
    return 0.5 * torch.mean(x * (x - denoised))


def build_skip(model_type="skip"):
    builder = skip if model_type == "skip" else dpwise_skip
    return builder(
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_out = Path(__file__).resolve().parent / "output_multi_image_sr"
    base_out.mkdir(parents=True, exist_ok=True)
    log_f = open(base_out / "multi_image_experiment.log", "w", buffering=1)

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")

    set14_dir = PROJECT_DIR / "data/sr/Set14"
    image_paths = sorted(set14_dir.glob("*.png"))
    log(f"Found {len(image_paths)} images in {set14_dir}: {[p.name for p in image_paths]}")

    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "dip-multi-image-sr"), name="set14_skip_vs_dpwise", reinit=True) if HAS_WANDB else None
    downsampler = Downsampler(n_planes=3, factor=4, kernel_type="lanczos2", phase=0.5, preserve_size=True).to(device)
    eval_configs = [("skip", "l2"), ("skip", "deepred_l2"), ("dpwise_skip", "l2"), ("dpwise_skip", "deepred_l2")]
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
        w, h = imgs["HR_pil"].size

        # Save reference images using the exact file stem
        imgs["HR_pil"].save(img_dir / f"{name}_original_HR.png")
        imgs["LR_pil"].save(img_dir / f"{name}_observed_LR.png")
        np_to_pil(imgs["bicubic_np"]).save(img_dir / f"{name}_bicubic_HR.png")

        psnr_bic = calc_psnr(np_to_torch(imgs["bicubic_np"]).to(device), hr_t)
        log(f"\n[{idx}/{len(image_paths)}] {name} ({w}x{h}) | Bicubic: {psnr_bic:.2f} dB")
        results[name] = {"bicubic": psnr_bic}

        net_input_saved = torch.rand((1, 32, h, w), device=device) * 0.1

        for model_type, loss_type in eval_configs:
            tag = f"{model_type}_{loss_type}"
            net = build_skip(model_type).to(device)
            opt = torch.optim.Adam(net.parameters(), lr=0.0005)
            params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            log(f"\n---> [{tag:>24}] Params: {params:,} | Image: {name} | 2500 steps")

            t0 = time.time()
            for step in range(1, 2501):
                opt.zero_grad(set_to_none=True)
                inp = net_input_saved + (torch.randn_like(net_input_saved) * 0.03)
                out_hr = net(inp)
                out_lr = downsampler(out_hr)

                loss = F.mse_loss(out_lr, lr_t)
                if loss_type == "deepred_l2":
                    loss = loss + 1e-3 * red_penalty(out_hr)

                loss.backward()
                opt.step()

                if step % 100 == 0 or step == 2500:
                    with torch.no_grad():
                        psnr_val = calc_psnr(out_hr, hr_t)
                    if step % 100 == 0 or step == 2500:
                        log(f"  [{tag:>24}] step {step:04d}/2500 | Loss: {loss.item():.6f} | PSNR: {psnr_val:.2f} dB | {time.time()-t0:.1f}s")
                    if run:
                        run.log({f"loss/{name}_{tag}": loss.item(), f"psnr/{name}_{tag}": psnr_val, "step": step})

            # Save final trained step output directly
            with torch.no_grad():
                np_to_pil(np.clip(torch_to_np(out_hr), 0, 1)).save(img_dir / f"{name}_HR_{tag}.png")
                results[name][tag] = psnr_val
            del net
            torch.cuda.empty_cache()

    # Final summary table
    log("\n" + "=" * 90)
    log(f"{'Image':<20} | {'Bicubic':<8} | {'skip_l2':<14} | {'skip_deepred':<14} | {'dpwise_l2':<14} | {'dpwise_deepred':<14}")
    log("-" * 90)
    tags = ["skip_l2", "skip_deepred_l2", "dpwise_skip_l2", "dpwise_skip_deepred_l2"]
    for img, res in results.items():
        row = f"{img:<20} | {res['bicubic']:<8.2f} | " + " | ".join(f"{res[t]:.2f} ({res[t]-res['bicubic']:+.2f})" for t in tags)
        log(row)
    log("-" * 90)
    avg_row = f"{'AVERAGE':<20} | {np.mean([r['bicubic'] for r in results.values()]):<8.2f} | " + " | ".join(
        f"{np.mean([r[t] for r in results.values()]):.2f}" for t in tags
    )
    log(avg_row)
    log("=" * 90)

    log_f.close()
    if run:
        run.finish()


if __name__ == "__main__":
    main()
