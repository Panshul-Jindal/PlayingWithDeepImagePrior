"""Super-Resolution across Deep Image Prior (DIP) Architectures.

Performs 4x Super-Resolution (4000 steps per model) comparing:
  - skip (DIP Hourglass / Skip encoder-decoder)
  - unet (Standard U-Net)
  - resnet (Residual CNN prior)
  - swin (Swin Transformer prior)
  - swin_skip (Recursive Swin-skip prior)
"""

import csv
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
from models.unet import UNet
from models.resnet import ResNet
from models.swin import swin
from models.swin_skip import swin_skip
from utils.common_utils import np_to_pil, np_to_torch, torch_to_np
from utils.sr_utils import get_baselines, load_LR_HR_imgs_sr

# --- Experiment Constants ---
STEPS = 4000
FACTOR = 4
INPUT_DEPTH = 32
LEARNING_RATE = 0.0005
REG_NOISE_STD = 0.03
SEED = 0
MODELS = ["skip", "unet", "resnet", "swin", "swin_skip"]
IMAGE_PATH = PROJECT_DIR / "data/sr/zebra_GT.png"
OUTPUT_DIR = Path(__file__).resolve().parent / "output_all_models_sr"


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


def build_model(name, input_depth=INPUT_DEPTH):
    name = name.lower()
    if name == "skip":
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
    elif name == "unet":
        return UNet(
            num_input_channels=input_depth,
            num_output_channels=3,
            feature_scale=4,
            more_layers=0,
            concat_x=False,
            upsample_mode="bilinear",
            pad="reflection",
            norm_layer=nn.BatchNorm2d,
            need_sigmoid=True,
            need_bias=True,
        )
    elif name == "resnet":
        return ResNet(
            num_input_channels=input_depth,
            num_output_channels=3,
            num_blocks=10,
            num_channels=64,
            need_residual=True,
            act_fun="LeakyReLU",
            need_sigmoid=True,
            norm_layer=nn.BatchNorm2d,
            pad="reflection",
        )
    elif name == "swin":
        return swin(
            input_depth=input_depth,
            output_channels=3,
            patch_size=4,
            embed_dim=32,
            depth=2,
            num_heads=4,
            window_size=8,
            need_sigmoid=True,
        )
    elif name == "swin_skip":
        return swin_skip(
            input_depth=input_depth,
            output_channels=3,
            patch_size=4,
            embed_dim=32,
            depth=2,
            num_heads=4,
            window_size=8,
            need_sigmoid=True,
        )
    raise ValueError(f"Unknown model architecture: '{name}'")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_name = IMAGE_PATH.stem.replace("_GT", "")
    log_file = open(OUTPUT_DIR / f"{image_name}_sr_all_models.log", "w", buffering=1)

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")

    log("=" * 80)
    log(" Deep Image Prior: Multi-Model Super-Resolution Comparison")
    log(f" Target: {IMAGE_PATH.name} | Factor: {FACTOR}x | Steps: {STEPS} | LR: {LEARNING_RATE} | Seed: {SEED}")
    log(f" Models: {', '.join(MODELS)}")
    log("=" * 80)

    run = None
    if HAS_WANDB:
        try:
            run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "dip-multi-model-sr"),
                name=f"all_models_sr_{image_name}_x{FACTOR}",
                reinit=True,
            )
        except Exception as e:
            log(f"[Warning] WandB init failed ({e}), continuing offline.")
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
    log(f" HR Resolution: {hr_w}x{hr_h} | LR Resolution: {lr_w}x{lr_h}")
    log(f" Baseline Bicubic: RGB={psnr_bic_rgb:.2f} dB, Y-shaved={psnr_bic_y:.2f} dB\n")

    imgs["HR_pil"].save(OUTPUT_DIR / f"{image_name}_original_HR.png")
    imgs["LR_pil"].save(OUTPUT_DIR / f"{image_name}_observed_LR.png")
    np_to_pil(imgs["bicubic_np"]).save(OUTPUT_DIR / f"{image_name}_bicubic_HR.png")

    downsampler = Downsampler(n_planes=3, factor=FACTOR, kernel_type="lanczos2", phase=0.5, preserve_size=True).to(device)
    for p in downsampler.parameters():
        p.requires_grad = False
    criterion = nn.MSELoss().to(device)

    net_input_saved = torch.rand((1, INPUT_DEPTH, hr_h, hr_w), device=device) * 0.1
    noise_buffer = torch.zeros_like(net_input_saved)

    results = {}

    for model_name in MODELS:
        net = build_model(model_name).to(device)
        params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

        log(f"---> [{model_name.upper():>9}] Params: {params:,} | Starting {STEPS} steps...")
        t0 = time.time()
        best_rgb, best_y = -1.0, -1.0
        losses, psnr_rgb_hist, psnr_y_hist = [], [], []

        for step in range(1, STEPS + 1):
            opt.zero_grad(set_to_none=True)
            inp = net_input_saved + (noise_buffer.normal_() * REG_NOISE_STD)

            out_HR = net(inp)
            out_LR = downsampler(out_HR)

            loss = criterion(out_LR, img_LR)
            loss.backward()
            opt.step()

            loss_val = loss.item()
            losses.append(loss_val)

            with torch.no_grad():
                psnr_rgb = calc_psnr(out_HR, img_HR)
                psnr_y = calc_psnr_y(out_HR, img_HR, shave=4)

            psnr_rgb_hist.append(psnr_rgb)
            psnr_y_hist.append(psnr_y)
            best_rgb = max(best_rgb, psnr_rgb)
            best_y = max(best_y, psnr_y)

            if run is not None:
                try:
                    run.log({f"loss/{model_name}": loss_val, f"psnr_rgb/{model_name}": psnr_rgb, f"psnr_y/{model_name}": psnr_y, "step": step})
                except Exception:
                    pass

            if step == 1 or step % 100 == 0 or step == STEPS:
                log(f"  [{model_name:>9}] step {step:04d}/{STEPS} | Loss: {loss_val:.6f} | RGB: {psnr_rgb:.2f} dB | Y: {psnr_y:.2f} dB | Elapsed: {time.time()-t0:.1f}s")

        # Save final output directly from last step
        with torch.no_grad():
            out_np = np.clip(torch_to_np(out_HR), 0.0, 1.0)
            np_to_pil(out_np).save(OUTPUT_DIR / f"{image_name}_HR_{model_name}.png")

        results[model_name] = {
            "params": params,
            "final_loss": loss_val,
            "final_rgb": psnr_rgb,
            "best_rgb": best_rgb,
            "final_y": psnr_y,
            "best_y": best_y,
            "losses": losses,
            "psnr_rgb": psnr_rgb_hist,
            "psnr_y": psnr_y_hist,
        }

        del net
        torch.cuda.empty_cache()

    # Save metrics CSV
    csv_path = OUTPUT_DIR / f"{image_name}_sr_metrics.csv"
    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        header = ["iteration"]
        for m in results:
            header.extend([f"{m}_loss", f"{m}_psnr_rgb", f"{m}_psnr_y"])
        writer.writerow(header)
        for s in range(STEPS):
            row = [s + 1]
            for m, res in results.items():
                row.extend([res["losses"][s], res["psnr_rgb"][s], res["psnr_y"][s]])
            writer.writerow(row)
    log(f"\nSaved metric log CSV to: {csv_path}")

    # Summary Table
    log("\n" + "=" * 110)
    log(" EXPERIMENT SUMMARY TABLE (4000 STEPS, FACTOR 4x)")
    log("=" * 110)
    log(f"{'Model':<12} | {'Params':<12} | {'Final Loss':<12} | {'Final RGB':<12} | {'Best RGB':<12} | {'Final Y':<12} | {'Best Y':<12} | {'Y Gain vs Bic':<15}")
    log("-" * 110)
    log(f"{'Bicubic':<12} | {'N/A':<12} | {'N/A':<12} | {psnr_bic_rgb:<12.2f} | {psnr_bic_rgb:<12.2f} | {psnr_bic_y:<12.2f} | {psnr_bic_y:<12.2f} | {'Baseline':<15}")

    for m_name, res in results.items():
        gain_y = res["final_y"] - psnr_bic_y
        log(
            f"{m_name:<12} | {res['params']:<12,} | "
            f"{res['final_loss']:<12.6f} | {res['final_rgb']:<12.2f} | {res['best_rgb']:<12.2f} | "
            f"{res['final_y']:<12.2f} | {res['best_y']:<12.2f} | {gain_y:+12.2f} dB"
        )
    log("=" * 110)

    if run is not None:
        try:
            run.finish()
        except Exception:
            pass

    log_file.close()
    print(f"\nDone! Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
