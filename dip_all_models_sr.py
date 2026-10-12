"""Super-Resolution across Deep Image Prior (DIP) Architectures.

Performs 4x Super-Resolution (6000 steps per model) comparing architectures from:
  src/playingwithdeepimageprior/deep-image-prior/models
including:
  - skip (DIP Hourglass / Skip encoder-decoder)
  - unet (Standard U-Net)
  - resnet (Residual CNN prior)
  - swin (Swin Transformer prior)
  - swin_skip (Recursive Swin-skip encoder-decoder prior)

Saves to the output folder:
  - Original Ground Truth HR image
  - Observed Low-Resolution (LR) image
  - Bicubic HR baseline
  - Super-resolved HR reconstruction for each evaluated model
  - Side-by-side visual comparison figure
  - Combined Loss and PSNR convergence curves
  - Comprehensive log file (text) and CSV recording losses and PSNRs at each iteration
"""

import argparse
import csv
import os
from pathlib import Path
import sys
import time

# Handle PIL deprecation of Image.ANTIALIAS for PIL >= 10
from PIL import Image
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = getattr(Image, "Resampling", Image).LANCZOS

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Ensure local deep-image-prior directory is importable
PROJECT_DIR = Path(__file__).resolve().parent / "src/playingwithdeepimageprior/deep-image-prior"
sys.path.insert(0, str(PROJECT_DIR))

from models.downsampler import Downsampler
from models.skip import skip
from models.unet import UNet
from models.resnet import ResNet
from models.swin import swin
from models.swin_skip import swin_skip
from utils.common_utils import np_to_pil, np_to_torch, pil_to_np, torch_to_np
from utils.sr_utils import get_baselines, load_LR_HR_imgs_sr

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def calculate_psnr(img1, img2, max_val=1.0):
    """Compute Peak Signal-to-Noise Ratio (dB) between two image tensors."""
    mse = torch.mean((img1.detach() - img2.detach()) ** 2)
    if mse == 0:
        return 100.0
    return (10.0 * torch.log10((max_val ** 2) / mse)).item()


def parameter_count(model):
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(model_name, input_depth, lr=0.0005, reg_noise=0.03, pad="reflection", upsample_mode="bilinear"):
    """Instantiate the requested DIP architecture from models directory."""
    name = model_name.lower()
    if name == "skip":
        model = skip(
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
    elif name == "unet":
        model = UNet(
            num_input_channels=input_depth,
            num_output_channels=3,
            feature_scale=4,
            more_layers=0,
            concat_x=False,
            upsample_mode=upsample_mode,
            pad=pad,
            norm_layer=nn.BatchNorm2d,
            need_sigmoid=True,
            need_bias=True,
        )
    elif name == "resnet":
        model = ResNet(
            num_input_channels=input_depth,
            num_output_channels=3,
            num_blocks=10,
            num_channels=64,
            need_residual=True,
            act_fun="LeakyReLU",
            need_sigmoid=True,
            norm_layer=nn.BatchNorm2d,
            pad=pad,
        )
    elif name == "swin":
        model = swin(
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
        model = swin_skip(
            input_depth=input_depth,
            output_channels=3,
            patch_size=4,
            embed_dim=32,
            depth=2,
            num_heads=4,
            window_size=8,
            need_sigmoid=True,
        )
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose from: ['skip', 'unet', 'resnet', 'swin', 'swin_skip']"
        )

    # Learning rate defaults to 0.0005 and jitter noise to 0.03 for all architectures
    return model, lr, reg_noise


def log_and_print(msg, log_file):
    """Print message to stdout and write to log file."""
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()


def train_single_model(
    model_name,
    net,
    lr,
    reg_noise_std,
    steps,
    net_input_saved,
    downsampler,
    img_HR_tensor,
    img_LR_tensor,
    criterion,
    device,
    log_file,
    wandb_run=None,
):
    """Execute DIP Super-Resolution optimization for a single model."""
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    noise_buffer = torch.zeros_like(net_input_saved)

    losses = []
    psnr_hr_history = []
    psnr_lr_history = []

    best_psnr = -1.0
    best_step = 0
    start_time = time.time()
    num_params = parameter_count(net)
    input_h, input_w = net_input_saved.shape[-2], net_input_saved.shape[-1]
    input_pixels = input_h * input_w

    log_and_print(
        f"\n[{model_name.upper()}] Starting {steps} steps...",
        log_file,
    )
    log_and_print(f"  Number of parameters:        {num_params:,}", log_file)
    log_and_print(f"  Number of pixels in input:   {input_pixels:,} ({input_w} x {input_h})", log_file)
    log_and_print(f"  Input tensor shape:          {tuple(net_input_saved.shape)} ({net_input_saved.numel():,} elements)", log_file)
    log_and_print(f"  Optimization settings:       lr={lr}, reg_noise_std={reg_noise_std}", log_file)

    for iteration in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise_buffer.normal_() * reg_noise_std)
        else:
            net_input = net_input_saved

        out_HR = net(net_input)
        out_LR = downsampler(out_HR)

        loss = criterion(out_LR, img_LR_tensor)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        with torch.no_grad():
            psnr_hr = calculate_psnr(out_HR, img_HR_tensor)
            psnr_lr = calculate_psnr(out_LR, img_LR_tensor)

        psnr_hr_history.append(psnr_hr)
        psnr_lr_history.append(psnr_lr)

        if psnr_hr > best_psnr:
            best_psnr = psnr_hr
            best_step = iteration

        if wandb_run is not None:
            wandb_run.log({
                f"loss/{model_name}": loss_val,
                f"psnr_hr/{model_name}": psnr_hr,
                f"psnr_lr/{model_name}": psnr_lr,
                "iteration": iteration,
            })

        if iteration == 1 or iteration % 100 == 0 or iteration == steps:
            elapsed = time.time() - start_time
            log_and_print(
                f"  [{model_name:>9}] step {iteration:05d}/{steps} | "
                f"Loss: {loss_val:.6f} | "
                f"PSNR_LR: {psnr_lr:.2f} dB | "
                f"PSNR_HR: {psnr_hr:.2f} dB | "
                f"Best HR: {best_psnr:.2f} dB (step {best_step}) | "
                f"Elapsed: {elapsed:.1f}s",
                log_file,
            )

    # Final inference without jitter
    with torch.no_grad():
        final_out_HR = net(net_input_saved)
        final_out_HR_np = np.clip(torch_to_np(final_out_HR), 0.0, 1.0)
        final_psnr_hr = calculate_psnr(final_out_HR, img_HR_tensor)

    total_time = time.time() - start_time
    log_and_print(
        f"[{model_name.upper()}] Finished in {total_time:.1f}s | Final PSNR: {final_psnr_hr:.2f} dB | Best: {best_psnr:.2f} dB",
        log_file,
    )

    return {
        "final_out_np": final_out_HR_np,
        "losses": losses,
        "psnr_hr": psnr_hr_history,
        "psnr_lr": psnr_lr_history,
        "final_psnr": final_psnr_hr,
        "best_psnr": best_psnr,
        "best_step": best_step,
        "param_count": parameter_count(net),
        "input_pixels": input_pixels,
        "total_time": total_time,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Super-Resolution across DIP models")
    parser.add_argument(
        "--image",
        type=str,
        default=str(PROJECT_DIR / "data/sr/zebra_GT.png"),
        help="Path to ground truth HR image",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["skip", "unet", "resnet", "swin", "swin_skip"],
        help="List of model architectures to evaluate",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=6000,
        help="Number of optimization epochs/steps per model (default: 6000)",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=4,
        help="Super-resolution scaling factor (default: 4)",
    )
    parser.add_argument(
        "--input_depth",
        type=int,
        default=32,
        help="Input noise channel depth (default: 32)",
    )
    parser.add_argument(
        "--kernel_type",
        type=str,
        default="lanczos2",
        help="Downsampler kernel (default: lanczos2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "output_all_models_sr"),
        help="Destination directory for outputs and logs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
        help="Learning rate for all models (default: 0.0005)",
    )
    parser.add_argument(
        "--reg_noise_std",
        type=float,
        default=0.03,
        help="Input noise jitter standard deviation for all models (default: 0.03)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Target image not found at {image_path}")

    image_name = image_path.stem.replace("_GT", "")
    log_path = output_dir / f"{image_name}_sr_all_models.log"
    csv_path = output_dir / f"{image_name}_sr_metrics.csv"
    log_file = open(log_path, "w", buffering=1)

    log_and_print("=" * 70, log_file)
    log_and_print(" Deep Image Prior: Multi-Model Super-Resolution Comparison", log_file)
    log_and_print("=" * 70, log_file)
    log_and_print(f"Device:           {device}", log_file)
    log_and_print(f"Target Image:     {image_path}", log_file)
    log_and_print(f"Factor:           {args.factor}x", log_file)
    log_and_print(f"Epochs/Steps:     {args.steps} per model", log_file)
    log_and_print(f"Input Noise Depth:{args.input_depth}", log_file)
    log_and_print(f"Models to train:  {', '.join(args.models)}", log_file)
    log_and_print(f"Output directory: {output_dir}", log_file)
    log_and_print("=" * 70, log_file)

    # Optional WandB Initialization
    wandb_run = None
    if HAS_WANDB and not args.no_wandb:
        try:
            wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "dip-multi-model-sr"),
                name=f"all_models_sr_{image_name}_x{args.factor}",
                config=vars(args),
                reinit=True,
            )
            wandb_run.define_metric("iteration")
            wandb_run.define_metric("loss/*", step_metric="iteration")
            wandb_run.define_metric("psnr_hr/*", step_metric="iteration")
            wandb_run.define_metric("psnr_lr/*", step_metric="iteration")
        except Exception as e:
            log_and_print(f"[Warning] Failed to initialize wandb ({e}). Continuing without wandb.", log_file)
            wandb_run = None

    # Load HR and synthesize LR observation (cropped to divisible by 32)
    imgs = load_LR_HR_imgs_sr(str(image_path), imsize=-1, factor=args.factor, enforse_div32="CROP")
    imgs["bicubic_np"], imgs["sharp_np"], imgs["nearest_np"] = get_baselines(
        imgs["LR_pil"], imgs["HR_pil"]
    )

    img_HR_tensor = np_to_torch(imgs["HR_np"]).to(device)
    img_LR_tensor = np_to_torch(imgs["LR_np"]).to(device)
    img_bicubic_tensor = np_to_torch(imgs["bicubic_np"]).to(device)
    img_nearest_tensor = np_to_torch(imgs["nearest_np"]).to(device)

    psnr_bicubic = calculate_psnr(img_bicubic_tensor, img_HR_tensor)
    psnr_nearest = calculate_psnr(img_nearest_tensor, img_HR_tensor)

    hr_w, hr_h = imgs["HR_pil"].size
    lr_w, lr_h = imgs["LR_pil"].size

    log_and_print(f"Ground Truth HR Dimensions: {hr_w} x {hr_h}", log_file)
    log_and_print(f"Downsampled LR Dimensions:  {lr_w} x {lr_h}", log_file)
    log_and_print(f"Baseline Bicubic PSNR:       {psnr_bicubic:.2f} dB", log_file)
    log_and_print(f"Baseline Nearest PSNR:       {psnr_nearest:.2f} dB", log_file)
    log_and_print("-" * 70, log_file)

    # Save Ground Truth HR, Observed LR, and Bicubic baseline images
    real_hr_path = output_dir / f"{image_name}_original_HR.png"
    observed_lr_path = output_dir / f"{image_name}_observed_LR.png"
    bicubic_hr_path = output_dir / f"{image_name}_bicubic_HR.png"

    imgs["HR_pil"].save(real_hr_path)
    imgs["LR_pil"].save(observed_lr_path)
    np_to_pil(imgs["bicubic_np"]).save(bicubic_hr_path)

    log_and_print(f"Saved Original HR image to: {real_hr_path}", log_file)
    log_and_print(f"Saved Observed LR image to: {observed_lr_path}", log_file)
    log_and_print(f"Saved Bicubic HR baseline:  {bicubic_hr_path}", log_file)

    # Setup Downsampler & Loss Criterion
    downsampler = Downsampler(
        n_planes=3, factor=args.factor, kernel_type=args.kernel_type, phase=0.5, preserve_size=True
    ).to(device)
    criterion = nn.MSELoss().to(device)

    # Shared fixed uniform input noise [0, 0.1]
    input_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    net_input_saved = torch.rand(
        (1, args.input_depth, hr_h, hr_w), generator=input_generator, device="cpu"
    ).to(device) * (1.0 / 10.0)

    results = {}
    model_output_paths = {}

    for model_name in args.models:
        log_and_print("\n" + "=" * 70, log_file)
        log_and_print(f" Preparing Model: {model_name.upper()}", log_file)
        log_and_print("=" * 70, log_file)

        try:
            net, lr, reg_noise = build_model(model_name, args.input_depth, lr=args.lr, reg_noise=args.reg_noise_std)
            net = net.to(device)
            num_params = parameter_count(net)
            input_pixels = hr_w * hr_h
            log_and_print(f"  Architecture:           {model_name.upper()}", log_file)
            log_and_print(f"  Trainable parameters:   {num_params:,}", log_file)
            log_and_print(f"  Number of input pixels: {input_pixels:,} ({hr_w} x {hr_h})", log_file)
        except Exception as e:
            log_and_print(f"[Error] Failed to build model {model_name}: {e}", log_file)
            continue

        res = train_single_model(
            model_name=model_name,
            net=net,
            lr=lr,
            reg_noise_std=reg_noise,
            steps=args.steps,
            net_input_saved=net_input_saved,
            downsampler=downsampler,
            img_HR_tensor=img_HR_tensor,
            img_LR_tensor=img_LR_tensor,
            criterion=criterion,
            device=device,
            log_file=log_file,
            wandb_run=wandb_run,
        )
        results[model_name] = res

        # Save this model's HR output image
        model_hr_path = output_dir / f"{image_name}_HR_{model_name}.png"
        np_to_pil(res["final_out_np"]).save(model_hr_path)
        model_output_paths[model_name] = model_hr_path
        log_and_print(f"Saved {model_name} HR output to: {model_hr_path}", log_file)

        # Free GPU memory before next model
        del net
        torch.cuda.empty_cache()

    # Save detailed CSV metrics log
    log_and_print("\nWriting iteration metrics to CSV...", log_file)
    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        header = ["iteration"]
        for m in results.keys():
            header.extend([f"{m}_loss", f"{m}_psnr_hr", f"{m}_psnr_lr"])
        writer.writerow(header)

        for step_idx in range(args.steps):
            row = [step_idx + 1]
            for m, res in results.items():
                loss_v = res["losses"][step_idx] if step_idx < len(res["losses"]) else ""
                psnr_hr_v = res["psnr_hr"][step_idx] if step_idx < len(res["psnr_hr"]) else ""
                psnr_lr_v = res["psnr_lr"][step_idx] if step_idx < len(res["psnr_lr"]) else ""
                row.extend([loss_v, psnr_hr_v, psnr_lr_v])
            writer.writerow(row)
    log_and_print(f"Saved metric CSV to: {csv_path}", log_file)

    # Generate Combined Visual Comparison Figure
    num_models = len(results)
    total_cols = 3 + num_models  # HR, LR, Bicubic + models
    fig_w = max(18, total_cols * 3.5)
    fig_comp, axes = plt.subplots(1, total_cols, figsize=(fig_w, 4.5))

    axes[0].imshow(imgs["HR_np"].transpose(1, 2, 0))
    axes[0].set_title(f"Original HR\n({hr_w}x{hr_h})", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(imgs["LR_np"].transpose(1, 2, 0))
    axes[1].set_title(f"Observed LR\n({lr_w}x{lr_h})", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(imgs["bicubic_np"].transpose(1, 2, 0))
    axes[2].set_title(f"Bicubic Baseline\nPSNR: {psnr_bicubic:.2f} dB", fontsize=10)
    axes[2].axis("off")

    for idx, (m_name, res) in enumerate(results.items()):
        ax_target = axes[3 + idx]
        ax_target.imshow(res["final_out_np"].transpose(1, 2, 0))
        gain = res["final_psnr"] - psnr_bicubic
        gain_sign = "+" if gain >= 0 else ""
        ax_target.set_title(
            f"{m_name.upper()}\nPSNR: {res['final_psnr']:.2f} dB ({gain_sign}{gain:.2f})",
            fontsize=10,
        )
        ax_target.axis("off")

    fig_comp.suptitle(f"Super-Resolution (x{args.factor}, {args.steps} epochs) - Architecture Comparison", fontsize=13)
    fig_comp.tight_layout()
    comparison_fig_path = output_dir / f"{image_name}_all_models_visual_comparison.png"
    fig_comp.savefig(comparison_fig_path, dpi=180)
    plt.close(fig_comp)
    log_and_print(f"Saved visual comparison montage: {comparison_fig_path}", log_file)

    # Generate Training Curves (Loss & PSNR vs iteration)
    fig_curves, (ax_l, ax_p) = plt.subplots(1, 2, figsize=(15, 5.5))
    iters = np.arange(1, args.steps + 1)

    for m_name, res in results.items():
        ax_l.plot(iters, res["losses"], label=f"{m_name} (final: {res['losses'][-1]:.4e})")
        ax_p.plot(iters, res["psnr_hr"], label=f"{m_name} ({res['final_psnr']:.2f} dB)")

    ax_l.set_xscale("log")
    ax_l.set_yscale("log")
    ax_l.set_xlabel("Iteration (log scale)")
    ax_l.set_ylabel("Reconstruction MSE Loss")
    ax_l.set_title("Training Loss Convergence")
    ax_l.grid(True, which="both", alpha=0.3)
    ax_l.legend()

    ax_p.axhline(psnr_bicubic, color="red", linestyle="--", label=f"Bicubic Baseline ({psnr_bicubic:.2f} dB)")
    ax_p.set_xscale("log")
    ax_p.set_xlabel("Iteration (log scale)")
    ax_p.set_ylabel("PSNR vs Original HR (dB)")
    ax_p.set_title("Super-Resolution PSNR Progress")
    ax_p.grid(True, which="both", alpha=0.3)
    ax_p.legend()

    fig_curves.suptitle(f"Super-Resolution Optimization Curves ({image_name}, x{args.factor})", fontsize=13)
    fig_curves.tight_layout()
    curves_fig_path = output_dir / f"{image_name}_all_models_training_curves.png"
    fig_curves.savefig(curves_fig_path, dpi=180)
    plt.close(fig_curves)
    log_and_print(f"Saved training curves plot:    {curves_fig_path}", log_file)

    # Summary Table
    log_and_print("\n" + "=" * 85, log_file)
    log_and_print(" EXPERIMENT SUMMARY TABLE", log_file)
    log_and_print("=" * 85, log_file)
    header_str = f"{'Model':<12} | {'Params':<12} | {'Input Pixels':<14} | {'Final Loss':<12} | {'Final PSNR':<12} | {'Best PSNR':<12} | {'Gain vs Bicubic':<15}"
    log_and_print(header_str, log_file)
    log_and_print("-" * 102, log_file)
    log_and_print(
        f"{'Bicubic':<12} | {'N/A':<12} | {f'{hr_w*hr_h:,}':<14} | {'N/A':<12} | {psnr_bicubic:<12.2f} | {psnr_bicubic:<12.2f} | {'Baseline':<15}",
        log_file,
    )

    for m_name, res in results.items():
        gain = res["final_psnr"] - psnr_bicubic
        gain_str = f"+{gain:.2f} dB" if gain >= 0 else f"{gain:.2f} dB"
        in_pix_str = f"{res.get('input_pixels', hr_w * hr_h):,}"
        log_and_print(
            f"{m_name:<12} | {res['param_count']:<12,} | {in_pix_str:<14} | "
            f"{res['losses'][-1]:<12.6f} | {res['final_psnr']:<12.2f} | {res['best_psnr']:<12.2f} | {gain_str:<15}",
            log_file,
        )
    log_and_print("=" * 102, log_file)

    if wandb_run is not None:
        wandb_run.log({
            "visual_comparison": wandb.Image(str(comparison_fig_path)),
            "training_curves": wandb.Image(str(curves_fig_path)),
        })
        for m_name, p in model_output_paths.items():
            wandb_run.log({f"reconstruction/{m_name}": wandb.Image(str(p))})
        wandb_run.finish()

    log_file.close()
    print(f"\nAll operations complete! Full results and log stored in: {output_dir}")


if __name__ == "__main__":
    main()
