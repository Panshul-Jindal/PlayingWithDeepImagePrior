"""Compare the recursive Swin-skip image prior on structured and unstructured targets."""

import os
from pathlib import Path
import sys
import wandb
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

PROJECT_DIR = Path(__file__).resolve().parent / "src/playingwithdeepimageprior/deep-image-prior"
sys.path.insert(0, str(PROJECT_DIR))

from models.swin_skip import swin_skip


STEPS = 8000
INPUT_DEPTH = 32
PATCH_SIZE = 4
EMBED_DIM = 32
SWIN_DEPTH = 2
NUM_HEADS = 4
WINDOW_SIZE = 8
LEARNING_RATE = 0.0001
NOISE_SIGMA = 25.0 / 255.0


def load_image(path):
    image = Image.open(path).convert("RGB")
    width = image.width - image.width % 32
    height = image.height - image.height % 32
    left = (image.width - width) // 2
    top = (image.height - height) // 2
    image = image.crop((left, top, left + width, top + height))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def make_targets(image, generator):
    noise = torch.randn(image.shape, generator=generator) * NOISE_SIGMA
    noisy_image = (image + noise).clamp(0.0, 1.0)
    permutation = torch.randperm(image.numel(), generator=generator)
    permuted_image = image.flatten()[permutation].reshape_as(image)
    white_noise = torch.rand(image.shape, generator=generator)
    return {
        "Image": image,
        "Image + noise": noisy_image,
        "Image shuffled": permuted_image,
        "U(0, 1) noise": white_noise,
    }


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def calculate_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1.detach() - img2.detach()) ** 2)
    if mse == 0:
        return 100.0
    return (10.0 * torch.log10((max_val ** 2) / mse)).item()


def train(target_name, target, orig_image, net_input, device, run):
    model = swin_skip(
        INPUT_DEPTH,
        output_channels=target.shape[1],
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        depth=SWIN_DEPTH,
        num_heads=NUM_HEADS,
        window_size=WINDOW_SIZE,
        need_sigmoid=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    losses = []
    psnrs = []

    print(f"{target_name}: {parameter_count(model):,} trainable parameters")
    for iteration in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        output = model(net_input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        losses.append(loss_val)

        with torch.no_grad():
            psnr_val = calculate_psnr(output, orig_image)
        psnrs.append(psnr_val)

        run.log({
            f"loss/{target_name}": loss_val,
            f"psnr/{target_name}": psnr_val,
            "iteration": iteration,
        })

        if iteration == 1 or iteration % 100 == 0:
            print(f"  iteration {iteration:05d}/{STEPS}: loss={loss_val:.6f}, psnr={psnr_val:.2f} dB")

    return losses, psnrs, model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = PROJECT_DIR / "data/denoising/F16_GT.png"
    image_name = image_path.stem.replace("_GT", "")
    architecture = "swin_skip"
    run_name = f"{architecture}-{image_name}-seed-{SEED}"

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "swin-image-prior"),
        name=run_name,
        config={
            "architecture": "recursive-swin-skip",
            "image": image_name,
            "seed": SEED,
            "steps": STEPS,
            "input_depth": INPUT_DEPTH,
            "patch_size": PATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "swin_depth": SWIN_DEPTH,
            "num_heads": NUM_HEADS,
            "window_size": WINDOW_SIZE,
            "learning_rate": LEARNING_RATE,
            "noise_sigma": NOISE_SIGMA,
        },
    )

    # Pin all target metrics to the explicit iteration axis so curves overlay
    wandb.define_metric("iteration")
    wandb.define_metric("loss/*", step_metric="iteration")
    wandb.define_metric("psnr/*", step_metric="iteration")

    image = load_image(image_path).to(device)

    input_generator = torch.Generator(device="cpu").manual_seed(SEED)
    target_generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    net_input = torch.rand(
        (1, INPUT_DEPTH, image.shape[-2], image.shape[-1]),
        generator=input_generator,
    ).to(device)
    targets = make_targets(image.cpu(), target_generator)
    targets = {name: target.to(device) for name, target in targets.items()}

    print(f"Device: {device}")
    print(f"Image: {image_path}")
    print(f"Tensor shape: {tuple(image.shape)}")
    print(f"Shared input shape: {tuple(net_input.shape)}")

    loss_results = {}
    psnr_results = {}
    for target_name, target in targets.items():
        losses, psnrs, _ = train(target_name, target, image, net_input, device, run)
        loss_results[target_name] = losses
        psnr_results[target_name] = psnrs

    iterations = np.arange(1, STEPS + 1)

    output_dir = Path(__file__).resolve().parent / "output_swin_test_skip"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loss comparison plot
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    for target_name, losses in loss_results.items():
        ax_loss.plot(iterations, losses, label=target_name)
    ax_loss.set_xscale("log")
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Iteration (log scale)")
    ax_loss.set_ylabel("MSE loss")
    ax_loss.set_title("Recursive Swin-skip image-prior fitting (MSE Loss)")
    ax_loss.grid(True, which="both", alpha=0.25)
    ax_loss.legend()
    fig_loss.tight_layout()

    loss_output_path = output_dir / "swin_skip_loss_comparison.png"
    fig_loss.savefig(loss_output_path, dpi=150)
    fig_loss.savefig(output_dir / "swin_skip_similar_to_dip.png", dpi=150)
    plt.close(fig_loss)
    print(f"Saved loss plot to {loss_output_path}")

    # 2. PSNR comparison plot
    fig_psnr, ax_psnr = plt.subplots(figsize=(10, 6))
    for target_name, psnrs in psnr_results.items():
        ax_psnr.plot(iterations, psnrs, label=target_name)
    ax_psnr.set_xscale("log")
    ax_psnr.set_xlabel("Iteration (log scale)")
    ax_psnr.set_ylabel("PSNR with original image (dB)")
    ax_psnr.set_title("Recursive Swin-skip image-prior fitting (PSNR with original image)")
    ax_psnr.grid(True, which="both", alpha=0.25)
    ax_psnr.legend()
    fig_psnr.tight_layout()

    psnr_output_path = output_dir / "swin_skip_psnr_comparison.png"
    fig_psnr.savefig(psnr_output_path, dpi=150)
    plt.close(fig_psnr)
    print(f"Saved PSNR plot to {psnr_output_path}")

    # 3. Combined comparison plot (Side-by-side like DIP paper Fig. 2)
    fig_comb, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for target_name, losses in loss_results.items():
        ax1.plot(iterations, losses, label=target_name)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Iteration (log scale)")
    ax1.set_ylabel("MSE loss")
    ax1.set_title("Reconstruction Error")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend()

    for target_name, psnrs in psnr_results.items():
        ax2.plot(iterations, psnrs, label=target_name)
    ax2.set_xscale("log")
    ax2.set_xlabel("Iteration (log scale)")
    ax2.set_ylabel("PSNR with original image (dB)")
    ax2.set_title("PSNR with Original Image")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend()

    fig_comb.suptitle("Recursive Swin-skip Image Prior Comparison", fontsize=14)
    fig_comb.tight_layout()
    comb_output_path = output_dir / "swin_skip_comparison.png"
    fig_comb.savefig(comb_output_path, dpi=150)
    plt.close(fig_comb)
    print(f"Saved combined comparison plot to {comb_output_path}")

    run.log({
        "loss_comparison": wandb.Image(str(loss_output_path)),
        "psnr_comparison": wandb.Image(str(psnr_output_path)),
        "comparison": wandb.Image(str(comb_output_path)),
    })
    run.finish()


if __name__ == "__main__":
    main()