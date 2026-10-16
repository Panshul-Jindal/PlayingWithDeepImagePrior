"""Compare the recursive Swin-skip image prior on structured and unstructured targets."""

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


def train(target_name, target, net_input, device, run):
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

    print(f"{target_name}: {parameter_count(model):,} trainable parameters")
    for iteration in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        output = model(net_input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        losses.append(loss_val)

        # Log periodically to eliminate network sync throttling
        if iteration == 1 or iteration % 10 == 0 or iteration == STEPS:
            run.log({
                f"loss/{target_name}": loss_val,
                "iteration": iteration,
            })

        if iteration == 1 or iteration % 100 == 0:
            print(f"  iteration {iteration:05d}/{STEPS}: loss={loss_val:.6f}")

    return losses, model


def main():
    run = wandb.init(
        project="swin_skip_similar_to_dip",
        name=f"swin_skip_similar_to_dip-seed-{SEED}",
        config={
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
            "architecture": "recursive-swin-skip",
        },
    )

    # Pin all target losses to the explicit iteration axis so curves overlay
    wandb.define_metric("iteration")
    wandb.define_metric("loss/*", step_metric="iteration")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = PROJECT_DIR / "data/denoising/F16_GT.png"
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

    results = {}
    for target_name, target in targets.items():
        results[target_name], _ = train(target_name, target, net_input, device, run)

    # Generate log-log MSE comparison plot
    fig, axis = plt.subplots(figsize=(10, 6))
    iterations = np.arange(1, STEPS + 1)
    for target_name, losses in results.items():
        axis.plot(iterations, losses, label=target_name)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Iteration (log scale)")
    axis.set_ylabel("MSE loss")
    axis.set_title("Recursive Swin-skip image-prior fitting")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / "swin_skip_similar_to_dip.png"
    fig.savefig(output_path, dpi=150)
    print(f"Saved loss plot to {output_path}")

    run.log({
        "loss_comparison": wandb.Image(str(output_path))
    })
    run.finish()


if __name__ == "__main__":
    main()