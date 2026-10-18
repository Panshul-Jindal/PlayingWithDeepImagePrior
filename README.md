# Playing with Deep Image Prior

> *We bent the losses, warped the space,*  
> *Tested transformers in the race,*  
> *Pushed priors through the gradient maze—*  
> *To see what truly makes them blaze.*  
>  
> *And in that quest, one truth stood tall:*  
> *You don't need million weights at all.*  
> *A slim **ConvNeXt** stepped in the light,*  
> *Stripped down in size, yet twice as bright—*  
> *Matching the classic DIP stride for stride,*  
> *With tiny parameters on its side!*  

---

## The Full Report
For the complete technical paper, mathematical formulations, comprehensive benchmark tables across Set14, and deep takeaways, read:  
👉 **[REPORT.md](REPORT.md)**

---

## Project Structure

```
.
├── src/playingwithdeepimageprior/deep-image-prior/
│   ├── models/           # Skip, UNet, ResNet, Swin, DPwise, ConvNeXt
│   ├── data/sr/Set14/    # Benchmark HR/LR evaluation images
│   └── utils/            # Downsampling operators, PSNR & metrics
├── dip_*.py              # Core Python experiment scripts
├── run_*.sh              # One-click execution bash scripts
├── output_*/             # Generated HR outputs, visual comparisons & CSV logs
├── pyproject.toml        # Environment specification (managed via uv)
└── RESEARCH_EXPERIMENTS.md # Master research findings & experiment log
```

---

## Reproducibility & Quickstart

This project uses **[uv](https://github.com/astral-sh/uv)** for fast, reproducible dependency management.

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/Panshul-Jindal/PlayingWithDeepImagePrior.git
cd PlayingWithDeepImagePrior

# Install dependencies using uv
uv sync
```

### 2. Run Experiments

You can run individual experiments directly or using their bash wrappers:

- **ConvNeXt Super-Resolution (The Star Discovery)**:
  ```bash
  uv run python dip_convnext_zebra.py
  # Or run across Set14:
  ./run_convnext_multi_image.sh
  ```

- **Vanilla DIP Baseline**:
  ```bash
  ./run_dip_sr.sh
  ```

- **Architecture & Loss Comparisons**:
  ```bash
  ./run_all_models_sr.sh      # Benchmark Skip, UNet, ResNet, Swin
  ./run_dip_losses.sh          # Benchmark L2, L1, TV, DeepRED
  ./run_dip_st_sr.sh           # Structure Tensor (ST-DIP)
  ```

Outputs (restored high-resolution images, comparison montages, and per-step metrics) are automatically saved into their respective `output_*/` folders.
