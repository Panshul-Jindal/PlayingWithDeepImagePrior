#!/usr/bin/env bash
#
# run_all_sr_experiments.sh - Sequentially run all Super-Resolution research experiments.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================================="
echo " Master Research Runner: DIP Super-Resolution Benchmarking"
echo " Time: $(date)"
echo " Directory: $SCRIPT_DIR"
echo "=========================================================="

# 1. Check if Experiment 1 (DeepRED) is still running or completed
if [[ -f "$SCRIPT_DIR/output_dip_red/zebra_hr_with_new_approach.png" ]]; then
    echo ">>> Experiment 1 (DeepRED) already completed."
else
    echo ">>> Running Experiment 1: DeepRED (arXiv:1903.10176)..."
    bash "$SCRIPT_DIR/run_dip_red.sh"
fi

# 2. Run Experiment 2: Single-Image Internal Learning (ZSSR UNet)
echo ""
echo "=========================================================="
echo ">>> Running Experiment 2: Single-Image Internal Learning (ZSSR UNet)..."
echo "=========================================================="
bash "$SCRIPT_DIR/run_internal_sr.sh"

# 3. Run Experiment 3: Single-Image Conditional Diffusion Model
echo ""
echo "=========================================================="
echo ">>> Running Experiment 3: Single-Image Conditional Diffusion SR..."
echo "=========================================================="
bash "$SCRIPT_DIR/run_diffusion_sr.sh"

echo ""
echo "=========================================================="
echo " All Research Experiments Completed Successfully!"
echo " Outputs available in:"
echo "   - $SCRIPT_DIR/output_dip_sr/"
echo "   - $SCRIPT_DIR/output_dip_red/"
echo "   - $SCRIPT_DIR/output_single_image_internal_sr/"
echo "   - $SCRIPT_DIR/output_single_image_diffusion_sr/"
echo "=========================================================="
