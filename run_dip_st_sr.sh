#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/output_dip_structure_tensor"

if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

echo "=========================================================="
echo " Starting Experiment 6: Structure Tensor DIP (ST-DIP)"
echo " Time: $(date)"
echo " Directory: $SCRIPT_DIR"
echo "=========================================================="

python -u "$SCRIPT_DIR/dip_sr_structure_tensor.py" 2>&1 | tee "$SCRIPT_DIR/output_dip_structure_tensor/experiment.log"
