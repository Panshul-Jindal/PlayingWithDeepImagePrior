#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/output_dip_red"

if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

echo "=========================================================="
echo " Starting Experiment 1: DeepRED Super-Resolution"
echo " Time: $(date)"
echo " Directory: $SCRIPT_DIR"
echo "=========================================================="

python -u "$SCRIPT_DIR/dip_red_sr.py" 2>&1 | tee "$SCRIPT_DIR/output_dip_red/experiment.log"
