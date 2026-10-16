#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="$SCRIPT_DIR/.venv/bin/activate"

# Output directories
OUT1="$SCRIPT_DIR/output_swin_test"
OUT2="$SCRIPT_DIR/output_swin_test_skip"

mkdir -p "$OUT1" "$OUT2"

# Activate venv
if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "ERROR: venv not found at $VENV_ACTIVATE"
    exit 1
fi

source "$VENV_ACTIVATE"

echo "=================================================="
echo "Swin Image Prior Experiments"
echo "Host: $(hostname)"
echo "Time: $(date)"
echo "Python: $(which python)"
echo "=================================================="

# --------------------------------------------------
# Experiment 1
# --------------------------------------------------

echo ""
echo "=================================================="
echo "1/2: Running swin_test.py"
echo "Log: $OUT1/experiment.log"
echo "Started: $(date)"
echo "=================================================="

python -u "$SCRIPT_DIR/swin_test.py" 2>&1 | tee "$OUT1/experiment.log"

echo ""
echo ">>> swin_test.py finished: $(date)"


# --------------------------------------------------
# Experiment 2
# --------------------------------------------------

echo ""
echo "=================================================="
echo "2/2: Running swin_test_skip.py"
echo "Log: $OUT2/experiment.log"
echo "Started: $(date)"
echo "=================================================="

python -u "$SCRIPT_DIR/swin_test_skip.py" 2>&1 | tee "$OUT2/experiment.log"

echo ""
echo ">>> swin_test_skip.py finished: $(date)"


# --------------------------------------------------
# Done
# --------------------------------------------------

echo ""
echo "=================================================="
echo "ALL EXPERIMENTS FINISHED"
echo "Finished: $(date)"
echo ""
echo "Logs:"
echo "  $OUT1/experiment.log"
echo "  $OUT2/experiment.log"
echo "=================================================="