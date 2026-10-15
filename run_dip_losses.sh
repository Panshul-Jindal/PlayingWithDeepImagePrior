#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-dip_losses_sr}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output_dip_losses_sr"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/screen_run.log"

if [[ "${1:-}" == "--run-tasks" || -n "${STY:-}" ]]; then
    if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$SCRIPT_DIR/.venv/bin/activate"
    fi

    echo "=========================================================="
    echo " Starting DIP Super-Resolution Loss & Regularizer Comparison"
    echo " Architecture: Skip | 4000 steps"
    echo " Losses:       l2, l1, tv, deepred_l2, deepred_l1"
    echo " Output dir:   $OUTPUT_DIR"
    echo " Log file:     $LOG_FILE"
    echo "=========================================================="

    python -u "$SCRIPT_DIR/dip_loss_regularizer_sr.py" 2>&1 | tee -a "$LOG_FILE"
    exit 0
fi

if command -v screen >/dev/null 2>&1; then
    if screen -list 2>/dev/null | grep -q "\.${SESSION_NAME}[[:space:]]"; then
        echo "Session '$SESSION_NAME' is already running. Attach with: screen -r $SESSION_NAME"
        exit 1
    fi

    echo "Launching DIP loss comparison in screen session: '$SESSION_NAME'..."
    screen -dmS "$SESSION_NAME" bash -c "\"$SCRIPT_DIR/run_dip_losses.sh\" --run-tasks; exec bash"

    echo "=========================================================="
    echo " Session '$SESSION_NAME' running in background!"
    echo "   - Attach:  screen -r $SESSION_NAME"
    echo "   - Detach:  Ctrl+A then D"
    echo "   - Live log: tail -f $LOG_FILE"
    echo "=========================================================="
else
    "$SCRIPT_DIR/run_dip_losses.sh" --run-tasks
fi
