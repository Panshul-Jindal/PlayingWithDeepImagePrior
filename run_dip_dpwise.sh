#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-dip_dpwise_sr}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output_dip_dpwise_sr"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/screen_run.log"

if [[ "${1:-}" == "--run-tasks" || -n "${STY:-}" ]]; then
    if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$SCRIPT_DIR/.venv/bin/activate"
    fi

    echo "=========================================================="
    echo " Starting DIP Depthwise-Separable Skip SR Experiment"
    echo " Architecture: Depthwise-Separable Skip (dpwise_skip.py)"
    echo " Losses:       l2, tv, deepred_l1, deepred_l2"
    echo " Output dir:   $OUTPUT_DIR"
    echo " Log file:     $LOG_FILE"
    echo "=========================================================="

    python -u "$SCRIPT_DIR/dip_dpwise_sr.py" 2>&1 | tee -a "$LOG_FILE"
    exit 0
fi

if command -v screen >/dev/null 2>&1; then
    if screen -list 2>/dev/null | grep -q "\.${SESSION_NAME}[[:space:]]"; then
        echo "Error: Screen session '$SESSION_NAME' is already running."
        echo "Attach with: screen -r $SESSION_NAME"
        exit 1
    fi

    echo "Launching DIP Depthwise-Separable experiment in screen session: '$SESSION_NAME'..."
    screen -dmS "$SESSION_NAME" bash -c "\"$SCRIPT_DIR/run_dip_dpwise.sh\" --run-tasks; exec bash"

    echo "=========================================================="
    echo " Session '$SESSION_NAME' started in background!"
    echo "   - Attach:   screen -r $SESSION_NAME"
    echo "   - Detach:   Ctrl+A then D"
    echo "   - Live log: tail -f $LOG_FILE"
    echo "=========================================================="
else
    "$SCRIPT_DIR/run_dip_dpwise.sh" --run-tasks
fi

