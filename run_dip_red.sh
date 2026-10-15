#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-dip_red}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output_dip_red"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/screen_run.log"

if [[ "${1:-}" == "--run-tasks" || -n "${STY:-}" ]]; then
    if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$SCRIPT_DIR/.venv/bin/activate"
    fi

    echo "=========================================================="
    echo " Starting DeepRED Super-Resolution Experiment"
    echo " Output dir: $OUTPUT_DIR"
    echo " Log file:   $LOG_FILE"
    echo "=========================================================="

    python -u "$SCRIPT_DIR/dip_red_sr.py" 2>&1 | tee -a "$LOG_FILE"
    exit 0
fi

if command -v screen >/dev/null 2>&1; then
    if screen -list 2>/dev/null | grep -q "\.${SESSION_NAME}[[:space:]]"; then
        echo "Error: Screen session '$SESSION_NAME' is already running."
        echo "Attach with: screen -r $SESSION_NAME"
        exit 1
    fi

    echo "Launching DeepRED experiment in screen session: '$SESSION_NAME'..."
    screen -dmS "$SESSION_NAME" bash -c "\"$SCRIPT_DIR/run_dip_red.sh\" --run-tasks; exec bash"

    echo "=========================================================="
    echo " Session '$SESSION_NAME' running in background!"
    echo "   - Attach:   screen -r $SESSION_NAME"
    echo "   - Detach:   Ctrl+A then D"
    echo "   - Live log: tail -f $LOG_FILE"
    echo "=========================================================="
else
    "$SCRIPT_DIR/run_dip_red.sh" --run-tasks
fi
