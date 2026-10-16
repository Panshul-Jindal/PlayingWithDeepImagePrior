#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-dip_sr}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/output_dip_sr"

if [[ "${1:-}" == "--run-tasks" || -n "${STY:-}" ]]; then
    if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
        source "$SCRIPT_DIR/.venv/bin/activate"
    fi
    echo "Starting DIP Super-Resolution reproduction..."
    python -u "$SCRIPT_DIR/dip_sr_test.py" 2>&1 | tee "$SCRIPT_DIR/output_dip_sr/experiment.log"
    exit 0
fi

if screen -list 2>/dev/null | grep -q "\.${SESSION_NAME}[[:space:]]"; then
    echo "Session '$SESSION_NAME' is already running. Attach with: screen -r $SESSION_NAME"
    exit 1
fi

screen -dmS "$SESSION_NAME" bash -c "\"$SCRIPT_DIR/run_dip_sr.sh\" --run-tasks; exec bash"
echo "=========================================================="
echo " DIP Super-Resolution running in screen session: '$SESSION_NAME'"
echo " Attach to view live progress: screen -r $SESSION_NAME"
echo " Detach without stopping:      Ctrl+A then D"
echo " Live log file:                $SCRIPT_DIR/output_dip_sr/experiment.log"
echo "=========================================================="
