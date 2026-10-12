#!/usr/bin/env bash
set -euo pipefail

# Default screen session name
SESSION_NAME="${1:-all_models_sr}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output_all_models_sr_1"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/screen_run.log"

# Internal execution handler when launched in screen or with --run-tasks
if [[ "${1:-}" == "--run-tasks" || -n "${STY:-}" ]]; then
    if [[ "${1:-}" == "--run-tasks" ]]; then
        shift
    fi

    if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$SCRIPT_DIR/.venv/bin/activate"
    fi

    echo "=========================================================="
    echo " Starting DIP Super-Resolution across models (6000 steps)"
    echo " Output directory: $OUTPUT_DIR"
    echo " Log file:         $LOG_FILE"
    echo "=========================================================="

    python -u "$SCRIPT_DIR/dip_all_models_sr.py" \
        --steps 6000 \
        --factor 4 \
        --lr 0.0005 \
        --reg_noise_std 0.03 \
        --models skip unet resnet swin swin_skip \
        --output_dir "$OUTPUT_DIR" \
        "$@" 2>&1 | tee -a "$LOG_FILE"

    echo "=========================================================="
    echo " Training completed for all models!"
    echo " Outputs and log available in: $OUTPUT_DIR"
    echo "=========================================================="
    exit 0
fi

# Check if screen session already exists
if command -v screen >/dev/null 2>&1; then
    if screen -list 2>/dev/null | grep -q "\.${SESSION_NAME}[[:space:]]"; then
        echo "Error: Screen session '$SESSION_NAME' is already running."
        echo "Attach with: screen -r $SESSION_NAME"
        exit 1
    fi

    echo "Launching DIP multi-model super-resolution in detached screen: '$SESSION_NAME'..."
    screen -dmS "$SESSION_NAME" bash -c "\"$SCRIPT_DIR/run_all_models_sr.sh\" --run-tasks \"$@\"; exec bash"

    echo "=========================================================="
    echo " Session '$SESSION_NAME' started successfully in background!"
    echo ""
    echo " Useful screen commands:"
    echo "   - View live output:        screen -r $SESSION_NAME"
    echo "   - Detach without stopping: Press Ctrl+A then D"
    echo "   - Monitor log directly:    tail -f $LOG_FILE"
    echo "   - Check session status:    screen -list"
    echo "=========================================================="
else
    echo "Screen is not installed or not in PATH. Running directly..."
    "$SCRIPT_DIR/run_all_models_sr.sh" --run-tasks "$@"
fi
