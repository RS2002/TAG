#!/bin/bash
# run.sh
# Entry script for training or evaluating the TAG model.

set -e

# --- Configuration ---
# Uncomment and set the following line if using a virtual environment
# source /path/to/venv/bin/activate

# Choose mode: train or eval
MODE=${1:-train}

# Optional checkpoint path for evaluation
CHECKPOINT=${2:-}

# --- Execution ---
case "$MODE" in
    train)
        echo "Starting training..."
        python train.py
        ;;
    eval)
        if [ -z "$CHECKPOINT" ]; then
            echo "Usage: ./run.sh eval <checkpoint_path>"
            exit 1
        fi
        echo "Starting evaluation with checkpoint: $CHECKPOINT"
        python eval.py --checkpoint "$CHECKPOINT"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: ./run.sh [train|eval] [checkpoint_path]"
        exit 1
        ;;
esac