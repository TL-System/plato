#!/bin/bash
# Usage: bash evaluate_model.sh <model_path> [optional: max_per_task]
# Comment: This script evaluates a HuggingFace-based language model using the NanoChat benchmark infrastructure.
# model_path: Path to the HuggingFace model to evaluate.
# max_per_task: (Optional) Maximum number of examples to evaluate per task.

export NANOCHAT_BASE_DIR="$PWD/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $NANOCHAT_BASE_DIR
fi

if [ -z "$2" ]; then
  MAX_PER_TASK=16
else
  MAX_PER_TASK=$2
fi
uv run evaluate_model.py --hf_path=$1 --max-per-task=$MAX_PER_TASK

