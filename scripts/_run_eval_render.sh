#!/usr/bin/env bash
# Helper: launch a rendered N1.7 eval on a checkpoint, detached. Args: <ckpt-dir> [n_ep] [max_steps]
set -u
CKPT="${1:?usage: _run_eval_render.sh <ckpt-dir> [n_ep] [max_steps]}"
NEP="${2:-3}"
MAXS="${3:-300}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$REPO/logs/eval_render_$(basename "$CKPT").log"
JSON="$REPO/logs/eval_render_$(basename "$CKPT").json"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gr00t-n17 || true
export DISPLAY="${DISPLAY:-:0}" MUJOCO_GL="${MUJOCO_GL:-glfw}"

cd "$REPO"
nohup python scripts/eval_gr00t_n17.py \
  --env-name OpenCabinet --split target \
  --n-episodes "$NEP" --n-action-steps 16 --max-steps "$MAXS" \
  --ckpt "$CKPT" --render --render-warmup-s 8 \
  --results-path "$JSON" > "$LOG" 2>&1 &
echo "launched PID $! -> $LOG"
