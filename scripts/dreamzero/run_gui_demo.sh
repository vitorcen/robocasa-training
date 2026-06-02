#!/usr/bin/env bash
# Live GUI demo: drive the RoboCasa OpenCabinet sim with the DreamZero NF4 policy and
# render a passive MuJoCo viewer so you can WATCH the arm move.
#
# Architecture (two conda envs, talk over ZMQ on localhost):
#   - policy server : `dreamzero` env, NF4 Wan2.1-14B + robocasa LoRA  (~20 GB VRAM on a 4090)
#   - sim client    : `robocasa` env, robocasa OpenCabinet + MuJoCo viewer (--render)
#
# IMPORTANT: run this from a REAL interactive terminal on the machine with the display
# (DISPLAY=:0). The policy server's import chain (groot->albumentations->scipy.ndimage)
# trips a flaky scipy doc-scrape bug ONLY when launched from a headless/automated context;
# in an interactive shell it loads fine.
#
# Usage:
#   bash robocasa-training/scripts/dreamzero/run_gui_demo.sh \
#       [CKPT_DIR] [MAX_STEPS] [N_ACTION_STEPS]
#   defaults: CKPT_DIR=checkpoints/dreamzero_robocasa_smoke/checkpoint-50  MAX_STEPS=48  N_ACTION_STEPS=16
#
# Note: NF4 inference is ~85 s per chunk on a 4090, so the arm moves in bursts of
# N_ACTION_STEPS, then pauses ~85 s while the next chunk is computed. A 50-step smoke
# checkpoint is barely trained — expect plausible but un-skilled motion.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # mujoco-experience/
cd "$REPO_ROOT"

CKPT_DIR="${1:-$REPO_ROOT/robocasa-training/checkpoints/dreamzero_robocasa_smoke/checkpoint-50}"
MAX_STEPS="${2:-48}"
N_ACTION_STEPS="${3:-16}"
PORT="${PORT:-5700}"
ENV_NAME="${ENV_NAME:-OpenCabinet}"
SPLIT="${SPLIT:-target}"
SERVER_ENV="${SERVER_ENV:-dreamzero}"
CLIENT_ENV="${CLIENT_ENV:-robocasa}"
export DISPLAY="${DISPLAY:-:0}"
export SCIPY_ARRAY_API=0

SRV_LOG="/tmp/dz_gui_server.log"
echo "[gui] checkpoint : $CKPT_DIR"
echo "[gui] server env : $SERVER_ENV   client env: $CLIENT_ENV   port: $PORT   DISPLAY: $DISPLAY"

# 1) start the policy server
rm -f "$SRV_LOG"
# setsid -> server is its own process-group leader; `kill -- -$SRV_PID` reaps the whole
# tree (conda-run wrapper + python grandchild). Plain kill orphaned the python -> 16 GB leak.
setsid conda run -n "$SERVER_ENV" --no-capture-output python -u \
    robocasa-training/scripts/dreamzero/serve_dreamzero_robocasa.py \
    --ckpt-dir "$CKPT_DIR" --port "$PORT" > "$SRV_LOG" 2>&1 &
SRV_PID=$!
trap 'echo "[gui] stopping server group $SRV_PID"; kill -9 -- -$SRV_PID 2>/dev/null; pkill -9 -f "serve_dreamzero_robocasa.py --ckpt-dir $CKPT_DIR" 2>/dev/null' EXIT

echo "[gui] loading NF4 model (3-4 min)… tail: $SRV_LOG"
for i in $(seq 1 60); do
    if grep -qa "ready on tcp" "$SRV_LOG" 2>/dev/null; then echo "[gui] server ready."; break; fi
    if ! kill -0 $SRV_PID 2>/dev/null; then
        echo "[gui] ERROR: server died during load. Last lines:"; tail -20 "$SRV_LOG"; exit 1
    fi
    sleep 10
done

# 2) run ONE episode with the passive viewer (GUI)
echo "[gui] launching RoboCasa sim client with --render (a MuJoCo window will open)…"
conda run -n "$CLIENT_ENV" --no-capture-output python -u scripts/_gr00t_eval_client.py \
    --env-name "$ENV_NAME" --split "$SPLIT" --port "$PORT" \
    --n-episodes 1 --max-steps "$MAX_STEPS" --n-action-steps "$N_ACTION_STEPS" \
    --render --render-warmup-s 4 --seed "${SEED:-0}"

echo "[gui] done."
