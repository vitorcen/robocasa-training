#!/usr/bin/env bash
# Pull a mid-training DreamZero checkpoint from the AutoDL box and run a HEADLESS
# RoboCasa OpenCabinet success-rate eval on the local 4090 (NF4 inference).
#
# Use this to track "is the LoRA improving?" as the 15k run produces checkpoint-2500,
# -5000, ... Each checkpoint is ~415 MB (LoRA + experiment_cfg), self-contained.
#
# Architecture mirrors run_gui_demo.sh but WITHOUT --render (headless, scriptable SR):
#   policy server : `dreamzero` env, NF4 Wan2.1-14B + robocasa LoRA  (~20 GB VRAM)
#   sim client    : `robocasa`  env, OpenCabinet sim, prints success_rate JSON
#
# Cost warning: NF4 inference is ~60-85 s per 16-step chunk on a 4090. One 1200-step
# episode ≈ 75 chunks ≈ 1.0-1.5 h. N_EPISODES×that. Keep N small for a trend signal.
#
# Usage:
#   bash robocasa-training/scripts/dreamzero/pull_and_eval_ckpt.sh <STEP> [N_EPISODES] [MAX_STEPS]
#   e.g.  bash .../pull_and_eval_ckpt.sh 2500 3 1200
#
# Env overrides:
#   AUTODL_SSH   : "ssh -p 32660 root@connect.westd.seetacloud.com" style endpoint args
#   AUTODL_PW    : ssh password (used via sshpass; never commit the value)
#   REMOTE_OUT   : remote output_dir (default /root/autodl-tmp/dreamzero_robocasa_opencabinet_lora)
#   PORT, SERVER_ENV, CLIENT_ENV, ENV_NAME, SPLIT

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

STEP="${1:?need checkpoint STEP, e.g. 2500}"
N_EPISODES="${2:-3}"
MAX_STEPS="${3:-1200}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"

PORT="${PORT:-5703}"
ENV_NAME="${ENV_NAME:-OpenCabinet}"
SPLIT="${SPLIT:-target}"
SERVER_ENV="${SERVER_ENV:-dreamzero}"
CLIENT_ENV="${CLIENT_ENV:-robocasa}"

SSH_PORT="${SSH_PORT:-32660}"
SSH_HOST="${SSH_HOST:-root@connect.westd.seetacloud.com}"
REMOTE_OUT="${REMOTE_OUT:-/root/autodl-tmp/dreamzero_robocasa_opencabinet_lora}"
LOCAL_OUT="$REPO_ROOT/robocasa-training/checkpoints/dreamzero_robocasa_opencabinet_lora"
CKPT_LOCAL="$LOCAL_OUT/checkpoint-$STEP"

# sshpass wrapper (password via env AUTODL_PW; falls back to key auth if unset)
ssh_cmd() { if [ -n "${AUTODL_PW:-}" ]; then sshpass -p "$AUTODL_PW" "$@"; else "$@"; fi; }

# 1) pull the checkpoint if not already local
if [ -d "$CKPT_LOCAL/experiment_cfg" ]; then
    echo "[pull-eval] checkpoint-$STEP already local, skipping scp"
else
    echo "[pull-eval] scp checkpoint-$STEP from AutoDL ..."
    mkdir -p "$LOCAL_OUT"
    ssh_cmd scp -P "$SSH_PORT" -o StrictHostKeyChecking=no -r \
        "$SSH_HOST:$REMOTE_OUT/checkpoint-$STEP" "$CKPT_LOCAL" || {
        echo "[pull-eval] ERROR: scp failed (does checkpoint-$STEP exist remotely yet?)"; exit 1; }
fi
du -sh "$CKPT_LOCAL" 2>/dev/null

# 2) start NF4 policy server
export SCIPY_ARRAY_API=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SRV_LOG="/tmp/dz_eval_server_$STEP.log"; rm -f "$SRV_LOG"
echo "[pull-eval] starting NF4 server on :$PORT (log $SRV_LOG)"
# setsid -> server is its own process-group leader, so `kill -- -$SRV_PID` reaps the
# whole tree (conda-run wrapper + the python grandchild). Plain `kill $SRV_PID` only
# hit the wrapper and orphaned the python, leaking ~16 GB VRAM each run.
setsid conda run -n "$SERVER_ENV" --no-capture-output python -u \
    robocasa-training/scripts/dreamzero/serve_dreamzero_robocasa.py \
    --ckpt-dir "$CKPT_LOCAL" --port "$PORT" > "$SRV_LOG" 2>&1 &
SRV_PID=$!
trap 'echo "[pull-eval] stopping server group $SRV_PID"; kill -9 -- -$SRV_PID 2>/dev/null; pkill -9 -f "serve_dreamzero_robocasa.py --ckpt-dir $CKPT_LOCAL" 2>/dev/null' EXIT

echo "[pull-eval] loading NF4 model (~3 min)…"
for i in $(seq 1 60); do
    if grep -qa "ready on tcp" "$SRV_LOG" 2>/dev/null; then echo "[pull-eval] server ready."; break; fi
    if ! kill -0 $SRV_PID 2>/dev/null; then
        echo "[pull-eval] ERROR: server died during load:"; tail -25 "$SRV_LOG"; exit 1; fi
    sleep 10
done

# 3) headless SR eval (POLICY_TIMEOUT_S high for slow NF4 first chunk)
echo "[pull-eval] running $N_EPISODES episode(s) x $MAX_STEPS steps headless …"
POLICY_TIMEOUT_S="${POLICY_TIMEOUT_S:-600}" SCIPY_ARRAY_API=0 \
conda run -n "$CLIENT_ENV" --no-capture-output python -u scripts/_gr00t_eval_client.py \
    --env-name "$ENV_NAME" --split "$SPLIT" --port "$PORT" \
    --n-episodes "$N_EPISODES" --max-steps "$MAX_STEPS" \
    --n-action-steps "$N_ACTION_STEPS" --seed 0

echo "[pull-eval] checkpoint-$STEP eval done."
