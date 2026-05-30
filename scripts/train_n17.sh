#!/usr/bin/env bash
# GR00T-N1.7 fine-tune launcher — LeIsaac SO-101 PickOrange.
#
# Replicate hi-space/GR00T-N1.7-3B-Pick-Orange recipe (current 14/15 SOTA).
#  - VLM backbone : nvidia/Cosmos-Reason2-2B (4.6 GB, already cached)
#  - Action head  : Gr00tN1d7, action_horizon=40, 4 diffusion steps
#  - Trainable    : projector + DiT + linear heads + VL-LN  (~600M)
#  - Frozen       : Cosmos vision encoder + LLM backbone
#
# Single 4090 24GB, bf16; adafactor + grad-ckpt squeeze. If OOM, flip
# `backbone_trainable_params_fp32 = False` inside launch_finetune_ckpt_n17.py.
#
# Env knobs:
#   GR00T_ROOT          Isaac-GR00T repo (default: REPO_ROOT/dependencies/Isaac-GR00T)
#   DATASET_DIR         LeRobot v3.0 dataset (default: LeIsaac v2-gr00t leisaac-pick-orange)
#   OUTPUT_DIR          ckpt + logs (default: LeIsaac/outputs/gr00t-n17-leisaac-pick-orange)
#   BASE_MODEL          Path 1 (cold, default on AutoDL) = /root/autodl-tmp/cosmos_raw
#                       Path 1 (HF download)             = nvidia/Cosmos-Reason2-2B
#                       Path 2 (warm)                    = hi-space/GR00T-N1.7-3B-Pick-Orange
#   MAX_STEPS           default 10000 (hi-space converged at 6000; we go longer + auto-eval-on-save)
#   SAVE_STEPS          default 1000   (10k step / 10 ckpts; fits 140GB autodl-tmp)
#   SAVE_ONLY_MODEL     default 1      (skip optimizer state, single ckpt 25→12 GB)
#   LOSS_PRUNE_TOP_K    default 5      (keep best-5 ckpts by train_loss + last 1 = 6 max)
#   GPU_PROFILE         "auto" (default) | "small24" | "big48" | "big96"
#                       auto detect via nvidia-smi:
#                         <30 GB  → small24  (4090 24GB squeeze: grad-ckpt + adafactor + per-step=2)
#                         30-60GB → big48    (no grad-ckpt + adamw + per-step=4)
#                         >60 GB  → big96    (no grad-ckpt + adamw + per-step=8; e.g. RTX PRO 6000 96GB)
#   GLOBAL_BATCH        auto from GPU_PROFILE (override to force)
#   GRAD_ACCUM          auto from GPU_PROFILE
#   OPTIM               auto from GPU_PROFILE: adafactor (small24) vs adamw_torch (big48/big96)
#   GRADIENT_CKPT       auto from GPU_PROFILE: 1 (small24) vs 0 (big48/big96)
#   MAX_GRAD_NORM       default upstream 1.0; set 0 to disable HF/Accelerate clipping
#   STABLE_GRAD_CLIP_PARAMS_DISABLE
#                       default 0: cache trainable params once, so clipping does not
#                       re-walk model.named_modules() every optimizer step
#   USE_WANDB           default 0
#
# Don't `set -x` here — Adafactor prints fewer things than Adam, log stays readable.

set -euo pipefail

# robocasa-training repo root = scripts/.. ; this script lives in scripts/.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GR00T_ROOT="${GR00T_ROOT:-$REPO_ROOT/dependencies/Isaac-GR00T}"
CONDA_ENV="${CONDA_ENV:-gr00t-n17}"
# OpenCabinet target/human, LeRobot v2.1 (the dir that carries meta/modality.json).
_DEFAULT_DATASET="$HOME/.cache/robocasa/datasets/v1.0/target/atomic/OpenCabinet/20250813/lerobot_old"
DATASET_DIR="${DATASET_DIR:-$_DEFAULT_DATASET}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/checkpoints/gr00t_n17_opencabinet}"
MODALITY_CFG="${MODALITY_CFG:-$REPO_ROOT/scripts/robocasa_config_n17.py}"
# Warm start from the official N1.7 VLA (already an embodied policy) is best for a
# single-task finetune; both this and the cold Cosmos backbone are in the local HF cache.
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-1200}"
# save_only_model=True dropped 332/1030 weight keys interacting with tune_top_llm_layers — keep full ckpt.
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-0}"
LOSS_PRUNE_TOP_K="${LOSS_PRUNE_TOP_K:-2}"
# Disable loss-driven pruning by default: it keeps only top-K-loss + last ckpts and was
# DELETING the 1000-multiple eval sweep points that KEEP_MULTIPLE wants to preserve
# (the two prune callbacks conflicted → SR curve got holes at 3000-6000). With it off,
# the step-multiple CheckpointPruneCallback (KEEP_MULTIPLE) governs and every sweep point
# survives. Set LOSS_PRUNE_DISABLE=0 to re-enable.
export LOSS_PRUNE_DISABLE="${LOSS_PRUNE_DISABLE:-1}"

# Auto-detect GPU profile from VRAM (resolves GPU_PROFILE=auto)
GPU_PROFILE="${GPU_PROFILE:-auto}"
if [[ "$GPU_PROFILE" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        _VRAM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}')
        if   [[ $_VRAM_GB -gt 60 ]]; then GPU_PROFILE="big96"
        elif [[ $_VRAM_GB -gt 30 ]]; then GPU_PROFILE="big48"
        else                              GPU_PROFILE="small24"; fi
        echo "[gr00t-n17-train] auto-detected GPU_PROFILE=$GPU_PROFILE (VRAM ${_VRAM_GB} GB)"
    else
        GPU_PROFILE="small24"
        echo "[gr00t-n17-train] nvidia-smi unavailable, defaulting to GPU_PROFILE=small24" >&2
    fi
fi

case "$GPU_PROFILE" in
    small24)  # RTX 4090 24 GB: every memory squeeze trick.
        # micro-batch=1 (global 4 / accum 4): RoboCasa feeds 3 cameras (vs LeIsaac's 2),
        # and the desktop's gnome-remote-desktop holds ~0.4 GB, so micro-batch=2 OOMs at
        # ~22.4 GB. micro-batch=1 fits with headroom. Bump global_batch on a freer GPU.
        _G=${GLOBAL_BATCH:-4};  _A=${GRAD_ACCUM:-4};  _O=${OPTIM:-adafactor};      _C=${GRADIENT_CKPT:-1} ;;
    big48)    # A100 40GB / L40 48GB: moderate
        _G=${GLOBAL_BATCH:-16}; _A=${GRAD_ACCUM:-4};  _O=${OPTIM:-adamw_torch};   _C=${GRADIENT_CKPT:-0} ;;
    big96)    # RTX PRO 6000 96 GB / H100 80GB: efficient
        _G=${GLOBAL_BATCH:-32}; _A=${GRAD_ACCUM:-4};  _O=${OPTIM:-adamw_torch};   _C=${GRADIENT_CKPT:-0} ;;
    *)
        echo "[gr00t-n17-train] ERROR: unknown GPU_PROFILE=$GPU_PROFILE (small24|big48|big96|auto)" >&2
        exit 1 ;;
esac
GLOBAL_BATCH=$_G; GRAD_ACCUM=$_A; OPTIM=$_O; GRADIENT_CKPT=$_C
export OPTIM GRADIENT_CKPT

# num_workers=0 on purpose: with >0 workers, N1.7 on this 4090 crashes randomly every
# ~50-1000 steps with corrupted-iterator ValueErrors ("too many/not enough values to
# unpack") — forked dataloader workers corrupting torch/CUDA state. num_workers=0 (main-
# process loading) eliminates the fork and trains stably; the OpenCabinet dataset is small
# + shard-cached so the throughput cost is negligible (~1.5 it/s either way). Verified:
# num_workers=4 crashed at step 1057; num_workers=0 sailed past. See gr00t-4090 skill.
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
USE_WANDB="${USE_WANDB:-0}"
export LOSS_PRUNE_TOP_K

if [[ ! -d "$DATASET_DIR" ]]; then
    echo "[gr00t-n17-train] ERROR: dataset not found: $DATASET_DIR" >&2
    exit 1
fi
if [[ ! -f "$DATASET_DIR/meta/modality.json" ]]; then
    echo "[gr00t-n17-train] ERROR: dataset missing meta/modality.json" >&2
    exit 1
fi
if [[ ! -d "$GR00T_ROOT" ]]; then
    echo "[gr00t-n17-train] ERROR: Isaac-GR00T repo not found: $GR00T_ROOT" >&2
    exit 1
fi
if [[ ! -f "$MODALITY_CFG" ]]; then
    echo "[gr00t-n17-train] ERROR: modality config not found: $MODALITY_CFG" >&2
    exit 1
fi

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/gr00t_n17_train_$(date +%Y%m%d_%H%M%S).log"

WANDB_FLAG=()
if [[ "$USE_WANDB" == "1" ]]; then
    WANDB_FLAG+=(--use_wandb)
fi
SAVE_ONLY_MODEL_FLAG=()
if [[ "$SAVE_ONLY_MODEL" == "1" ]]; then
    SAVE_ONLY_MODEL_FLAG+=(--save_only_model)
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Same cuBLAS workaround as N1.6 wrapper (torch 2.7.1 bf16 non-contig matmul bug).
export DISABLE_ADDMM_CUDA_LT="${DISABLE_ADDMM_CUDA_LT:-1}"
# torch.compile(action_head) is an optional +5-10% throughput patch in the launcher,
# but on this box (torch 2.7.1 inductor) it dies AOT-compiling the head with
# "'aten' object has no attribute 'map'". It's pure speed, not correctness — default
# OFF here. Set COMPILE_ACTION_HEAD_DISABLE=0 to re-enable if a future torch fixes it.
export COMPILE_ACTION_HEAD_DISABLE="${COMPILE_ACTION_HEAD_DISABLE:-1}"
# Disable the launcher's non_blocking=True H2D pipeline-overlap patch. It's a throughput
# optimization, but non_blocking copies without strict sync are a classic intermittent
# CUDA-corruption source — here they caused random `clip_grad_norm_ → named_parameters`
# ValueErrors ("too many/not enough values to unpack") at random steps. With it OFF (plus
# num_workers=0) training runs stably. Verified: ON → crash at 1331; OFF → past 1750 clean.
export PIPELINE_OVERLAP_DISABLE="${PIPELINE_OVERLAP_DISABLE:-1}"

echo "[gr00t-n17-train] launching:"
echo "  gr00t_root=$GR00T_ROOT"
echo "  dataset=$DATASET_DIR"
echo "  output=$OUTPUT_DIR"
echo "  base=$BASE_MODEL  steps=$MAX_STEPS  save_steps=$SAVE_STEPS"
echo "  save_only_model=$SAVE_ONLY_MODEL  loss_prune_top_k=$LOSS_PRUNE_TOP_K  save_total_limit=$SAVE_TOTAL_LIMIT"
echo "  gpu_profile=$GPU_PROFILE  optim=$OPTIM  grad_ckpt=$GRADIENT_CKPT"
echo "  max_grad_norm=${MAX_GRAD_NORM:-1.0}  stable_grad_clip_params=$([[ ${STABLE_GRAD_CLIP_PARAMS_DISABLE:-0} == 1 ]] && echo off || echo on)"
echo "  global_batch=$GLOBAL_BATCH  grad_accum=$GRAD_ACCUM  (per-step ≈ $((GLOBAL_BATCH / GRAD_ACCUM)))"
echo "  modality_cfg=$MODALITY_CFG"
echo "  log=$LOG_FILE"

cd "$GR00T_ROOT"

WRAPPER="$REPO_ROOT/scripts/launch_finetune_n17.py"

# Prefer the clean uv .venv built from Isaac-GR00T's uv.lock (the exact validated dep set
# that trains N1.7 successfully). The conda env cloned from another env had a mismatched
# CUDA/lib stack (cudnn etc.) that caused random process-memory corruption (9 distinct
# scramble errors) — the uv venv fixes the root. Falls back to conda if .venv is absent.
PYBIN="$GR00T_ROOT/.venv/bin/python"
if [[ -x "$PYBIN" ]]; then
    echo "  python=$PYBIN (uv venv from uv.lock)"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    PYBIN=python
    echo "  conda_env=$CONDA_ENV  python=$(which python) (FALLBACK — uv .venv not found)"
fi

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
GR00T_ROOT="$GR00T_ROOT" \
"$PYBIN" -u \
    "$WRAPPER" \
        --base_model_path "$BASE_MODEL" \
        --dataset_path "$DATASET_DIR" \
        --modality_config_path "$MODALITY_CFG" \
        --embodiment_tag NEW_EMBODIMENT \
        --num_gpus 1 \
        --output_dir "$OUTPUT_DIR" \
        --save_steps "$SAVE_STEPS" \
        --save_total_limit "$SAVE_TOTAL_LIMIT" \
        --max_steps "$MAX_STEPS" \
        --warmup_ratio 0.05 \
        --weight_decay 1e-5 \
        --learning_rate 1e-4 \
        --global_batch_size "$GLOBAL_BATCH" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
        --shard_size 1024 \
        --num_shards_per_epoch 100000 \
        --episode_sampling_rate 0.1 \
        "${SAVE_ONLY_MODEL_FLAG[@]}" \
        "${WANDB_FLAG[@]}" \
        2>&1 | tee "$LOG_FILE"
