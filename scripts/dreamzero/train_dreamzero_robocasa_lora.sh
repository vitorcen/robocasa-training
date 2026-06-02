#!/bin/bash
# DreamZero bf16 LoRA finetune on RoboCasa OpenCabinet (PandaOmron).
# Ported from LeIsaac train_dreamzero_bf16_lora.sh: num_views 2->3, data=robocasa_relative.
# Target: 1× ≥80 GB GPU (A100/H100 80G or RTX Pro 6000 96G). CUDA 12.8 + PyTorch 2.7.
#
# Prerequisites (do once, AutoDL 无卡模式 OK):
#   1. bash convert_robocasa_to_gear.sh /root/autodl-tmp/opencabinet
#   2. cp robocasa_relative.yaml $DREAMZERO_REPO/groot/vla/configs/data/dreamzero/
#   3. cd $DREAMZERO_REPO && pip install -e .   (already done on this box)
#
# Usage:
#   SMOKE=1 bash train_dreamzero_robocasa_lora.sh     # 50-step smoke (GPU 模式, verify OOM/loss/save)
#   bash train_dreamzero_robocasa_lora.sh             # full 15k run

set -e
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Multi-GPU NCCL (single GPU ignores these):
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

if [ -d /root/miniconda3/bin ] && [[ ":$PATH:" != *":/root/miniconda3/bin:"* ]]; then
    export PATH="/root/miniconda3/bin:$PATH"
fi
TORCHRUN=${TORCHRUN:-$(command -v torchrun || echo /root/miniconda3/bin/torchrun)}

# ============ USER CONFIGURATION ============
DREAMZERO_REPO=${DREAMZERO_REPO:-/root/autodl-tmp/dreamzero-repo}
ROBOCASA_DATA_ROOT=${ROBOCASA_DATA_ROOT:-/root/autodl-tmp/opencabinet}
OUTPUT_DIR=${OUTPUT_DIR:-/root/autodl-tmp/dreamzero_robocasa_opencabinet_lora}
LORA_RANK=${LORA_RANK:-4}
LORA_ALPHA=${LORA_ALPHA:-4}
if [ "${SMOKE:-0}" = "1" ]; then
    MAX_STEPS=${MAX_STEPS:-50}
    SAVE_STEPS=${SAVE_STEPS:-50}
else
    MAX_STEPS=${MAX_STEPS:-15000}   # 500 demo (8× LeIsaac) -> start higher than leisaac's 10k
    SAVE_STEPS=${SAVE_STEPS:-2500}
fi
WAN_CKPT_DIR=${WAN_CKPT_DIR:-/root/autodl-tmp/wan2.1-i2v-14b-480p}
TOKENIZER_DIR=${TOKENIZER_DIR:-/root/autodl-tmp/umt5-xxl}
NUM_GPUS=${NUM_GPUS:-1}
# =============================================

for p in "$ROBOCASA_DATA_ROOT" "$WAN_CKPT_DIR" "$TOKENIZER_DIR" "$DREAMZERO_REPO"; do
    [ -d "$p" ] || { echo "ERROR: required path not found: $p"; exit 1; }
done
[ -f "$ROBOCASA_DATA_ROOT/meta/embodiment.json" ] || {
    echo "ERROR: $ROBOCASA_DATA_ROOT/meta/embodiment.json missing — run convert_robocasa_to_gear.sh first"; exit 1; }

cd "$DREAMZERO_REPO"

"$TORCHRUN" --nproc_per_node "$NUM_GPUS" --standalone groot/vla/experiment/experiment.py \
    report_to=none \
    data=dreamzero/robocasa_relative \
    wandb_project=dreamzero_robocasa \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-4 \
    save_steps=$SAVE_STEPS \
    training_args.warmup_ratio=0.05 \
    output_dir="$OUTPUT_DIR" \
    per_device_train_batch_size=1 \
    max_steps=$MAX_STEPS \
    weight_decay=1e-5 \
    save_total_limit=5 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=2 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    frame_seqlen=880 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2_offload.json" \
    save_lora_only=true \
    "+training_args.save_only_model=true" \
    max_chunk_size=2 \
    robocasa_data_root="$ROBOCASA_DATA_ROOT" \
    dit_version="$WAN_CKPT_DIR" \
    text_encoder_pretrained_path="$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
    image_encoder_pretrained_path="$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    vae_pretrained_path="$WAN_CKPT_DIR/Wan2.1_VAE.pth" \
    tokenizer_path="$TOKENIZER_DIR"

# Notes:
#   - num_views=3: agentview_left + agentview_right + eye_in_hand (natively matches DreamZero 3-view pretrain)
#   - LoRA r=4 default; bump to 16 ONLY if loss plateaus by step 2000. Never >=32 for video-diffusion LoRA.
#   - max_steps=15000 ~= 4-6h on 80-96G; pair with mid-freq interleave eval @ 1200 sim-steps.
#   - save_lora_only + save_only_model => tiny ckpts (~200 MB), 5 kept ~= 1 GB.
