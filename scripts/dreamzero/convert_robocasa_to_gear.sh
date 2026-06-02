#!/bin/bash
# Convert RoboCasa OpenCabinet (PandaOmron) LeRobot v2.1 dataset -> DreamZero GEAR format.
# Metadata-only (no parquet/video change), ~1 min, runs in AutoDL 无卡模式 (no GPU needed).
#
# Embodiment uses the BUILT-IN DreamZero tag `robocasa_panda_omron` (embedding id 13) —
# better than reusing xdof: gives a meaningful embodiment embedding for this body.
#
# Dim splits read from dataset meta/modality.json (GR00T-style, confirmed):
#   state[16]: base_pos[0:3] base_rot[3:7] eef_pos[7:10] eef_rot[10:14] gripper_qpos[14:16]
#   action[12]: base_motion[0:4] ctrl_mode[4:5] eef_pos[5:8] eef_rot[8:11] gripper_close[11:12]
#
# relative-action: ONLY eef_pos (3-dim) — dimension-safe (state eef_pos is also 3-dim).
#   eef_rot NOT made relative: state eef_rot is 4-dim (quat) vs action 3-dim (axis-angle) → mismatch.
#   base/ctrl_mode/gripper stay absolute. Revisit after smoke test if loss/eval looks wrong.
#
# Usage (on AutoDL):
#   bash convert_robocasa_to_gear.sh /root/autodl-tmp/opencabinet

set -e
DATASET_PATH=${1:-/root/autodl-tmp/opencabinet}
DREAMZERO_REPO=${DREAMZERO_REPO:-/root/autodl-tmp/dreamzero-repo}
PYTHON=${PYTHON:-$(command -v python || command -v python3 || echo /root/miniconda3/bin/python)}

if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: dataset not found at $DATASET_PATH"; exit 1
fi

# Backup original modality.json (the GR00T one) before DreamZero rewrites it
if [ -f "$DATASET_PATH/meta/modality.json" ] && [ ! -f "$DATASET_PATH/meta/modality.json.gr00t-bak" ]; then
    cp "$DATASET_PATH/meta/modality.json" "$DATASET_PATH/meta/modality.json.gr00t-bak"
    echo "Backed up modality.json -> modality.json.gr00t-bak"
fi

"$PYTHON" "$DREAMZERO_REPO/scripts/data/convert_lerobot_to_gear.py" \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag robocasa_panda_omron \
    --state-keys  '{"base_pos":[0,3],"base_rot":[3,7],"eef_pos":[7,10],"eef_rot":[10,14],"gripper_qpos":[14,16]}' \
    --action-keys '{"base_motion":[0,4],"ctrl_mode":[4,5],"eef_pos":[5,8],"eef_rot":[8,11],"gripper_close":[11,12]}' \
    --relative-action-keys eef_pos \
    --task-key task \
    --fps 20 \
    --action-horizon 24 \
    --force

echo "--- GEAR metadata generated ---"
ls -la "$DATASET_PATH/meta/"
