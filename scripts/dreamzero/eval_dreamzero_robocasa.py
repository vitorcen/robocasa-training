"""Offline 'does the arm move?' sanity check for a DreamZero RoboCasa LoRA checkpoint.

Loads NF4 Wan2.1-14B + the robocasa LoRA on a single 4090, feeds ONE real observation
from the OpenCabinet dataset (3 camera frames + 16-d state + task), runs the AR
flow-matching action head, and reports the predicted 12-d action chunk stats.

"Arm moves" == actions are finite, non-constant, non-zero (the inference pipeline works
end-to-end). A 50-step checkpoint is ~untrained, so we only assert plumbing, not skill.

Run:
  conda run -n dreamzero python robocasa-training/scripts/dreamzero/eval_dreamzero_robocasa.py \
      --ckpt-dir <.../checkpoint-50>
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import pandas as pd

# Paths overridable via env vars; defaults derive from $HOME for portability
# (no hardcoded absolute user paths in committed scripts).
DREAMZERO_REPO = os.environ.get("DREAMZERO_REPO", str(Path.home() / "work/dreamzero-repo"))
LEISAAC_INFER = os.environ.get(
    "LEISAAC_DREAMZERO_INFER",
    str(Path.home() / "work/isaaclab-experience/LeIsaac/scripts/inference/dreamzero"))
sys.path.insert(0, DREAMZERO_REPO)
sys.path.insert(0, LEISAAC_INFER)

DS = os.environ.get(
    "ROBOCASA_DS",
    str(Path.home() / ".cache/robocasa/datasets/v1.0/target/atomic/OpenCabinet/20250813/lerobot_old"))


def _find_wan_snapshot() -> str:
    env = os.environ.get("WAN_SNAPSHOT_DIR")
    if env:
        return env
    base = Path.home() / ".cache/huggingface/hub/models--Wan-AI--Wan2.1-I2V-14B-480P/snapshots"
    snaps = sorted(base.glob("*")) if base.exists() else []
    return str(snaps[0]) if snaps else str(base / "MISSING")


WAN = _find_wan_snapshot()

# robocasa action[12] split (from meta/modality.json)
ACTION_KEYS = ["action.base_motion", "action.ctrl_mode", "action.eef_pos",
               "action.eef_rot", "action.gripper_close"]


def _first_frame(view: str) -> np.ndarray:
    """Decode first frame of an episode_0 view video -> (H,W,3) uint8."""
    mp4 = f"{DS}/videos/chunk-000/observation.images.{view}/episode_000000.mp4"
    try:
        import decord
        vr = decord.VideoReader(mp4)
        return vr[0].asnumpy().astype(np.uint8)
    except Exception:
        import imageio.v3 as iio
        return np.asarray(iio.imread(mp4, index=0)).astype(np.uint8)


def build_obs() -> dict:
    df = pd.read_parquet(f"{DS}/data/chunk-000/episode_000000.parquet")
    state = np.asarray(df["observation.state"].iloc[0], dtype=np.float64)  # (16,)
    obs = {
        "video.robot0_agentview_left":  _first_frame("robot0_agentview_left")[None],   # (1,H,W,3)
        "video.robot0_agentview_right": _first_frame("robot0_agentview_right")[None],
        "video.robot0_eye_in_hand":     _first_frame("robot0_eye_in_hand")[None],
        "state.base_pos":      state[0:3][None],
        "state.base_rot":      state[3:7][None],
        "state.eef_pos":       state[7:10][None],
        "state.eef_rot":       state[10:14][None],
        "state.gripper_qpos":  state[14:16][None],
        "annotation.task":     "Open the cabinet door.",
    }
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--cfg-dir", default=None)
    ap.add_argument("--wan-dir", default=WAN)
    args = ap.parse_args()
    cfg_dir = args.cfg_dir or str(Path(args.ckpt_dir) / "experiment_cfg")

    from dreamzero_inference_loader import build_dreamzero_inference_model
    import dreamzero_policy as dzp
    from tianshou.data import Batch

    print("[eval] init single-gpu distributed", flush=True)
    dzp._init_single_gpu_distributed()

    print(f"[eval] building NF4 model from {args.ckpt_dir}", flush=True)
    trained_model, full_cfg, metadata_dict = build_dreamzero_inference_model(
        ckpt_dir=args.ckpt_dir, experiment_cfg_dir=cfg_dir, wan_snap_dir=args.wan_dir,
    )
    print("[eval] wrapping sim policy (embodiment=robocasa_panda_omron)", flush=True)
    sim_policy = dzp._build_fake_groot_sim_policy(
        trained_model, full_cfg, metadata_dict, embodiment_str="robocasa_panda_omron",
    )

    obs = build_obs()
    print("[eval] obs shapes:", {k: (np.asarray(v).shape if not isinstance(v, str) else v) for k, v in obs.items()}, flush=True)

    import time
    with torch.no_grad():
        t0 = time.perf_counter()
        result_batch, video_pred = sim_policy.lazy_joint_forward_causal(Batch(obs=obs))
        dt = time.perf_counter() - t0
    print(f"[eval] inference took {dt:.1f}s", flush=True)

    # Collect action sub-keys -> (T, dim) each, concat to action[12]
    parts = []
    for k in ACTION_KEYS:
        sub = k.split(".")[-1]
        v = getattr(result_batch.act, sub, None)
        if v is None:
            v = getattr(result_batch.act, k, None)
        if v is None:
            print(f"[eval] WARNING: missing action key {k}", flush=True)
            continue
        v = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
        while v.ndim >= 3 and v.shape[0] == 1:
            v = v[0]
        if v.ndim == 1:
            v = v[:, None]
        print(f"[eval]   {k}: shape={v.shape} min={v.min():.3f} max={v.max():.3f}", flush=True)
        parts.append(v)

    act = np.concatenate(parts, axis=-1)  # (T, 12)
    print("\n==================== RESULT ====================", flush=True)
    print(f"action chunk shape : {act.shape}  (expect (24, 12))", flush=True)
    print(f"finite (no NaN/Inf): {np.isfinite(act).all()}", flush=True)
    print(f"min/max/mean/std   : {act.min():.4f} / {act.max():.4f} / {act.mean():.4f} / {act.std():.4f}", flush=True)
    nonzero = np.abs(act).max(axis=0) > 1e-4
    varies = act.std(axis=0) > 1e-4
    print(f"per-dim nonzero    : {nonzero.tolist()}", flush=True)
    print(f"per-dim varies(t)  : {varies.tolist()}", flush=True)
    arm_moves = bool(np.isfinite(act).all() and nonzero.any() and (act.std() > 1e-3))
    print(f"\n>>> ARM MOVES (pipeline OK): {arm_moves} <<<", flush=True)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[eval] peak VRAM: {peak:.2f} GB / 24 GB", flush=True)


if __name__ == "__main__":
    main()
