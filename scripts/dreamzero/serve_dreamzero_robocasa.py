"""DreamZero (NF4 Wan2.1-14B + robocasa LoRA) pickle/ZMQ policy server for RoboCasa eval.

Speaks the SAME wire protocol as scripts/_gr00t_inference_server.py so the shared
robocasa sim client (scripts/_gr00t_eval_client.py, supports --render GUI) drives it:
    req  = {"op": "get_action", "obs": {state.*:(1,D), video.*:(1,H,W,3), annotation..:[str]}}
    resp = {action.* : (T, D)}            # T action steps, env.step keys
    req  = {"op": "reset"} -> {"ok": True}
    req  = {"op": "shutdown"} -> {"ok": True}, exit

Runs in the `dreamzero` conda env (bitsandbytes NF4). ~20 GB VRAM on a 4090.

Usage:
  conda run -n dreamzero python robocasa-training/scripts/dreamzero/serve_dreamzero_robocasa.py \
      --ckpt-dir <.../checkpoint-XXXX> --port 5700
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys
from pathlib import Path

# This import ordering mirrors the verified-working eval/test scripts. Two non-obvious
# rules to keep the groot->albumentations->skimage->scipy.ndimage import chain from
# crashing (scipy doc-scrape `int += None`) when launched detached/background:
#   1) set SCIPY_ARRAY_API=0 before numpy/torch;  2) do NOT import pyzmq before that chain.
os.environ.setdefault("SCIPY_ARRAY_API", "0")

import numpy as np
import torch
import pandas  # noqa: F401  (kept: this mirrors the verified-working eval script's import set;
#                            pandas pre-warms scipy so the later albumentations->scipy chain
#                            doesn't crash. Do not remove.)

# Paths overridable via env vars; defaults derive from $HOME for portability
# (no hardcoded absolute user paths in committed scripts).
DREAMZERO_REPO = os.environ.get("DREAMZERO_REPO", os.path.expanduser("~/work/dreamzero-repo"))
LEISAAC_INFER = os.environ.get(
    "LEISAAC_DREAMZERO_INFER",
    os.path.expanduser("~/work/isaaclab-experience/LeIsaac/scripts/inference/dreamzero"))
sys.path.insert(0, DREAMZERO_REPO)
sys.path.insert(0, LEISAAC_INFER)

# Defensive neutralisation of a scipy doc-scrape crash: scipy's array-API backend
# registration (run when albumentations->skimage lazily import scipy.ndimage) parses
# function docstrings via NumpyDocString, which in this env trips a Reader `_l += 1`
# TypeError. __init__ pre-fills _parsed_data with empty sections before _parse(), so
# swallowing a _parse failure leaves a valid (empty) doc and lets the import proceed.
try:
    import scipy._lib._docscrape as _ds  # noqa: E402  (does not import scipy.ndimage)
    _ds_orig_parse = _ds.NumpyDocString._parse
    def _ds_safe_parse(self):
        try:
            _ds_orig_parse(self)
        except Exception:
            pass
    _ds.NumpyDocString._parse = _ds_safe_parse
except Exception:
    pass

# Module-level heavy imports. zmq stays out of module scope (imported in main).
from dreamzero_inference_loader import build_dreamzero_inference_model  # noqa: E402
import dreamzero_policy as dzp  # noqa: E402
from tianshou.data import Batch  # noqa: E402


def _find_wan_snapshot() -> str:
    env = os.environ.get("WAN_SNAPSHOT_DIR")
    if env:
        return env
    base = Path(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-I2V-14B-480P/snapshots"))
    snaps = sorted(base.glob("*")) if base.exists() else []
    return str(snaps[0]) if snaps else str(base / "MISSING")


WAN = _find_wan_snapshot()

# env obs state key -> policy modality state key (from our meta/modality.json)
STATE_MAP = {
    "state.base_position": "state.base_pos",
    "state.base_rotation": "state.base_rot",
    "state.end_effector_position_relative": "state.eef_pos",
    "state.end_effector_rotation_relative": "state.eef_rot",
    "state.gripper_qpos": "state.gripper_qpos",
}
# policy action sub-key -> env.step action key
ACTION_MAP = {
    "base_motion": "action.base_motion",
    "ctrl_mode": "action.control_mode",
    "eef_pos": "action.end_effector_position",
    "eef_rot": "action.end_effector_rotation",
    "gripper_close": "action.gripper_close",
}


def _map_obs(obs: dict) -> dict:
    """Rename client/env obs keys -> policy modality keys. Keeps the (1,...) T dim."""
    out = {}
    for k, v in obs.items():
        if k.startswith("state."):
            nk = STATE_MAP.get(k)
            if nk is not None:
                out[nk] = np.asarray(v, dtype=np.float64)
        elif k.startswith("video."):
            out[k] = np.asarray(v)            # video.robot0_* names already match
        elif k.startswith("annotation"):
            t = v[0] if isinstance(v, (list, tuple)) else v
            if isinstance(t, bytes):
                t = t.decode("utf-8", "replace")
            out["annotation.task"] = str(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--cfg-dir", default=None)
    ap.add_argument("--wan-dir", default=WAN)
    ap.add_argument("--port", type=int, default=5700)
    args = ap.parse_args()
    cfg_dir = args.cfg_dir or str(Path(args.ckpt_dir) / "experiment_cfg")

    dzp._init_single_gpu_distributed()
    print(f"[dz-server] building NF4 model from {args.ckpt_dir}", flush=True)
    trained_model, full_cfg, metadata_dict = build_dreamzero_inference_model(
        ckpt_dir=args.ckpt_dir, experiment_cfg_dir=cfg_dir, wan_snap_dir=args.wan_dir)
    sim_policy = dzp._build_fake_groot_sim_policy(
        trained_model, full_cfg, metadata_dict, embodiment_str="robocasa_panda_omron")
    print("[dz-server] policy ready", flush=True)

    ah = sim_policy.trained_model.action_head

    # --- Keep UMT5-XXL prompt encoding ON CPU --------------------------------
    # The loader's default encode_prompt hauls the ~11GB UMT5 onto the GPU for the
    # first (cache-miss) encode, spiking peak VRAM to ~20GB. Alongside the MuJoCo
    # client's render context that overflows a 24GB card on the very first step.
    # UMT5 already lives on CPU here, so run the forward there and ship only the
    # (small) embedding to CUDA. Constant OpenCabinet prompt -> cached after step 1.
    _cpu_prompt_cache: dict = {}
    def _cpu_encode_prompt(input_ids, attention_mask):
        key_t = input_ids.detach().cpu() if torch.is_tensor(input_ids) else torch.as_tensor(input_ids)
        key = key_t.numpy().tobytes()
        if key in _cpu_prompt_cache:
            return _cpu_prompt_cache[key].clone()
        te = ah.text_encoder
        if te is None:
            raise RuntimeError("text encoder freed but uncached prompt requested")
        ids = key_t
        mask = attention_mask.detach().cpu() if torch.is_tensor(attention_mask) else torch.as_tensor(attention_mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        emb = te(ids, mask)                       # UMT5 forward on CPU (one-time, cached)
        emb = emb.clone().to(dtype=torch.bfloat16)
        for i, v in enumerate(seq_lens):
            emb[:, v:] = 0
        emb = emb.to("cuda")                      # only the embedding crosses to GPU
        _cpu_prompt_cache[key] = emb.detach().clone()
        return emb
    ah.encode_prompt = _cpu_encode_prompt
    print("[dz-server] patched: UMT5 prompt encode stays on CPU (no 11GB GPU spike)", flush=True)

    def _reset():
        for attr, val in (("current_start_frame", 0), ("language", None),
                          ("clip_feas", None), ("ys", None), ("_episode_clip_cache", None)):
            if hasattr(ah, attr):
                setattr(ah, attr, val)

    def _act_lookup(act, sub):
        # sim_policy sets batch.act = unnormalized_action, a dict keyed by the FULL
        # policy key "action.<sub>" (e.g. "action.eef_pos"), not the short "<sub>".
        # Support dict OR attr access, full key first then short fallback.
        full = f"action.{sub}"
        if isinstance(act, dict):
            for k in (full, sub):
                if k in act:
                    return act[k]
            return None
        for k in (full, sub):
            v = getattr(act, k, None)
            if v is not None:
                return v
        return None

    @torch.no_grad()
    def _get_action(obs: dict) -> dict:
        pobs = _map_obs(obs)
        result_batch, _ = sim_policy.lazy_joint_forward_causal(Batch(obs=pobs))
        out = {}
        for sub, envk in ACTION_MAP.items():
            v = _act_lookup(result_batch.act, sub)
            if v is None:
                continue
            v = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
            while v.ndim >= 3 and v.shape[0] == 1:
                v = v[0]
            if v.ndim == 1:
                v = v[:, None]
            out[envk] = v.astype(np.float32)   # (T, D)
        return out

    import zmq  # imported late: see top-of-file note (pyzmq-before-scipy crashes)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{args.port}")
    print(f"[dz-server] ready on tcp://*:{args.port}", flush=True)

    while True:
        try:
            req = pickle.loads(sock.recv())
        except Exception as e:
            sock.send(pickle.dumps({"error": str(e)}))
            continue
        op = req.get("op")
        if op == "shutdown":
            sock.send(pickle.dumps({"ok": True})); break
        if op == "reset":
            _reset(); sock.send(pickle.dumps({"ok": True})); continue
        if op != "get_action":
            sock.send(pickle.dumps({"error": f"unknown op {op}"})); continue
        try:
            import time
            t0 = time.perf_counter()
            act = _get_action(req["obs"])
            dt = time.perf_counter() - t0
            print(f"[dz-server] get_action {dt:.1f}s -> {[ (k, v.shape) for k,v in act.items() ]}", flush=True)
            sock.send(pickle.dumps(act))
        except Exception as e:
            import traceback; traceback.print_exc()
            sock.send(pickle.dumps({"error": str(e)}))


if __name__ == "__main__":
    main()
