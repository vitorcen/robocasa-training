"""ACT ZMQ inference server (runs in `lerobot` conda env).

Wire protocol: SAME as scripts/_gr00t_inference_server.py — pickle-over-ZMQ REP.
    request  = {"op": "get_action", "obs": <dict of np.ndarray>}
    response = <dict of np.ndarray, each shape (T, ...)>  per modality-keyed action
    request  = {"op": "shutdown"} -> {"ok": True}, then exit

The whole point of matching GR00T's wire format is that scripts/_gr00t_eval_client.py
runs UNCHANGED — it's the same eval loop, just talking to a different policy.

Input obs from robocasa gym (per dependencies/robocasa/robocasa/wrappers/gym_wrapper.py):
    video.robot0_agentview_left      (H, W, 3) float32 in [0,1]
    video.robot0_agentview_right     (H, W, 3) float32 in [0,1]
    video.robot0_eye_in_hand         (H, W, 3) float32 in [0,1]
    state.base_position              (3,) float32
    state.base_rotation              (4,) float32   (quat)
    state.end_effector_position_relative (3,) float32
    state.end_effector_rotation_relative (4,) float32   (quat)
    state.gripper_qpos               (2,) float32
    annotation.human.task_description  str

The client wraps each entry with a leading T=1 dim before sending.

We reassemble:
    observation.images.<cam>  : (1, 3, H, W) float32   (CHW for ACT)
    observation.state         : (1, 16) float32       (concat per modality.json order)

Action output (12-dim) is split per modality.json action slicing:
    action.base_motion             [0:4]
    action.control_mode            [4:5]
    action.end_effector_position   [5:8]
    action.end_effector_rotation   [8:11]
    action.gripper_close           [11:12]
each with leading T (chunk replay length).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import zmq

# Local-only HF: ACT ckpt is on disk under output_dir/checkpoints/.../pretrained_model
# LeRobot's PreTrainedConfig.from_pretrained handles local dirs natively, but its
# revision check would still hit the hub if we passed a fake repo_id. We patch the
# hub-version resolver to be a no-op before importing the policy class.
import lerobot.datasets.utils as _U
import lerobot.datasets.dataset_metadata as _M
import lerobot.datasets.lerobot_dataset as _L
_id = lambda repo_id, rev: rev
_U.get_safe_version = _id
_M.get_safe_version = _id
_L.get_safe_version = _id

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor.pipeline import DataProcessorPipeline

# Concat order for observation.state — must match the modality.json shipped with
# the RoboCasa dataset (and therefore what ACT was trained on).
STATE_KEYS_AND_DIMS = [
    ("state.base_position", 3),
    ("state.base_rotation", 4),
    ("state.end_effector_position_relative", 3),
    ("state.end_effector_rotation_relative", 4),
    ("state.gripper_qpos", 2),
]
# Action slicing for the response dict — same modality.json.
ACTION_SLICES = [
    ("action.base_motion", slice(0, 4)),
    ("action.control_mode", slice(4, 5)),
    ("action.end_effector_position", slice(5, 8)),
    ("action.end_effector_rotation", slice(8, 11)),
    ("action.gripper_close", slice(11, 12)),
]
CAM_KEYS = [
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
]


def _strip_time_dim(arr):
    """Client added T=1 to every key (see _gr00t_eval_client._add_time_dim).
    Strip it back here so shapes match what the policy expects.
    """
    a = np.asarray(arr)
    if a.ndim >= 1 and a.shape[0] == 1:
        return a[0]
    return a


def _build_batch(obs, device):
    """robocasa raw obs (with T=1 added by client) -> ACT input batch on device."""
    batch = {}
    # State: concat 16 dims in fixed order.
    state_parts = []
    for k, d in STATE_KEYS_AND_DIMS:
        v = _strip_time_dim(obs[k]).astype(np.float32)
        if v.shape[-1] != d:
            raise ValueError(f"{k} expected dim={d}, got {v.shape}")
        state_parts.append(v)
    state = np.concatenate(state_parts, axis=-1)  # (16,)
    batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)

    # Images: HWC float32 [0,1] -> CHW, add batch dim.
    for raw_key in CAM_KEYS:
        img = _strip_time_dim(obs[raw_key]).astype(np.float32)
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"{raw_key} expected (H,W,3), got {img.shape}")
        chw = np.transpose(img, (2, 0, 1))  # (3,H,W)
        target_key = raw_key.replace("video.", "observation.images.")
        batch[target_key] = torch.from_numpy(chw).unsqueeze(0).to(device)

    # Task text — ACT itself doesn't use it, but some processors may demand the
    # key. Pass through if available; otherwise omit.
    if "annotation.human.task_description" in obs:
        v = obs["annotation.human.task_description"]
        if isinstance(v, (list, tuple)):
            v = v[0]
        batch["task"] = str(v)
    return batch


def _split_action_chunk(action_chunk):
    """action_chunk: (T, 12) numpy float32 -> dict of (T, sub-dim) per modality slice."""
    out = {}
    for key, sl in ACTION_SLICES:
        out[key] = action_chunk[:, sl]
    return out


def _resolve_ckpt(model_path):
    """Accept either a pretrained_model dir directly, or a parent that contains
    checkpoints/last/pretrained_model (the layout lerobot-train writes).
    """
    p = Path(model_path)
    if (p / "config.json").exists():
        return p
    for cand in [
        p / "checkpoints" / "last" / "pretrained_model",
        p / "pretrained_model",
    ]:
        if (cand / "config.json").exists():
            return cand
    # Look for the highest-numbered checkpoint dir
    ckpt_root = p / "checkpoints"
    if ckpt_root.exists():
        numeric = sorted(
            [d for d in ckpt_root.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name),
        )
        if numeric:
            cand = numeric[-1] / "pretrained_model"
            if (cand / "config.json").exists():
                return cand
    raise FileNotFoundError(f"could not find a pretrained_model dir under {model_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True,
                    help="ACT ckpt dir (the pretrained_model dir, or a parent containing one).")
    ap.add_argument("--n-action-steps", type=int, default=50,
                    help="How many steps of the predicted chunk to return per request. "
                         "Must be <= policy chunk_size (default 100).")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--temporal-ensemble", type=float, default=None,
                    help="Enable ACT temporal ensembling at inference with this coeff "
                         "(e.g. 0.01). The ckpt was trained with coeff=None so we build "
                         "the ensembler manually. In TE mode the server returns one "
                         "ensembled action per query (client must use n_action_steps=1 and "
                         "send an episode-start reset).")
    args = ap.parse_args()

    ckpt = _resolve_ckpt(args.model_path)
    print(f"[serve_act] loading ACT from {ckpt} ...", flush=True)
    policy = ACTPolicy.from_pretrained(str(ckpt))
    policy.to(args.device)
    policy.eval()
    chunk_size = policy.config.chunk_size

    te = args.temporal_ensemble
    if te is not None:
        # Trained with coeff=None -> no ensembler built at init. Construct it now
        # so select_action() runs the TE path (re-query every step, weight
        # overlapping chunk predictions -> smoother closed-loop, less drift).
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        policy.config.temporal_ensemble_coeff = te
        policy.temporal_ensembler = ACTTemporalEnsembler(te, chunk_size)
        print(f"[serve_act] TEMPORAL ENSEMBLING on, coeff={te} (per-step re-query)", flush=True)
    policy.reset()
    n_steps = min(args.n_action_steps, chunk_size)
    print(f"[serve_act] chunk_size={chunk_size}, returning n_action_steps={n_steps}", flush=True)

    # CRITICAL: the policy's predict_action_chunk returns *normalized* actions
    # (zero-mean, unit-var per dim). Without running the postprocessor, the env
    # gets garbage values and the robot flies off the kitchen. The preprocessor
    # likewise normalizes obs.state and obs.images before they reach the model.
    print(f"[serve_act] loading preprocessor + postprocessor from {ckpt} ...", flush=True)
    preproc = DataProcessorPipeline.from_pretrained(
        str(ckpt), config_filename="policy_preprocessor.json"
    )
    postproc = DataProcessorPipeline.from_pretrained(
        str(ckpt), config_filename="policy_postprocessor.json"
    )
    print(f"[serve_act] pre/post processors ready", flush=True)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{args.port}")
    print(f"[serve_act] ready on tcp://*:{args.port}", flush=True)

    while True:
        try:
            req = pickle.loads(sock.recv())
        except Exception as e:
            print(f"[serve_act] bad request: {e}", file=sys.stderr, flush=True)
            sock.send(pickle.dumps({"error": str(e)}))
            continue
        op = req.get("op")
        if op == "shutdown":
            sock.send(pickle.dumps({"ok": True}))
            print("[serve_act] shutting down", flush=True)
            break
        if op == "reset":
            # Episode boundary: clear the temporal ensembler / action queue so
            # state doesn't bleed across episodes. Sent by the client when
            # --send-reset is set (TE mode).
            policy.reset()
            sock.send(pickle.dumps({"ok": True}))
            continue
        if op != "get_action":
            sock.send(pickle.dumps({"error": f"unknown op {op}"}))
            continue
        try:
            raw_batch = _build_batch(req["obs"], args.device)
            # preproc normalizes images (MEAN_STD with imagenet stats) + state
            # (MEAN_STD from dataset stats.json). Without this the model sees
            # off-distribution inputs and outputs garbage.
            batch = preproc.process_observation(raw_batch)
            with torch.no_grad():
                if te is not None:
                    # select_action runs predict_action_chunk + temporal_ensembler
                    # update, returning ONE ensembled (normalized) action.
                    a = policy.select_action(batch)            # (1, 12) normalized
                    actions_real = postproc.process_action(a)  # (1, 12) real
                    chunk = actions_real.cpu().numpy().astype(np.float32)  # (1, 12) -> T=1
                else:
                    actions = policy.predict_action_chunk(batch)  # (1, chunk_size, 12) normalized
                    # postproc inverts the action normalizer learned at training
                    # time — turns mean-0 unit-var output back into real actions.
                    actions_real = postproc.process_action(actions)
                    chunk = actions_real[0, :n_steps].cpu().numpy().astype(np.float32)  # (T, 12)
            sock.send(pickle.dumps(_split_action_chunk(chunk)))
        except Exception as e:
            import traceback
            traceback.print_exc()
            sock.send(pickle.dumps({"error": str(e)}))


if __name__ == "__main__":
    main()
