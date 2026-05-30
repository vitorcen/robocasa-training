"""GR00T-N1.7 ZMQ inference server (runs in the `gr00t-n17` conda env).

Same pickle-over-ZMQ REP protocol as the N1.5 server
(mujoco-experience/scripts/_gr00t_inference_server.py), so the existing sim
client scripts/_gr00t_eval_client.py talks to it UNCHANGED:
  request  = {"op": "get_action", "obs": <dict of np.ndarray>} -> action dict
  request  = {"op": "reset"}      -> {"ok": True}
  request  = {"op": "shutdown"}   -> {"ok": True}, then exit

Two N1.7 specifics vs the N1.5 server:
- API moved to gr00t.policy.gr00t_policy.Gr00tPolicy(embodiment_tag, model_path,
  device); the modality config comes from the CHECKPOINT's saved processor, not
  from DATA_CONFIG_MAP. We still import robocasa_config_n17 first so the
  NEW_EMBODIMENT modality is registered (needed when the processor re-resolves
  it at load time).
- get_action() returns a (action, info) TUPLE in N1.7 — we unpack and send only
  the action dict, matching what the client expects.

Not invoked directly; spawned by eval_gr00t_n17.py / watchdog_gr00t.sh.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import zmq

# Register the RoboCasa NEW_EMBODIMENT modality config before constructing the policy.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import robocasa_config_n17  # noqa: F401  (registration side-effect)


def _to_numpy(d):
    out = {}
    for k, v in d.items():
        if hasattr(v, "cpu"):
            v = v.cpu().numpy()
        elif hasattr(v, "numpy"):
            v = v.numpy()
        out[k] = np.asarray(v)
    return out


def _flat_to_nested(flat):
    """Adapt the shared client's N1.5-style FLAT obs to N1.7's NESTED+BATCHED shape.

    The sim client (_gr00t_eval_client.py) sends flat keys with a leading T dim
    (from _add_time_dim):
        video.<cam>: uint8 (T, H, W, C)
        state.<key>: float32 (T, D)
        annotation.<...>: [str]            (a T-length list of strings)

    N1.7's Gr00tPolicy.get_action / _unbatch_observation expect a nested dict,
    batched (B added in front):
        {"video":   {<cam>: uint8  (B, T, H, W, C)},
         "state":   {<key>: float32(B, T, D)},
         "language":{<key>: list[list[str]]  (B, T)}}
    N1.5's Gr00tPolicy ate the flat form directly; N1.7 refactored to nested, so
    without this the server dies with `KeyError: 'video'` on the first step.
    Idempotent: if already nested (has a dict 'video'), pass through.
    """
    if isinstance(flat.get("video"), dict):
        return flat
    nested = {"video": {}, "state": {}, "language": {}}
    for k, v in flat.items():
        if k.startswith("video."):
            nested["video"][k[len("video."):]] = np.asarray(v)[None]            # +B
        elif k.startswith("state."):
            nested["state"][k[len("state."):]] = np.asarray(v, dtype=np.float32)[None]
        elif k.startswith("annotation"):
            lv = v if isinstance(v, list) else [v]                              # T-list
            nested["language"][k] = [lv]                                        # +B
    return nested


def _format_action(action):
    """Adapt N1.7's action dict back to what the shared client / robocasa env want.

    N1.7 `_get_action` returns BARE modality keys with a leading batch dim:
        {base_motion:(B,T,4), control_mode:(B,T,1), end_effector_position:(B,T,3),
         end_effector_rotation:(B,T,3), gripper_close:(B,T,1)}
    The robocasa env step expects `action.<key>` (the modality.json sub-keys, with
    the `action.` prefix) and the client's _slice_step indexes step t from a (T, D)
    chunk — so strip the B dim and re-prefix. Symmetric to _flat_to_nested on the
    obs side. (N1.5 already returned `action.<key>` (T,D), hence no client change.)
    """
    out = {}
    for k, v in action.items():
        v = np.asarray(v)
        if v.ndim >= 1:                      # drop batch B=1 → (T, D)
            v = v[0]
        out[k if k.startswith("action.") else f"action.{k}"] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--strict", action="store_true",
                    help="Enforce N1.7 observation/action validation (default off, "
                         "matching the lenient N1.5 server).")
    ap.add_argument("--port", type=int, default=5555)
    args = ap.parse_args()

    from gr00t.policy.gr00t_policy import Gr00tPolicy

    print(f"[server-n17] loading policy from {args.model_path} ...", flush=True)
    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_path,
        device=args.device,
        strict=args.strict,
    )

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{args.port}")
    print(f"[server-n17] ready on tcp://*:{args.port}", flush=True)

    while True:
        try:
            req = pickle.loads(sock.recv())
        except Exception as e:
            print(f"[server-n17] bad request: {e}", file=sys.stderr, flush=True)
            sock.send(pickle.dumps({"error": str(e)}))
            continue
        op = req.get("op")
        if op == "shutdown":
            sock.send(pickle.dumps({"ok": True}))
            print("[server-n17] shutting down", flush=True)
            break
        if op == "reset":
            try:
                if hasattr(policy, "reset"):
                    policy.reset()
            except Exception as e:
                print(f"[server-n17] reset warn: {e}", file=sys.stderr, flush=True)
            sock.send(pickle.dumps({"ok": True}))
            continue
        if op != "get_action":
            sock.send(pickle.dumps({"error": f"unknown op {op}"}))
            continue
        try:
            result = policy.get_action(_flat_to_nested(req["obs"]))
            # N1.7 returns (action, info); N1.5 returned action only.
            action = result[0] if isinstance(result, tuple) else result
            sock.send(pickle.dumps(_format_action(_to_numpy(action))))
        except Exception as e:
            import traceback
            traceback.print_exc()
            sock.send(pickle.dumps({"error": str(e)}))


if __name__ == "__main__":
    main()
