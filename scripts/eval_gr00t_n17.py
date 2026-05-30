"""Orchestrator: run a locally-finetuned GR00T-N1.7 policy against a RoboCasa env.

Mirrors mujoco-experience/scripts/robocasa_eval_gr00t.py (the N1.5 path) but:
  - server = robocasa-training/scripts/serve_gr00t_n17.py in the `gr00t-n17` env
  - client = mujoco-experience/scripts/_gr00t_eval_client.py in `robocasa` env
    (REUSED UNCHANGED — identical ZMQ wire protocol).

Usage:
    python scripts/eval_gr00t_n17.py --env-name OpenCabinet --n-episodes 10 \
        --ckpt checkpoints/gr00t_n17_opencabinet/checkpoint-8000
"""
import argparse
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent          # robocasa-training/
PARENT_ROOT = REPO_ROOT.parent                              # mujoco-experience/
CLIENT = PARENT_ROOT / "scripts" / "_gr00t_eval_client.py"  # reuse the shared client
SERVER = REPO_ROOT / "scripts" / "serve_gr00t_n17.py"


def _find_free_port(preferred):
    s = socket.socket()
    try:
        s.bind(("", preferred)); s.close(); return preferred
    except OSError:
        s.close()
    s = socket.socket(); s.bind(("", 0)); port = s.getsockname()[1]; s.close()
    return port


def _conda_cmd(env_name, py_cmd):
    return (
        f"source $(conda info --base)/etc/profile.d/conda.sh && "
        f"conda activate {env_name} && "
        f"export PYTHONDONTWRITEBYTECODE=1 && "
        f"exec python -u {py_cmd}"
    )


def _server_cmd(server_env, py_cmd):
    """Run the N1.7 policy server via the clean uv .venv (built from Isaac-GR00T's
    uv.lock) if present — that's the validated dep set. A conda env cloned from
    another env has a mismatched CUDA/lib stack that randomly corrupts the process
    (see gr00t-4090 skill). Falls back to `conda activate <server_env>` if no .venv."""
    venv_py = REPO_ROOT / "dependencies" / "Isaac-GR00T" / ".venv" / "bin" / "python"
    if venv_py.exists():
        return f"export PYTHONDONTWRITEBYTECODE=1 && exec {shlex.quote(str(venv_py))} -u {py_cmd}"
    return _conda_cmd(server_env, py_cmd)


def _wait_port_open(host, port, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", default="OpenCabinet")
    ap.add_argument("--split", default="target", choices=["pretrain", "target"])
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--n-action-steps", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--ckpt", required=True,
                    help="N1.7 checkpoint dir (e.g. checkpoints/gr00t_n17_opencabinet/checkpoint-8000)")
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--port", type=int, default=5557,
                    help="Default 5557 to avoid the N1.5 (5555) and ACT (5556) eval ports.")
    ap.add_argument("--server-warmup-s", type=int, default=300)
    ap.add_argument("--server-env", default="gr00t-n17")
    ap.add_argument("--client-env", default="robocasa")
    ap.add_argument("--results-path", default=None)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--render-warmup-s", type=float, default=5.0)
    args = ap.parse_args()

    ckpt = os.path.abspath(args.ckpt)
    if not os.path.isdir(ckpt):
        sys.exit(f"N1.7 ckpt dir not found: {ckpt}\n"
                 f"hint: run scripts/train_n17.sh (or watchdog_gr00t.sh) first.")

    port = _find_free_port(args.port)
    if port != args.port:
        print(f"[orch-n17] port {args.port} busy, using {port}", flush=True)

    server_py = (
        f"{SERVER} --model-path {shlex.quote(ckpt)} "
        f"--embodiment-tag {args.embodiment_tag} --port {port}"
    )
    client_py = (
        f"{CLIENT} --env-name {args.env_name} --split {args.split} "
        f"--n-episodes {args.n_episodes} --n-action-steps {args.n_action_steps} "
        f"--max-steps {args.max_steps} --port {port}"
    )
    if args.results_path:
        client_py += f" --results-path {shlex.quote(args.results_path)}"
    if args.render:
        client_py += f" --render --render-warmup-s {args.render_warmup_s}"

    server_cmd = ["bash", "-c", _server_cmd(args.server_env, server_py)]
    client_cmd = ["bash", "-c", _conda_cmd(args.client_env, client_py)]

    print(f"[orch-n17] starting GR00T-N1.7 server in env [{args.server_env}] ...", flush=True)
    server = subprocess.Popen(server_cmd, stdout=sys.stdout, stderr=sys.stderr,
                              preexec_fn=os.setsid)
    try:
        if not _wait_port_open("localhost", port, args.server_warmup_s):
            server.terminate()
            sys.exit(f"[orch-n17] server failed to bind port {port} within "
                     f"{args.server_warmup_s}s — check server logs above")
        print(f"[orch-n17] server up on {port}; starting client ...", flush=True)
        ret = subprocess.call(client_cmd)
    finally:
        print("[orch-n17] tearing down server ...", flush=True)
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
            server.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)
            except Exception:
                pass

    sys.exit(ret)


if __name__ == "__main__":
    main()
