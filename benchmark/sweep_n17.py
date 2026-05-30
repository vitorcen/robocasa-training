#!/usr/bin/env python3
"""N1.7 checkpoint SWEEP — eval a grid of training checkpoints (every 2k steps by
default) on the SAME fixed 30 RoboCasa kitchens the frozen baselines used, so we can
plot SR-vs-step and find the sweep peak (underfit -> peak -> overfit knee).

Each ckpt is evaluated by the EXISTING orchestrator scripts/eval_gr00t_n17.py, which
spawns the N1.7 policy server (uv .venv) + the shared seed-locked driver/worker client
(_gr00t_eval_client.py). Fairness is enforced by:
  - SEED_BASE=0          -> episode i is always the SAME kitchen as the baselines'
  - --max-steps 1200     -> identical step budget to authoritative N1.5/pi0.5
  - --n-action-steps 16  -> identical replay horizon (mean-success-steps stays comparable)
  - EVAL_EP_RETRIES=3    -> sim-DNF (heap-corruption / hang) retried, honest fails are not

Writes one JSON per ckpt to benchmark/results/sweep/ and prints an SR-vs-step table.
The peak ckpt's 30-round JSON is what later feeds the final 3-policy leaderboard.

    python benchmark/sweep_n17.py                      # 8k..16k step 2k, 30 rounds
    STEPS="8000 12000 16000" ROUNDS=30 python benchmark/sweep_n17.py
    python benchmark/sweep_n17.py --dry-run            # just show the plan

Env knobs: STEPS, ROUNDS, MAX_STEPS, N_ACTION_STEPS, SEED_BASE, EVAL_EP_RETRIES,
           EP_TIMEOUT_S, PER_CKPT_TIMEOUT_S, BASE_PORT, CKPT_DIR.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent                       # robocasa-training/
ORCH = REPO_ROOT / "scripts" / "eval_gr00t_n17.py"
SWEEP_DIR = BENCH_DIR / "results" / "sweep"

CKPT_DIR = Path(os.environ.get(
    "CKPT_DIR", REPO_ROOT / "checkpoints" / "gr00t_n17_opencabinet"))
STEPS = [int(s) for s in os.environ.get("STEPS", "8000 10000 12000 14000 16000").split()]
ROUNDS = int(os.environ.get("ROUNDS", "30"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "1200"))     # MUST match authoritative baselines
N_ACTION_STEPS = int(os.environ.get("N_ACTION_STEPS", "16"))
SEED_BASE = os.environ.get("SEED_BASE", "0")             # same 30 kitchens as baselines
ENV_NAME = os.environ.get("ENV_NAME", "OpenCabinet")
SPLIT = os.environ.get("SPLIT", "target")
BASE_PORT = int(os.environ.get("BASE_PORT", "5610"))
# generous per-ckpt wall: 30 rounds x (up to 1200 steps + retries) + server warmup
PER_CKPT_TIMEOUT_S = int(os.environ.get("PER_CKPT_TIMEOUT_S", str(ROUNDS * 120 + 900)))


def _recompute(json_path):
    """SR excluding sim-DNF (steps==0 AND not success), + mean success steps. Mirrors
    make_leaderboard.py so the sweep table and the final leaderboard agree exactly."""
    try:
        d = json.load(open(json_path))
    except Exception:
        return None
    res = d.get("results", [])
    real = [r for r in res if not (r.get("steps", 0) == 0 and not r.get("success"))]
    succ = [r for r in real if r.get("success")]
    steps_succ = [r["steps"] for r in succ if r.get("steps")]
    return {
        "sr": (len(succ) / len(real)) if real else None,
        "n_success": len(succ),
        "n_real": len(real),
        "sim_dnf": len(res) - len(real),
        "mean_steps": (sum(steps_succ) / len(steps_succ)) if steps_succ else None,
    }


def _eval_ckpt(step, port):
    ckpt = CKPT_DIR / f"checkpoint-{step}"
    out_json = SWEEP_DIR / f"N1.7-ckpt-{step}.json"
    out_log = SWEEP_DIR / f"N1.7-ckpt-{step}.log"
    if not ckpt.is_dir():
        return {"step": step, "status": "missing-ckpt", "detail": str(ckpt)}

    cmd = [sys.executable, str(ORCH),
           "--env-name", ENV_NAME, "--split", SPLIT,
           "--n-episodes", str(ROUNDS), "--max-steps", str(MAX_STEPS),
           "--n-action-steps", str(N_ACTION_STEPS),
           "--ckpt", str(ckpt), "--results-path", str(out_json), "--port", str(port)]

    env = dict(os.environ)
    env["SEED_BASE"] = SEED_BASE                 # lock the 30 kitchens
    env.setdefault("EVAL_EP_RETRIES", "3")       # retry sim-DNF, never honest fails
    env.pop("MUJOCO_GL", None)                   # headless

    print(f"\n{'='*72}\n[sweep] ckpt-{step}  port={port}  {ROUNDS} rounds x {MAX_STEPS} steps "
          f"(seed_base={SEED_BASE})\n  {' '.join(cmd)}\n{'='*72}", flush=True)
    t0 = time.time()
    try:
        with open(out_log, "w") as lf:
            subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=lf,
                           stderr=subprocess.STDOUT, timeout=PER_CKPT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"[sweep] ckpt-{step} TIMED OUT after {PER_CKPT_TIMEOUT_S}s", flush=True)
    wall = time.time() - t0

    rec = _recompute(out_json)
    if rec is None:
        return {"step": step, "status": "DNF", "detail": f"no json (see {out_log.name})",
                "wall_s": wall}
    rec.update({"step": step, "status": "ok", "wall_s": wall})
    return rec


def _table(rows):
    rows = sorted(rows, key=lambda r: r["step"])
    L = ["", "# N1.7 OpenCabinet checkpoint sweep", "",
         f"- {ROUNDS} rounds x {MAX_STEPS} steps, seed_base={SEED_BASE} "
         f"(same kitchens as baselines), SR excludes sim-DNF",
         "",
         "| step | SR | successes | mean succ steps | sim-DNF | wall |",
         "|------|----|-----------|-----------------|---------|------|"]
    best = None
    for r in rows:
        if r.get("status") != "ok" or r.get("sr") is None:
            L.append(f"| {r['step']} | _{r.get('status','?')}_ | — | — | — | — |")
            continue
        ms = f"{r['mean_steps']:.0f}" if r.get("mean_steps") else "—"
        wall = f"{r['wall_s']/60:.0f}m" if r.get("wall_s") else "—"
        mark = ""
        if best is None or r["sr"] > best["sr"]:
            best = r
        L.append(f"| {r['step']} | **{r['sr']*100:.1f}%** | {r['n_success']}/{r['n_real']} "
                 f"| {ms} | {r['sim_dnf']} | {wall} |")
    if best:
        L += ["", f"**peak: ckpt-{best['step']}  SR={best['sr']*100:.1f}%  "
                  f"({best['n_success']}/{best['n_real']}, mean {best['mean_steps']:.0f} steps)**"]
    return "\n".join(L) + "\n", best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] ckpt_dir={CKPT_DIR}\n[sweep] grid={STEPS}\n[sweep] out={SWEEP_DIR}")
    if args.dry_run:
        for s in STEPS:
            ck = CKPT_DIR / f"checkpoint-{s}"
            print(f"  ckpt-{s}: {'OK' if ck.is_dir() else 'MISSING'}  {ck}")
        return

    rows = []
    for i, step in enumerate(STEPS):
        rows.append(_eval_ckpt(step, BASE_PORT + i))
        md, _ = _table(rows)                       # incremental so a crash leaves a partial table
        (SWEEP_DIR / "sweep_table.md").write_text(md)
        print(md, flush=True)

    md, best = _table(rows)
    (SWEEP_DIR / "sweep_table.md").write_text(md)
    print("\n\n" + md)
    if best:
        print(f"[sweep] peak ckpt-{best['step']} -> "
              f"benchmark/results/sweep/N1.7-ckpt-{best['step']}.json "
              f"is the final-leaderboard N1.7 entry")


if __name__ == "__main__":
    main()
