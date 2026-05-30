#!/usr/bin/env python3
"""RoboCasa OpenCabinet benchmark — eval every policy in policies.tsv over N rounds,
then print + save a Leaderboard ranked by success rate.

Each row's eval_script is one of the existing orchestrators (eval_gr00t_n17.py /
robocasa_eval_gr00t.py / robocasa_eval_pi05.py) — they each spawn their own policy
server (in the right conda/uv env) plus the shared robocasa sim client, and write a
results JSON (success_rate / n_completed / results[]). This runner just sequences them
on the single GPU, distinct ports, then aggregates.

Usage:
    python robocasa-training/benchmark/run_benchmark.py                 # 20 rounds, 1200 steps
    ROUNDS=20 MAX_STEPS=1200 ENV_NAME=OpenCabinet python .../run_benchmark.py
    python .../run_benchmark.py --only GR00T-N1.7-OpenCabinet           # one policy

Env knobs: ROUNDS, MAX_STEPS, N_ACTION_STEPS, ENV_NAME, SPLIT, EVAL_EP_RETRIES,
           PER_POLICY_TIMEOUT_S.
"""
import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent.parent          # mujoco-experience repo root
TSV = BENCH_DIR / "policies.tsv"
RESULTS_DIR = BENCH_DIR / "results"
LEADERBOARD_MD = BENCH_DIR / "leaderboard.md"

ROUNDS = int(os.environ.get("ROUNDS", "20"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "1200"))      # OpenCabinet needs ~400-1150 steps
N_ACTION_STEPS = int(os.environ.get("N_ACTION_STEPS", "16"))
ENV_NAME = os.environ.get("ENV_NAME", "OpenCabinet")
SPLIT = os.environ.get("SPLIT", "target")
PER_POLICY_TIMEOUT_S = int(os.environ.get("PER_POLICY_TIMEOUT_S", str(ROUNDS * 200 + 600)))
BASE_PORT = int(os.environ.get("BASE_PORT", "5600"))


def _read_tsv():
    rows = []
    with open(TSV) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if parts[0] == "name":          # header
                continue
            if len(parts) < 3:
                continue
            name, eval_script, ckpt = parts[0], parts[1], parts[2]
            extra = parts[3] if len(parts) > 3 else "-"
            rows.append({"name": name.strip(), "eval_script": eval_script.strip(),
                         "ckpt": ckpt.strip(), "extra": extra.strip()})
    return rows


def _run_one(p, port):
    """Run one policy's eval; return its parsed result dict (or a DNF marker)."""
    script = REPO_ROOT / p["eval_script"]
    out_json = RESULTS_DIR / f"{p['name']}.json"
    out_log = RESULTS_DIR / f"{p['name']}.log"
    if not script.exists():
        return {"status": "missing-script", "detail": str(script)}
    if not p["ckpt"] or p["ckpt"] in ("-", "FILL_ME_pi05_ckpt_path") or p["ckpt"].startswith("FILL_ME"):
        return {"status": "skipped", "detail": f"no ckpt ({p['ckpt']})"}

    cmd = [sys.executable, str(script),
           "--env-name", ENV_NAME, "--split", SPLIT,
           "--n-episodes", str(ROUNDS), "--max-steps", str(MAX_STEPS),
           "--n-action-steps", str(N_ACTION_STEPS),
           "--ckpt", p["ckpt"], "--results-path", str(out_json), "--port", str(port)]
    render = os.environ.get("BENCH_RENDER", "0") == "1"
    if render:
        cmd += ["--render", "--render-warmup-s", os.environ.get("RENDER_WARMUP_S", "3")]
    if p["extra"] and p["extra"] != "-":
        cmd += shlex.split(p["extra"])

    env = dict(os.environ)
    env.setdefault("EVAL_EP_RETRIES", "3")        # auto-retry robocasa sim flakiness
    if render:
        env.setdefault("DISPLAY", ":0")           # GUI: MuJoCo viewer on the physical display
    else:
        env.pop("MUJOCO_GL", None)                # headless: no viewer

    print(f"\n{'='*70}\n[bench] {p['name']}  (port {port}, {ROUNDS} rounds x {MAX_STEPS} steps)\n"
          f"  cmd: {' '.join(shlex.quote(c) for c in cmd)}\n{'='*70}", flush=True)
    t0 = time.time()
    try:
        with open(out_log, "w") as lf:
            subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT,
                           timeout=PER_POLICY_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"[bench] {p['name']} TIMED OUT after {PER_POLICY_TIMEOUT_S}s", flush=True)
    wall = time.time() - t0

    if not out_json.exists():
        return {"status": "DNF", "detail": f"no results json (see {out_log.name})", "wall_s": wall}
    try:
        d = json.load(open(out_json))
    except Exception as e:
        return {"status": "DNF", "detail": f"bad json: {e}", "wall_s": wall}
    res = d.get("results", [])
    succ = [r for r in res if r.get("success")]
    steps_succ = [r["steps"] for r in succ if r.get("steps")]
    return {
        "status": "ok",
        "success_rate": d.get("success_rate", 0.0),
        "n_completed": d.get("n_completed", len(res)),
        "n_success": len(succ),
        "mean_steps_success": (sum(steps_succ) / len(steps_succ)) if steps_succ else None,
        "wall_s": wall,
    }


def _leaderboard(results):
    def key(item):
        r = item[1]
        return r.get("success_rate", -1) if r.get("status") == "ok" else -1
    ranked = sorted(results.items(), key=key, reverse=True)

    lines = []
    lines.append(f"# RoboCasa {ENV_NAME} Leaderboard")
    lines.append("")
    lines.append(f"- task: `{ENV_NAME}` (split `{SPLIT}`) · rounds: {ROUNDS} · max_steps: {MAX_STEPS} "
                 f"· n_action_steps: {N_ACTION_STEPS}")
    lines.append("")
    lines.append("| # | Policy | Success rate | Successes | Completed | Mean steps (success) | Wall |")
    lines.append("|---|--------|-------------|-----------|-----------|----------------------|------|")
    for i, (name, r) in enumerate(ranked, 1):
        if r.get("status") == "ok":
            sr = f"**{r['success_rate']*100:.1f}%**"
            ms = f"{r['mean_steps_success']:.0f}" if r.get("mean_steps_success") else "—"
            wall = f"{r['wall_s']/60:.0f}m"
            lines.append(f"| {i} | {name} | {sr} | {r['n_success']}/{r['n_completed']} | "
                         f"{r['n_completed']}/{ROUNDS} | {ms} | {wall} |")
        else:
            lines.append(f"| — | {name} | _{r['status']}_ | — | — | — | {r.get('detail','')} |")
    md = "\n".join(lines) + "\n"
    LEADERBOARD_MD.write_text(md)
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run just this policy name from the TSV")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = _read_tsv()
    if args.only:
        rows = [r for r in rows if r["name"] == args.only]
        if not rows:
            sys.exit(f"no policy named {args.only} in {TSV}")

    results = {}
    for i, p in enumerate(rows):
        results[p["name"]] = _run_one(p, BASE_PORT + i)
        # incremental leaderboard so a later crash still leaves a partial board
        _leaderboard(results)

    print("\n\n" + _leaderboard(results))
    print(f"[bench] leaderboard saved to {LEADERBOARD_MD}")


if __name__ == "__main__":
    main()
