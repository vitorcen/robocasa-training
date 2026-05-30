#!/usr/bin/env python3
"""Standalone Leaderboard generator — recomputes SR from each policy's per-episode
results[], NOT from the (sometimes-stale) success_rate field that the eval client writes
incrementally. Robust to that bug + to DNF runner aggregation. Re-runnable any time.

    python robocasa-training/benchmark/make_leaderboard.py
"""
import json
import os
from pathlib import Path

BENCH = Path(__file__).resolve().parent
RES = BENCH / "results"
OUT = BENCH / "leaderboard.md"

ENV_NAME = os.environ.get("ENV_NAME", "OpenCabinet")
SPLIT = os.environ.get("SPLIT", "target")


def load():
    rows = []
    for j in sorted(RES.glob("*.json")):
        try:
            d = json.load(open(j))
        except Exception:
            continue
        res = d.get("results", [])
        # sim-crash episodes (steps==0 AND not success) are NOT model failures — exclude
        # from the denominator; report them separately as "sim_dnf".
        real = [r for r in res if not (r.get("steps", 0) == 0 and not r.get("success"))]
        sim_dnf = len(res) - len(real)
        succ = [r for r in real if r.get("success")]
        steps_succ = [r["steps"] for r in succ if r.get("steps")]
        sr = (len(succ) / len(real)) if real else None
        rows.append({
            "name": j.stem,
            "sr": sr,
            "n_success": len(succ),
            "n_real": len(real),
            "sim_dnf": sim_dnf,
            "mean_steps": (sum(steps_succ) / len(steps_succ)) if steps_succ else None,
        })
    return rows


def main():
    rows = load()
    rows.sort(key=lambda r: (r["sr"] is not None, r["sr"] or -1), reverse=True)
    L = [f"# RoboCasa {ENV_NAME} Leaderboard", "",
         f"- task `{ENV_NAME}` (split `{SPLIT}`) · SR recomputed from per-episode results "
         f"(sim-crash episodes excluded from denominator)", "",
         "| # | Policy | Success rate | Successes | Mean steps (success) | sim-DNF |",
         "|---|--------|-------------|-----------|----------------------|---------|"]
    for i, r in enumerate(rows, 1):
        if r["sr"] is None:
            L.append(f"| — | {r['name']} | _no data_ | — | — | {r['sim_dnf']} |")
            continue
        ms = f"{r['mean_steps']:.0f}" if r["mean_steps"] else "—"
        L.append(f"| {i} | {r['name']} | **{r['sr']*100:.1f}%** | "
                 f"{r['n_success']}/{r['n_real']} | {ms} | {r['sim_dnf']} |")
    md = "\n".join(L) + "\n"
    OUT.write_text(md)
    print(md)
    print(f"[leaderboard] saved to {OUT}")


if __name__ == "__main__":
    main()
