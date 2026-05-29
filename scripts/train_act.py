"""Auto-sliced ACT training with periodic eval — mandatory watcher built in.

Why: long unattended training without periodic eval is the documented anti-pattern
(see ~/.claude memory `feedback-train-with-watcher`). Symptoms it prevents:
  - DP v0.4.0 training 10h with train_loss 0.554 → 0.011 (looks converged), then
    final eval revealed 0/15 success — wasted 9h that 1× quick eval would catch.
  - DreamerV3 6h burn with zero interpretable signal because no slice eval ran.

This script slices total_steps into N segments. After each, it pauses training,
spawns a 3-episode RoboCasa eval (via scripts/robocasa_eval_act.py), appends
the success rate to logs/sr_curve.csv, and classifies the trajectory as one of:

  DEAD      — last 3 evals all < 5% SR. Policy isn't learning the task.
  UNDERFIT  — recent SR plateau ≤ 1.05× previous window. Need more capacity/data.
  OVERFIT   — peak passed ≥2 evals ago and current < 0.7 × peak. Catastrophic forgetting.
  PROGRESS  — still climbing or near peak.

Early-stop: touching `<output_dir>/.eval_abort` between segments terminates the
loop (used by external watcher / human). The DEAD classifier does NOT auto-kill
by default — pass --dead_kills to opt in (matches memory's "default: trigger
abort but don't hard-kill" rule).

Hardware note (this machine, RTX 4090 24G):
  ACT training fills 22.75/24 GiB at bs=8 + bf16 → train + eval cannot share
  the GPU. That's why slices are SEQUENTIAL (train, idle, eval, idle, train),
  not the "watcher polls ckpt dir" mode used on multi-GPU rigs.

Usage:
    python scripts/train_act.py                         # 100k steps, 10 slices, 3 ep eval
    python scripts/train_act.py --total-steps 50000 --n-slices 5
    python scripts/train_act.py --no-eval               # skip eval (still bad practice)
    python scripts/train_act.py --eval-episodes 5 --eval-env-name OpenCabinet

Anything after a `--` is forwarded verbatim to lerobot-train per segment.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # robocasa-training/
PARENT_REPO = REPO_ROOT.parent                       # mujoco-experience/
DEFAULT_DATASET_ROOT = "/home/david/.cache/robocasa/datasets/v1.0/target/atomic/OpenCabinet/20250813/lerobot"
DEFAULT_OUTPUT = REPO_ROOT / "checkpoints" / "act_opencabinet"
DEFAULT_LOG_DIR = REPO_ROOT / "logs"
EVAL_ORCHESTRATOR = PARENT_REPO / "scripts" / "robocasa_eval_act.py"


def _run_segment(target_steps, output_dir, dataset_root, batch_size, num_workers,
                 persistent_workers, video_backend, frame_cache_dir, save_freq,
                 log_freq, resume, passthrough, log_file, init_from=None):
    """Spawn lerobot-train as a subprocess up to target_steps. Returns exit code."""
    scripts_dir = str(Path(__file__).resolve().parent)
    patch_code = (
        f"import sys; sys.path.insert(0, {scripts_dir!r}); "
        "import lerobot.datasets.utils as U, lerobot.datasets.dataset_metadata as M, "
        "lerobot.datasets.lerobot_dataset as L; "
        "_id = lambda repo_id, rev: rev; "
        "U.get_safe_version = _id; M.get_safe_version = _id; L.get_safe_version = _id; "
    )
    if frame_cache_dir:
        # Precached .npy memmaps replace torchcodec decode entirely → no
        # DataLoader-worker SEGV, ~0.005s/read. This is the real fix; the
        # PID-safe patch below is moot when the cache is used but kept as a
        # fallback for runs without a cache.
        patch_code += (
            f"import frame_cache_patch; frame_cache_patch.apply({frame_cache_dir!r}); "
        )
    else:
        # No cache: at least make torchcodec fork-safe (partial mitigation).
        patch_code += "import pid_safe_decoder_patch; pid_safe_decoder_patch.apply(); "
    runner = f"{patch_code}from lerobot.scripts.lerobot_train import main; main()"
    # First finetune segment: init weights from a pretrained ckpt via --policy.path
    # (loads the pretrained ACT config+weights for finetuning). Once we've saved a
    # finetune ckpt, later segments resume from output_dir instead. Fresh-from-
    # scratch runs (no init_from) use --policy.type=act.
    policy_init = (["--policy.type=act"] if (init_from is None or resume)
                   else [f"--policy.path={init_from}"])
    cli = [
        *policy_init,
        "--policy.push_to_hub=false",
        "--policy.device=cuda",
        "--policy.use_amp=true",
        "--dataset.repo_id=local/robocasa_opencab",
        f"--dataset.root={dataset_root}",
        f"--dataset.video_backend={video_backend}",
        f"--output_dir={output_dir}",
        "--job_name=act_opencabinet",
        f"--steps={target_steps}",
        f"--batch_size={batch_size}",
        f"--num_workers={num_workers}",
        f"--persistent_workers={'true' if persistent_workers else 'false'}",
        f"--save_freq={save_freq}",
        f"--log_freq={log_freq}",
        "--save_checkpoint=true",
        "--wandb.enable=false",
    ]
    if resume:
        cli += [
            "--resume=true",
            f"--config_path={output_dir}/checkpoints/last/pretrained_model/train_config.json",
        ]
    cli += list(passthrough)

    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    with open(log_file, "ab") as logf:
        logf.write(f"\n\n===== segment up to step {target_steps} (resume={resume}) =====\n".encode())
        logf.flush()
        cmd = [sys.executable, "-c", runner, *cli]
        # stdbuf to keep tqdm + INFO lines flushed
        cmd = ["stdbuf", "-oL", "-eL", *cmd]
        return subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)


def _run_eval(ckpt_dir, n_episodes, env_name, results_path, n_action_steps,
              max_steps, log_file):
    """Spawn scripts/robocasa_eval_act.py and wait. Returns exit code."""
    cmd = [
        sys.executable, str(EVAL_ORCHESTRATOR),
        "--env-name", env_name,
        "--n-episodes", str(n_episodes),
        "--ckpt", str(ckpt_dir),
        "--results-path", str(results_path),
        "--n-action-steps", str(n_action_steps),
        "--max-steps", str(max_steps),
    ]
    with open(log_file, "ab") as logf:
        logf.write(f"\n\n===== eval at ckpt {ckpt_dir} ({n_episodes} ep) =====\n".encode())
        logf.flush()
        return subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT)


def _classify(history):
    """history = list of (step, success_rate). Returns one of
    DEAD / UNDERFIT / OVERFIT / PROGRESS / WAITING."""
    if not history:
        return "WAITING"
    n = len(history)
    sr_values = [h[1] for h in history]
    current = sr_values[-1]

    # DEAD: 3 evals deep and all near zero
    if n >= 3 and max(sr_values[-3:]) < 0.05:
        return "DEAD"

    # OVERFIT: peak was ≥2 evals ago AND current < 70% of peak
    peak = max(sr_values)
    peak_idx = max(range(n), key=lambda i: sr_values[i])
    if peak >= 0.10 and peak_idx <= n - 3 and current < 0.7 * peak:
        return "OVERFIT"

    # UNDERFIT: 6+ evals deep, no meaningful gain between halves
    if n >= 6:
        prev_max = max(sr_values[-6:-3])
        recent_max = max(sr_values[-3:])
        if prev_max >= 0.10 and recent_max <= 1.05 * prev_max:
            return "UNDERFIT"

    return "PROGRESS"


def _write_status(status_path, status, history, target_total, abort_path):
    """Atomic-ish status file for `train_status.sh` / human watchers."""
    obj = {
        "status": status,
        "total_steps_target": target_total,
        "history": [{"step": s, "success_rate": sr} for s, sr in history],
        "peak": max((sr for _, sr in history), default=None),
        "current": history[-1][1] if history else None,
        "abort_marker": str(abort_path),
        "ts": time.time(),
    }
    tmp = status_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(status_path)


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--total-steps", type=int, default=100000,
                    help="Total training steps across all slices.")
    ap.add_argument("--n-slices", type=int, default=10,
                    help="Split total-steps into N slices, eval after each. "
                         "Defaults to 10 per ~/.claude memory feedback-train-with-watcher.")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4,
                    help="DataLoader workers. Safe at 4 now that pid_safe_decoder_patch "
                         "fixes the fork-inherited torchcodec handle SEGV (validated: 57 "
                         "worker respawns, 0 crashes). num_workers=0 was the old slow "
                         "workaround (data_s 0.21s >> updt_s 0.032s, ~6x slower).")
    ap.add_argument("--persistent_workers", type=lambda x: x.lower() == 'true', default=False,
                    help="Keep False: workers respawn each epoch so any torchcodec frame-"
                         "buffer leak is cleared; the PID-safe patch makes respawn cheap+safe.")
    ap.add_argument("--video_backend", default="torchcodec",
                    choices=["pyav", "torchcodec"],
                    help="torchcodec = GPU decode (fast, ~17 step/s with num_workers=0). "
                         "pyav = ffmpeg CPU decode (avoid: 56x slower on this dataset).")
    ap.add_argument("--max_segment_retries", type=int, default=3,
                    help="If a segment's lerobot-train subprocess exits non-zero (e.g. "
                         "rare DataLoader SEGV), resume from last ckpt and retry up to "
                         "this many times before giving up.")
    ap.add_argument("--resume_from_last", action="store_true",
                    help="Skip the empty-dir check; pick up from the last SR data point "
                         "in the csv and resume training from output_dir/checkpoints/last/.")
    ap.add_argument("--frame_cache_dir", default=None,
                    help="Dir of precached .npy frame memmaps (from precache_videos.py). "
                         "If omitted, auto-detects <dataset_root>/frame_cache. When present, "
                         "training reads frames from the cache (no torchcodec → no SEGV, fast).")
    ap.add_argument("--no_frame_cache", action="store_true",
                    help="Force torchcodec decode even if a frame cache exists.")
    ap.add_argument("--init-from", default=None,
                    help="Finetune: init policy weights from this pretrained_model dir "
                         "(--policy.path) on the FIRST segment, instead of training from "
                         "scratch. Later segments resume from output_dir. Use for "
                         "pretrain(mimicgen)-then-finetune(human).")
    ap.add_argument("--save_freq", type=int, default=5000,
                    help="lerobot ckpt save frequency (steps). Frequent (5k) so a crash "
                         "loses little and the resume point keeps advancing.")
    ap.add_argument("--log_freq", type=int, default=500)
    # Eval-side
    ap.add_argument("--eval-episodes", type=int, default=3,
                    help="Per-slice eval episodes. Keep small (3) for speed; "
                         "scale via --final-eval-episodes for the final ckpt.")
    ap.add_argument("--final-eval-episodes", type=int, default=20,
                    help="Episodes for the FINAL slice (after the last train segment) "
                         "to get a statistically-honest peak number.")
    ap.add_argument("--eval-env-name", default="OpenCabinet")
    ap.add_argument("--eval-n-action-steps", type=int, default=50)
    ap.add_argument("--eval-max-steps", type=int, default=400)
    ap.add_argument("--no-eval", action="store_true")
    # Watcher behaviour
    ap.add_argument("--dead_kills", action="store_true",
                    help="If set, DEAD classification touches .eval_abort to terminate. "
                         "Default off — DEAD only warns; you decide.")
    # Output locations
    ap.add_argument("--sr-csv", default=str(DEFAULT_LOG_DIR / "sr_curve.csv"))
    ap.add_argument("--status-json", default=str(DEFAULT_LOG_DIR / "auto_eval.status.json"))
    ap.add_argument("--train-log", default=str(DEFAULT_LOG_DIR / "train_act.log"))
    ap.add_argument("-h", "--help", action="store_true")
    args, passthrough = ap.parse_known_args()

    if args.help:
        ap.print_help()
        return

    output_dir = Path(args.output_dir).resolve()
    sr_csv = Path(args.sr_csv)
    sr_csv.parent.mkdir(parents=True, exist_ok=True)
    status_json = Path(args.status_json)
    train_log = Path(args.train_log)
    # Abort marker lives next to the CSV (logs/), not under output_dir, because
    # lerobot-train refuses to start with an existing output_dir + resume=False.
    abort_marker = sr_csv.parent / ".eval_abort"
    if abort_marker.exists():
        abort_marker.unlink()

    # lerobot-train refuses to start with an existing output_dir + resume=False.
    # Normally we want a clean start, but --resume_from_last lets the user
    # continue an interrupted multi-slice run.
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume_from_last:
        print(f"[train_act] ERROR: output_dir already exists and is non-empty:\n  {output_dir}\n"
              f"  Pass --resume_from_last to continue, "
              f"or `rm -rf {output_dir}` to start fresh.", file=sys.stderr)
        sys.exit(2)
    if not EVAL_ORCHESTRATOR.exists():
        print(f"[train_act] missing eval orchestrator: {EVAL_ORCHESTRATOR}", file=sys.stderr)

    # Slice boundaries: cumulative train targets. 10 slices of 10k for the default
    # 100k run → eval points at [10k, 20k, ..., 100k].
    if args.no_eval or args.n_slices <= 1:
        slice_targets = [args.total_steps]
    else:
        per = args.total_steps // args.n_slices
        slice_targets = [per * (i + 1) for i in range(args.n_slices)]
        if slice_targets[-1] != args.total_steps:
            slice_targets[-1] = args.total_steps

    # Frequent saves so a crash/retry loses little and resume point advances.
    # (The old bug: save_freq = slice_targets[0] = 75k meant a crash at 84k
    # rewound to the 75k resume point every retry → no-progress loop.)
    save_freq_per_seg = max(min(args.save_freq, slice_targets[0]), 1)

    # Resolve frame cache: explicit dir, else auto-detect <dataset_root>/frame_cache.
    frame_cache_dir = None
    if not args.no_frame_cache:
        cand = Path(args.frame_cache_dir) if args.frame_cache_dir else (Path(args.dataset_root) / "frame_cache")
        if cand.is_dir() and any(cand.glob("*.npy")):
            frame_cache_dir = str(cand)

    print(f"[train_act] mode: {'sliced+eval' if not args.no_eval else 'single-shot, NO EVAL'}", flush=True)
    print(f"[train_act] frame cache: {frame_cache_dir or 'OFF (torchcodec decode)'}", flush=True)
    print(f"[train_act] save_freq: {save_freq_per_seg}", flush=True)
    print(f"[train_act] slice targets: {slice_targets}", flush=True)
    print(f"[train_act] eval: {args.eval_episodes} ep per slice on {args.eval_env_name}", flush=True)
    print(f"[train_act] outputs:")
    print(f"  ckpts:  {output_dir}/checkpoints/")
    print(f"  csv:    {sr_csv}")
    print(f"  status: {status_json}")
    print(f"  log:    {train_log}")
    print(f"  abort:  touch {abort_marker} to stop between slices")
    print(flush=True)

    # Init CSV
    if not sr_csv.exists():
        with open(sr_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "step", "n_episodes", "success_rate", "mean_steps",
                "wall_time_s", "status",
            ])

    history = []  # list of (step, success_rate)
    completed_through_step = 0

    # Resume mode: load prior data points, skip slices we already evaluated.
    if args.resume_from_last and sr_csv.exists():
        with open(sr_csv, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    history.append((int(row["step"]), float(row["success_rate"])))
                except (KeyError, ValueError):
                    continue
        if history:
            completed_through_step = max(s for s, _ in history)
            remaining = [t for t in slice_targets if t > completed_through_step]
            print(f"[train_act] resume: csv has {len(history)} datapoints up to step "
                  f"{completed_through_step}; remaining slices: {remaining}", flush=True)
            slice_targets = remaining
            if not slice_targets:
                print("[train_act] all slices already evaluated; nothing to do.", flush=True)
                return

    for i, target in enumerate(slice_targets):
        if abort_marker.exists():
            print(f"[train_act] .eval_abort detected before slice {i+1}; exiting.", flush=True)
            break

        # First attempt: resume if we have prior data, fresh otherwise.
        # Retries: always resume from last ckpt (lerobot save_freq guarantees one).
        ret = None
        for attempt in range(args.max_segment_retries + 1):
            resume = (i > 0) or (args.resume_from_last and target > completed_through_step) or (attempt > 0)
            tag = f"SLICE {i+1}/{len(slice_targets)}"
            if attempt > 0:
                tag += f" RETRY {attempt}/{args.max_segment_retries}"
            print(f"\n[train_act] === {tag} — train to step {target} (resume={resume}) ===", flush=True)
            ret = _run_segment(
                target_steps=target,
                output_dir=output_dir,
                dataset_root=args.dataset_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                persistent_workers=args.persistent_workers,
                video_backend=args.video_backend,
                frame_cache_dir=frame_cache_dir,
                save_freq=save_freq_per_seg,
                log_freq=args.log_freq,
                resume=resume,
                passthrough=passthrough,
                log_file=train_log,
                init_from=args.init_from,
            )
            if ret == 0:
                break
            # Retry only if a checkpoint exists to resume from.
            last_ckpt_cfg = output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
            if not last_ckpt_cfg.exists():
                print(f"[train_act] no ckpt at {last_ckpt_cfg.parent}; cannot retry.", file=sys.stderr)
                break
            print(f"[train_act] slice {i+1} attempt {attempt+1} failed (ret={ret}); "
                  f"will resume from last ckpt and retry.", file=sys.stderr)
            time.sleep(5)  # let any orphan worker / cuda context clean up
        if ret != 0:
            print(f"[train_act] slice {i+1} exhausted retries (last ret={ret}); see {train_log}",
                  file=sys.stderr)
            sys.exit(ret)

        if args.no_eval:
            continue

        # Give cuda context from the just-exited lerobot-train process time to
        # release VRAM + free handles. Without this we've seen slice-1 eval
        # crash with ret=245 (client SIGSEGV during warm-up reset), reproducibly
        # only when training -> eval transition is too fast. 5s is enough in
        # practice; cheaper than retry logic.
        time.sleep(5)

        ckpt_dir = output_dir / "checkpoints" / "last" / "pretrained_model"
        if not (ckpt_dir / "config.json").exists():
            print(f"[train_act] WARN: ckpt missing at {ckpt_dir}, skipping eval", flush=True)
            continue

        is_final = (i == len(slice_targets) - 1)
        n_ep = args.final_eval_episodes if is_final else args.eval_episodes
        results_path = output_dir / "eval_results" / f"step_{target}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n[train_act] === EVAL at step {target} ({n_ep} ep) ===", flush=True)
        t0 = time.time()
        ret = _run_eval(ckpt_dir, n_ep, args.eval_env_name, results_path,
                        args.eval_n_action_steps, args.eval_max_steps, train_log)
        # The client writes results incrementally after each episode, so even
        # when it SIGSEGVs on a later reset (ret=245) the file still holds the
        # episodes that finished. Accept a partial result rather than dropping
        # the data point — a 2/3-episode SR is still a usable curve point.
        if not results_path.exists():
            print(f"[train_act] eval at step {target} produced no results "
                  f"(ret={ret}). Missing data point.", flush=True)
            continue

        r = json.loads(results_path.read_text())
        completed = r.get("n_completed", len(r.get("results", [])))
        if completed == 0:
            print(f"[train_act] eval at step {target} completed 0 episodes "
                  f"(ret={ret}). Missing data point.", flush=True)
            continue
        if ret != 0 or completed < n_ep:
            print(f"[train_act] eval at step {target} partial: {completed}/{n_ep} "
                  f"episodes (client ret={ret}, likely renderer SIGSEGV). "
                  f"Using partial SR.", flush=True)
        sr = float(r.get("success_rate", 0.0))
        mean_steps = (sum(e["steps"] for e in r.get("results", [])) / max(len(r["results"]), 1)
                      if r.get("results") else 0)
        wall = float(r.get("wall_time_s", time.time() - t0))
        n_ep_actual = completed

        history.append((target, sr))
        status = _classify(history)
        with open(sr_csv, "a", newline="") as f:
            csv.writer(f).writerow([target, n_ep_actual, sr, mean_steps, wall, status])
        _write_status(status_json, status, history, args.total_steps, abort_marker)

        peak = max(s for _, s in history)
        peak_step = next(s for s, sr_ in history if sr_ == peak)
        print(f"\n[train_act] step {target}: SR = {sr:.1%}  "
              f"(peak {peak:.1%} @ step {peak_step})  STATUS = {status}", flush=True)
        print(f"[train_act] csv -> {sr_csv}", flush=True)

        if status == "DEAD":
            print(f"[train_act] DEAD: last 3 evals < 5% SR. "
                  f"{'Auto-aborting (--dead_kills).' if args.dead_kills else 'NOT auto-killing — pass --dead_kills to opt in.'}",
                  flush=True)
            if args.dead_kills:
                abort_marker.touch()
                break

    print("\n[train_act] done.", flush=True)
    if history:
        print(f"[train_act] SR curve: {[(s, f'{sr:.1%}') for s, sr in history]}", flush=True)


if __name__ == "__main__":
    main()
