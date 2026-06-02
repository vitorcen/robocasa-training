#!/usr/bin/env bash
# GR00T-N1.7 RoboCasa training watchdog: sliced train-while-eval + auto-resume on crash.
#
# Single 4090: training and eval CANNOT coexist (training fills ~22 GB). So this
# is a TIME-SLICED serial loop (satisfies the "long training must have sliced
# eval + early-stop" rule): train a slice -> kill -> wait GPU drain -> eval the
# new ckpts -> resume. On a crash it auto-resumes from the latest ckpt.
#
# INTERLEAVE + EARLY-STOP (default ON): instead of training straight to MAX_STEPS
# and only evaluating at the end, the run is cut into INTERLEAVE_STEPS slices. After
# each slice the new boundary ckpt is evaluated; if closed-loop SR stops improving for
# EARLYSTOP_PATIENCE consecutive evals, the run stops early (the peak is already past).
# Slices stop via a StopAtStep callback in launch_finetune_n17.py that keeps --max_steps
# at the FULL target, so the LR scheduler is one continuous curve (no per-slice sawtooth).
# Set INTERLEAVE_STEPS=0 for the legacy "train to MAX_STEPS, eval at end" behavior.
#
# Background (N1.x on 4090 24GB, bf16 + grad-ckpt): training can crash with
# `d.is_cuda() INTERNAL ASSERT FAILED` randomly. The ckpt up to that point is
# intact. Resume can OOM because the saved optimizer.pt blows the transient
# headroom -> we drop optimizer.pt before each resume (HF re-inits the optimizer;
# loss recovers in ~50 step).
#
# Env:
#   MAX_STEPS / SAVE_STEPS       final target + ckpt cadence
#   INTERLEAVE_STEPS             slice size = eval cadence (default 1500; 0 = eval-at-end)
#   EARLYSTOP_PATIENCE           stop after this many non-improving evals (default 3; 0 = off)
#   EARLYSTOP_MIN_DELTA          SR gain (0-1) required to count as improvement (default 0.0)
#   GLOBAL_BATCH / GRAD_ACCUM    passed through to train_n17.sh
#   EVAL_STEPS_MULTIPLE          eval ckpts at multiples of this (auto-aligned to INTERLEAVE_STEPS)
#   EVAL_N_EPISODES              sim episodes per eval
#   EVAL_MAX_STEPS               sim steps per episode (default 1200; 400 underreports slow successes)
#   EVAL_ENV_NAME                RoboCasa task (default OpenCabinet)
#   EVAL_ACTION_HORIZON          client n_action_steps replayed per chunk
#   MAX_RETRIES                  max watchdog cycles before giving up
#   OUTPUT_DIR                   train output (ckpt dir)
#   CONDA_ENV                    N1.7 conda env (default gr00t-n17), used by train + eval
#
# Stops once HF Trainer reports global_step >= MAX_STEPS, or early-stop triggers.

set -uo pipefail

# robocasa-training repo root = scripts/.. ; this script lives in scripts/.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MAX_STEPS="${MAX_STEPS:-8000}"
# SAVE_STEPS small (250) on purpose: N1.x throws random CUDA faults on the 4090, and
# the watchdog can only resume from a SAVED checkpoint. If the first save is too late
# (e.g. 500) and a fault hits before it, the watchdog restarts from scratch and can
# loop forever making zero progress. 250 lands a ckpt early; KEEP_MULTIPLE(2000)+last-few
# pruning keeps these temporary ckpts from filling disk. See gr00t-4090-finetune skill.
SAVE_STEPS="${SAVE_STEPS:-250}"
EVAL_STEPS_MULTIPLE="${EVAL_STEPS_MULTIPLE:-1000}"  # eval ckpts at multiples of this (0 = disabled)
EVAL_N_EPISODES="${EVAL_N_EPISODES:-10}"
EVAL_ENV_NAME="${EVAL_ENV_NAME:-OpenCabinet}"
EVAL_SPLIT="${EVAL_SPLIT:-target}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-1200}"            # sim steps per episode (400 underreports slow successes)
EVAL_WALL_S="${EVAL_WALL_S:-2400}"                  # hard wall per eval ckpt
EVAL_ACTION_HORIZON="${EVAL_ACTION_HORIZON:-16}"
EVAL_POLICY_PORT="${EVAL_POLICY_PORT:-5557}"
CONDA_ENV="${CONDA_ENV:-gr00t-n17}"
# micro-batch=1 on 4090 (global 4 / accum 4): matches train_n17.sh small24. Do NOT
# raise to 8 here — that forces micro-batch=2 and OOMs with RoboCasa's 3 cameras.
GLOBAL_BATCH="${GLOBAL_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
# MAX_STALL = consecutive cycles WITHOUT checkpoint progress before giving up. This is the
# real stop condition: as long as the latest checkpoint keeps advancing, the watchdog grinds
# on indefinitely (crashes are expected on this box). MAX_RETRIES is just a far-away absolute
# backstop so a truly wedged loop can't spin forever.
MAX_STALL="${MAX_STALL:-40}"
MAX_RETRIES="${MAX_RETRIES:-2000}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/checkpoints/gr00t_n17_opencabinet}"
EVAL_LOG="${EVAL_LOG:-$REPO_ROOT/logs/gr00t_n17_ckpts.csv}"

# --- Interleave + early-stop (default ON) ---------------------------------
INTERLEAVE_STEPS="${INTERLEAVE_STEPS:-1500}"        # slice size; 0 = legacy eval-at-end
EARLYSTOP_PATIENCE="${EARLYSTOP_PATIENCE:-3}"       # non-improving evals before stop; 0 = off
EARLYSTOP_MIN_DELTA="${EARLYSTOP_MIN_DELTA:-0.0}"   # SR improvement threshold (0-1)
if (( INTERLEAVE_STEPS > 0 )); then
    # Boundary ckpts land on INTERLEAVE_STEPS multiples; eval + keep must align to them,
    # else the boundary ckpt is neither evaluated nor preserved. Lock both to the slice size.
    if (( EVAL_STEPS_MULTIPLE == 0 || INTERLEAVE_STEPS % EVAL_STEPS_MULTIPLE != 0 )); then
        EVAL_STEPS_MULTIPLE="$INTERLEAVE_STEPS"
    fi
    KEEP_MULTIPLE="${KEEP_MULTIPLE:-$INTERLEAVE_STEPS}"
    if (( INTERLEAVE_STEPS % KEEP_MULTIPLE != 0 )); then
        KEEP_MULTIPLE="$INTERLEAVE_STEPS"
    fi
    export KEEP_MULTIPLE
fi

WATCHDOG_LOG_DIR="$REPO_ROOT/logs/gr00t_watchdog"
mkdir -p "$WATCHDOG_LOG_DIR"
WATCHDOG_LOG="$WATCHDOG_LOG_DIR/watchdog_$(date +%Y%m%d_%H%M%S).log"

echo "[watchdog] target=$MAX_STEPS save_steps=$SAVE_STEPS global=$GLOBAL_BATCH accum=$GRAD_ACCUM"
echo "[watchdog] output=$OUTPUT_DIR" | tee -a "$WATCHDOG_LOG"
if (( INTERLEAVE_STEPS > 0 )); then
    echo "[watchdog] interleave=$INTERLEAVE_STEPS eval_mult=$EVAL_STEPS_MULTIPLE keep_mult=${KEEP_MULTIPLE:-} eval_steps=$EVAL_MAX_STEPS earlystop_patience=$EARLYSTOP_PATIENCE" | tee -a "$WATCHDOG_LOG"
else
    echo "[watchdog] interleave=OFF (eval at end), eval_mult=$EVAL_STEPS_MULTIPLE eval_steps=$EVAL_MAX_STEPS" | tee -a "$WATCHDOG_LOG"
fi
echo "[watchdog] log=$WATCHDOG_LOG"

cleanup_procs() {
    # Broad pkill first (covers most cases)
    pkill -f "launch_finetune_n17\|gr00t/experiment\|serve_gr00t_n17\|eval_gr00t_n17\|_gr00t_eval_client" 2>/dev/null
    sleep 3
    pkill -9 -f "launch_finetune_n17\|gr00t/experiment\|serve_gr00t_n17\|eval_gr00t_n17\|_gr00t_eval_client" 2>/dev/null
    sleep 2
    # Bulletproof orphan kill: only target known eval/inference scripts to avoid
    # nuking our own legit training process.  These leave detached children that
    # escape process-group pkill.
    for pat in "serve_gr00t_n17" "_gr00t_eval_client"; do
        for pid in $(pgrep -f "$pat" 2>/dev/null); do
            kill -9 "$pid" 2>/dev/null && echo "  [watchdog] killed orphan PID $pid ($pat)"
        done
    done
    sleep 1
}

# Wait until GPU memory drops below threshold (MiB). Returns 0 on success,
# 1 if still busy after timeout.  Use before launching memory-hungry steps.
# Fallback: if natural drain timed out and ALLOW_GPU_RESET=1, attempts
# `sudo -n nvidia-smi --gpu-reset` (passwordless) to clear CUDA driver zombies.
wait_gpu_free() {
    local threshold_mib="${1:-3000}"
    local timeout_s="${2:-60}"
    local interval=2
    local waited=0
    while (( waited < timeout_s )); do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        used=${used:-99999}
        if (( used < threshold_mib )); then
            echo "  [watchdog] GPU free (${used} MiB) after ${waited}s"
            return 0
        fi
        sleep $interval
        waited=$(( waited + interval ))
    done
    local final
    final=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "  [watchdog] GPU still ${final} MiB after ${timeout_s}s — attempt gpu-reset" >&2
    if sudo -n nvidia-smi --gpu-reset >/dev/null 2>&1; then
        sleep 3
        local after
        after=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        echo "  [watchdog] post-reset GPU=${after} MiB"
        (( after < threshold_mib )) && return 0
    else
        echo "  [watchdog] WARN: sudo gpu-reset not available (no passwordless sudo?)" >&2
    fi
    return 1
}

latest_step() {
    # Latest checkpoint-N step number, or empty if none.
    ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null \
        | sed 's|.*/checkpoint-||' \
        | sort -n | tail -1
}

drop_optimizer() {
    local step="$1"
    local ckpt="$OUTPUT_DIR/checkpoint-$step"
    if [[ -f "$ckpt/optimizer.pt" ]]; then
        mv "$ckpt/optimizer.pt" "$ckpt/optimizer.pt.bak" 2>/dev/null
        echo "  [watchdog] moved optimizer.pt → .bak (free $(du -h "$ckpt/optimizer.pt.bak" | cut -f1))"
    fi
}

# HF Trainer honors trainer_state.json's save_steps over CLI args on resume.
# Patch it in place so our SAVE_STEPS env knob actually takes effect.
patch_save_steps() {
    local step="$1"
    local ts="$OUTPUT_DIR/checkpoint-$step/trainer_state.json"
    [[ -f "$ts" ]] || return 0
    python3 - "$ts" "$SAVE_STEPS" <<'PY'
import json, sys
path, save_steps = sys.argv[1], int(sys.argv[2])
with open(path) as f:
    d = json.load(f)
old = d.get("save_steps")
d["save_steps"] = save_steps
with open(path, "w") as f:
    json.dump(d, f, indent=2)
print(f"  [watchdog] patched {path}: save_steps {old} → {save_steps}")
PY
}

trainer_state_step() {
    local step="$1"
    local ts="$OUTPUT_DIR/checkpoint-$step/trainer_state.json"
    [[ -f "$ts" ]] || { echo ""; return; }
    python3 -c "import json,sys; print(json.load(open('$ts')).get('global_step',''))" 2>/dev/null
}

# Selective ckpt pruning:
#   - Keep ALL ckpts at multiples of $KEEP_MULTIPLE (default 500)
#   - Keep only last $KEEP_TEMPORARY non-multiple ckpts
prune_checkpoints() {
    local keep_mult="${KEEP_MULTIPLE:-500}"
    local keep_temp="${KEEP_TEMPORARY:-3}"
    python3 - "$OUTPUT_DIR" "$keep_mult" "$keep_temp" <<'PY'
import os, sys, shutil, re
out, keep_mult, keep_temp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
if not os.path.isdir(out):
    sys.exit(0)
ckpts = []
for d in os.listdir(out):
    m = re.match(r"^checkpoint-(\d+)$", d)
    if m:
        ckpts.append((int(m.group(1)), os.path.join(out, d)))
ckpts.sort()
permanent = {s for s, _ in ckpts if s > 0 and s % keep_mult == 0}
temporary = [(s, p) for s, p in ckpts if s not in permanent]
to_keep_temp = set(s for s, _ in temporary[-keep_temp:])
removed = []
for s, p in temporary:
    if s in to_keep_temp:
        continue
    shutil.rmtree(p, ignore_errors=True)
    removed.append(s)
if removed:
    print(f"  [watchdog] prune: removed temporary {removed}, kept permanent {sorted(permanent)}, last temp {sorted(to_keep_temp)}")
PY
}

# Early-stop signal: read EVAL_LOG, sort by step, find the running-best SR and how many
# trailing evals failed to beat it by EARLYSTOP_MIN_DELTA. Exit 0 => stop (peak is past),
# 1 => keep going. Needs > EARLYSTOP_PATIENCE points so a single noisy dip can't trip it.
check_early_stop() {
    (( EARLYSTOP_PATIENCE > 0 )) || return 1
    [[ -f "$EVAL_LOG" ]] || return 1
    python3 - "$EVAL_LOG" "$EARLYSTOP_PATIENCE" "$EARLYSTOP_MIN_DELTA" <<'PY'
import sys, csv
log, patience, min_delta = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
rows = []
try:
    with open(log) as f:
        for row in csv.DictReader(f):
            try:
                rows.append((int(row["step"]), float(row["success_rate"])))
            except (ValueError, KeyError, TypeError):
                continue  # skip NaN / malformed
except FileNotFoundError:
    sys.exit(1)
rows.sort()
if len(rows) < patience + 1:
    sys.exit(1)  # not enough evidence yet
best = -1.0
no_improve = 0
for _, sr in rows:
    if sr > best + min_delta:
        best, no_improve = sr, 0
    else:
        no_improve += 1
sys.exit(0 if no_improve >= patience else 1)
PY
}

eval_unevaluated_ckpts() {
    # Find ckpts at multiples of EVAL_STEPS_MULTIPLE that aren't yet in EVAL_LOG.
    # Run eval one by one (GPU serial), append CSV.
    (( EVAL_STEPS_MULTIPLE > 0 )) || return 0
    [[ -d "$OUTPUT_DIR" ]] || return 0
    mkdir -p "$(dirname "$EVAL_LOG")"
    [[ -f "$EVAL_LOG" ]] || echo "step,n_episodes,n_completed,success_rate,n_action_steps" > "$EVAL_LOG"

    # Sort by the numeric step (sed first), NOT `sort -t- -k2` — the output dir path can
    # contain dashes (e.g. mujoco-experience) which scrambles a dash-delimited sort.
    for step in $(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sed 's|.*/checkpoint-||' | sort -n); do
        local ckpt_dir="$OUTPUT_DIR/checkpoint-$step"
        (( step % EVAL_STEPS_MULTIPLE == 0 )) || continue
        # Skip if already evaluated (step appears as first CSV column)
        if awk -F, -v s="$step" 'NR>1 && $1==s {found=1} END{exit !found}' "$EVAL_LOG" 2>/dev/null; then
            continue
        fi

        echo "  [watchdog] EVAL ckpt-$step  n_episodes=$EVAL_N_EPISODES  h=$EVAL_ACTION_HORIZON" | tee -a "$WATCHDOG_LOG"

        # Training already torn down by caller; ensure GPU is drained before eval.
        wait_gpu_free 3000 90 | tee -a "$WATCHDOG_LOG"

        local eval_log="$REPO_ROOT/logs/gr00t_n17_ckpt_eval_${step}.log"
        local results_json="$REPO_ROOT/logs/gr00t_n17_ckpt_eval_${step}.json"

        # eval_gr00t_n17.py spawns the N1.7 server (gr00t-n17) + sim client (robocasa)
        # and writes success_rate incrementally to results_json.
        CONDA_ENV="$CONDA_ENV" \
            timeout "$EVAL_WALL_S" python "$REPO_ROOT/scripts/eval_gr00t_n17.py" \
                --env-name "$EVAL_ENV_NAME" --split "$EVAL_SPLIT" \
                --n-episodes "$EVAL_N_EPISODES" \
                --n-action-steps "$EVAL_ACTION_HORIZON" \
                --max-steps "$EVAL_MAX_STEPS" \
                --ckpt "$ckpt_dir" --port "$EVAL_POLICY_PORT" \
                --server-env "$CONDA_ENV" \
                --results-path "$results_json" 2>&1 | tee "$eval_log" | tail -8 | tee -a "$WATCHDOG_LOG"

        # Stop any eval leftovers (frees GPU for next train cycle)
        pkill -f "serve_gr00t_n17\|_gr00t_eval_client" 2>/dev/null
        sleep 3

        # Parse success_rate from the client's JSON summary.
        local sr ncomp
        if [[ -f "$results_json" ]]; then
            sr=$(python3 -c "import json;d=json.load(open('$results_json'));print(f\"{d.get('success_rate',0):.3f}\")" 2>/dev/null)
            ncomp=$(python3 -c "import json;print(json.load(open('$results_json')).get('n_completed',0))" 2>/dev/null)
        fi
        sr="${sr:-NaN}"; ncomp="${ncomp:-0}"
        echo "$step,$EVAL_N_EPISODES,$ncomp,$sr,$EVAL_ACTION_HORIZON" >> "$EVAL_LOG"
        echo "  [watchdog] ➜ step=$step  success_rate=$sr  ($ncomp/$EVAL_N_EPISODES ep)" | tee -a "$WATCHDOG_LOG"
    done
}

retry=0
retry=0; stall=0; prev_ckpt=-1
while (( stall < MAX_STALL && retry < MAX_RETRIES )); do
    retry=$((retry + 1))
    cur=$(latest_step || echo "")
    cur_num=${cur:-0}
    # Progress check: did the PREVIOUS cycle advance the latest checkpoint? If yes, the run is
    # making headway despite crashes → reset the stall counter. Only count consecutive
    # no-progress cycles toward MAX_STALL. (prev_ckpt=-1 on the first iter → never a false stall.)
    if (( cur_num > prev_ckpt )); then
        stall=0
    else
        stall=$((stall + 1))
    fi
    prev_ckpt=$cur_num
    actual_step=$(trainer_state_step "$cur" 2>/dev/null)

    echo | tee -a "$WATCHDOG_LOG"
    echo "===== [watchdog] cycle $retry  stall=$stall/$MAX_STALL  latest_ckpt=${cur:-none}  actual_step=${actual_step:-0}  target=$MAX_STEPS =====" | tee -a "$WATCHDOG_LOG"

    # Step 1: clear any stale procs (eval server + training)
    cleanup_procs
    pkill -f "serve_gr00t_n17\|_gr00t_eval_client" 2>/dev/null
    sleep 2

    # Step 2: eval any pending ckpts BEFORE training (so we see ckpt-N quality
    # without waiting for next cycle to crash).  Skips ckpts already in CSV.
    eval_unevaluated_ckpts

    # Step 2b: EARLY-STOP — if SR has stopped improving for EARLYSTOP_PATIENCE evals, the
    # peak is behind us; stop (the best ckpt is preserved by KEEP_MULTIPLE pruning).
    if (( INTERLEAVE_STEPS > 0 )) && check_early_stop; then
        echo "[watchdog] ✋ EARLY-STOP — closed-loop SR did not improve for $EARLYSTOP_PATIENCE consecutive evals (peak is past)" | tee -a "$WATCHDOG_LOG"
        break
    fi

    # Step 3: cleanup eval procs + WAIT for GPU memory to actually drain
    # (CUDA driver may keep ~7-9 GB resident for ~30s after python proc dies)
    pkill -f "serve_gr00t_n17\|_gr00t_eval_client" 2>/dev/null
    sleep 3
    wait_gpu_free 3000 90 | tee -a "$WATCHDOG_LOG"

    # Step 4: check if we already hit target after the eval pause
    if [[ -n "$actual_step" ]] && (( actual_step >= MAX_STEPS )); then
        echo "[watchdog] ✅ DONE — latest ckpt-$cur reports global_step=$actual_step >= $MAX_STEPS" | tee -a "$WATCHDOG_LOG"
        break
    fi

    # Step 5: prep ckpt for resume (patch save_steps).
    # NOTE: do NOT drop optimizer.pt here — that was an N1.6/Adam workaround (optimizer.pt
    # ~3GB blew the resume transient alloc). N1.7 uses adafactor → optimizer.pt is ~10MB,
    # so keeping it gives a cleaner, more stable resume (matches the manual resumes that
    # never startup-crashed). Set DROP_OPTIMIZER=1 to re-enable if you ever hit resume OOM.
    if [[ -n "$cur" ]]; then
        [[ "${DROP_OPTIMIZER:-0}" == "1" ]] && drop_optimizer "$cur" | tee -a "$WATCHDOG_LOG"
        patch_save_steps "$cur" | tee -a "$WATCHDOG_LOG"
    fi

    # Step 5b: SETTLE before relaunch. Rapid relaunch after a crash startup-segfaults
    # (exit 139) because the CUDA context/driver hasn't fully reset — manual resumes with
    # a clean, settled GPU never did this. Drain hard (<1800 MiB) then a fixed pause.
    pkill -9 -f "launch_finetune_n17" 2>/dev/null || true
    wait_gpu_free 1800 120 | tee -a "$WATCHDOG_LOG"
    sleep "${RELAUNCH_SETTLE_S:-15}"

    # Step 5c: compute this slice's stop boundary. STOP_AT_STEP is the next INTERLEAVE_STEPS
    # multiple above the resumed step, capped at MAX_STEPS. The launcher's StopAtStep callback
    # stops the slice there while keeping --max_steps at the full target (scheduler intact).
    # INTERLEAVE_STEPS=0 → STOP_AT_STEP=0 → no callback → train straight to MAX_STEPS.
    base_step=${actual_step:-0}
    base_step=${base_step:-0}
    if (( INTERLEAVE_STEPS > 0 )); then
        slice_target=$(( ( base_step / INTERLEAVE_STEPS + 1 ) * INTERLEAVE_STEPS ))
        (( slice_target > MAX_STEPS )) && slice_target=$MAX_STEPS
    else
        slice_target=0
    fi

    # Step 6: train (until StopAtStep boundary, next crash, or MAX_STEPS)
    cycle_log="$WATCHDOG_LOG_DIR/cycle_${retry}_$(date +%H%M%S).log"
    [[ "$slice_target" != "0" ]] && echo "  [watchdog] slice: $base_step → $slice_target (target $MAX_STEPS)" | tee -a "$WATCHDOG_LOG"
    MAX_STEPS="$MAX_STEPS" SAVE_STEPS="$SAVE_STEPS" \
        STOP_AT_STEP="$slice_target" \
        GLOBAL_BATCH="$GLOBAL_BATCH" GRAD_ACCUM="$GRAD_ACCUM" \
        OUTPUT_DIR="$OUTPUT_DIR" \
        bash "$REPO_ROOT/scripts/train_n17.sh" >"$cycle_log" 2>&1
    rc=$?
    last_step=$(grep -oE "checkpoint-[0-9]+" "$cycle_log" | tail -1 | sed 's/checkpoint-//' || echo "?")
    echo "  [watchdog] cycle $retry exit=$rc  ckpt-after-cycle=$last_step" | tee -a "$WATCHDOG_LOG"

    # Custom pruning: keep all multiples of KEEP_MULTIPLE + last KEEP_TEMPORARY others
    prune_checkpoints | tee -a "$WATCHDOG_LOG"

    if [[ "$rc" == "0" ]]; then
        # Clean exit = either a slice boundary (interleave) or the real finish. Re-read the
        # latest checkpoint's global_step to tell them apart.
        fin_ckpt=$(latest_step || echo "")
        fin_step=$(trainer_state_step "$fin_ckpt" 2>/dev/null)
        fin_step=${fin_step:-0}
        if (( fin_step >= MAX_STEPS )); then
            echo "[watchdog] clean exit, training finished (step=$fin_step)" | tee -a "$WATCHDOG_LOG"
            # One last eval pass to cover the final ckpts
            cleanup_procs
            pkill -f "serve_gr00t_n17\|_gr00t_eval_client" 2>/dev/null
            sleep 2
            eval_unevaluated_ckpts
            break
        else
            echo "[watchdog] slice boundary at step=$fin_step (< $MAX_STEPS) — eval + resume next slice" | tee -a "$WATCHDOG_LOG"
            # Loop continues: top-of-loop eval_unevaluated_ckpts evaluates this boundary
            # ckpt, then early-stop is checked, then the next slice trains.
        fi
    fi
    sleep 2
done

cleanup_procs

if (( stall >= MAX_STALL )); then
    echo "[watchdog] ⚠️  gave up — $MAX_STALL consecutive cycles with NO checkpoint progress (genuinely stuck)" | tee -a "$WATCHDOG_LOG"
elif (( retry >= MAX_RETRIES )); then
    echo "[watchdog] ⚠️  reached absolute backstop MAX_RETRIES=$MAX_RETRIES without finishing" | tee -a "$WATCHDOG_LOG"
fi

final=$(latest_step)
echo "[watchdog] final ckpt: checkpoint-$final" | tee -a "$WATCHDOG_LOG"
touch "$OUTPUT_DIR/.training_done"
