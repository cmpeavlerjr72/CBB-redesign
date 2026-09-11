#!/usr/bin/env bash
# CBB engine AWS sweep entrypoint: parity gate -> chunked seed sweep -> push.
#
# Adapted from cfb-props-sim's scripts/cloud/entrypoint.sh, with one structural
# change forced by this repo's tools: `scripts/run_engine.py` is NOT
# chunk-resumable the way cfb's `build_sweep_dist_v10.py` is (that script
# resumes from an existing output parquet on every invocation; this one always
# writes a single results dir once, at the end of the process, with no
# incremental checkpoint file). So instead of one process handling internal
# chunking, THIS script chunks by running run_engine.py multiple times, each
# with its own --seed-offset and its own --tag suffix, pushing after each
# chunk. A spot kill costs at most one chunk's seeds, same guarantee cfb's
# --chunk gives, achieved without changing the engine.
#
#   docker run --rm -e HF_TOKEN="$HF_TOKEN" -v "$PWD/out:/out" cbb-sweep \
#       --tag F2_2025_s200_r2event --seeds 200 --chunk-seeds 50 --workers 96
#
# See docs/ops/aws_launch_chain.md for the full rollover procedure, the
# parity gate details, and the cost table.
set -euo pipefail

APP="${APP_DIR:-/app}"
OUT="${OUT_DIR:-/out}"
PY="${PYTHON:-python}"

TAG=""
FOLD="F2"
SEASON=2025
SEEDS=200
CHUNK_SEEDS=50          # 0 = one single call for all seeds (no intermediate pushes)
SEED_OFFSET_START=0
WORKERS=""              # default: nproc
GAMES_PER_BLOCK=60
SEEDS_PER_BLOCK=25
MAX_GAMES=0             # 0 = all games in the fold
ENGINE_EVENT="round2_s1"
ENGINE_CLOCK="reference"
ENGINE_ROTATION="reference"
ENGINE_FG3="decision8"
PARITY="gate"           # gate | only | skip
PARITY_REF="docs/ops/parity_reference_windows.json"
PARITY_GAMES=60
PARITY_SEEDS=5
PUSH="on"               # on | off  (hf_sync_data.py push --dirs results)
BOOTSTRAP="none"        # none | raw | results | engine_inputs

usage() {
  cat <<'EOF'
Usage: run_aws_sweep.sh --tag TAG [options]

  --tag TAG              REQUIRED. Base tag; each chunk writes
                         results/engine_v0/TAG_off<N>_n<K>/ and the parity
                         smoke writes results/engine_v0/TAG_paritycheck/.
  --fold F                default F2
  --season N              default 2025
  --seeds N               total seeds across all chunks (default 200)
  --chunk-seeds N         seeds per run_engine.py invocation (default 50).
                         0 = one call for all --seeds (no incremental push).
  --seed-offset-start N   first seed index (default 0)
  --workers N             joblib/process workers (default: nproc)
  --games-per-block N     default 60 (RESUME.md guidance: small blocks so a
                         time-budget cutoff still returns usable output)
  --seeds-per-block N     default 25
  --max-games N           0 = all games (default). Set for a smoke subset.
  --engine-event MODE     reference | round2_s1 (default round2_s1, the
                         adopted round-2 winners)
  --engine-clock MODE     reference | reference_empirical (default reference)
  --engine-rotation MODE  reference (only wired mode; default reference)
  --engine-fg3 MODE       decision8 | artifact (default decision8)
  --parity gate|only|skip default gate. "only" runs the parity smoke +
                         digest compare and exits before any real sweep --
                         run this FIRST on a fresh box. "skip" requires the
                         gate to have already passed on this exact image in
                         this session; it is not a way past a failure.
  --parity-ref PATH       reference digest (default
                         docs/ops/parity_reference_windows.json, emitted on
                         the home Windows box by scripts/digest_engine_run.py
                         --emit)
  --parity-games N        smoke game count (default 60)
  --parity-seeds N        smoke seed count (default 5)
  --push on|off           hf_sync_data.py push --dirs results after each
                         chunk and at the end (default on). NOTE: that push
                         uploads the WHOLE local results/ tree as one commit
                         (upload_folder, not per-file incremental) -- see
                         docs/ops/aws_launch_chain.md section 4.
  --bootstrap none|raw|results|engine_inputs
                         pull a gitignored bulk dir from HF before running
                         (default none). NOTE: `engine_inputs`
                         (data/processed/models/engine/, incl.
                         event_round2_s1_*/) must already be present at
                         `docker build` time -- Dockerfile.cbb's build-time
                         checks run before this entrypoint ever executes, so
                         this flag does not substitute for
                         `hf_sync_data.py pull --dirs engine_inputs` before
                         the build. It exists for re-pulling a fresher copy
                         at container start if the image was built with an
                         older one, or for non-Docker use.
  -h, --help              this text

Environment: HF_TOKEN required when --push on or --bootstrap != none.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --tag)               TAG="$2"; shift 2 ;;
    --fold)              FOLD="$2"; shift 2 ;;
    --season)            SEASON="$2"; shift 2 ;;
    --seeds)             SEEDS="$2"; shift 2 ;;
    --chunk-seeds)       CHUNK_SEEDS="$2"; shift 2 ;;
    --seed-offset-start) SEED_OFFSET_START="$2"; shift 2 ;;
    --workers)           WORKERS="$2"; shift 2 ;;
    --games-per-block)   GAMES_PER_BLOCK="$2"; shift 2 ;;
    --seeds-per-block)   SEEDS_PER_BLOCK="$2"; shift 2 ;;
    --max-games)         MAX_GAMES="$2"; shift 2 ;;
    --engine-event)      ENGINE_EVENT="$2"; shift 2 ;;
    --engine-clock)      ENGINE_CLOCK="$2"; shift 2 ;;
    --engine-rotation)   ENGINE_ROTATION="$2"; shift 2 ;;
    --engine-fg3)        ENGINE_FG3="$2"; shift 2 ;;
    --parity)            PARITY="$2"; shift 2 ;;
    --parity-ref)        PARITY_REF="$2"; shift 2 ;;
    --parity-games)      PARITY_GAMES="$2"; shift 2 ;;
    --parity-seeds)      PARITY_SEEDS="$2"; shift 2 ;;
    --push)              PUSH="$2"; shift 2 ;;
    --bootstrap)         BOOTSTRAP="$2"; shift 2 ;;
    -h|--help)           usage; exit 0 ;;
    *) echo "run_aws_sweep: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$TAG" ]; then
  echo "run_aws_sweep: --tag is required" >&2; usage >&2; exit 2
fi
case "$TAG" in
  *[!A-Za-z0-9_-]*) echo "run_aws_sweep: --tag must be [A-Za-z0-9_-] only" >&2; exit 2 ;;
esac

cd "$APP"
mkdir -p "$OUT"
LOG="$OUT/${TAG}.log"
exec > >(tee -a "$LOG") 2>&1

: "${WORKERS:=$(nproc)}"

STATUS_FILE="$OUT/STATUS_${TAG}.txt"
PHASE="init"
status() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  $1" >> "$STATUS_FILE"; }

sync_results_to_out() {   # never fails the caller; /out is the durable side
  mkdir -p "$OUT/results"
  cp -rf "$APP/results/." "$OUT/results/" 2>/dev/null || true
}

push_evidence() {   # never fails the caller
  if [ "$PUSH" = "on" ] && [ -n "${HF_TOKEN:-}" ]; then
    sync_results_to_out
    $PY -u scripts/hf_sync_data.py push --dirs results --max-attempts 6 \
      || echo "[cloud] WARNING: results push failed (continuing)"
  fi
  cp -f "$STATUS_FILE" "$OUT/" 2>/dev/null || true
  cp -f "$LOG" "$OUT/" 2>/dev/null || true
}

on_exit() {
  rc=$?
  if [ "$rc" -eq 0 ]; then
    status "END COMPLETE"
  else
    status "END FAILED exit=$rc phase=$PHASE"
    echo "[cloud] EXIT $rc during phase '$PHASE'" >&2
  fi
  push_evidence
}
trap on_exit EXIT
# See cfb-props-sim/docs/ops/cloud_sweep_runbook.md section on the boxtest1
# kill test: a signal kill must never masquerade as success, and bash defers
# traps while a foreground command runs -- so long-running calls are
# backgrounded + wait'ed (wait is interruptible), and the child is killed
# explicitly so the evidence push does not compete with a full worker fleet.
CHILD=""
trap 'if [ -n "$CHILD" ]; then kill -TERM "$CHILD" 2>/dev/null; fi; exit 143' TERM INT HUP

echo "=============================================================="
echo "[cloud] tag=$TAG fold=$FOLD season=$SEASON seeds=$SEEDS chunk=$CHUNK_SEEDS workers=$WORKERS"
echo "[cloud] adapters: EVENT=$ENGINE_EVENT CLOCK=$ENGINE_CLOCK ROTATION=$ENGINE_ROTATION FG3=$ENGINE_FG3"
echo "[cloud] started $(date -u +%Y-%m-%dT%H:%M:%SZ)  host=$(uname -srm)  nproc=$(nproc)"
echo "[cloud] git $(git rev-parse HEAD 2>/dev/null || echo unknown)"
PYTHONIOENCODING=utf-8 $PY - <<'PYVER'
import joblib, lightgbm, numpy, pandas, platform, scipy, sklearn
print(f"[cloud] python {platform.python_version()} | numpy {numpy.__version__} | "
      f"lightgbm {lightgbm.__version__} | pandas {pandas.__version__} | "
      f"scikit-learn {sklearn.__version__} | joblib {joblib.__version__} | "
      f"scipy {scipy.__version__}")
PYVER
echo "[cloud] threads: OMP=${OMP_NUM_THREADS:-unset} MKL=${MKL_NUM_THREADS:-unset} " \
     "HASHSEED=${PYTHONHASHSEED:-unset}"
echo "=============================================================="

if [ "$PUSH" = "on" ] || [ "$BOOTSTRAP" != "none" ]; then
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "[cloud] FATAL: HF_TOKEN not set, required for --push on / --bootstrap $BOOTSTRAP." >&2
    echo "[cloud] Pass it at RUN time: docker run -e HF_TOKEN=... (never in the image)." >&2
    exit 2
  fi
fi

status "RUNNING tag=$TAG seeds=$SEEDS chunk=$CHUNK_SEEDS workers=$WORKERS"
status "config git=$(git rev-parse HEAD 2>/dev/null || echo unknown) fold=$FOLD season=$SEASON"

# ---- 0. bootstrap (rarely needed -- data/processed is git-tracked except
#         event_round2_s1_*/, which must already be in the image; see the
#         Dockerfile.cbb build-time check) --------------------------------
PHASE="bootstrap"
case "$BOOTSTRAP" in
  none)    echo "[cloud] bootstrap: none (tracked data/processed + the image's own engine_inputs is the input set)" ;;
  raw)     echo "[cloud] bootstrap: pulling data/raw from HF"; $PY -u scripts/hf_sync_data.py pull --dirs raw --max-attempts 6 ;;
  results) echo "[cloud] bootstrap: pulling results from HF"; $PY -u scripts/hf_sync_data.py pull --dirs results --max-attempts 6 ;;
  engine_inputs) echo "[cloud] bootstrap: re-pulling data/processed/models/engine from HF"; $PY -u scripts/hf_sync_data.py pull --dirs engine_inputs --max-attempts 6 ;;
  *) echo "[cloud] FATAL: --bootstrap must be none|raw|results|engine_inputs" >&2; exit 2 ;;
esac

# ---- 1. PARITY GATE -------------------------------------------------------
PHASE="parity-gate"
case "$PARITY" in
  gate|only)
    echo "[cloud] --- PARITY GATE: ${PARITY_GAMES} games x ${PARITY_SEEDS} seeds ---"
    PARITY_TAG="${TAG}_paritycheck"
    $PY -u scripts/run_engine.py --fold "$FOLD" --season "$SEASON" \
        --seeds "$PARITY_SEEDS" --max-games "$PARITY_GAMES" --workers "$WORKERS" \
        --games-per-block "$PARITY_GAMES" --seeds-per-block "$PARITY_SEEDS" \
        --tag "$PARITY_TAG" --results-dir results/engine_v0
    ENGINE_EVENT="$ENGINE_EVENT" ENGINE_CLOCK="$ENGINE_CLOCK" \
    ENGINE_ROTATION="$ENGINE_ROTATION" ENGINE_FG3="$ENGINE_FG3" \
    $PY -u scripts/digest_engine_run.py --compare "$PARITY_REF" \
        --results "results/engine_v0/${PARITY_TAG}"
    rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "[cloud] FATAL: parity gate FAILED. No sweep, no upload." >&2
      echo "[cloud] Take the printed field diff to the PM. Do not 'just run it anyway'," >&2
      echo "[cloud] and do not loosen the comparison -- tolerance is a PM decision." >&2
      exit 4
    fi
    status "PARITY PASS"
    echo "[cloud] --- parity PASS ---"
    if [ "$PARITY" = "only" ]; then
      echo "[cloud] --parity only: gate passed, exiting without simulating."
      exit 0
    fi ;;
  skip)
    echo "[cloud] WARNING: parity gate SKIPPED by request. Only defensible if the" \
         "gate already passed on this exact image in this session." ;;
  *) echo "[cloud] FATAL: --parity must be gate|only|skip" >&2; exit 2 ;;
esac

# ---- 2. the sweep, chunked by repeated run_engine.py invocations ----------
# run_engine.py itself has no incremental checkpoint (unlike cfb's sweep
# runner, it writes its results parquet exactly once, at process end), so
# resumability comes from chunking BETWEEN invocations, not within one.
done_seeds=0
if [ "$CHUNK_SEEDS" -le 0 ]; then
  CHUNK_SEEDS="$SEEDS"
fi
T0=$(date +%s)
while [ "$done_seeds" -lt "$SEEDS" ]; do
  this=$(( SEEDS - done_seeds < CHUNK_SEEDS ? SEEDS - done_seeds : CHUNK_SEEDS ))
  offset=$(( SEED_OFFSET_START + done_seeds ))
  subtag="${TAG}_off${offset}_n${this}"
  PHASE="sweep-${subtag}"
  echo "[cloud] --- chunk: seeds [$offset, $((offset+this)) ) -> $subtag --- $(date -u +%H:%M:%SZ)"
  ENGINE_EVENT="$ENGINE_EVENT" ENGINE_CLOCK="$ENGINE_CLOCK" \
  ENGINE_ROTATION="$ENGINE_ROTATION" ENGINE_FG3="$ENGINE_FG3" \
  $PY -u scripts/run_engine.py --fold "$FOLD" --season "$SEASON" \
      --seeds "$this" --seed-offset "$offset" --workers "$WORKERS" \
      --games-per-block "$GAMES_PER_BLOCK" --seeds-per-block "$SEEDS_PER_BLOCK" \
      --max-games "$MAX_GAMES" --tag "$subtag" --results-dir results/engine_v0 &
  CHILD=$!
  rc=0
  wait "$CHILD" || rc=$?
  CHILD=""
  if [ "$rc" -ne 0 ]; then
    echo "[cloud] run_engine.py chunk exited rc=$rc" >&2
    exit "$rc"
  fi
  done_seeds=$(( done_seeds + this ))
  status "CHUNK done_seeds=$done_seeds/$SEEDS tag=$subtag"
  sync_results_to_out
  if [ "$PUSH" = "on" ]; then
    $PY -u scripts/hf_sync_data.py push --dirs results --max-attempts 6 \
      || echo "[cloud] WARNING: per-chunk push failed (continuing)"
  fi
done
T1=$(date +%s)
echo "[cloud] sweep complete: $done_seeds/$SEEDS seeds in $(( (T1-T0)/60 )) min $(( (T1-T0)%60 ))s"

# ---- 3. final summary + archive push --------------------------------------
PHASE="summary"
$PY - "$TAG" <<'PYSUM'
import glob, json, sys
tag = sys.argv[1]
import pandas as pd
total_g = total_p = 0
for d in sorted(glob.glob(f"results/engine_v0/{tag}_off*_n*")):
    gp = f"{d}/games.parquet"
    try:
        g = pd.read_parquet(gp, columns=["game_id", "seed"])
        total_g += len(g)
        print(f"[cloud] {d}: {len(g):,} (game,seed) rows, "
              f"{g['game_id'].nunique()} games, {g['seed'].nunique()} seeds")
    except Exception as e:
        print(f"[cloud] {d}: could not read ({e})")
print(f"[cloud] TOTAL across chunks: {total_g:,} (game,seed) rows")
PYSUM

if [ "$PUSH" = "on" ]; then
  echo "[cloud] --- final push: hf_sync_data.py push --dirs results ---"
  $PY -u scripts/hf_sync_data.py push --dirs results --max-attempts 6
  status "ARCHIVED results/ (all chunks under tag $TAG)"
else
  echo "[cloud] push disabled; results are in $APP/results/engine_v0/ and mirrored to $OUT/results/"
fi

echo "=============================================================="
echo "[cloud] PHASE SUMMARY  sweep=OK  seeds=$done_seeds/$SEEDS  push=$([ "$PUSH" = on ] && echo OK || echo off)"
echo "[cloud] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================="
