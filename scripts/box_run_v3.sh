#!/usr/bin/env bash
# box_run_v3.sh -- versioned sibling of box_run_v2.sh (AWS operator 2026-10-01 day): passes EVERY ENGINE_* variable
# in the caller's environment through to the container (v2 had a fixed list that misses ENGINE_CHANCE_TIME,
# ENGINE_SHARED_SHOOTING, ENGINE_EVENT_TEAM_BLOCK, ENGINE_LATE_GAME, ...), plus HF_TOKEN, CBB_TRUTH, BOX_DOCKER_ARGS.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${CBB_IMAGE:-cbb-sweep}"
ENVARGS=()
for v in $(env | grep -o '^ENGINE_[A-Za-z0-9_]*' || true); do ENVARGS+=(-e "$v"); done
exec docker run --rm -e HF_TOKEN -e CBB_TRUTH -e PYTHONIOENCODING=utf-8 ${ENVARGS[@]+"${ENVARGS[@]}"} \
  ${BOX_DOCKER_ARGS:-} \
  -v "$PWD:/app" -w /app --entrypoint python "$IMAGE" -u "$@"
