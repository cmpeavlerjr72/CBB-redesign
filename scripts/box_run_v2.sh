#!/usr/bin/env bash
# box_run_v2.sh -- versioned sibling of scripts/box_run.sh (lane J, not edited). Lane G 2026-09-30.
# Adds (1) BOX_DOCKER_ARGS: extra `docker run` flags, word-split, for the tagged-input overlay bind mounts
# (`engine_v3_<tag>/docker_mounts.txt`), and (2) pass-through of the default-off engine switches and the truth switch.
#   BOX_DOCKER_ARGS="$(sed "s#\$PWD#$PWD#g" data/processed/models/engine_v3_S1/docker_mounts.txt | tr '\n' ' ')" \
#     ENGINE_SHOT_BLOCK=K2_Ocell scripts/box_run_v2.sh scripts/run_po4b_closed_loop_sample_v1.py ...
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${CBB_IMAGE:-cbb-sweep}"
exec docker run --rm \
  -e HF_TOKEN -e ENGINE_INPUTS_VERSION -e ENGINE_CLOCK -e ENGINE_EVENT -e ENGINE_FG_MAKE \
  -e ENGINE_REBOUND -e ENGINE_ROTATION -e ENGINE_FG3 -e ENGINE_FREE_THROW \
  -e ENGINE_SHOT_BLOCK -e ENGINE_FOUL_JOINT -e ENGINE_TEAM_RATE_DRAW -e ENGINE_FOUL_ACCRUAL -e CBB_TRUTH \
  ${BOX_DOCKER_ARGS:-} \
  -v "$PWD:/app" -w /app --entrypoint python "$IMAGE" -u "$@"
