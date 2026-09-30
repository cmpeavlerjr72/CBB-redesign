#!/usr/bin/env bash
# Run any repo command inside the cbb-sweep image with the HOST clone bind-mounted over /app.
# The image is only the pinned Python environment (lane J 2026-09-30, aws_launch_chain.md
# section 17); code, data/ and results/ come from the host clone, so HF-pulled data needs no
# rebuild and outputs land on the host for `hf_sync_data.py push`.
#
#   scripts/box_run.sh scripts/train_rebound_v3_par_v1.py --stage 2 ...
#   scripts/box_run.sh scripts/hf_sync_data.py push --dirs model_artifacts results --max-attempts 6
#   BOX_CMD=bash scripts/box_run.sh          # not supported; use `docker run -it` by hand
#
# HF_TOKEN is passed by NAME (`-e HF_TOKEN`), never on the command line or in the image.
# Thread env vars are the image's baked pins (1); the par_v1 trainers also force them.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${CBB_IMAGE:-cbb-sweep}"
exec docker run --rm \
  -e HF_TOKEN -e ENGINE_INPUTS_VERSION -e ENGINE_CLOCK -e ENGINE_EVENT -e ENGINE_FG_MAKE \
  -e ENGINE_REBOUND -e ENGINE_ROTATION -e ENGINE_FG3 -e ENGINE_FREE_THROW \
  -v "$PWD:/app" -w /app --entrypoint python "$IMAGE" -u "$@"
