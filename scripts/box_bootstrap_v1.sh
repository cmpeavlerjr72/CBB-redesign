#!/usr/bin/env bash
# Box bootstrap, tonight's session (operator 2026-09-30). Runs ON the box, as ec2-user.
#   scripts/box_bootstrap_v1.sh install            # docker + git (copy this file up with scp before the clone exists)
#   scripts/box_bootstrap_v1.sh clone <SHA>        # clone + checkout
#   scripts/box_bootstrap_v1.sh pull               # HF pull in a python:3.12-slim container (needs ~/.hf_env)
#   scripts/box_bootstrap_v1.sh build              # docker build -f Dockerfile.cbb -t cbb-sweep .
#   scripts/box_bootstrap_v1.sh check              # inputs check, thread probe
# ~/.hf_env holds `export HF_TOKEN=...` (written from the operator's machine over ssh stdin, mode 600, never echoed).
set -uo pipefail
REPO=https://github.com/cmpeavlerjr72/CBB-redesign.git
D="$HOME/cbb"
case "${1:-}" in
install)
  sudo dnf install -y docker git >/dev/null && sudo systemctl start docker && sudo usermod -aG docker "$USER"
  sudo chmod 666 /var/run/docker.sock
  docker --version; git --version ;;
clone)
  [ -d "$D/.git" ] || git clone "$REPO" "$D"
  cd "$D" && git fetch origin main && git checkout -q "${2:?SHA}" && git rev-parse HEAD ;;
pull)
  cd "$D" && . ~/.hf_env
  docker run --rm -e HF_TOKEN -v "$PWD:/w" -w /w python:3.12-slim sh -c \
    "pip install -q huggingface_hub==1.31.0 && python scripts/hf_sync_data.py pull --dirs engine_inputs model_artifacts team_rate_tables raw engine_inputs_v3 --max-attempts 6" ;;
build)
  cd "$D" && docker build -f Dockerfile.cbb -t cbb-sweep . ;;
check)
  cd "$D" && . ~/.hf_env
  docker run --rm -v "$PWD:/app" -w /app --entrypoint python cbb-sweep scripts/ops_box_inputs_check_v1.py | tail -5
  docker run --rm -v "$PWD:/app" -w /app --entrypoint python cbb-sweep scripts/ops_lgbm_thread_probe_v1.py ;;
*) sed -n 2,9p "$0"; exit 2 ;;
esac
