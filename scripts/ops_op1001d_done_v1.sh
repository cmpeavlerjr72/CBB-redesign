#!/usr/bin/env bash
# local: write <req>.done.md for a single-arm read. done.sh <req> <title> <window> <tag> <flagregex> <extra text>
cd "$(dirname "$0")/.."; REQ=$1; TITLE=$2; WIN=$3; TAG=$4; FL=$5; EXTRA=${6:-}
G=results/engine_v0/v3full_grade/${TAG}__verified.md
FLAGS=$(.venv/Scripts/python.exe -c "import json;m=json.load(open('results/engine_v0/$TAG/run_meta.json'));print({k:v for k,v in m['adapter_flags'].items() if k.startswith('ENGINE_') and k not in ('ENGINE_EVENT','ENGINE_ROTATION','ENGINE_FG3','ENGINE_FG_MAKE','ENGINE_REBOUND','ENGINE_FREE_THROW','ENGINE_ROTATION_SCHEME','ENGINE_INPUTS_VERSION')})")
TAB=$(.venv/Scripts/python.exe scripts/ops_op1001d_gatetab_v1.py $G docs/tests/v3box_grades_2026-10-01/v3full_COMB9GCTKD_s200_o0__verified.md | cut -c1-230)
cat > docs/ops/box_queue/$REQ.done.md <<EOT
# $REQ DONE ($TITLE) (AWS operator, 2026-10-01)

- Box: spot c7a.48xlarge i-0cd355979d9a32718 (us-east-2c). Clone \`~/cbb\` at \`8eca67b\` (engine identical to \`573aa2e\`; parity vs v9 PASS bit-identical 12:48:45Z). Wall window (Z): $WIN, 96 workers shared with another read.
- Run as written through \`scripts/box_op1001d_jobs_v1.sh\` / \`box_run_v3.sh\` (all \`ENGINE_*\` reach the container).
- run_meta adapter flags (non-default-input flags): \`$FLAGS\`. Expected: $FL
- Synced (scp, no overwrite): \`results/engine_v0/$TAG/\` and \`$G\`.
$EXTRA
- Headline gate rows, arm vs reference (\`v3full_COMB9GCTKD_s200_o0\`, verified). Raw gate tolerances, not Decision 12 floors; the PM rules:

$TAB
EOT
echo wrote docs/ops/box_queue/$REQ.done.md
