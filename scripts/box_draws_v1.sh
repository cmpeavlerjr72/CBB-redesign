#!/usr/bin/env bash
# Stage C draw files on the S1 tag (operator 2026-09-30). Same commands as box_v3_sims_v1.sh `builds` (draw part only).
set -uo pipefail
cd "$(dirname "$0")/.."
M=data/processed/models; TRD=$M/engine_v3_trdraw; B=scripts/box_run_v2.sh
E3=data/processed/team_rate_features_E3_v4.parquet; VAR=data/processed/team_rate_variance_O1a_v3.parquet
mkdir -p $TRD
$B scripts/build_engine_inputs_trdraw_v1.py --inputs-dir $M/engine_v3_S1 --variance none --K 1 --table $E3 --out $TRD/S1_K1
$B scripts/build_engine_inputs_trdraw_v1.py --inputs-dir $M/engine_v3_S1 --variance e3 --K 64 --seed 20260930 --table $E3 --out $TRD/S2_e3_K64
$B scripts/build_engine_inputs_trdraw_v1.py --inputs-dir $M/engine_v3_S1 --variance o1a --K 64 --seed 20260930 --table $E3 --variance-table $VAR --out $TRD/S3_o1a_K64
ls -la $TRD
