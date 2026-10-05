#!/usr/bin/env bash
# chain_ft_prior_loop_v1.sh -- free_throw section 20 DESCRIPTIVE closed loop for P1 (not a winner). Player-FT worker 2026-10-05.
# parity (flag off, v9) -> F2/F1 opening windows (P1 vs existing player_day1 srv o0/o1000) -> F2/F1 every-11th season samples.
set -u
cd /c/Users/devuser/CBB-clean-sheet
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
W=16
R=results/ft_prior/runs
ts(){ date +%H:%M:%S; }
echo "[chain] start $(ts)"
for IN in engine_v3 engine_v3_FTP_P1; do
  $PY scripts/run_engine.py --fold F2 --season 2025 --seeds 5 --max-games 60 --workers 6 \
     --input-dir data/processed/models/$IN --tag ftprior_parity_${IN} > $R/parity_${IN}.log 2>&1
  $PY scripts/digest_engine_run.py --compare docs/ops/parity_reference_windows_v9.json --results results/engine_v0/ftprior_parity_${IN} > $R/parity_${IN}_cmp.log 2>&1
  echo "[chain] parity $IN rc=$? $(ts)"; tail -2 $R/parity_${IN}_cmp.log
done
F1OV=data/processed/models/fold1_v1/overrides_V2.json
run_f2(){ # tag ids input flag offset
  ENGINE_FT_SCORE=$4 $PY scripts/run_engine_window_v1.py --ids-file $2 --fold F2 --season 2025 --input-dir data/processed/models/$3 \
    --seeds 50 --seed-offset $5 --workers $W --tag $1 --results-dir $R > $R/$1.log 2>&1; echo "[chain] $1 rc=$? $(ts)"; }
run_f1(){
  ENGINE_ROTATION_SCHEME=static ENGINE_FT_SCORE=$4 $PY scripts/run_engine_window_v1.py --ids-file $2 --fold F1 --season 2024 \
    --input-dir data/processed/models/$3 --seeds 50 --seed-offset $5 --workers $W --tag $1 --results-dir $R --overrides $F1OV > $R/$1.log 2>&1
  echo "[chain] $1 rc=$? $(ts)"; }
WF2=data/processed/models/engine_v3_anonwin/window_ids_F2.parquet
WF1=data/processed/models/engine_v3_anonwin_f1/window_ids_F1.parquet
SF2=results/ft_prior/sample_F2_2025_every11.parquet
SF1=data/processed/models/fold1_v1/sample_F1_2024_every11.parquet
run_f2 F2win_P1_o0 $WF2 engine_v3_FTP_P1 P1 0
run_f1 F1win_P1_o0 $WF1 engine_v3_f1_FTP_P1 P1 0
run_f2 F2smp_P1_o0 $SF2 engine_v3_FTP_P1 P1 0
run_f2 F2smp_srv_o0 $SF2 engine_v3 reference 0
run_f2 F2smp_srv_o1000 $SF2 engine_v3 reference 1000
run_f1 F1smp_P1_o0 $SF1 engine_v3_f1_FTP_P1 P1 0
run_f1 F1smp_srv_o0 $SF1 engine_v3_f1 reference 0
run_f1 F1smp_srv_o1000 $SF1 engine_v3_f1 reference 1000
echo "[chain] DONE $(ts)"
