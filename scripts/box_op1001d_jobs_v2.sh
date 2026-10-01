#!/usr/bin/env bash
# operator 2026-10-01 day: request jobs, run in ~/cbb (all request commits are ancestors of the clone's SHA). jobs.sh <JOB> <WORKERS>
set -uo pipefail
cd "${BOXDIR:-$HOME/cbb}"; . ~/.hf_env; export HF_TOKEN PYTHONIOENCODING=utf-8 CBB_TRUTH=verified_v1
J="${1:?job}"; W="${2:-90}"; ts() { date -u +%H:%M:%SZ; }
R=scripts/box_run_v3.sh; RE=results/engine_v0
G() { $R scripts/eval_gates.py --results $RE/$1 --season ${2:-2025} --out $RE/v3full_grade/$1__verified.md; }
chunks() { # chunks <tag> <first_offset> [extra env handled by caller]
  local T=$1 O0=$2 k o sub
  for k in 0 1 2 3 4 5 6 7; do o=$((O0+25*k)); sub=${T}_off${o}_n25; [ -f $RE/$sub/run_meta.json ] && continue
    echo "[chunk] $sub $(ts)"; $R scripts/run_engine.py --fold F2 --season 2025 --seeds 25 --seed-offset $o --workers $W \
      --games-per-block 60 --seeds-per-block 25 --tag $sub --results-dir $RE --input-dir data/processed/models/engine_v3 || { echo "CHUNK FAILED $sub"; return 4; }; done
  $R scripts/concat_engine_runs.py --tag $T --results-dir $RE --overwrite; }
clockarm() { local MODE=$1 T=$2; ( export ENGINE_CLOCK=$MODE; chunks $T 0 ) || return $?
  python3 -c "import json;print('  ENGINE_CLOCK in run_meta:', json.load(open('$RE/$T/run_meta.json'))['adapter_flags'])"; G $T; }
BS() { $R scripts/ops_pair_bootstrap_v1.py --results-dir $RE "$@"; }
FL=d1001D_S2f1_s200_o1000,d1001D_S2f2_s200_o2000,d1001D_S2f3_s200_o3000,d1001D_S2f4_s200_o4000
mkdir -p results/d1001D results/d1001I results/laneF results/laneI_1001 $RE/v3full_grade
case "$J" in
D1_floors) bash scripts/box_op1001d_floors_v2.sh ~/cbb $W 1 2 3 4 ;;
D1_retrain)
  B=data/processed/models/full_retrain_v1
  [ -d $B/FRa_box_v1 ] || { sudo cp -r $B/FR_box_v1 $B/FRa_box_v1 && sudo rm -rf $B/FRa_box_v1/gate; }
  [ -d $B/FTa_box_v2 ] || { sudo cp -r $B/FT_box_v2 $B/FTa_box_v2 && sudo rm -rf $B/FTa_box_v2/gate; }
  sudo chown -R ec2-user $B
  NOISE=$(ls $RE/v3full_grade/d1001D_S2f{1,2,3,4}_s200_o*__verified.md | paste -sd, -)
  REF=docs/tests/v3box_grades_2026-10-01/v3full_COMB9GCTKD_s200_o0__verified.md
  for V in "F_R FRa_box_v1" "F_T FTa_box_v2"; do set -- $V
    $R scripts/chain_full_retrain_v1.py --variant $1 --tag $2 --stages parity,gate --gate-stack adopted --cores $W --gate-mode full --gate-seeds 200 --gate-offsets 0 --gate-workers $W --gate-ref $REF --gate-noise $NOISE || echo "[chain $2] rc=$?"; done
  BS --ref v3full_COMB9GCTKD_s200_o0 --floors $FL --arms fr1_FRa_box_v1_full_s200_o0_ev4_sv2,fr1_FTa_box_v2_full_s200_o0_ev4_sv2 --out-json results/d1001D/boot_vs_S2.json --out-md results/d1001D/boot_vs_S2.md ;;
H1) clockarm v5b_r8M2D_glat_pmean laneH_v3full_M2D_s200_o0 ;;
H2) clockarm v5b_r8K2M_glat_pmean laneH_v3full_K2M_s200_o0 ;;
H3) clockarm v5b_r8K2_glat_pmean laneH_v3full_K2_s200_o0 ;;
F1) T=laneF_v3in_s200_o0; ( export ENGINE_SHOT_BLOCK=K2_Ocell_v3in; $R scripts/run_engine.py --fold F2 --season 2025 --seeds 200 --seed-offset 0 --workers $W --games-per-block 60 --seeds-per-block 25 --input-dir data/processed/models/engine_v3 --tag $T --results-dir $RE ) && G $T
  BS --ref v3full_COMB9GCTKD_s200_o0 --floors $FL --arms $T --out-json results/laneF/boot_v3in_vs_S2.json --out-md results/laneF/boot_v3in_vs_S2.md ;;
I1)
  M=data/processed/models
  $R scripts/build_engine_inputs_v3_tag_v1.py --tag I_RBTO --team-rate-table data/processed/team_rate_features_E3_v4.parquet --no-table-for po,fg --rb-artifacts $M/rebound/round_stageb/TO/team_rate_features_E3_v4/artifacts/s2_F2_O_seed0 --team-rate-missing raise || exit 11
  $R scripts/build_engine_anchor_offsets_v1.py --input-dir $M/engine_v3_I_RBTO --families rb || exit 12
  $R scripts/diag_rbto_parity_v1.py $M/engine_v3_I_RBTO results/laneI_1001/rbto_parity_box.json || exit 13
  $R scripts/build_engine_inputs_v3_ftbio_v1.py --tag I_N1 || exit 14
  ( export BOX_DOCKER_ARGS="-e CBB_NJOBS=1"; $R scripts/train_free_throw_v3_newcomer_artifacts.py --arm N1 --parity-input-dir $M/engine_v3_I_N1 ) || exit 15
  echo '{"adapters.RB_S1_MANIFEST": "data/processed/models/engine_v3_I_RBTO/overlay/data/processed/models/rebound/s1_confirm/S1_weekly/F2/manifest.json"}' > results/laneI_1001/ov_rbto.json
  echo '{"adapters.FT_S1_MANIFEST": "data/processed/models/free_throw/laneI_N1/S1_conf_aligned/F2/manifest.json"}' > results/laneI_1001/ov_n1.json
  ( export ENGINE_SEASON_ANCHOR=/app/data/processed/models/engine_v3_I_RBTO/anchor_offsets_F2_2025.npz
    $R scripts/run_engine_overlay_v1.py --overrides results/laneI_1001/ov_rbto.json --runner full -- --fold F2 --season 2025 --seeds 200 --seed-offset 0 --workers $W --games-per-block 60 --seeds-per-block 25 --input-dir data/processed/models/engine_v3_I_RBTO --tag d1001I_RBTO_s200_o0 --results-dir $RE ) || exit 16
  $R scripts/run_engine_overlay_v1.py --overrides results/laneI_1001/ov_n1.json --runner full -- --fold F2 --season 2025 --seeds 200 --seed-offset 0 --workers $W --games-per-block 60 --seeds-per-block 25 --input-dir data/processed/models/engine_v3_I_N1 --tag d1001I_N1_s200_o0 --results-dir $RE || exit 17
  G d1001I_RBTO_s200_o0; G d1001I_N1_s200_o0
  BS --ref v3full_COMB9GCTKD_s200_o0 --floors $FL --arms d1001I_RBTO_s200_o0,d1001I_N1_s200_o0 --out-json results/d1001I/boot_vs_S2.json --out-md results/d1001I/boot_vs_S2.md ;;
I2)
  $R scripts/build_engine_inputs_v3_A1_v1.py || exit 11
  echo '{}' > results/laneI_1001/ov_ref.json
  $R scripts/run_engine_overlay_v1.py --overrides results/laneI_1001/ov_ref.json --runner full -- --fold F2 --season 2025 --seeds 200 --seed-offset 0 --workers $W --games-per-block 60 --seeds-per-block 25 --input-dir data/processed/models/engine_v3_I_A1 --tag d1001I_A1_s200_o0 --results-dir $RE || exit 16
  G d1001I_A1_s200_o0
  BS --ref v3full_COMB9GCTKD_s200_o0 --floors $FL --arms d1001I_A1_s200_o0 --out-json results/d1001I/boot_A1_vs_S2.json --out-md results/d1001I/boot_A1_vs_S2.md ;;
D2)
  $R scripts/chain_fold1_v1.py --stages gate --gate-mode full --gate-seeds 200 --gate-offsets 0 --gate-workers $W --cores $W || echo "[f1 o0] rc=$?"
  $R scripts/chain_fold1_v1.py --stages gate --gate-mode full --gate-seeds 200 --gate-offsets 0,1000,2000,3000,4000 --gate-workers $W --cores $W || echo "[f1 all] rc=$?"
  CBB_TRUTH=verified_v1 $R scripts/ops_pair_bootstrap_season_v1.py --season 2024 -- --results-dir $RE --ref f1c_V1_full_s200_o0 --floors f1c_V1_full_s200_o1000,f1c_V1_full_s200_o2000,f1c_V1_full_s200_o3000,f1c_V1_full_s200_o4000 --arms f1c_V2_full_s200_o0 --out-json results/d1001D/boot_f1_V2_vs_V1.json --out-md results/d1001D/boot_f1_V2_vs_V1.md ;;
L1) bash scripts/box_late_game_r3_v1.sh tierA ~/cbb $W clk_Dt clk_Dtt clk_D ;;
B1)
  for ARM in FL1 FN FL; do T=d1001B_${ARM}_s200_o0; [ -f $RE/$T/run_meta.json ] || { if [ $ARM = FN ]; then FORM=reference; else FORM=$ARM; fi
    ( export ENGINE_FT_SCORE=FTn ENGINE_TEAM_FORM=$FORM; $R scripts/run_engine.py --fold F2 --season 2025 --seeds 200 --seed-offset 0 --workers $W --games-per-block 60 --seeds-per-block 25 --input-dir data/processed/models/engine_v3 --tag $T --results-dir $RE ) || echo "[B1 $ARM] rc=$?"; }
    G $T; grep -o '"ENGINE_FT_SCORE": "[^"]*"\|"ENGINE_TEAM_FORM": "[^"]*"' $RE/$T/run_meta.json | sort -u | tr '\n' ' '; echo; done
  BS --ref v3full_COMB9GCTKD_s200_o0 --floors $FL --arms d1001B_FL1_s200_o0,d1001B_FN_s200_o0,d1001B_FL_s200_o0 --out-json results/d1001B/boot_r16_vs_S2.json --out-md results/d1001B/boot_r16_vs_S2.md ;;
D3)
  B=data/processed/models/full_retrain_v1
  ENVS="--env ENGINE_EVENT=round2_s1 --env ENGINE_CLOCK=v5b_glat_pmean --env ENGINE_ROTATION=reference --env ENGINE_FG3=decision8 --env CBB_TRUTH=verified_v1 --env ENGINE_SHOT_BLOCK=K2_Ocell --env ENGINE_FOUL_JOINT=R8a --env ENGINE_SHARED_SHOOTING=G3 --env ENGINE_CHANCE_TIME=KD"
  for T in FR_box_v1:FRa_R8a FT_box_v2:FTa_R8a; do SRC=${T%%:*}; NAME=${T##*:}
    $R scripts/run_with_env_v1.py $ENVS -- scripts/run_engine_overlay_v1.py --overrides $B/$SRC/inputs/set/inputs/overrides.json --runner full -- --fold F2 --season 2025 --seeds 200 --seed-offset 0 --workers $W --games-per-block 60 --seeds-per-block 25 --tag d1001D_${NAME}_s200_o0 --results-dir $RE --input-dir $B/$SRC/inputs/set/inputs || echo "[D3 $NAME] rc=$?"
    G d1001D_${NAME}_s200_o0; done
  BS --ref v3full_COMB9GCTKD_s200_o0 --floors $FL --arms d1001D_FRa_R8a_s200_o0,d1001D_FTa_R8a_s200_o0 --out-json results/d1001D/boot_R8a_vs_S2.json --out-md results/d1001D/boot_R8a_vs_S2.md ;;
F2)
  export ENGINE_OT_STATS=1
  B=data/processed/models/full_retrain_v1
  REF=docs/tests/v3box_grades_2026-10-01/v3full_COMB9GCTKD_s200_o0__verified.md
  NOISE=$(ls $RE/v3full_grade/d1001D_S2f{1,2,3,4}_s200_o*__verified.md | paste -sd, -)
  [ -d $B/FRb_box_v1 ] || { sudo cp -r $B/FR_box_v1 $B/FRb_box_v1 && sudo rm -rf $B/FRb_box_v1/gate; }
  [ -d $B/FTb_box_v2 ] || { sudo cp -r $B/FT_box_v2 $B/FTb_box_v2 && sudo rm -rf $B/FTb_box_v2/gate; }
  sudo chown -R ec2-user $B
  $R scripts/chain_full_retrain_v1.py --variant F_R --tag FRc_box_v1 --ot-foul-carry --preflight-only > logs/f2_preflight.log 2>&1; tail -n5 logs/f2_preflight.log
  $R scripts/chain_full_retrain_v1.py --variant F_R --tag FRc_box_v1 --ot-foul-carry --gate-stack adopted --cores $W --parallel --gate-mode full --gate-seeds 200 --gate-offsets 0 --gate-workers $W --gate-ref $REF --gate-noise $NOISE || echo "[F2 FRc] rc=$?"
  $R scripts/chain_full_retrain_v1.py --variant F_R --tag FRb_box_v1 --stages parity,gate --gate-stack adopted --cores $W --gate-mode full --gate-seeds 200 --gate-offsets 0 --gate-workers $W --gate-ref $REF --gate-noise $NOISE || echo "[F2 FRb] rc=$?"
  $R scripts/chain_full_retrain_v1.py --variant F_T --tag FTc_box_v1 --reuse-from FRc_box_v1 --ot-foul-carry --gate-stack adopted --cores $W --parallel --gate-mode full --gate-seeds 200 --gate-offsets 0 --gate-workers $W --gate-ref $REF --gate-noise $NOISE || echo "[F2 FTc] rc=$?"
  $R scripts/chain_full_retrain_v1.py --variant F_T --tag FTb_box_v2 --stages parity,gate --gate-stack adopted --cores $W --gate-mode full --gate-seeds 200 --gate-offsets 0 --gate-workers $W --gate-ref $REF --gate-noise $NOISE || echo "[F2 FTb] rc=$?"
  mkdir -p results/laneF
  BS --ref v3full_COMB9GCTKD_s200_o0 --floors $FL --arms fr1_FRc_box_v1_full_s200_o0_ev4_sv2,fr1_FRb_box_v1_full_s200_o0_ev4_sv2,fr1_FTc_box_v1_full_s200_o0_ev4_sv2,fr1_FTb_box_v2_full_s200_o0_ev4_sv2 --out-json results/laneF/boot_carry.json --out-md results/laneF/boot_carry.md ;;
*) echo "unknown job $J"; exit 2 ;;
esac
