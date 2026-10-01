#!/usr/bin/env bash
# operator 2026-10-01: queue job definitions, run in ~/cbb4. jobs.sh <JOB>
set -uo pipefail
cd ~/cbb4; . ~/.hf_env; export HF_TOKEN PYTHONIOENCODING=utf-8
ts() { date -u +%H:%M:%SZ; }
P() { docker run --rm -e PYTHONIOENCODING=utf-8 -e CBB_TRUTH=verified_v1 -e HF_TOKEN -v ~/cbb4:/app -w /app --entrypoint python cbb-sweep -u "$@"; }
G=docs/tests/v3box_grades_2026-09-30
NOISE=$G/v3full_S0f1_s200_o1000__verified.md,$G/v3full_S0f2_s200_o2000__verified.md,$G/v3full_S0f3_s200_o3000__verified.md,$G/v3full_S0f4_s200_o4000__verified.md
GATE="--gate-mode full --gate-seeds 200 --gate-offsets 0 --gate-workers 90 --gate-ref $G/v3full_S0_s200_o0__verified.md --gate-noise $NOISE"
agg() { local S=$1 A=$2 N=$3; local d=results/aggregation_v1/${S}_${A}_s${N}_o0; [ -f $d/games.parquet ] && return 0
  echo "  [agg] $S $A s$N $(ts)"; P scripts/exp_aggregation_swap_v1.py --stack $S --arm $A --all-games --seeds $N --workers 90 --games-per-block 16 > logs/laneA_${S}_${A}_s$N.log 2>&1 || { echo "  [agg] FAILED $S $A"; return 1; }; }
laneB() { local ARM=$1 T=laneB_v3full_$1_s200_o0
  for o in 0 25 50 75 100 125 150 175; do sub=${T}_off${o}_n25; [ -f results/engine_v0/$sub/run_meta.json ] && continue
    docker run --rm -e PYTHONIOENCODING=utf-8 -e CBB_TRUTH=verified_v1 -e ENGINE_EVENT=round2_s1 -e ENGINE_CLOCK=v5b_glat_pmean \
      -e ENGINE_ROTATION=reference -e ENGINE_FG3=decision8 -e ENGINE_SHARED_SHOOTING=$ARM -v ~/cbb4:/app -w /app --entrypoint python cbb-sweep -u \
      scripts/run_engine_v3evb_v1.py --fold F2 --season 2025 --seeds 25 --seed-offset $o --workers 90 --games-per-block 60 --seeds-per-block 25 \
      --tag $sub --results-dir results/engine_v0 --input-dir data/processed/models/engine_v3 > logs/laneB_$sub.log 2>&1 || return 4; done
  P scripts/concat_engine_runs.py --tag $T --results-dir results/engine_v0 --overwrite > logs/laneB_concat_$ARM.log 2>&1; }
r9() { local k="$*"; local m="logs/LANEC_OK_$(echo $k | tr ' ' '_')"; [ -f $m ] && return 0
  bash scripts/box_r9_v1.sh "$@" > "logs/laneC_$(echo $k | tr ' ' '_').log" 2>&1 && touch $m || { echo "  [r9] FAILED $k"; return 1; }; }
hfpush() { P scripts/hf_sync_data.py push --dirs results model_artifacts --max-attempts 3 > logs/push_$1.log 2>&1; echo "  [push] $1 rc=$? $(ts)"; }
case "$1" in
D_FR) scripts/box_run.sh scripts/chain_full_retrain_v1.py --variant F_T --tag FR_box_v1 --cores 90 --preflight-only > logs/laneD_preflight.log 2>&1; tail -3 logs/laneD_preflight.log
      scripts/box_run.sh scripts/chain_full_retrain_v1.py --variant F_R --tag FR_box_v1 --cores 90 --parallel $GATE > logs/laneD_FR.log 2>&1; rc=$?; hfpush D_FR; exit $rc ;;
D_FT) until [ -f logs/JOB_D_FR.done -o -f logs/JOB_D_FR.failed ]; do sleep 20; done
      scripts/box_run.sh scripts/chain_full_retrain_v1.py --variant F_T --tag FT_box_v1 --reuse-from FR_box_v1 --cores 90 --parallel $GATE > logs/laneD_FT.log 2>&1; rc=$?; hfpush D_FT; exit $rc ;;
A3_XF) E3=data/processed/team_rate_features_E3_v4.parquet; T=data/processed/models
      P scripts/build_engine_inputs_v3_tag_v1.py --tag X_F_laneA --team-rate-table $E3 --team-rate-missing raise --no-table-for po,rb --fg-artifacts $T/fg_make/round_stageb/T/team_rate_features_E3_v4/B1 > logs/laneA_build_XF.log 2>&1; echo "  build XF rc=$?"
      P scripts/build_engine_inputs_v3_tag_v1.py --tag X_PR_laneA --team-rate-table $E3 --team-rate-missing raise --no-table-for fg --po-artifacts $T/possession_outcome/round_stageb/T/team_rate_features_E3_v4 --rb-artifacts $T/rebound/round_stageb/T/team_rate_features_E3_v4/artifacts/s2_F2_A0B0C0_seed0 > logs/laneA_build_XPR.log 2>&1; echo "  build XPR rc=$?"
      P scripts/build_engine_inputs_v3_tag_v1.py --tag S0_laneA > logs/laneA_build_S0.log 2>&1; echo "  build S0_laneA rc=$?"
      P scripts/build_engine_inputs_v3_tag_v1.py --tag S1_laneA --team-rate-table $E3 --team-rate-missing raise --variance-table data/processed/team_rate_variance_O1a_v3.parquet --po-artifacts $T/possession_outcome/round_stageb/T/team_rate_features_E3_v4 --fg-artifacts $T/fg_make/round_stageb/T/team_rate_features_E3_v4/B1 --rb-artifacts $T/rebound/round_stageb/T/team_rate_features_E3_v4/artifacts/s2_F2_A0B0C0_seed0 > logs/laneA_build_S1.log 2>&1; echo "  build S1_laneA rc=$?"
      P scripts/build_engine_inputs_v3_tag_v1.py --tag R2_laneA --po-artifacts $T/possession_outcome/round_stageb/R2_seed1 --fg-artifacts $T/fg_make/round_stageb/R2_seed1/B1 --rb-artifacts $T/rebound/round_stageb/R2_seed1/artifacts/s2_F2_A0B0C0_seed1 > logs/laneA_build_R2.log 2>&1; echo "  build R2_laneA rc=$?"
      agg X_F FULL 200 ;;
A3_XPR) agg X_PR FULL 200 ;;
B_G3) laneB G3 ;;
B_U1) laneB U1 ;;
C_t1) for A in R9ao1 R9ao3; do r9 sample $A 90 0; done; for A in S0 R8b R9ao1 R9ao3; do r9 tap $A 90 0; done
      for O in 3000 4000; do r9 sample S0 90 $O; done; for O in 1000 2000 3000 4000; do r9 tap S0 90 $O; done ;;
C_t2a) r9 full R9ao1 90 0 ;;
C_t2b) r9 full R9ao3 90 0 ;;
A1_t1_S0) for A in TEAM OFF DEF ALL PO FG RB RAT; do agg S0 $A 48; done ;;
A1_t1_S1) for A in TEAM OFF DEF ALL PO FG RB RAT; do agg S1 $A 48; done ;;
A2) agg R2 FULL 200 ;;
A1_t2) for A in RAT_PO PACE PLY; do for S in S0 S1; do agg $S $A 48; done; done ;;
H_A2|H_A2f1) flock ~/cbb5.lock bash ~/op_clone.sh cbb5 759be01 || exit 9
      cd ~/cbb5; if [ "$1" = H_A2 ]; then T=laneH_v3full_A2_s200_o0; O0=0; else T=laneH_v3full_A2f1_s200_o1000; O0=1000; fi
      export ENGINE_EVENT=round2_s1 ENGINE_CLOCK=v5b_r7A2_glat_pmean ENGINE_ROTATION=reference ENGINE_FG3=decision8 CBB_TRUTH=verified_v1
      IN=data/processed/models/engine_v3_S0; export BOX_DOCKER_ARGS; BOX_DOCKER_ARGS="$(sed "s#\$PWD#$PWD#g" "$IN/docker_mounts.txt" | tr '\n' ' ')"
      scripts/box_run_v2.sh scripts/ops_overlay_check_v1.py --input-dir "$IN" --root /app > logs/laneH_overlay.log 2>&1 || exit 3
      for k in 0 1 2 3 4 5 6 7; do o=$((O0 + 25*k)); sub=${T}_off${o}_n25; [ -f results/engine_v0/$sub/run_meta.json ] && continue; echo "  [H] $sub $(ts)"
        scripts/box_run_v2.sh scripts/run_engine.py --fold F2 --season 2025 --seeds 25 --seed-offset $o --workers 90 --games-per-block 60 --seeds-per-block 25 --tag $sub --results-dir results/engine_v0 --input-dir $IN > logs/laneH_$sub.log 2>&1 || exit 4; done
      scripts/box_run_v2.sh scripts/concat_engine_runs.py --tag $T --results-dir results/engine_v0 --overwrite > logs/laneH_concat_$T.log 2>&1
      python3 -c "import json;print('  ENGINE_CLOCK in run_meta:', json.load(open('results/engine_v0/$T/run_meta.json'))['adapter_flags']['ENGINE_CLOCK'])" ;;
C2A|B2|C2B_*|C2C_*) flock ~/cbb6.lock bash ~/op_clone.sh cbb6 7840d40 || exit 9
      cd ~/cbb6; for t in v3full_S0_s200_o0 v3full_S0f1_s200_o1000 v3full_S0f2_s200_o2000 v3full_S0f3_s200_o3000 v3full_S0f4_s200_o4000 v3full_COMB_s200_o0; do [ -d results/engine_v0/$t ] || sudo cp -r ~/cbb/results/engine_v0/$t results/engine_v0/; done; sudo chown -R ec2-user results; mkdir -p results/foul_r9/full results/shared_shooting/full
      FL=v3full_S0f1_s200_o1000,v3full_S0f2_s200_o2000,v3full_S0f3_s200_o3000,v3full_S0f4_s200_o4000
      BS() { CBB_TRUTH=verified_v1 scripts/box_run_v2.sh scripts/ops_pair_bootstrap_v1.py --results-dir results/engine_v0 "$@"; }
      case "$1" in
      C2A) bash scripts/box_r9_v1.sh full COMB9 90 0 200 > logs/laneC2_full_COMB9.log 2>&1 || exit 4
           BS --ref v3full_S0_s200_o0 --floors $FL --arms v3full_COMB9_s200_o0,v3full_COMB_s200_o0 --out-json results/foul_r9/full/pair_COMB9_vs_S0.json --out-md results/foul_r9/full/pair_COMB9_vs_S0.md > logs/laneC2_boot1.log 2>&1 || exit 5
           BS --ref v3full_COMB_s200_o0 --floors $FL --arms v3full_COMB9_s200_o0 --out-json results/foul_r9/full/pair_COMB9_vs_COMB.json --out-md results/foul_r9/full/pair_COMB9_vs_COMB.md > logs/laneC2_boot2.log 2>&1 || exit 6 ;;
      B2) T=v3full_COMB9G_s200_o0
           for o in 0 25 50 75 100 125 150 175; do sub=${T}_off${o}_n25; [ -f results/engine_v0/$sub/run_meta.json ] && continue; echo "  [B2] $sub $(ts)"
             docker run --rm -e PYTHONIOENCODING=utf-8 -e CBB_TRUTH=verified_v1 -e ENGINE_EVENT=round2_s1 -e ENGINE_ROTATION=reference -e ENGINE_FG3=decision8 \
               -e ENGINE_CLOCK=v5b_r6L2_glat_pmean -e ENGINE_SHOT_BLOCK=K2_Ocell -e ENGINE_FOUL_JOINT=R9ao3 -e ENGINE_SHARED_SHOOTING=G3 -v ~/cbb6:/app -w /app --entrypoint python cbb-sweep -u \
               scripts/run_engine_v3evb_v1.py --fold F2 --season 2025 --seeds 25 --seed-offset $o --workers 90 --games-per-block 60 --seeds-per-block 25 \
               --tag $sub --results-dir results/engine_v0 --input-dir data/processed/models/engine_v3 > logs/laneB2_$sub.log 2>&1 || exit 4; done
           P2() { docker run --rm -e PYTHONIOENCODING=utf-8 -e CBB_TRUTH=verified_v1 -v ~/cbb6:/app -w /app --entrypoint python cbb-sweep -u "$@"; }
           P2 scripts/concat_engine_runs.py --tag $T --results-dir results/engine_v0 --overwrite > logs/laneB2_concat.log 2>&1
           P2 scripts/eval_gates.py --results results/engine_v0/$T --season 2025 --out results/engine_v0/v3full_grade/${T}__verified.md > logs/laneB2_grade.log 2>&1
           until [ -f ~/cbb4/logs/JOB_C2A.done -o -f ~/cbb4/logs/JOB_C2A.failed ]; do sleep 20; done
           BS --ref v3full_S0_s200_o0 --floors $FL --arms $T,v3full_COMB9_s200_o0 --out-json results/shared_shooting/full/pair_COMB9G_vs_S0.json --out-md results/shared_shooting/full/pair_COMB9G_vs_S0.md > logs/laneB2_boot1.log 2>&1 || exit 5
           BS --ref v3full_COMB9_s200_o0 --floors $FL --arms $T --out-json results/shared_shooting/full/pair_COMB9G_vs_COMB9.json --out-md results/shared_shooting/full/pair_COMB9G_vs_COMB9.md > logs/laneB2_boot2.log 2>&1 || exit 6 ;;
      C2B_*) A=${1#C2B_}; bash scripts/box_r9_v1.sh tapfull $A 90 0 200 > logs/laneC2_tapfull_$A.log 2>&1 ;;
      C2C_*) O=${1#C2C_}; bash scripts/box_r9_v1.sh tapfull S0 90 $O 200 > logs/laneC2_tapfull_S0_$O.log 2>&1 ;;
      esac ;;
A4) flock ~/cbb7.lock bash ~/op_clone.sh cbb7 acad375 || exit 9
      cd ~/cbb7; docker run --rm -e HF_TOKEN -v "$PWD:/w" -w /w python:3.12-slim sh -c "pip install -q huggingface_hub==1.31.0 && python scripts/hf_sync_data.py pull --dirs model_artifacts --only 'fg_make/round_aggfix/Tfix_seed*/**' --max-attempts 4" > logs/pull_a4.log 2>&1; echo "  pull rc=$?"; sudo chown -R ec2-user data
      P7() { docker run --rm -e PYTHONIOENCODING=utf-8 -e CBB_TRUTH=verified_v1 -v ~/cbb7:/app -w /app --entrypoint python cbb-sweep -u "$@"; }
      FG=data/processed/models/fg_make/round_aggfix; SS=team_rate_features_E3_v4; TAB="--team-rate-table data/processed/$SS.parquet --team-rate-missing raise"
      P7 scripts/build_engine_inputs_v3_tag_v1.py --tag X_Tfix_laneA $TAB --no-table-for po,rb --fg-artifacts $FG/Tfix_seed0/$SS/B1 > logs/a4_b1.log 2>&1; echo "  build X_Tfix rc=$?"
      P7 scripts/build_engine_inputs_v3_tag_v1.py --tag S1fix_laneA $TAB --po-artifacts data/processed/models/possession_outcome/round_stageb/T/$SS --fg-artifacts $FG/Tfix_seed0/$SS/B1 --rb-artifacts data/processed/models/rebound/round_stageb/T/$SS/artifacts/s2_F2_A0B0C0_seed0 > logs/a4_b2.log 2>&1; echo "  build S1fix rc=$?"
      P7 scripts/build_engine_inputs_v3_tag_v1.py --tag X_Tfix1_laneA $TAB --no-table-for po,rb --fg-artifacts $FG/Tfix_seed1/$SS/B1 > logs/a4_b3.log 2>&1; echo "  build X_Tfix1 rc=$?"
      for st in X_Tfix S1fix X_Tfix1; do [ -f results/aggregation_v1/${st}_FULL_s200_o0/games.parquet ] && continue; echo "  [A4] $st $(ts)"
        P7 scripts/exp_aggregation_swap_v1.py --stack $st --arm FULL --all-games --seeds 200 --workers 90 --games-per-block 16 > logs/a4_$st.log 2>&1 || echo "  [A4] FAILED $st"; done ;;
I_K|I_C12) flock ~/cbb8.lock bash ~/op_clone.sh cbb8 38eafdc || exit 9
      cd ~/cbb8; for t in v3full_S0_s200_o0 v3full_S0f1_s200_o1000 v3full_S0f2_s200_o2000 v3full_S0f3_s200_o3000 v3full_S0f4_s200_o4000 v3full_COMB9_s200_o0; do [ -d results/engine_v0/$t ] || sudo cp -r ~/cbb6/results/engine_v0/$t results/engine_v0/; done; sudo chown -R ec2-user results; mkdir -p results/ppp_decomp/full
      FL=v3full_S0f1_s200_o1000,v3full_S0f2_s200_o2000,v3full_S0f3_s200_o3000,v3full_S0f4_s200_o4000
      if [ "$1" = I_K ]; then CT=K; RT=v3full_COMB9CTK_s200_o0; L=COMB9K; else CT=C12; RT=v3full_COMB9CT12_s200_o0; L=COMB9C12; fi
      bash scripts/box_chance_time_v1.sh COMB9 $CT 90 0 200 > logs/laneI_$CT.log 2>&1 || exit 4; grep "^\[env\]" logs/laneI_$CT.log
      BS() { CBB_TRUTH=verified_v1 scripts/box_run_v2.sh scripts/ops_pair_bootstrap_v1.py --results-dir results/engine_v0 "$@"; }
      [ -f results/engine_v0/v3full_grade/${RT}__verified.md ] || CBB_TRUTH=verified_v1 scripts/box_run_v2.sh scripts/eval_gates.py --results results/engine_v0/$RT --season 2025 --out results/engine_v0/v3full_grade/${RT}__verified.md > logs/laneI_grade_$CT.log 2>&1
      BS --ref v3full_COMB9_s200_o0 --floors $FL --arms $RT --out-json results/ppp_decomp/full/pair_${L}_vs_COMB9.json --out-md results/ppp_decomp/full/pair_${L}_vs_COMB9.md > logs/laneI_boot1_$CT.log 2>&1 || exit 5
      BS --ref v3full_S0_s200_o0 --floors $FL --arms $RT,v3full_COMB9_s200_o0 --out-json results/ppp_decomp/full/pair_${L}_vs_S0.json --out-md results/ppp_decomp/full/pair_${L}_vs_S0.md > logs/laneI_boot2_$CT.log 2>&1 || exit 6 ;;
D3) flock ~/cbb9.lock bash ~/op_clone.sh cbb9 e998b22 || exit 9
      cd ~/cbb9; sudo cp -an ~/cbb4/data/. ~/cbb9/data/; sudo chown -R ec2-user ~/cbb9/data
      for t in v3full_S0_s200_o0 v3full_S0f1_s200_o1000 v3full_S0f2_s200_o2000 v3full_S0f3_s200_o3000 v3full_S0f4_s200_o4000 fr1_FR_box_v1_full_s200_o0_ev4 fr1_FT_box_v1_full_s200_o0_ev4; do [ -d results/engine_v0/$t ] || sudo cp -r ~/cbb/results/engine_v0/$t results/engine_v0/; done; sudo chown -R ec2-user results
      R=data/processed/models/full_retrain_v1; [ -d $R/FT_box_v2 ] || cp -r $R/FT_box_v1 $R/FT_box_v2
      G=docs/tests/v3box_grades_2026-09-30
      scripts/box_run.sh scripts/chain_full_retrain_v1.py --variant F_T --tag FT_box_v2 --reuse-from FR_box_v1 --cores 90 --parallel --redo fg_train,inputs,gate \
        --gate-mode full --gate-seeds 200 --gate-offsets 0 --gate-workers 90 --gate-ref $G/v3full_S0_s200_o0__verified.md \
        --gate-noise $G/v3full_S0f1_s200_o1000__verified.md,$G/v3full_S0f2_s200_o2000__verified.md,$G/v3full_S0f3_s200_o3000__verified.md,$G/v3full_S0f4_s200_o4000__verified.md > logs/laneD3.log 2>&1; rc=$?; tail -4 logs/laneD3.log; [ $rc -eq 0 ] || exit $rc
      for k in 1 2 3 4; do scripts/box_run.sh scripts/diag_pair_gate_reports.py --a $R/FR_box_v1/gate/fr1_FR_box_v1_full_s200_o0_ev4__verified.md --b $R/FT_box_v2/gate/fr1_FT_box_v2_full_s200_o0_ev4__verified.md \
        --label-a FR_box_v1 --label-b FT_box_v2 --noise $G/v3full_S0f${k}_s200_o${k}000__verified.md --out $R/FT_box_v2/gate/pair_FR_box_v1_vs_FT_box_v2_n${k}.md > logs/laneD3_pair$k.log 2>&1; done
      FL=v3full_S0f1_s200_o1000,v3full_S0f2_s200_o2000,v3full_S0f3_s200_o3000,v3full_S0f4_s200_o4000
      CBB_TRUTH=verified_v1 scripts/box_run_v2.sh scripts/ops_pair_bootstrap_v1.py --ref v3full_S0_s200_o0 --floors $FL --arms fr1_FT_box_v2_full_s200_o0_ev4,fr1_FT_box_v1_full_s200_o0_ev4,fr1_FR_box_v1_full_s200_o0_ev4 --labels FT_box_v2,FT_box_v1,FR_box_v1 --out-json $R/FT_box_v2/gate/boot_vs_S0.json --out-md $R/FT_box_v2/gate/boot_vs_S0.md > logs/laneD3_boot1.log 2>&1
      CBB_TRUTH=verified_v1 scripts/box_run_v2.sh scripts/ops_pair_bootstrap_v1.py --ref fr1_FR_box_v1_full_s200_o0_ev4 --floors $FL --arms fr1_FT_box_v2_full_s200_o0_ev4 --labels FT_box_v2 --out-json $R/FT_box_v2/gate/boot_vs_FR_box_v1.json --out-md $R/FT_box_v2/gate/boot_vs_FR_box_v1.md > logs/laneD3_boot2.log 2>&1
      scripts/box_run_v2.sh scripts/hf_sync_data.py push --dirs results model_artifacts --max-attempts 3 > logs/push_D3.log 2>&1; echo "  push rc=$?" ;;
I_KD) flock ~/cbb10.lock bash ~/op_clone.sh cbb10 6edbe94 || exit 9
      cd ~/cbb10; for t in v3full_S0_s200_o0 v3full_S0f1_s200_o1000 v3full_S0f2_s200_o2000 v3full_S0f3_s200_o3000 v3full_S0f4_s200_o4000 v3full_COMB9_s200_o0 v3full_COMB9CTK_s200_o0; do [ -d results/engine_v0/$t ] || sudo cp -r ~/cbb8/results/engine_v0/$t results/engine_v0/; done; sudo chown -R ec2-user results; mkdir -p results/ppp_decomp/full
      FL=v3full_S0f1_s200_o1000,v3full_S0f2_s200_o2000,v3full_S0f3_s200_o3000,v3full_S0f4_s200_o4000; RT=v3full_COMB9CTKD_s200_o0
      bash scripts/box_chance_time_v1.sh COMB9 KD 90 0 200 > logs/laneI_KD.log 2>&1 || exit 4; grep "^\[env\]" logs/laneI_KD.log
      [ -f results/engine_v0/v3full_grade/${RT}__verified.md ] || CBB_TRUTH=verified_v1 scripts/box_run_v2.sh scripts/eval_gates.py --results results/engine_v0/$RT --season 2025 --out results/engine_v0/v3full_grade/${RT}__verified.md > logs/laneI_grade_KD.log 2>&1
      CBB_TRUTH=verified_v1 scripts/box_run_v2.sh scripts/ops_pair_bootstrap_v1.py --results-dir results/engine_v0 --ref v3full_S0_s200_o0 --floors $FL --arms $RT,v3full_COMB9CTK_s200_o0,v3full_COMB9_s200_o0 --out-json results/ppp_decomp/full/pair_COMB9KD_vs_S0.json --out-md results/ppp_decomp/full/pair_COMB9KD_vs_S0.md > logs/laneI_boot1_KD.log 2>&1 || exit 5
      CBB_TRUTH=verified_v1 scripts/box_run_v2.sh scripts/ops_pair_bootstrap_v1.py --results-dir results/engine_v0 --ref v3full_COMB9_s200_o0 --floors $FL --arms $RT --out-json results/ppp_decomp/full/pair_COMB9KD_vs_COMB9.json --out-md results/ppp_decomp/full/pair_COMB9KD_vs_COMB9.md > logs/laneI_boot2_KD.log 2>&1 || exit 6 ;;
G_G4) flock ~/cbb5.lock bash ~/op_clone.sh cbb5 759be01 || { echo "  cbb5 v6 parity FAIL, trying v7"; cd ~/cbb5; docker run --rm -v ~/cbb5:/app -v ~/cbb5/out:/out cbb-sweep --tag box1001_parity_cbb5v7 --parity only --parity-ref docs/ops/parity_reference_windows_v7.json --workers 48 --push off > logs/parity_v7.log 2>&1; grep -q "parity PASS" logs/parity_v7.log || { echo "  v7 parity FAIL too: refusing"; exit 9; }; echo "  v7 parity PASS"; }
      cd ~/cbb5; for t in v3full_S0_s200_o0 v3full_S0f1_s200_o1000 v3full_S0f2_s200_o2000 v3full_S0f3_s200_o3000 v3full_S0f4_s200_o4000; do sudo cp -rn ~/cbb/results/engine_v0/$t results/engine_v0/; done; sudo chown -R ec2-user results
      docker run --rm -e HF_TOKEN -v "$PWD:/w" -w /w python:3.12-slim sh -c "pip install -q huggingface_hub==1.31.0 && python scripts/hf_sync_data.py pull --dirs model_artifacts --only 'fg_make/round4_site/**' --max-attempts 4" > logs/pull_g4.log 2>&1; echo "  pull rc=$?"
      bash scripts/box_fullread_laneG_v1.sh preflight > logs/laneG_preflight.log 2>&1; tail -2 logs/laneG_preflight.log; grep -q "preflight OK" logs/laneG_preflight.log || exit 8
      bash scripts/box_fullread_laneG_v1.sh run 90 0 > logs/laneG_run.log 2>&1 || exit 4
      bash scripts/box_fullread_laneG_v1.sh grade > logs/laneG_grade.log 2>&1 ;;
R18_prep) cd ~/cbb
      docker run --rm -e HF_TOKEN -v "$PWD:/w" -w /w python:3.12-slim sh -c "pip install -q huggingface_hub==1.31.0 && python scripts/hf_sync_data.py pull --dirs results --only 'engine_v0/v3full_S2_*/**' 'engine_v0/v3full_S3_*/**' --max-attempts 4" > logs/pull_r18.log 2>&1; echo "  pull rc=$?"
      bash scripts/box_builds_v1.sh S1 > logs/r18_build_S1.log 2>&1; echo "  S1 tag rc=$?"
      bash scripts/box_draws_v1.sh > logs/r18_draws.log 2>&1; echo "  draws rc=$?" ;;
R18_S1K2O|R18_S2|R18_S3) A=${1#R18_}; until [ -f logs/JOB_R18_prep.done -o -f logs/JOB_R18_prep.failed ]; do sleep 15; done; cd ~/cbb
      bash scripts/box_fullread_v1.sh run $A 90 0 200 25 > logs/fullread_${A}_o0.log 2>&1 && bash scripts/box_fullread_v2.sh grade v3full_${A}_s200_o0 > logs/grade_$A.log 2>&1; rc=$?
      scripts/box_run_v2.sh scripts/hf_sync_data.py push --dirs results --max-attempts 3 > logs/push_$A.log 2>&1; exit $rc ;;
*) echo "unknown job $1"; exit 2 ;;
esac
