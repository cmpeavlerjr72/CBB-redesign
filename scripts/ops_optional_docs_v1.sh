#!/usr/bin/env bash
# Optional-arm gate docs from the locally graded pulls (operator 2026-09-30). verified truth only.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8; PY=.venv/Scripts/python.exe
G=results/engine_v0/v3box_grade; D=docs/tests; DATE=2026-09-30
P50F=v3part50_S0f1,v3part50_S0f2,v3part50_S0f3,v3part50_S0f4
SMF=v3box_S0f1_s200_o1000,v3box_S0f2_s200_o2000
mk() {  # mk ARM LABEL HAS50 SAMPLETAG DESC
  local arm=$1 lab=$2 has50=$3 st=$4 desc=$5 out=$D/engine_gates_F2_2025_s200_v3_${1}_$DATE.md
  { echo "# Engine gates, arm $arm ($lab), 2026-09-30 (optional step; spot reclaimed mid-run)"; echo
    echo "$desc"; echo
    echo "Verified truth only (\`CBB_TRUTH=verified_v1\`). Multi-draw floor = SD over the reference's draws; movement flagged BEYOND only when |move| > 2 x that SD (a candidate difference, not a decision). Floor draws are unpaired reruns of S0 on other seeds."; echo
    if [ "$has50" = 1 ]; then
      echo "## A. Full size, 5,705 verified games, seeds 0-49 only (spot was reclaimed at 19:45:58Z before seeds 50-199), paired against S0 on the same 50 seeds; floor = S0 reruns on seeds 1000-1049, 2000-2049, 3000-3049, 4000-4049"; echo
      $PY scripts/ops_v3_pair_table_v1.py --dir $G --truth verified --ref v3part50_S0 --floors $P50F --arms v3part50_S1,v3part50_$arm --labels S1,$arm --out $G/tmp_$arm.md > /dev/null; tail -n +2 $G/tmp_$arm.md; echo
    fi
    echo "## B. 500-game verified sample (stride500_verified_minswap_v1), 200 seeds, paired against S0 on the same games and seeds; floor = S0 reruns on seeds 1000-1199 and 2000-2199 (two draws; S0f3 and S0f4 sample runs were lost)"; echo
    $PY scripts/ops_v3_pair_table_v1.py --dir $G --truth verified --ref v3box_S0_s200_o0 --floors $SMF --arms v3box_S1_s200_o0,$st --labels S1,$arm --out $G/tmp2_$arm.md > /dev/null; tail -n +2 $G/tmp2_$arm.md
  } > $out; echo wrote $out
}
mk S2 "S1 + per-game draw from E3 variance, K=64" 1 v3box_S2_s200_o0 "S2 = S1 stack (Stage B T artifacts, E3 v4) plus a per-game team-rate draw from the E3 estimation variance (draw file built on the S1 tag, seed 20260930)."
mk S3 "S1 + per-game draw from O1a variance, K=64" 1 v3box_S3_s200_o0 "S3 = S1 stack plus the draw from the O1a variance (same normals as S2; the O1a variance file had to be copied to the box by scp because the team_rate_tables HF key does not match team_rate_variance_*.parquet)."
mk K2O "served artifacts, ENGINE_SHOT_BLOCK=K2_Ocell" 0 v3box_K2O_s200_o0 "K2O = S0 (served artifacts on the v3 inputs) with the drawn block flag K2_Ocell."
mk R8b "served artifacts, ENGINE_FOUL_JOINT=R8b" 0 v3box_R8b_s200_o0 "R8b = S0 with ENGINE_FOUL_JOINT=R8b (round-7 CL2 reuse)."
mk R8bS "served artifacts, ENGINE_FOUL_JOINT=R8bS" 0 v3box_R8bS_s200_o0 "R8bS = S0 with ENGINE_FOUL_JOINT=R8bS (shared whistle latent)."
