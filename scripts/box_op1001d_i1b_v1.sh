#!/usr/bin/env bash
# operator: d1001_I_1 reads only (builds and both parity checks already PASSED in the first attempt; it failed because
# results/laneI_1001 was root-owned so the overrides json could not be written). i1b.sh <WORKERS>
set -uo pipefail
cd ~/cbb; . ~/.hf_env; export HF_TOKEN PYTHONIOENCODING=utf-8 CBB_TRUTH=verified_v1
W=${1:-64}; R=scripts/box_run_v3.sh; RE=results/engine_v0; M=data/processed/models
sudo chown -R ec2-user results; 
echo '{"adapters.RB_S1_MANIFEST": "data/processed/models/engine_v3_I_RBTO/overlay/data/processed/models/rebound/s1_confirm/S1_weekly/F2/manifest.json"}' > results/laneI_1001/ov_rbto.json
echo '{"adapters.FT_S1_MANIFEST": "data/processed/models/free_throw/laneI_N1/S1_conf_aligned/F2/manifest.json"}' > results/laneI_1001/ov_n1.json
FL=d1001D_S2f1_s200_o1000,d1001D_S2f2_s200_o2000,d1001D_S2f3_s200_o3000,d1001D_S2f4_s200_o4000
( export ENGINE_SEASON_ANCHOR=/app/$M/engine_v3_I_RBTO/anchor_offsets_F2_2025.npz
  $R scripts/run_engine_overlay_v1.py --overrides results/laneI_1001/ov_rbto.json --runner full -- --fold F2 --season 2025 --seeds 200 --seed-offset 0 --workers $W --games-per-block 60 --seeds-per-block 25 --input-dir $M/engine_v3_I_RBTO --tag d1001I_RBTO_s200_o0 --results-dir $RE ) || echo "[RBTO] rc=$?"
$R scripts/run_engine_overlay_v1.py --overrides results/laneI_1001/ov_n1.json --runner full -- --fold F2 --season 2025 --seeds 200 --seed-offset 0 --workers $W --games-per-block 60 --seeds-per-block 25 --input-dir $M/engine_v3_I_N1 --tag d1001I_N1_s200_o0 --results-dir $RE || echo "[N1] rc=$?"
for t in d1001I_RBTO_s200_o0 d1001I_N1_s200_o0; do $R scripts/eval_gates.py --results $RE/$t --season 2025 --out $RE/v3full_grade/${t}__verified.md; done
sudo chown -R ec2-user results
$R scripts/ops_pair_bootstrap_v1.py --results-dir $RE --ref v3full_COMB9GCTKD_s200_o0 --floors $FL --arms d1001I_RBTO_s200_o0,d1001I_N1_s200_o0 --out-json results/d1001I/boot_vs_S2.json --out-md results/d1001I/boot_vs_S2.md
echo "[i1b] done $(date -u +%H:%M:%SZ)"
