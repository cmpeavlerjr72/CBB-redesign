# po_tov_1: PO section 30 arm `T1` (as-of league TOV level offset on `first`) at full size on both folds. DESCRIPTIVE read

- **Commit:** this file's commit or later on `main`. Engine `src/` is unchanged. `T1` is served through two pieces:
  - the existing default-off `ENGINE_SEASON_ANCHOR=<npz>` (`po_first`, TOV column only);
  - a private `ENGINE_DIR` (overlay) that holds the anchored `first` artifacts.
- **Spec and results:** `docs/models/possession_outcome/experiments.md` sections 30-31.
- **Label:** DESCRIPTIVE. T1 is NOT the registered winner: it fails E3 on the F2 d15-45 TOV gap. The PM decides whether a full read is wanted.

## 0. Setup
```
python scripts/hf_sync_data.py pull --dirs model_artifacts --only 'possession_outcome/tovlevel/engine_scratch_F1_T1/**' \
  'possession_outcome/tovlevel/engine_scratch_F2_T1/**' 'fold1_v1/**' 'engine_f1/**' 'engine_v3_f1/**'
T=data/processed/models/possession_outcome/tovlevel
```
Each scratch dir carries its `offsets_*.npz` and `overrides_*.json`. Run the plain-default parity check against v9 as usual. Skip it if it already passed at this engine src.

## 1. Fold 2 (5,710 x 200, offset 0)
Paired with `v3full_COMB9GCTKD_s200_o0`. Floors: `d1001D_S2f{1..4}`.
```
export CBB_TRUTH=verified_v1
ENGINE_SEASON_ANCHOR=$T/engine_scratch_F2_T1/offsets_F2_2025.npz scripts/box_run_v3.sh scripts/run_engine_overlay_v2.py \
  --overrides $T/engine_scratch_F2_T1/overrides_F2_T1.json --runner full -- --fold F2 --season 2025 --seeds 200 \
  --seed-offset 0 --workers 90 --input-dir data/processed/models/engine_v3 --tag potov1_T1_F2_s200_o0 --results-dir results/engine_v0
scripts/box_run_v3.sh scripts/eval_gates.py --results results/engine_v0/potov1_T1_F2_s200_o0 --season 2025 \
  --out results/engine_v0/v3full_grade/potov1_T1_F2_s200_o0__verified.md
scripts/box_run_v3.sh scripts/ops_pair_bootstrap_v1.py --results-dir results/engine_v0 --ref v3full_COMB9GCTKD_s200_o0 \
  --floors d1001D_S2f1_s200_o1000,d1001D_S2f2_s200_o2000,d1001D_S2f3_s200_o3000,d1001D_S2f4_s200_o4000 \
  --arms potov1_T1_F2_s200_o0 --out-json results/potov1/boot_T1_F2.json --out-md results/potov1/boot_T1_F2.md
```

## 2. Fold 1 (5,635 x 200)
Paired with `f1c_V2_full_s200_o0`.
```
ENGINE_ROTATION_SCHEME=static ENGINE_SEASON_ANCHOR=$T/engine_scratch_F1_T1/offsets_F1_2024.npz scripts/box_run_v3.sh \
  scripts/run_engine_overlay_v2.py --overrides $T/engine_scratch_F1_T1/overrides_F1_T1.json --runner full -- --fold F1 \
  --season 2024 --seeds 200 --seed-offset 0 --workers 90 --input-dir data/processed/models/engine_v3_f1 \
  --tag potov1_T1_F1_s200_o0 --results-dir results/engine_v0
scripts/box_run_v3.sh scripts/eval_gates.py --results results/engine_v0/potov1_T1_F1_s200_o0 --season 2024 \
  --out results/engine_v0/v3full_grade/potov1_T1_F1_s200_o0__verified.md
```
- `f1c_V2_full_s200_o0` was run from a dirty tree. If its engine src differs from this commit, rerun V2 o0 alongside, the same way `chain_fold1_v1.py` does.
- **Env check.** Confirm all of these in `run_meta.json`; if any is missing, the run is not T1:
  - `adapter_flags.ENGINE_SEASON_ANCHOR` is set;
  - the event source path contains `engine_scratch_F?_T1`;
  - F1 shows `ENGINE_ROTATION_SCHEME=static`, as `f1c_V2` does.

## Sync back
- `results/engine_v0/potov1_T1_F{1,2}_s200_o0/{games.parquet,run_meta.json}`, their `__verified.md`, and `results/potov1/*`.
- Re-grade the decomposition locally with `scripts/diag_total_bias_decomp_v1.py`.

## The one line to decide
Does T1 move G9 total bias toward 0 on fold 1 without a G1-G9 veto on either fold, and what does the fold-2 d15-45 TOV level cost at full size? Fold 1's TOV channel is -1.00 pts for the full season, so the expected move is about +1 pt.
