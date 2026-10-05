# Box request laneI_2: COMB9 + chance-time feed `KD` at full size, paired with COMB9 (lane I, 2026-10-01 01:07 EDT)

**Label: POST-HOC DIAGNOSTIC read for a PM ruling (not a registered SHIP-DECISION).** `KD` is addendum 1c (`docs/models/chance_time/experiments.md` s6). It is K with the chance-1 time table also conditioned on the possession's drawn duration. On the full-size K read (laneI_1), G9 total bias went -1.49 -> -0.56 (PASS), and the G5 total SD ratio and h/a corr moved AWAY. KD is built to keep the level fix without that G5 loss. Local 500 x 32 against R: points per game +1.10 (23 floors). Within-game corr(possessions, eFG), total SD and h/a points corr all fall within R's seed floor; under K they were outside it. Priority: same as laneI_1, behind registered SHIP-DECISION requests. This one matters more than laneI_1's tier B (C12). If C12 has not run, skip it.

**Commit:** `6edbe94` (origin/main) or later. Tables `data/processed/models/chance_time/F2/lut_v3.npz` are tracked. Inputs and overlay are as for COMB9 / laneI_1. `chmod +x scripts/*.sh`.

**Tier A (about 16 min at 90 workers):**
```
bash scripts/box_chance_time_v1.sh COMB9 KD 90 0 200
CBB_TRUTH=verified_v1 python scripts/ops_pair_bootstrap_v1.py --results-dir results/engine_v0 --ref v3full_S0_s200_o0 \
  --floors v3full_S0f1_s200_o1000,v3full_S0f2_s200_o2000,v3full_S0f3_s200_o3000,v3full_S0f4_s200_o4000 \
  --arms v3full_COMB9CTKD_s200_o0,v3full_COMB9CTK_s200_o0,v3full_COMB9_s200_o0 --out-json results/ppp_decomp/full/pair_COMB9KD_vs_S0.json --out-md results/ppp_decomp/full/pair_COMB9KD_vs_S0.md
CBB_TRUTH=verified_v1 python scripts/ops_pair_bootstrap_v1.py --results-dir results/engine_v0 --ref v3full_COMB9_s200_o0 \
  --floors v3full_S0f1_s200_o1000,v3full_S0f2_s200_o2000,v3full_S0f3_s200_o3000,v3full_S0f4_s200_o4000 \
  --arms v3full_COMB9CTKD_s200_o0 --out-json results/ppp_decomp/full/pair_COMB9KD_vs_COMB9.json --out-md results/ppp_decomp/full/pair_COMB9KD_vs_COMB9.md
```
-> `results/engine_v0/v3full_COMB9CTKD_s200_o0/` and `results/engine_v0/v3full_grade/v3full_COMB9CTKD_s200_o0__verified.md`. Flag check as in laneI_1: the `[env]` line shows `ENGINE_CHANCE_TIME=KD`, and the rows differ from COMB9.

**Sync back (same paths):** `results/engine_v0/v3full_COMB9CTKD_s200_o0/{games.parquet,run_meta.json}`, its `__verified.md`, `results/ppp_decomp/full/pair_COMB9KD_*`.

**Decision needed:** none; run as written if the queue still accepts it (filed before 01:30 EDT). Write `laneI_2.done.md` (or `.failed.md`).
