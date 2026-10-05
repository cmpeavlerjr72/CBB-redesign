# laneC_2 DONE: tier A (COMB9 full + both pairings) and tier B (tapfull COMB9, COMB, S0); tier C status at the bottom (AWS operator, 2026-10-01)

- Box: on-demand c7a.48xlarge (i-0393f93baca3f0f66). Clone `~/cbb6` at `7840d40`. Engine flag-off parity vs v6: PASS bit-identical (03:16Z). S0 tag built in that clone. `v3full_S0_s200_o0`, the S0f1..f4 floor reads and `v3full_COMB_s200_o0` were copied into it (COMB pulled from HF).
- laneC_1 tier 2 (standalone R9ao1 / R9ao3 full) was dropped, as you amended.

**Tier A:** `bash scripts/box_r9_v1.sh full COMB9 90 0 200`.
- Timing: 03:15:41Z -> 03:31Z, about 2 min per 25-seed chunk. The script printed both grades (verified and legacy).
- Both `ops_pair_bootstrap_v1.py` commands ran exactly as written: done 03:33:51Z.
- On your question about the second pairing: with `--ref v3full_COMB_s200_o0`, the tool takes the draw SD over [COMB, S0f1, S0f2, S0f3, S0f4]. That mixes the COMB ref with S0 reruns, so recompute that floor locally from the S0 draws, as you planned.

**Tier B:** `tapfull COMB9` 03:33:51Z -> 03:48:24Z; `tapfull COMB` 03:41:14Z -> 03:56:26Z; `tapfull S0` 03:48:24Z -> 04:03:54Z. All exited 0.

**Synced locally**, to the same paths:
- scp: `results/engine_v0/v3full_COMB9_s200_o0/{games.parquet,run_meta.json}`;
- scp: `results/engine_v0/v3full_grade/v3full_COMB9_s200_o0__{verified,current}.md`;
- scp: `results/foul_r9/full/pair_COMB9_vs_{S0,COMB}.{md,json}`;
- tar: all 24 `results/engine_v0/r9tapfull_{COMB9,COMB,S0}_s200_o0_off{0..175}_n25/` dirs (games.parquet, games_v2.parquet, half_agg.parquet, run_meta.json; about 9 MB each);
- pulled from HF, already local: `v3full_COMB_s200_o0` games, `v3full_S0_s200_o0` and `v3full_S0f{1..4}_*` games plus their `__verified.md` grades.

The operator also wrote a gate doc in the format of the Decision 11 set: `docs/tests/engine_gates_F2_2025_s200_v3_COMB9_full_2026-09-30.md` (committed).

Tier C DONE: `tapfull S0 90 <O> 200` for O = 1000 (05:05:48Z -> 05:20:45Z), 2000 (05:08:32Z -> 05:23:38Z), 3000 (05:23:38Z -> 05:38:08Z) and 4000 (05:38:08Z -> 05:52:42Z). All 32 `results/engine_v0/r9tapfull_S0_s200_o{1000,2000,3000,4000}_off*_n25/` dirs were synced locally (tar).
