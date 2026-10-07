# Adoption: clock round-8 K2 (`v5b_r8K2_glat_pmean`) as the served clock (served stack v3)

PM decision 2026-10-07, before the Oct 10 freeze, under the PM mandate of 2026-10-05.

## Evidence

The paired 200-seed full-size reads on both folds share seeds 0-199 with served v2.

| gate | F2 served v2 | F2 K2 | F1 served v2 | F1 K2 | F1 real |
|---|---|---|---|---|---|
| G5 total SD ratio | 0.925 FAIL | 0.973 PASS | 0.918 FAIL | 0.961 PASS | 1.0 |
| G5 home/away score corr | 0.126 | 0.170 | 0.147 | 0.188 (FAIL) | 0.252 |
| G9 total slope | unchanged | unchanged | 0.937 | 0.941 | 1.0 |
| G9 total bias | | | -1.71 | -1.46 | 0 |
| G1 poss mean / SD | | | 69.27 / 5.58 | 69.23 / 5.68 | 68.39 / 5.48 |

- **Gate statuses:** identical on both folds. One G9 month cell (Dec, F1) moved from -0.496 to -0.502 against a 0.5 limit. That is edge noise, not a regression.
- **G2:** 3/9 cells pass on served v2 and 4/9 on K2.
- **G7 OT rate:** 0.031 on served v2 and 0.034 on K2, against a real 0.060.

Sources:
- F2 read: box `d1001_H_3`, `docs/tests/g5_variance_decomp_2026-10-05.md`.
- F1 read: `docs/tests/engine_gates_F1_2024_s200_v3_K2_full_2026-10-07.md`, spec `docs/ops/box_queue/d1007_K2F1.md`, which was committed before the run.

## Why this is a sub-model fix, not a compensation

The G5 decomposition owns the pace x efficiency gap to the clock. Possession time is drawn before the outcome. K2 conditions first-chance time on the chance end class, and it moves the FG-value covariance channel to reality: -45.6 becomes -23.4, against a real -22.1.

The Decision 11 warning still stands. On K2, regulation possession variance runs slightly high (G1 SD 5.68 vs 5.48, inside tolerance), while shared PPP is short. The shared-PPP gap belongs to the FT block (whistle plus the structural FT-make coupling) and to G7 (OT rate). It is not closed here, and it must not be read as closed: once those owners are fixed, the next G5 read can overshoot the SD ratio.

## Caveats

- One seed set per fold. There is no separate noise-floor retrain for the loop read; the offline round-8 selection carried the floor.
- Fold 1 uses static rotation in both arms.

## Decision

`ENGINE_CLOCK=v5b_r8K2_glat_pmean` becomes the engine default. Served v2 stays reachable as `adapters.SERVED_V2`. The parity reference moves to v10. The retrain chain and the seal-week build must produce K2 clock artifacts for 2026-27 serving.

## Implementation (worker, 2026-10-07)

**Code.**
- `adapters.CLOCK_DEFAULT = "v5b_r8K2_glat_pmean"` is read by `Adapters.load`.
- `adapters.SERVED_V2` holds the full served-v2 env (clock L2, K2_Ocell, R9ao3, G3, KD, event team block v3).
- `clock_adapter_v3.ADOPTED_MODES` gains K2. This is a label only.
- `run_aws_sweep.sh` now defaults its `ENGINE_CLOCK` to K2, which avoids a hybrid stack.

**Proofs** (fold 2, 2025, `engine_v3`, 60 games x seeds 0-4, games and players compared with exact equality):

| run | vs | result |
|---|---|---|
| post-change, SERVED_V2 env | parity v9 | PASS, bit-identical digest `e0a42353...` |
| post-change, no ENGINE_* set | pre-change `ENGINE_CLOCK=v5b_r8K2_glat_pmean` | games 300/300, players 4,955/4,955 identical |
| new reference v10 (same generator, default env) | - | digest `db23f493a4ab...`, file sha256 `238a2f5d...` |
| engine served from a `train_clock_k2_chain_v1.py` refit root (design `clock/r6_L2`, possessions v4) through `run_engine_overlay_v1` | v10 run | games and players identical; source check PASS |

**Retrain chain.**
- `chain_full_retrain_v1.py` gains stage `clock_k2`, which is default on (`--no-clock-k2` opts out).
- The stage refits K2 on the clock stage's design and on the chance tables of the chain's possessions version.
- The gate serves `ENGINE_CLOCK=v5b_r8K2_glat_pmean` with the clock overrides re-pointed at that root. Gate tags get the suffix `_sv3`.
- Identity on the served inputs: the manifests are equal and the six first-chance pmfs differ by at most 0.0. B1_sigma is equal (0.04710077845445259).
- The pickle bytes differ, but the contents are the same.

**2026-27 serving.**
- No family is refit through 2025-26.
- The clock manifest path does not depend on the season, so a 2026-27 game is served by the last fold-2 refit: 2025-04-01, trained through 2025-03-31. This is carried forward, the same way as `build_season_2027_artifacts_v1.py`.
- Refitting through 2025-26 needs the sealed season, so it stays a PM retrain-set decision.

**Seal week.**
- `ops_seal_week_v1.py` gains a first stage, `parity_v10` (`scripts/run_parity_smoke_v1.py`), which runs the smoke and compares it to v10.
- It also gains a `clock_serving` stage (`scripts/diag_clock_serving_2027_v1.py`). The stage checks that the K2 set is present and selected for the slate. Under `--execute` it pulls a missing set from HF.
- Dry run: 19 stages, 14 OK, 4 SEAL_OK, 1 MANUAL, 0 FAIL.

**Tests.** The suite gives 748 passed, 1 skipped:
- `test_served_defaults_v2.py` now pins K2, SERVED_V2 and the v9/v10 flags.
- `test_ops_seal_week.py` now checks the stage order.
