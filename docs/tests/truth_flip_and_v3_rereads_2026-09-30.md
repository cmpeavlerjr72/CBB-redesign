# Truth-default flip, parity v7, verified default sample, and v3 re-reads of earlier paired rounds (lane E, 2026-09-30 / 10-01)

Nothing is adopted. No served default changed. No engine file edited by this lane. Every re-read: v3 inputs (`data/processed/models/engine_v3`), verified same-rule stride sample (500 games), grading truth = the NEW DEFAULT (verified; `CBB_TRUTH` unset), 25 seeds, 2 workers, all of tonight's new flags unset. Each run's `run_meta.json` carries the engine commit and dirty paths; they are tabulated in section 6.

## 1. Truth default flip (committed and pushed first, `0becb37`)

- `eval.reference`: unset `CBB_TRUTH` now means `verified_v1`; opt-out is `CBB_TRUTH=legacy_v0` (or an explicit `verified_finals=False`); any other value raises. `load_gate_targets` follows (`data/reference/verified_v1`). Change is 16 lines + tests; ledger row added.
- Full suite: 645 passed, 1 skipped.
- Gate table check: `scripts/eval_gates.py --results results/engine_v0/clk6_R_s25 --season 2025`, default vs explicit `CBB_TRUTH=verified_v1`: reports byte-identical (diff empty), 500 games both. `CBB_TRUTH=legacy_v0` on the same run differs (20 diff lines). So the flip is a no-op for lanes that already set `verified_v1`.
- Scope of the flip: it only changes what `load_actual_games` / `load_gate_targets` return. The foul-joint grader reads the team box directly and is independent of the flip (the box has no row for unplayed games).

## 2. Parity reference v7 (`docs/ops/parity_reference_windows_v7.json`, committed after `0becb37`)

- Same procedure as v6 (`run_engine.py --seeds 5 --max-games 60 --workers 2`), served stack, but `--input-dir data/processed/models/engine_v3`. SHA256 `34cd58dddf7d54359f8ae615bc6b9169d32c4a933de21ae513ada1068ede1736`; engine commit `489269f` (tree dirty only in `.gitignore`).
- Truth flip does not change the sim: a v2-inputs rerun at HEAD `f72bcb0` compared against v6 with `digest_engine_run.py --compare` -> PASS, bit-identical `0d4ddccc...`.
- What differs between v6 and v7, and why: only the input arrays (v3 = honest live-replay inputs instead of v2). 174 of 300 (game, seed) rows are identical to the v2 run, 126 differ (mean home pts 75.41 -> 75.25, possessions 68.79 -> 68.54 on the 60 games). Flags in the digest are unchanged except the label `ENGINE_INPUTS_VERSION` (`v1`, the file-stem label of `arrays_F2_2025.npz` in the v3 directory; the run_meta also records `v3-replay`).
- Further proof of default-path stability: `e3_lg2_R_s25` (tapped, HEAD `e5dd38c`) and `e3_fj_R_s25` (tapped, `3938a8d`) are bit-identical on all 28 `games.parquet` columns to lane B's untapped `clk6_R_s25` (`7cdb01e`), 12,500 rows.
- Trap found: lane N's `laneN_R_s25_o*` runs (inputs `engine_v3_N_R`) are NOT bit-identical to `clk6_R_s25` (26 of 28 columns differ at offset 0) even though the array and event-block files hash equal. They are not valid draws of the served stack on `engine_v3` and were NOT used as floor draws.

## 3. Default closed-loop sample (`d16346a`)

`scripts/run_po4b_closed_loop.py::subset_rows` now defaults to `stride500_verified_v1_F2_2025.parquet` (same rule on the verified universe; verified: equals the file's 500 ids; 10 in common with the old sample). `CBB_SAMPLE=legacy_stride` reproduces the old sample (verified equal to `po4b_R_s25`'s `game_ids`). Every runner importing `PO4B.subset_rows` follows. The `--sample-file` wrapper without a file now falls through to the default. `run_meta` records `sample_mode`.

## 4. Method

- Floors, Decision 12: four seed-offset draws of the served stack on the same inputs and sample, at offsets 1000, 2000, 3000, 4000 (`clk6_Rf1_s25`, `clk6_Rf2_s25`, `e3_Rf3_s25`, `e3_Rf4_s25`), reference = offset 0 (`clk6_R_s25`, `e3_*_R_s25`, all bit-identical). Floor of a line = max over the four of |draw - reference|. Paired game-bootstrap SE is each grader's own and is reported where the grader has it.
- Graders unedited, wrapped: `scripts/grade_lanee_lg2_v3_v1.py`, `scripts/grade_lanee_fj_v3_v1.py` (a junction view dir for the hardcoded floor pair; old grade jsons restored), shot-block grader run once per floor pair, then floors re-maxed (`floors_toward = X / max floor`, X is draw-independent).
- Timing measured: 500 x 25 on 2 workers took 380-1,272 s per run depending on box contention (late-game R 1,054 s, W_D 1,272 s; later arms 380-560 s). Chains: `scripts/chain_lanee_v3_v3.sh` (and `chain_lanee_lg2_v3_v1.sh` for the first two lg2 runs).
- Not touched per PM order: `ENGINE_SHOT_BLOCK=K2_Ocell`, `ENGINE_FOUL_JOINT=R8b` (and `CL2`, identical to R8b), clock `L2` (operator lane, box).

## 5. Per-round tables (old verdict -> new verdict)

### 5.1 Late-game round 2 (`docs/tests/late_game_round2_2026-09-30.md`), 7 runs, 6 arms

Reference R on v3: P(0)/P(1) 0.578 (old v2 R 0.551); ties 391; G9 total bias +0.393 (old -0.96), G5 total ratio 0.912 (old 0.835). Floors (max of 4): ratio 0.104 (old single-pair 0.062), G1 mean 0.077, G1 SD 0.039, G9 total 0.086, half share 0.0013 (single tapped draw, `e3_lg2_Rfloor`). Power rule (floor <= 0.10, ties >= 300): floor 0.104 is just over, so the ratio line is marginally underpowered on this sample; the +3 floor reading of W_D is a large effect but is labelled as such.

| arm | ratio old -> new | floors old -> new | OT in band | old vetoes | new vetoes (floors worse than R) | flip |
|---|---|---|---|---|---|---|
| W_C2 | 0.543 -> 0.595 | -0.13 -> +0.16 (SE 0.043) | no / no | G1 mean, half share | G1 mean (2.4), **G9 total bias (4.7)**, half share (1.0) | veto set widens by G9 total |
| **W_D** | 0.854 -> 0.918 | **+4.88 -> +3.26** (SE 0.057) | no (0.0437) / no (0.0448; band 0.046-0.055) | G1 mean, G1 SD, half share | G1 mean (3.2), G1 SD (3.4), **G9 total bias (6.5)**, half share (1.3) | same verdict (moves primary, fails vetoes, ratio < 1 so "does not fix"); G9 total bias newly fails |
| E_BL3 | 0.579 -> 0.523 | +0.46 -> -0.53 | no / no | none | none | no flip (PASS, no movement) |
| E_L0S0 (control) | 0.576 -> 0.593 | +0.41 -> +0.14 | no / no | none | none | no flip |
| W_C2+E_BL3 | 0.607 -> 0.627 | +0.90 -> +0.47 | no / no | G1 mean, half share | G1 mean, G9 total bias, half share | G9 total newly fails |
| W_D+E_BL3 (exploratory) | 0.851 -> 0.880 | +4.82 -> +2.90 | no / no | G1 mean, G1 SD, half share | G1 mean, G1 SD, G9 total bias, half share | G9 total newly fails |

First-half bit-identity holds in every arm (12,500 of 12,500). New R on v3 already over-predicts totals (+0.39), so any window over-production now shows as a G9 total-bias veto; this is a reading of the lines, not a defect assignment. Detail: `results/late_game/round2/grade_e3_v3.json` (gitignored).

### 5.2 Foul joint rounds 7 and 8 (`foul_joint_round_2026-09-30.md`, `foul_round8_whistle_2026-09-30.md`), 10 runs

New reference `e3_fj_R_s25` on v3: FTA/FGA 0.3170, G5 total ratio 0.9116 (old 0.8835), score corr 0.125, team FT slope 0.617 (old 0.958), team SD ratio 0.770, max occupancy gap 13.6 pp (old 13.1). Floors (max of 4): FTA/FGA 0.00102, G5 total 0.0111, score corr 0.0078, G9 total 0.086, G1 mean 0.077, G1 SD 0.039. V1/V2/V4 use bootstrap SEs (grader's own), V3 uses floors. "toward" = floors toward the target; V3 components in floors: G5 total, score corr, G1 mean, G1 SD, G5 margin, G9.

| arm | old V1-V4 (old G5-total floors) | new V1-V4 | new FTA/FGA (floors toward) | new max occ gap pp (old) | new V3 failing component | flip |
|---|---|---|---|---|---|---|
| R8a (= CL2a) | T,T,**F**,T (-4.6) | T,T,T,T, **eligible** | 0.3171 (+0.06) | 8.0 (7.7) | none (score corr -0.9, G1 SD -1.3) | **V3 now passes; grader-eligible** |
| R8aS | T,T,F,T (-12.7) | T,T,**F**,T | 0.3168 (-0.25) | 8.5 (7.9) | score corr -1.4 | stays failing, different line |
| R8bU | T,T,F,T (-7.6) | T,T,**F**,T | 0.3310 (+10.95) | 5.1 (6.4) | G9 total bias -2.5 | stays failing, different line |
| R8bS | T,T,F,T (-8.7) | T,T,**F**,T | 0.3316 (+10.28) | 5.1 (6.3) | G9 total bias -2.4 | stays failing, different line |
| CL1 | round-7 doc: all arms fail V3 (G5 total) | T,T,T,T, **eligible** | 0.3158 (-1.17) | 9.1 | none | **eligible now** |
| CL3 | fail V3 (G5 total) | T,T,**F**,T | 0.3339 (+8.05) | 5.4 (6.8) | G9 total bias -3.2 | stays failing, different line |
| CL4 | fail V3 (G5 total) | T,T,**F**,T | 0.3195 (+2.45) | 8.4 (7.7) | G9 total bias -2.9 | stays failing, different line |
| F5e (foul6 accrual LUT) | not graded by this grader earlier | **F**,T,T,T | 0.2988 (-17.9) | 27.2 | V1 (occupancy) | NEW read: undershoots on v3 |
| F5 | same | **F**,T,T,T | 0.2801 (-36.2) | 36.0 | V1 (occupancy) | NEW read: undershoots on v3 |

Not run: `R8b` / `CL2` (box lane). Readings, not conclusions: (i) the "G5 total SD ratio blocks everything" veto of rounds 7-8 came from the old reference being under-dispersed (0.8835); on v3 the reference is 0.912 and the arms move it by only -0.3 to -0.7 floors, so that veto no longer binds. (ii) In its place, the trip-class arms (R8b*, CL3, CL4) move total bias from +0.39 to about +0.95 (G9 -2.4 to -3.2 floors): the compensation moves rather than disappears. (iii) The accrual-only arms R8a and CL1 now clear all four pre-registered vetoes, but their FTA/FGA effect is +0.06 and -1.17 floors toward the target (primary effect is occupancy, 13.6 -> 8.0 / 9.1 pp). (iv) Team FT slope on the v3 reference is 0.617 (old 0.958): the responsiveness line is worse on v3 for every arm and is not repaired by any arm tested (R8bS 0.726, CL4 0.738 best). (v) The F5 / F5e LUTs were fitted on the pre-fix foul state; on v3 they are far from calibrated (occupancy gaps 27-36 pp). CL4 ran with another lane's uncommitted `loop.py` / `chance_time.py` in the tree and R8aS with `adapters.py` dirty (section 6); their reads should be repeated on a clean tree before anyone leans on them. The old per-arm "decision" status in the ledger is unchanged; this doc only restates the grader's rule outcomes.

### 5.3 Shot block round 2 (`shot_block_drawn_flag_2026-09-30.md`), 2 runs (K2_Ocell skipped: pending, box)

Reference `clk6_R_s25` on v3: G4 OREB% 0.2821 (target 0.2984), G9 total bias +0.393, team offence OREB slope 0.909. Floors (max of 4): G4 OREB 0.0008, G4 TOV 0.0003, G1 mean 0.077, G5 total 0.0111, G9 total 0.086, team slope 0.0019.

| arm | primary G4 OREB%, floors toward: old -> new | old vetoes | new vetoes | team off slope: old -> new | G9 total bias | flip |
|---|---|---|---|---|---|---|
| K2_Ocell_noteam | +37.5 -> **+9.38** (OREB 0.2896; effect +0.0075) | G5 total ratio (-11.3), G4 TOV (-1.5) | G4 TOV (-1.33; floor 0.0003 is tiny), **team slope falls** (0.909 -> 0.848, -0.061 = -33 floors) | 0.550 (-0.35 floors) -> 0.848 (-33 floors vs this R) | +0.393 -> +0.950 (-6.45 floors; not a pre-registered veto, reported) | G5 total veto clears; team-slope veto fires; primary shrinks |
| K2 | old per-line values not in the stored grade json; doc says both K2 arms fire G5 total and G4 TOV | as above | G4 TOV (-1.33), team slope falls (0.845) | -> 0.845 | +0.943 (-6.36) | same pattern as the noteam arm; primary +9.25 floors |

Neither arm is put forward by the grader (`put_forward=False`). OREB% pooled gate status flips FAIL -> PASS for both (0.2896 vs target), as before. Underpowered: none of the lines above (25 seeds, 500 games); the G4 OREB primary is far outside its floor, G4 TOV (floor 0.0003) is not meaningful against a 0.0004 move.

## 6. Commits and trees used (from each run's `run_meta.json`)

| run | engine commit | dirty src |
|---|---|---|
| e3_lg2_R | e5dd38c | `engine/loop.py` (other lane; output bit-identical to clk6_R) |
| e3_lg2_W_D | a67962d | none in `src/` |
| e3_lg2_W_C2, E_BL3 | acad375 | none in `src/` |
| e3_lg2_Rfloor | 1489cab | none |
| e3_lg2_E_L0S0 | 1d2aff2 | none |
| e3_lg2_W_C2_E_BL3, W_D_E_BL3 | e978716 | none |
| e3_fj_R | 3938a8d | none in `src/` |
| e3_fj_R8a | 1729088 | none in `src/` |
| e3_fj_R8aS | e53e950 | **`engine/adapters.py` (other lane)** |
| e3_fj_R8bU, R8bS | 759be01 | none |
| e3_fj_CL1 | daa759a | none in `src/` |
| e3_fj_CL3 | 1489cab | none |
| e3_fj_CL4 | 141af9d | **`engine/loop.py`, `engine/chance_time.py` (other lane)** |
| e3_fj_F5e / F5 | 9fb7d9e / e978716 | none |
| e3_Rf3, e3_Rf4 | 58348b3 / 80ab753 | none in `src/` |
| e3_sb_K2 / K2Onoteam | bafee9d / 28c03f6 | none |

## 7. Summary of flips (v2 / old truth / old sample -> v3 / verified)

| round | line | old -> new | flipped? |
|---|---|---|---|
| late game | W_D primary | +4.88 -> +3.26 floors; ratio 0.854 -> 0.918; still < 1 | no |
| late game | W_C2 primary | -0.13 -> +0.16 floors | no (null both) |
| late game | all duration arms, vetoes | G1 mean / half share (/ G1 SD) -> adds G9 total bias | veto set widens |
| late game | E arms | pass -> pass | no |
| foul joint | R8a, CL1 vetoes | V3 fail -> all four pass | **yes** (grader-eligible; Decision 11 applies, nothing adopted) |
| foul joint | R8aS, R8bU, R8bS, CL3, CL4 | V3 fail -> V3 fail | no (failing line moves from G5 total SD to score corr or G9 total bias) |
| foul joint | team FT slope on reference | 0.958 -> 0.617 | reference moved |
| foul accrual F5 / F5e | on v3 | V1 fails (occupancy 27-36 pp) | new read |
| shot block | K2_Ocell_noteam / K2 primary | +37.5 -> +9.4 / +9.25 floors | no (still passes) |
| shot block | vetoes | G5 total + G4 TOV -> G4 TOV + team slope falls | G5 total clears, slope veto fires |

## 8. NOT RUN, and the adoption caveat

- NOT RUN: `ENGINE_SHOT_BLOCK=K2_Ocell`, `ENGINE_FOUL_JOINT=R8b` / `CL2`, clock `L2` (box lane, PM order); clock round-6 `L1` (`ENGINE_ANDONE_LABEL=training`; refused on the OT veto; resume: `CBB_TRUTH` unset, `python scripts/run_po4b_closed_loop.py --arm round2_s1 --input-dir data/processed/models/engine_v3 --seeds 25 --workers 2 --no-players --tag e3_clk6_L1_s25` with `ENGINE_ANDONE_LABEL=training` exported, graded with `scripts/grade_clock_r6_v1.py`); po4b round 4b `G2` / `G3` (`ENGINE_EVENT=round4b_G2/G3`, 09-18, refused): their event artifacts and per-row team block were built on v2 inputs and were not rebuilt for v3, so a v3 run would silently mix a v2 event block with v3 team arrays (the overlay trap in `engine_inputs_v3_tag_path_2026-09-30.md`); not attempted. Late-game combos and controls all ran; 200-seed gates were not run.
- After the adoption commit the served defaults change. An "S0" reference from then on needs the explicit pre-adoption flag values (`ENGINE_CLOCK=v5b_glat_pmean`, `ENGINE_EVENT=round2_s1`, `ENGINE_FG_MAKE=round4_B1`, `ENGINE_REBOUND=s1_weekly`, ... as in the digest flags of `parity_reference_windows_v7.json`); every re-read above used those defaults at the commits in section 6, and `e3_lg2_R_s25` / `clk6_R_s25` are the S0 references.
- Resume (any one arm): `bash scripts/chain_lanee_v3_v3.sh` skips tags whose `games.parquet` exists; graders: `scripts/grade_lanee_lg2_v3_v1.py --out ...`, `scripts/grade_lanee_fj_v3_v1.py <tags> --out ...`, `scripts/grade_shot_block_closed_loop_v1.py clk6_R_s25 <arms> --floor <draw>,clk6_R_s25` (four floor draws, floors re-maxed).
- Incidents: none involving other lanes' processes. I killed only my own chain shells (PIDs 522075, 522261, 524448) twice to reorder the queue; the in-flight python runs were left to finish. The shot-block grader overwrites `results/shot_block_round2/closed_loop_grade_v1.json` on each run; the original was restored from `results/laneE_old/`.
