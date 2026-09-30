# Foul accrual + FT-trip production, joint round (possession-outcome round 7) -- 2026-09-30

Lane A. **NOTHING IS ADOPTED AND NO DEFAULT IS CHANGED.** The PM decides.
Pre-registration: `docs/models/possession_outcome/experiments.md` section 20 (commit `aa0ee91`,
pushed 11:22 EDT, before any fit). Results: section 21. Wall clock of the round: 10:53-12:45 EDT.

## 0. Verdict in one table

| arm | what it serves | offline (F2 floors vs block ref; F1) | closed-loop FTA/FGA (target 0.32955) | floors toward | V1 occupancy | V2 team slope | V3 other gates | V4 team SD | eligible |
|---|---|---|---:|---:|---|---|---|---|---|
| `CL0` served | constant 0.123346, served PO | -- | 0.31953 | 0 | max gap 13.07 pp | 0.958 | -- | 0.8209 | ref |
| `CL1` | `A1` table (half x true count) | +16.5; F1 +15.9 | 0.31855 | -0.55 | 8.66 pass | 0.983 pass | G5 total ratio -6.3 fl **FAIL** | 0.8106 **FAIL** | no |
| `CL2a` | `A2` GBM accrual, true state | +18.1; F1 +17.5 | 0.31983 | +0.17 | 7.73 pass | 0.994 pass | G5 total -4.6 fl **FAIL** | 0.8213 pass | no |
| `CL2` | `A2` + `T2c` trip offsets | T +8.9; F1 +7.9 | 0.33488 | +2.63 | 6.21 pass | 0.846 **FAIL** | G5 total -13.9 fl **FAIL** | 0.7863 **FAIL** | no |
| `CL3` | `DO2` (defence + offensive fouls) + `T2c` | DO +50.8; F1 +55.5 | 0.33764 | +1.09 | 6.81 pass | 1.030 pass | G5 total -11.4 fl **FAIL** | 0.8084 **FAIL** | no |
| `CL4` | `DO2` + `T2lab` (round-6 T2 definition) | T2lab **-28.0**; F1 -27.3 | 0.32376 | +2.37 | 7.69 pass | 0.963 pass | G5 total -9.1 fl **FAIL** | 0.9367 pass | no |
| `F5e` (09-18, comparison) | round-6 labelled-state accrual | -- | 0.30215 | -9.76 | 26.82 FAIL | 1.045 | G5 total -2.2 fl FAIL | 0.8323 | no |

**No arm is eligible.** Every arm fails V3 on one line only, the G5 total-points SD ratio
(`cbb_sim.eval.gates` definition: mean within-game sim SD / SD(actual - sim mean); reference
0.8835, floor 0.00105). G1 mean and SD, the G5 margin ratio and the G9 total bias stay inside
2 floors or move toward for every arm. Best primary: `CL2` (+2.63 floors, overshoots by +0.53 pp)
and `CL4` (+2.37, undershoots by -0.58 pp). `CL4` is the only arm that passes V1, V2 and V4, but
its trip term is the arm that loses 28 floors offline when fed the engine's state.

## 1. Step 1 -- instrumentation and where the trips are lost

`scripts/run_foul_joint_tap_v1.py`: an in-process tap (`StreamBook`, `categorical`,
`state.new_state`, `_foul_p` wrapped in its own process; `src/cbb_sim/` untouched by it). It writes
a per-possession `poss_tap.parquet` (open state, both team-foul counts, trip classes drawn, trip
fouls, silent/offensive draws, FTA, FGA) and `games_v2.parquet` (= `games.parquet` + team ids,
neutral, date; `games.parquet` unchanged). **Proved neutral:** `fj_R_s25` is bit-identical to
`po4b_R_s25` (12,500 x 28) and `fj_F5e_s25` to 09-18's `foul6_F5e_s25`.

Actual side: `chances_2025` joined to a replay rebuilt with pre-open foul counts,
`scripts/build_foul_accrual_design_v2.py` -> `round6/foul_accrual_poss_v2.parquet` (sibling; v1
untouched). Decompositions: `scripts/diag_foul_joint_decomp_v1.py` (labelled state) and
`_v2.py` (engine-definition state). Units per possession, so +2 possessions/game cancel.

**Overall (per possession, 2025):**

| | actual | served | F5e | CL2a | CL2 | CL3 |
|---|---:|---:|---:|---:|---:|---:|
| fouls (all) | 0.2235 | 0.2721 | 0.2212 | 0.2462 | 0.2494 | 0.2512 |
| non-trip fouls | 0.0714 | 0.1237 | 0.0796 | 0.0969 | 0.0939 | 0.0947 |
| shooting trips | 0.0725 | 0.0728 | 0.0745 | 0.0723 | 0.0701 | 0.0698 |
| bonus trips | 0.0587 | 0.0533 | 0.0447 | 0.0547 | 0.0632 | 0.0646 |
| FTA/FGA (event layer; box 0.3296) | 0.3267 | 0.3195 | 0.3021 | 0.3198 | 0.3349 | 0.3376 |

**By game minute (bonus trips per possession, sim - actual):** served +0.000..+0.006 in minutes
0-19, **-0.008 / -0.026 / -0.029 / -0.053** in 25-29 / 30-34 / 35-37 / 38-40. Shooting trips and
and-ones within +-0.009 everywhere. The trip shortfall is located: second-half bonus trips.

**By half, FTA/FGA:** actual H1 0.2367 / H2 0.4126 / OT 0.739; served 0.2540 / 0.3789 / 1.157;
`CL2a` 0.2428 / 0.3917 / 1.148; `CL2` 0.2506 / 0.4175 / 0.704; `CL3` 0.2520 / 0.4215 / 0.710.
Final 2:00 of regulation: actual 0.9366; served 0.7858; `CL2` 0.8701; `CL3` 0.8802; `CL4` 0.8644.

**Why (the decomposition closes: fouls = trip + non-trip per possession in every cell):**

1. The served constant is not the non-trip rate. Honest non-trip events under the engine's
   attribution are 0.0915 per possession (binary; 0.094 as a count), not 0.1233.
2. **The actual state labels are shifted.** `_GameMachine._open` reads the team-foul counter
   AFTER fouls committed while no possession is open, so `def_team_fouls` / `in_bonus` at open
   include the possession's own pre-open fouls (0.173 defence fouls/poss, 16.5% of possessions).
   The engine's state is the count before them. On the engine definition, actual non-trip fouls
   rise with the count before the bonus (H2 counts 0-5: 0.108-0.128/poss) and are ~0 in the
   bonus; the served engine draws 0.123 at every count, bonus included.
3. The served PO learned the shift: true count 5 shooting trips actual 0.109 vs engine 0.082;
   in-bonus shooting trips actual 0.034-0.036 vs engine 0.050-0.064.
4. The count distribution is over-dispersed and does not self-regulate: minute 35-37 mean count
   7.56 engine vs 7.53 actual, SD 3.45 vs 2.48, P(in bonus) 0.658 vs 0.789. Actual pre-bonus
   arrival depends on the foul DIFFERENTIAL (def - off = -4: 0.264 / 0.321 per poss H1 / H2; 0:
   0.159 / 0.233); the served engine is flat at 0.210-0.243.

So the served 0.123346 was compensating for missing state structure (the "0.0242 hidden
compensation" of 09-18 is this, not a trip-count shortfall: total fouls are HIGH in the served
engine). F5e failed in closed loop because it was fitted on the shifted labels (`A2lab`, the
same law fitted on labels and queried with the engine's state, is -10.6 floors offline).

## 2. Offline (both folds, one blind grader)

`scripts/train_foul_joint_v1.py` (24 LightGBM fits n_jobs=1 on 6 joblib workers, 238 s; 4
logistic T0 fits), `scripts/grade_foul_joint_v1.py` -> `round7/grade_round7.json`. Scored on
test rows fed the TRUE state. Applied floor 0.000804 (measured reseed floors <= 1.3e-4; every T
arm and every cell arm is deterministic).

| block | arm | F2 log loss | F2 floors vs ref | F1 floors | calib gap pp (F2) | team SD ratio | prior-quintile slope |
|---|---|---:|---:|---:|---:|---:|---:|
| A | `A0` const | 0.3088302 | 0 | 0 | +0.64 | 0 | 0 |
| A | `A1` cells | 0.2955651 | +16.50 | +15.92 | +0.63 | 0.471 | -0.35 |
| A | `A2lab` | 0.3173176 | **-10.56** | -9.99 | +0.02 | 0.489 | +0.22 |
| A | **`A2`** | 0.2942592 | **+18.12** | +17.54 | +0.62 | 0.585 | -0.38 |
| A | `A2_D9a` opp-adj | 0.2938705 | +18.61 (tie with A2: +0.48) | +17.93 | +0.63 | 0.460 | +0.11 |
| A | `A2_D9b` conf flag | 0.2942009 | +18.20 (tie) | +17.62 | +0.62 | 0.585 | -0.38 |
| D/O | `DO0` | 0.3611394 | 0 | 0 | -0.09 / +0.79 | 0 | 0 |
| D/O | **`DO2`** | 0.3203018 | **+50.79** | +55.50 | -0.09 / +0.70 | 0.78 / 0.48 | -0.37 / +0.22 |
| T | `T0` | 0.3413912 | 0 | 0 | -0.18 / +0.21 | 0.747 / 0.349 | 0.77 / 0.30 |
| T | `T1c` | 0.3374354 | +4.92 | +4.44 | +0.02 / +0.17 | 0.765 / 0.384 | 0.79 / 0.27 |
| T | `T2lab` | 0.3638793 | **-27.97** | -27.31 | -0.25 / -0.01 | 0.936 / 0.503 | 0.98 / 0.30 |
| T | **`T2c`** | 0.3342244 | **+8.91** | +7.92 | +0.00 / +0.18 | 0.776 / 0.424 | 0.81 / 0.28 |

(T and D/O columns: bonus / shooting and defence / offence.) Winners `A2`, `DO2`, `T2c`, all
confirmed on fold 1. Decision 9 arms tie with `A2` (`D9a` +0.48 floors, `D9b` +0.07): NULL again.

Segments (F2, calibration gap pp): `A2` by true count 0-3 / 4-5 / 6 / 7-8 / 9 / 10+ = +0.56 /
+0.61 / +0.87 / +0.81 / +0.63 / +0.77 (`A0`: -1.2 / -2.3 / +7.9 / +7.9 / +7.8 / +7.9; `A2lab`: -2.2 /
+1.4 / **+11.1** / +1.8 / +0.6 / +0.7); by half +0.72 / +0.52; site +0.64 / +0.62 / +0.57; conf /
non-conf +0.74 / +0.42; month Nov +0.04 ... Mar +0.88 (Apr n=2,077 fine). The uniform +0.6 pp is the
2025 level drift of the non-trip rate (train pool 2022-24 higher), not a state defect. `T2c`
bonus by true count: +0.01 / -0.16 / +0.08 / +0.16 / +0.01 / +0.18 (`T0`: -0.61 at 4-5, -1.34 at 9;
`T2lab`: **-12.0 at 6, +8.6 at 10+**). All cells powered (>= 17,000 rows).

Responsiveness: **no accrual arm is team-responsive** (prior-quintile slopes negative or ~0;
the engine cannot serve as-of team foul rates, `A2` is state-only by design). The T block
inherits the served PO's team slope (bonus 0.77-0.81, shooting ~0.28-0.30).

## 3. Wiring and parity

New module `src/cbb_sim/engine/foul_joint.py`; `loop.py` gets four minimal hunks (import, load,
the trip-offset call after `ad.event.predict`, the accrual branch). Flag `ENGINE_FOUL_JOINT`
(unset / `reference` = served path). LUTs from the FOLD-2 TRAIN fits only
(`scripts/build_foul_joint_lut_v1.py`; binning cost `A2` 0.00055, `D2` 0.00041, `O2` 0.00005 log
loss). The defence and offence draws share the one `foul_accrual` uniform: no RNG family added.

Parity, both PASS: (i) `run_engine.py --seeds 5 --max-games 60` digest sha256 `0d4ddccc...`
equals `docs/ops/parity_reference_windows_v6.json`; (ii) `fj_PARITY_s25` (full 500 x 25 default
path, edited engine, HEAD `005c7dd`, engine code as committed in `26f2ff6`) vs `po4b_R_s25`: games 28/28 and players 16/16 columns
bit-identical. `tests/test_engine.py` + `tests/test_smoke.py`: 22 passed.

## 4. Closed loop, multi-level

Grader `scripts/grade_foul_joint_closed_loop_v1.py` -> `results/foul_joint/closed_loop_grade_v1.{json,txt}`.
Floors from `po4b_R_s25_floor` vs `po4b_R_s25`.

**Occupancy by game minute, gap vs the TRUE-state actual (pp).** Actual: 0.0 / 2.6 / 21.4 / 55.6 /
0.3 / 12.3 / 51.5 / 78.9 / 91.0 %.

| arm | 5-9 | 10-14 | 15-19 | 25-29 | 30-34 | 35-37 | 38-40 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| served | +3.5 | +9.6 | +5.9 | -2.6 | -10.7 | -13.1 | -12.2 | 13.07 |
| CL2a | +1.5 | +4.4 | +1.1 | -1.0 | -6.0 | -7.7 | -7.4 | 7.73 |
| CL2 | +1.7 | +6.2 | +4.4 | +0.8 | -0.1 | -1.3 | -2.3 | 6.21 |
| CL3 | +1.9 | +6.8 | +5.5 | +1.7 | +1.8 | +0.1 | -1.3 | 6.81 |
| CL4 | +2.4 | +6.0 | +1.2 | +1.2 | -3.9 | -7.4 | -7.7 | 7.69 |

Team-foul count SD at minute 35-37 (actual 2.48): served 3.45, CL2a 2.83, CL2 2.60, CL3 2.57, CL4 3.17.

**Per team** (prior-season 2024 FT-rate quintiles, same 500 games): gap Q1 / Q5 pp served -2.05 /
-2.20, CL2 -0.34 / -0.87, CL3 -0.33 / -0.23, CL4 -1.35 / -1.48. Slope ratio in the table of section 0.

**Per game** (FTA/FGA per game, sim mean vs actual): MAE served 0.0884, CL2 0.0880, CL3 0.0878,
CL4 0.0893; correlation 0.178 / 0.195 / 0.196 / 0.155. **Home whistle** (non-neutral, FTA/FGA
home - away): actual +5.59 pp; served +3.32, CL2 +3.19, CL3 +3.50, CL4 +5.34.

**Other gates:** G1 possessions 70.02 (round-7 arms within 0.21 floors except CL4 +0.74 toward);
G1 SD gap vs box +0.34 (CL4 +0.24, toward). G5 margin ratio 0.969 (every round-7 arm within 0.8 floors; F5e -1.45).
**G5 total ratio 0.8835 -> 0.869-0.879 in every arm (-4.6 to -13.9 floors): the veto that blocks
everything.** G9 total bias -1.26 (CL2 -1.10, CL3 -1.01: toward). OREB% 0.2836 -> 0.2827-0.2839.

## 5. Reading (for the PM; nothing here is a decision)

1. **The accrual law on the engine-definition state fixes the bonus timing** (`CL2a`: max
   occupancy gap 13.1 -> 7.7 pp, H1 FTA/FGA toward actual) **but not the aggregate** (+0.17
   floors): at a given occupancy the second half still produces too few trips.
2. **The trip-class repair closes the second half** (`CL2`: H2 0.4175 vs 0.4126, late occupancy
   within 2.3 pp) **and overshoots the aggregate** through the first half (H1 0.2506 vs 0.2367;
   minutes 10-19 still +4 to +6 pp over-occupied). A likely owner of the overshoot, NOT tested:
   finding 7 below (the pbp count lags the true count, so offsets learned on pbp counts are
   queried at engine counts that run ahead of them).
3. **G5 total SD ratio narrows in every arm.** The served over-dispersed foul count was carrying
   within-game total variance: a second hidden compensation, now exposed. Under the
   pre-registered V3 it blocks eligibility; the G5 narrowness itself is owned by lane B.
4. `T2c` compresses team FT-rate spread (V4 0.821 -> 0.786) and loses team slope (0.846);
   the offensive-foul mechanism (`CL3`) restores the slope (1.030) but not the spread.
5. `CL4` looks best on vetoes and on the home whistle, but its trip term is the arm that is
   -28 floors offline on the engine's state (bonus -12 pp at count 6): it reaches its aggregate
   through a mis-specified conditional, and its per-game correlation falls (0.155).
6. The round-6 contradiction is resolved: F5/F5e failed because their state labels include the
   possession's own pre-open fouls, not because the engine under-produces trip fouls.
7. **Found after pre-registration, not an arm:** 2.26% of 2025 possessions hold an FT trip with no
   foul row of their own (15% matched by an extra row in the previous possession); ~1.55 fouls per
   team-game, against the box-minus-pbp foul gap of 1.60 (box 16.96 vs pbp replay 15.36). The pbp
   team-foul counter, and so every bonus/count label, lags the true count by these. Proposed next
   object (not pre-registered): state v3 = v2 + those trips counted into the running count, then
   re-fit `A2`/`T2c` and re-run `CL2`/`CL3`.

## 6. What this round does NOT establish

- No arm is eligible; nothing is proposed for adoption. The closed loop is one 25-seed paired run
  per arm; the V2/V4 team statistics are on ~3 team-games per team per seed (actual side on the
  same games) and have no measured noise band of their own.
- The trip term is a logit offset on the served PO through its logistic proxy `T0`, not a refit
  of the served LightGBM (declared in 20.2).
- The accrual LUTs evaluate no team rates (the engine does not carry them).
- OT, final 2:00: reported, not targeted (late-game lane).
- Technical FTs (-0.28 pp of the box gap) are outside this round.

## 7. Resume commands

    export PYTHONIOENCODING=utf-8
    .venv/Scripts/python.exe scripts/build_foul_accrual_design_v2.py              # 35 s, 4 workers
    .venv/Scripts/python.exe scripts/train_foul_joint_v1.py --n-jobs 6            # ~5 min
    .venv/Scripts/python.exe scripts/grade_foul_joint_v1.py
    .venv/Scripts/python.exe scripts/build_foul_joint_lut_v1.py
    # closed loop (6 workers, ~7-14 min each on a shared box); reference tag has no --env
    .venv/Scripts/python.exe scripts/run_foul_joint_tap_v1.py --tag fj_R_s25 --workers 6
    for a in CL1 CL2a CL2 CL3 CL4; do .venv/Scripts/python.exe scripts/run_foul_joint_tap_v1.py \
        --tag fj_${a}_s25 --env ENGINE_FOUL_JOINT=$a --workers 6; done
    .venv/Scripts/python.exe scripts/grade_foul_joint_closed_loop_v1.py fj_R_s25 fj_CL1_s25 \
        fj_CL2a_s25 fj_CL2_s25 fj_CL3_s25 fj_CL4_s25 fj_F5e_s25
    .venv/Scripts/python.exe scripts/diag_foul_joint_decomp_v2.py results/foul_joint/decomp_v2_all.json \
        fj_R_s25 fj_CL2a_s25 fj_CL2_s25 fj_CL3_s25
    # parity: run_engine.py --seeds 5 --max-games 60 --workers 2 --tag <t>; then
    .venv/Scripts/python.exe scripts/digest_engine_run.py --compare docs/ops/parity_reference_windows_v6.json --results results/engine_v0/<t>

`data/processed/models/possession_outcome/round7/` (~400 MB of predictions + 1.4 MB of LUTs) is
gitignored under `data/processed/models/*/round*/`; the arms need the LUTs, so they are rebuilt by
the two commands above (not synced to HF by this lane).

## 8. Incidents

None. Every process this lane ran was its own (parents identified by command line and PID); no
other process was touched. `loop.py` was edited in one commit (`26f2ff6`) with only this lane's
four hunks after a CRLF restore. Three of this lane's own new scripts were re-saved with CRLF line
endings between commits (whole-file churn in their own history only).
