# Foul round 9: first-half FT-trip production on the corrected team-foul state (possession-outcome round 9) -- 2026-09-30

Lane C, overnight 2026-09-30. **NOTHING IS ADOPTED AND NO DEFAULT IS CHANGED.** The PM decides.
Pre-registration `docs/models/possession_outcome/experiments.md` section 26 (commit `265e6fd`, pushed 21:35 EDT
before any fit); results section 28. Wall clock 20:41-23:20 EDT (timestamps in section 8).

## 0. Verdict

| line (pre-registered) | R8b | R9ao1 (AO1, tie-rule winner) | R9ao3 (AO3, responsive) | floor | verdict |
|---|---:|---:|---:|---:|---|
| Offline AO, fold 2 log loss vs AO0 (served constant) | -- | +1.28 floors (> 2 boot SE) | +1.35 floors | 0.000804 | AO1, AO2 (+2.26), AO3 all beat AO0; F1 confirms all; AO1-AO2-AO3 within 1 floor of each other -> tie rule picks AO1 |
| Offline T (trip mix x clock) vs T2c | -- | -- | -- | 0.000804 | **REFUTED**: T3t -3.2, T3s -4.2 floors (F1 -3.0 / -3.6) |
| **Primary H1 FTA/FGA** (actual 0.2383), 500 x 200 | 0.2503 | **0.2408** | 0.2409 | 0.00039 | **+24.5 / +24.3 floors toward** |
| Primary H2 FTA/FGA (actual 0.4186) | 0.4159 | 0.4185 | 0.4186 | 0.0017 | +1.6 / +1.5 toward (inside 2) |
| Pooled event-layer FTA/FGA (0.3301) | 0.3338 | 0.3304 | 0.3305 | 0.0014 | +2.5 toward |
| G3/G4 box FTA/FGA (0.3295) | 0.3338 | 0.3304 | 0.3305 | 0.00031 | +10.9 / +10.5 toward |
| Team FT-rate slope ratio (prior quintiles; S0 draws 0.69-0.77) | 0.728 | 0.753 | 0.800 | paired SE 0.036 / 0.095 | no fall (veto passes) |
| G1-G8, G9 margin and slope, TOV (G4 tov_pct), G5 by component | -- | all inside 2 floors | all inside 2 floors | 5 S0 draws | pass |
| **G9 total bias** (0) | -0.634 | **-0.825** | -0.816 | 0.061 (draw SD; paired SE 0.016) | **AWAY -3.1 / -3.0 floors: VETO** |

**R9ao1 and R9ao3 close 79% of the H1 overshoot and fail one veto, G9 total bias.** That veto failure is the
exposure of a compensation, not a new defect (section 5): R8b's first-half free-throw excess (+0.38 FTA, +0.26
points per game) was offsetting part of the engine's total under-prediction. R9's total bias equals S0's
(-0.825 vs -0.822). Under Decision 11 this reads as **VALIDATED-PENDING-SHIP-ACTION** (the PM's call; not
adopted, not refused). The full-size 5,710 x 200 reads (tier 2) were queued on the box and had not run when
this was written (section 7).

## 1. Step 1: the first-half overshoot, decomposed (closed)

Instrument: `scripts/run_foul_joint_tap_v2.py` (round-7 tap plus `--sample-file`, start type and
`--agg-halves`). It runs on v3 inputs (S0 tag) with the box's S0 event-block mount reproduced in-process
(`R9_ENGINE_DIR`, `scripts/ops_r9_engine_dir_v1.py`).

**Parity: local runs are bit-identical to the box on every shared row (26 game columns).**
- S0 and R8b, 60 games x 3 seeds, before and after the wiring;
- R9ao1 and R9ao3 at 25 seeds against the box's 200-seed runs;
- the box's tapped runs against its untapped runs (100,000 rows each).

Decomposition script: `scripts/diag_foul_r9_h1_decomp_v1.py`; outputs `results/foul_r9/decomp_h1_v1.txt`
and `decomp_h1_r9_v1.txt`. Actual = the 2025 event layer on the CORRECTED (v2) state: the counter minus the
possession's own pre-open fouls, with and-one fouls added back (round-8 amendment). Sample = the verified
500 games; 25 seeds.

**Shift-share of the H1 FTA/FGA gap.** Cells are rule state x H1 minute bucket; the mean of both
substitution orders is shown, and the identity closes exactly.

| component | R8b | R9ao1 |
|---|---:|---:|
| occupancy (state x minute weights) | +0.0061 | +0.0031 |
| and-one FTA | +0.0052 | +0.0000 |
| shooting-trip FTA | +0.0067 | +0.0061 |
| bonus-trip FTA | -0.0059 | -0.0063 |
| other FTA (technical) | -0.0004 | -0.0004 |
| FGA | +0.0002 | +0.0001 |
| **total = sim - actual** | **+0.0119** (0.2502 vs 0.2383) | **+0.0027** (0.2410) |

Reading the components:
- **Trip production per state is right.** Shooting and bonus trips offset (net +0.0008). In the bonus R8b
  draws a few too many shooting trips and too few bonus trips; total trips per state match (one-and-one
  0.178 vs 0.181 per poss).
- **The reset at half is right.** The open count at the first H2 possession is 0 in the engine and in the
  data.
- **Accrual reaches the bonus too early, mostly through the and-one constant.** Fouls per team-half H1, R8b
  minus actual: +0.36, of which:
  - pace (34.23 vs 33.94 defensive possessions): +0.06;
  - per possession: +0.30, made of and-ones +0.15, non-trip (defence and offence lumped) +0.09, and bonus
    trips +0.08 (a consequence of the early bonus).
- **Time to the bonus.** Median minute of the first possession opened in the bonus: 14.2 (R8b) / 14.4
  (R9ao1) vs 15.0 actual. P(bonus by minute 10): 10.3% / 9.1% vs 6.3%.

**The and-one constant is the owner.** The served engine draws and-ones at a per-shot-class constant
(`rules.and_one_rate_given_made`, flagged `provisional_and_one`, never bake-offed). The data's rate rises
through each half:
- per made FG, minute 0-4 ... 38-40 (2025): 0.034 / 0.047 / 0.052 / 0.053 / 0.058 / 0.070 / 0.068 / 0.071 /
  0.067;
- H1 0.045-0.048 vs H2 0.065-0.069 in every season 2022-2025.

So the engine's H1 and-ones run +27% per possession (0.0214 vs 0.0169) and its H2 and-ones -7%.

**Segments (H1 FTA/FGA, R8b vs actual on the sample).**
- **Site:** away offence 0.237 vs 0.209, home 0.264 vs 0.258. The home whistle is lane G's round.
- **Defence prior-season foul-rate quintile, Q1 -> Q5:** R8b 0.241 -> 0.270 vs actual (full season)
  0.210 -> 0.274. The slope is compressed; the accrual law (A2) carries no team rate.
- **Start type:** DREB 0.275 vs 0.242, made_FG 0.221 vs 0.225.
- **Per game x offence:** MAE 0.127 / 0.125 (R8b / R9ao1), correlation 0.200 / 0.220.

## 2. Offline (both folds, one blind grader)

- Trainer: `scripts/train_foul_r9_v1.py` (90 s).
- Grader: `scripts/grade_foul_r9_v1.py`; output `data/processed/models/possession_outcome/round9/grade_round9.json`.
- AO rows: 1,114,777 made-FG chances, 2022-2025.

| block | arm | F2 log loss | F2 floors vs ref | boot SE of delta | F1 floors |
|---|---|---:|---:|---:|---:|
| AO | AO0 per-class constant (served definition) | 0.200686 | 0 | -- | 0 |
| AO | AO1 class x period x clock cells | 0.199654 | +1.28 | 0.000088 | +1.40 |
| AO | AO3 AO1 + prior-season team terms | 0.199604 | +1.35 | 0.000089 | +1.49 |
| AO | AO2 GBM on class + true state | 0.198867 | +2.26 | 0.000116 | +2.24 |
| T | T2c (R8b's term) | 0.360942 | 0 | -- | 0 |
| T | T3t T2c x clock | 0.363493 | -3.17 | 0.000094 | -3.00 |
| T | T3s half x clock x rule state x diff | 0.364321 | -4.20 | 0.000100 | -3.65 |

**AO winner.** AO2 - AO1 = 0.98 floors and AO3 - AO1 = 0.06 floors, so all three are ties. The
pre-registered tie order (AO0 < AO1 < AO3 < AO2) selects **AO1**.

**Responsiveness, offence / defence prior-quintile slope ratio:**

| arm | offence | defence |
|---|---:|---:|
| AO0 | 0.40 | 0.09 |
| AO1 | 0.40 | 0.08 |
| AO2 | 0.36 | 0.08 |
| AO3 | **1.22** | **1.27** |

- AO1 is **NOT RESPONSIVE** on the defence side (the standing CLAUDE.md rule). AO3 is the responsive form.
- Both went to the closed loop. **The PM picks between the pre-registered tie rule (AO1) and the standing
  responsiveness rule (AO3).** They are indistinguishable on every closed-loop line.

**Calibration by half (pp):**
- AO0: H1 +0.86, H2 -0.74; minute 0-4 +1.98, minutes 25-37 -1.0 to -1.1.
- AO1: H1 0.00, H2 +0.08; every minute bucket within +-0.27.
- Class and site gaps are unchanged by any arm (away +0.21 pp: the home whistle).

**Level.** The chance-layer class rates (rim 0.0929, jump2 0.0433, three 0.0062) sit about 6% below the
served constants measured on the fg_make design (0.0989 / 0.0467 / 0.0067). R9 carries the event-layer level,
the same layer every actual in this round is read from. Closed-loop and-ones per possession: R9ao1
0.0168 / 0.0244 (H1 / H2) vs actual 0.0169 / 0.0246.

**Block T has no winner.** The clock dimension fragments T2c's cells and loses on both folds, so R8b's trip
term stands. The bonus-state trip MIX is a +-0.006 offset that nets to zero (section 1). It is not pursued.

## 3. Wiring and parity

| commit | what |
|---|---|
| `525fabd` | new module `src/cbb_sim/engine/foul_r9.py` |
| `525fabd` | `foul_joint.load` delegates `R9*` names (3 lines) |
| `525fabd` | one `loop.py` hunk at the and-one draw: same `and_one` uniform; the scalar is used unless `fj.ao` exists |
| `bd03ea9` | LUTs `round9/lut_ao_AO{1,3}_F2.npz` + `ao_team_prior_v1.parquet` (fold-2 train fits only), force-added |

Arms: `ENGINE_FOUL_JOINT=R9ao1|R9ao3` = R8b + the and-one LUT.

Parity:
- `run_engine.py` 60 x 5, digest vs `docs/ops/parity_reference_windows_v6.json`: **PASS, bit-identical**
  (HEAD `222a570`);
- unset flag and `R8b`: bit-identical to `v3box_S0_s200_o0` / `v3box_R8b_s200_o0` on the shared rows;
- box: flag-off v6 parity PASS (operator, clone `7d38de5`).

## 4. Closed loop (500 verified games x 200 seeds, v3 inputs, verified truth)

Run on the box (request `docs/ops/box_queue/laneC_1.md`, done note `laneC_1.done.md`). The same lines at
500 x 25 locally agree within noise (H1 +13.3 floors with 5 local draws; `results/foul_r9/cl_s25_v2.md`).

Floors (Decision 12):
- half lines: max(SD over 5 S0 seed-offset draws 0/1000/2000/3000/4000, paired game-bootstrap SE);
- gate lines: SD over the same 5 S0 draws.

Graders: `scripts/grade_foul_r9_closed_loop_v1.py` -> `results/foul_r9/cl_s200_v1.{md,json}`;
`scripts/diag_foul_r9_gate_delta_v1.py` -> `results/foul_r9/gates_s200/delta_vs_R8b_5draw.md`;
paired totals `scripts/diag_foul_r9_g9_paired_v1.py`.

**Gate lines, toward target vs R8b (floors; + = closer):**

| gate | R9ao1 | R9ao3 |
|---|---:|---:|
| G1 possessions mean | +0.63 | +0.63 |
| G1 possessions SD | -0.64 | -0.35 |
| G3 3PA share | +0.91 | +0.91 |
| G3 rim share | -0.77 | -0.77 |
| G4 TOV% | -0.71 | -0.71 |
| G4 OREB% | +1.35 | +1.35 |
| G4 eFG% | 0.00 | 0.00 |
| G5 margin SD ratio | +0.33 | +0.22 |
| G5 total SD ratio | -0.33 | +0.11 |
| G5 score correlation | +0.84 | +0.91 |
| G5 margin | +2.51 | +1.86 |
| G5 total | -0.07 | -0.01 |
| G6 home margin (non-neutral) | -0.22 | -0.53 |
| G7 OT rate | +1.70 | +1.70 |
| G8 (every line) | within -0.91 | within -0.91 |
| G9 margin bias | -0.27 | -0.65 |
| G9 calibration slope | +0.60 | +0.51 |
| **G9 total bias** | **-3.13** | **-2.97** |

**Paired per-game deltas vs R8b** (game bootstrap; Decision 12's second floor component):

| line | R9ao1 - R8b | R9ao3 - R8b | S0 - R8b |
|---|---|---|---|
| total points | -0.19 (SE 0.016) | -0.18 (SE 0.017) | -0.19 |
| FTM | -0.26 (SE 0.011) | -0.26 | -1.11 |
| FTA | -0.38 | -0.37 | -1.52 |
| FG points | +0.07 (SE 0.014) | +0.08 | +0.92 |
| margin | +0.01 (SE 0.023) | +0.03 | +0.04 |

## 5. Multi-level reading

1. **Overall.** H1 FTA/FGA 0.2503 -> 0.2408 (actual 0.2383). H2 is unchanged or slightly closer
   (0.4185 vs 0.4186). Pooled 0.3304 vs 0.3301 (event layer) and vs 0.3295 (box).
2. **The G9 total veto is a compensation exposed.** S0 -> R8b raised FTM by 1.11 per game and total points
   by only 0.19, because FT trips replace shots. R8b's total bias moved toward 0 (-0.82 -> -0.63) partly
   through first-half free throws the data does not have. R9 removes those (-0.26 FTM per game) and returns
   the total bias to S0's level (-0.825 vs -0.822). The under-predicted total belongs to the G9 / aggregation
   owner (PM ruling 2026-09-30, "aggregation over-spread is the new G9 owner"), not to the foul stack.
   Decision 11 applies.
3. **Per possession type and state** (25-seed taps, `decomp_h1_r9_v1.txt`). R9ao1's H1 FTA/FGA residual
   +0.0027 is occupancy (+0.0031): H1 bonus occupancy 0.212 vs 0.198, minute 10-14 0.251 vs 0.214. Per team-half
   fouls are now +0.17 (pace +0.05, non-trip +0.12). The non-trip part is A2's uniform +0.6 pp 2025 level
   drift (round 7). By start type, H1 FTA/FGA is DREB 0.264 vs 0.242 and made_FG 0.213 vs 0.225. And-ones by
   start type match actual within 0.0021 per poss.
4. **Per team.** FT-rate prior-quintile slope ratio:
   - R8b 0.728, R9ao1 0.753, R9ao3 0.800, against an S0 draw range of 0.685-0.773. The rise is not
     separable from noise.
   - Team SD ratio is 0.32-0.33 for every arm (actual 1): team FT-rate spread is compressed engine-wide and
     is not this round's object.
   - Defence foul-rate quintile, H1: R9ao1 0.232 -> 0.260 vs actual 0.210 -> 0.274. Still compressed; the
     accrual law has no team term.
5. **Per game** (FTA/FGA per game-side): MAE 0.1127 -> 0.1125, correlation 0.265 -> 0.266.
6. **Late game** (minute 38-40 bonus occupancy): S0 0.787, R8b 0.885, R9ao1 0.890; actual 0.920.

## 6. What this round does not establish

- **Full size (5,710 x 200) was not read** (tier 2 queued behind other lanes, section 7). The G5 ratio lines
  are inside 1 floor at 500 x 200; full size is the decisive read for them.
- **AO1 vs AO3 is not separable in the closed loop.** Picking one is a rule choice (section 2).
- The residual H1 occupancy has two owners outside this round:
  - A2's level drift: refit cadence; Decision 9's open dimension;
  - pace (+1.6 possessions per team-game, G1).
- Each is reported, not compensated.
- The fg_make-design vs event-layer and-one level gap (6%) is reported, not resolved. The truth for every
  line here is the event layer.

## 7. Not run, and resume commands

Tier 2 of `laneC_1` (full-size R9ao1 / R9ao3, paired with the operator's `v3full_{S0,R8b}_s200_o0` and S0
f1-f4), on the box, from the repo root:

    bash scripts/box_r9_v1.sh full R9ao1 90 0
    bash scripts/box_r9_v1.sh full R9ao3 90 0
    # then locally, with the operator's full-size grade files copied into one dir as <tag>__verified.md:
    .venv/Scripts/python.exe scripts/diag_foul_r9_gate_delta_v1.py --dir <dir> --ref v3full_R8b_s200_o0 \
        --arms v3full_R9ao1_s200_o0,v3full_R9ao3_s200_o0 \
        --floor-draws v3full_S0_s200_o0,v3full_S0f1_s200_o1000,v3full_S0f2_s200_o2000,v3full_S0f3_s200_o3000,v3full_S0f4_s200_o4000 \
        --out <dir>/delta_vs_R8b.md

Local reproduction:

    export PYTHONIOENCODING=utf-8 CBB_TRUTH=verified_v1
    .venv/Scripts/python.exe scripts/train_foul_r9_v1.py --n-jobs 3 && .venv/Scripts/python.exe scripts/grade_foul_r9_v1.py
    .venv/Scripts/python.exe scripts/build_foul_r9_lut_v1.py
    .venv/Scripts/python.exe scripts/build_engine_inputs_v3_tag_v1.py --tag <T>
    .venv/Scripts/python.exe scripts/ops_r9_engine_dir_v1.py <T> results/foul_r9/engine_dir_S0
    bash results/foul_r9/run_tap_S0_R8b.sh; bash results/foul_r9/run_r9_local.sh; bash results/foul_r9/run_floor_taps.sh
    .venv/Scripts/python.exe scripts/diag_foul_r9_h1_decomp_v1.py results/foul_r9/decomp_h1_r9_v1 \
        r9tapB_S0_s25 r9tapB_R8b_s25 r9tapB_R9ao1_s25 r9tapB_R9ao3_s25
    bash results/foul_r9/run_grade_s200.sh

## 8. Wall clock and incidents

All times EDT, from `Get-Date` and the run logs:

| time | event |
|---|---|
| 20:41 | start |
| 20:44 | first S0 tap on v3 inputs |
| 21:01 | overlay tag built; first taps discarded (below) |
| 21:03-21:33 | decomposition taps |
| 21:35 | pre-registration pushed |
| 21:36-21:38 | offline fits |
| 21:39 | wiring committed and parity |
| 21:42-22:09 | R9 taps |
| 22:09-23:04 | floor taps (2 workers) |
| 23:00 | box tier 1 done (02:31-03:00Z) |
| 23:05 | 200-seed grading |

Incidents:
1. **My first S0 / R8b taps on v3 inputs (20:44-21:01) were invalid.** They read the v1 event team block,
   because the box's docker mount was not reproduced. The parity check caught it (no shared row matched). I
   stopped my own R8b tap (PIDs 37812 / 57884 and its three workers) and deleted both outputs. No other
   process was touched.
2. I restarted my own floor-tap run once (22:09) to drop from 3 to 2 workers, so the graders could run
   inside the 3-core cap.
3. `docs/models/possession_outcome/experiments.md` and `change_ledger.md` were rebuilt once from HEAD plus my
   append. My first write normalised the files' mixed line endings; I caught it before commit, and the
   committed diff is the append only.
4. Section numbering: lane G had taken section 25 at 21:10, so this round is section 26 (pre-registration);
   lane G then took section 27, so the results are section 28.
