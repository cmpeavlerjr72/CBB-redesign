# G9 margin over-spread: closed decomposition by channel, S0 vs S1 (Lane A, overnight 2026-09-30)

DIAGNOSTIC ONLY. Nothing is adopted, served or re-defaulted. No engine module was edited. Every counterfactual is an in-memory swap of input arrays, or a substitution of artifact sets through the tagged-inputs builder.

**Pre-registration**
- `docs/models/aggregation/experiments.md` section 1, commit f14886f (20:58 EDT), before any swap arm ran.
- Addenda A-D (sections 1.8-1.11), each committed before it ran: 3f25bac, 3938a8d, 2302801, 30cf4fb. Addendum D's header says "22:00 EDT"; the commit is 21:51.
- Fix round: section 2, commit 0dc0171. The box round itself is NOT run.
- Addenda E, F, F2 and G (sections 2.1-2.5): each committed before it ran (8a0f44c, bfe5c4c, 3fbd1dd, 0aaa257, 8200a93). Several of these headers carry a time written by estimate, 2-9 minutes later than the commit's wall clock. The commit times are the record.
- Ledger: one row in `docs/models/change_ledger.md` section A.

**Scripts**
- `scripts/exp_aggregation_swap_v1.py`: the paired closed-loop swap runner. It serves each stack's overlay in-process.
- `scripts/diag_aggregation_harness_v1.py`: the deterministic expected-points harness.
- `scripts/diag_aggregation_overspread_v1.py`: parts `identity`, `harness`, `rates`, `factorial`, `loop`.
- `scripts/diag_aggregation_addendum{B,C,D}_v1.py` and `scripts/diag_aggregation_sim_vs_harness_v1.py`.
- `scripts/diag_aggregation_{g2,overfit,parity,devoverride,tfix}_v1.py`.
- Trainer wrappers: `scripts/train_fg_make_v4_par_g2_v1.py` and `scripts/train_fg_make_v4_par_rawfix_v1.py`. Both wrap `train_fg_make_v4_par_v1.py`, which is unchanged.
- Outputs are in `results/aggregation_v1/` (`analysis_*_v1.json`, `harness_*.parquet`, the box reads `X_F_FULL_s200_o0`, `X_PR_FULL_s200_o0`).

**Data**
- Fold 2 (2024-25): 5,705 graded games, 5,700 of them with box channels.
- Truth: verified finals, `CBB_TRUTH=verified_v1` set explicitly.
- Close: ESPN BET from `lines_close_v2_verified`, 5,380 games lined.
- 2025-26 was not touched.

**Stacks**
- Each stack was rebuilt locally with `build_engine_inputs_v3_tag_v1.py`:
  - `S0_laneA`: defaults (= `engine_v3`).
  - `S1_laneA`: the box's `S1` command (E3 v4 table plus the Stage B `T` artifacts, pulled from HF).
  - `R_laneA` and `R2_laneA`: served features with the Stage B retrains, seed 0 and seed 1.
  - Factorial stacks `X_*` and seed-floor stacks `Z_*`.
- Reproduction checks (0 differences in every case):
  - S0 matches the box's 200-seed rows on 9,136 rows x 26 columns.
  - S1 matches on 480 rows.
  - The box's `X_F` and `X_PR` reads match on 120 rows each.
  - The box's swap arms `S1_TEAM`, `S1_PO` and `S0_FG` match on 80 rows each.
  - The local fg_make `T` retrain (`G0`) reproduces the box `T`'s log losses and harness margin exactly.
- The builder's npz sha256 values differ between builds of identical content. The npz is a zip whose metadata differs between builds; the rows are identical. This covers the operator's "S1_laneA DIFFERS" note.

**Compute**
- The 4-core cap was held, with one exception, disclosed here. At 21:10-21:12 five compute processes overlapped for about 90 s: two harness runs and three loop workers that were still loading.
- No AWS action by this lane. Box work went through requests `laneA_1` .. `laneA_4`.
- Local fg_make retrains (G2, G0, Tfix) used `--n-jobs 4`, about 2 min each.

## 0. Verdicts per pre-registered line

| line | result | verdict |
|---|---|---|
| Owner of the S0 miss (k >= 0.02, > 2 floors, same sign vs the close) | possession_outcome team-RATE response: k 0.042 (SE 0.009; close 0.062). fg_make team-rate response: k 0.025 (SE 0.008, retrain floor 0.009; close 0.032). Own ratings: k -0.011, beta 1.02 (not over-spread) | **possession_outcome and fg_make rate responses own the S0 miss** (harness, section 2) |
| Owner of the S1 deterioration (Delta k >= 0.01, > 2 floors, same sign vs the close) | fg_make: Delta k +0.046 (SE 0.011; close +0.043). Factorial main effect of fg_make's `T` retrain: +0.035 of +0.043, 4.0 x its own retrain-seed floor | **fg_make's E3 retrain owns it**, through its response to the OFFENCE make-rate features. Mechanism: a train/serve skew in the shooter feature (rows F-G below; section 5.6) |
| Same, rebound | Delta k_RB +0.017 (2.9 floors) in the swap lens; rebound's own retrain +0.003 in the factorial (1.5 floors) | not an owner: the RB swap component grows because fg_make `T` changes the miss volume rebound acts on |
| "Combination" verdict | Per-sub-model and own terms carry more than 100% of the change. Interactions: I1 -0.004 (1.3 floors), I2 +0.006 (2.6 floors, 14%), factorial three-model interaction +0.002 (4%). Part A cross terms FALL by 0.024 | **MIXED by the letter of the rule**, because I2 is 2.6 floors. In substance the "combination" statement is not supported: the over-spread is additive, sub-model by sub-model |
| Closure | Harness partitions close exactly (no MC; residual 0). Sim-anchored: S0 `not_harnessed` +0.0047 vs an MC expectation of +0.0082 (1.9 SE) | S0 **CLOSED**. S1 **NOT CLOSED by -0.012** (section 2.4): partly localised to the harness's PPP aggregation step |
| Addendum A (factorial; owner if main effect >= 40%, > 2 own seed floors, close same sign) | fg_make 82% (close 86%), 4.0 floors; possession_outcome 1.0 floor; rebound 1.5 floors; interaction 4% | **fg_make owns**; combination < 25% |
| Addendum B (same-season memorisation via monthly refits) | X_F - S0 with fg_make scored by its pre-season refit: +0.043 (served schedule: +0.035) | **REFUTED** |
| Addendum D (closed loop, box, 5,710 x 200) | X_F (S0 + fg_make `T` only) carries 94% of S1's slope drop (6.0 floors); X_PR (S1 with fg_make served) carries 6% (0.3 floors) | **CONFIRMED** |
| Full-size swap arms (laneA_1, 5,710 x 48) | Per-arm MC-corrected slopes reproduce the harness pattern. Component k's are unreadable except for ratings and offence/defence (reliability of PO/FG/RB 0.23-0.32 < 0.5) | per-arm CONFIRMS the harness; per-component UNDERPOWERED (section 8.2) |
| Sim-level retrain-seed floor (laneA_2, R2 5,710 x 200) | R2 - S0 slope -0.0057 (close -0.0029). S1 - S0 = -0.0232: 4.1 retrain floors (close 7.6) | S1's loss is real beyond retraining noise; the "15 floor-SD" overstated it |
| Addendum E: training-season overfit (H_fit) | Pre-season refits scored on their own design rows: T and R have the same in-sample minus out-of-sample gaps (differences -0.03 / -0.09 / -0.03) | **NOT SUPPORTED** (PARTIAL by the letter: the one class beyond 0.05 goes the wrong way) |
| Addendum F: train/serve parity | Team rates exact in both stacks. **S1's served `shooter_shrunk_dev_c` is NOT what `T` was trained on** (corr 0.94-0.97); S0 exact | **skew found** |
| Addendum F2: does the skew carry the gap? | Serving `T` its training shooter devs closes 62% (SE 7%) of the X_F - S0 harness gap (close 63%) | **carrier** (>= 50%) |
| Addendum G: skew-free retrain `Tfix` | X_Tfix - S0 harness +0.012 / +0.004 (seeds 0 / 1; floor 0.007), against X_F's +0.035. Shot-level log loss still beats R | **the skew explains the fg_make owner** (inside 2 floors) |
| Closed loop of Tfix (laneA_4) | see section 8.3 | see section 8.3 |

## 1. Method

**Where the team signal enters.** Every scoring-stage team signal enters through input arrays:
- possession_outcome reads its round-2 team block: 8 style rates and 4 own-rating columns.
- fg_make, rebound and clock read `team_static`: make/allow rates, OREB/DREB rates, the same four ratings, and tempo.
- fg_make also reads its shooter block.

**Arms.** An arm sets one family to its league value for both teams: centred features to 0, tempo to neutral.

**Partitions.** The per-game margin X then splits exactly into three kinds of piece: the all-swapped baseline, one delta per family, and an interaction remainder.
- L1: PO, FG, RB, ratings, shooter skill (plus PACE in the loop), I1.
- L2: offence-team features, defence-team features, I2.
- L3: ratings inside PO vs elsewhere.

**The k identity.** Regress a reference R on all the components jointly. R is either the realised margin Y (the G9 line) or the close C. Then

1 - slope(R on X) = sum_j (1 - beta_j) cov(Z_j, X) / var(X)

holds exactly. beta_j < 1 means component j moves the prediction more than reality moves.

**Why a harness.** The pilot showed the closed loop is too noisy to carry this per component: paired per-seed correlation was 0.52, and a component's reliability was 0.27 at 16 seeds. So the same arms are also run in a deterministic harness:
- It predicts each sub-model's rates per (game, offence side) through the engine's own adapters, at fixed reference states.
- It aggregates them into expected points: PPP = [sum_k P(FGA_k) v_k p_k + 2 f P(trip)] / [1 - p_oreb * misses].
- It reproduces the sim's 200-seed mean margin at correlation 0.989 in both stacks. The slope of X on X_h is 0.985 for S0 and 0.967 for S1.
- What it does not capture is carried as its own component, `not_harnessed`.

## 2. Part C: harness decomposition

### 2.1 Arms

Columns: SD of the predicted margin; slope of the realised margin on it; slope of the close on it.

| arm swapped | S0 SD | S0 slope Y | S0 slope C | S1 SD | S1 slope Y | S1 slope C |
|---|---:|---:|---:|---:|---:|---:|
| none (FULL) | 9.64 | 0.912 | 0.904 | 10.04 | 0.869 | 0.868 |
| PO rates | 8.89 | 0.982 | 0.981 | 9.05 | 0.948 | 0.953 |
| FG rates | 9.18 | 0.956 | 0.946 | 9.12 | 0.957 | 0.953 |
| RB rates | 9.41 | 0.926 | 0.922 | 9.39 | 0.914 | 0.915 |
| ratings (all consumers) | 3.46 | 1.901 | 1.829 | 4.63 | 1.673 | 1.662 |
| ratings in PO only | 7.53 | 1.148 | 1.136 | 8.19 | 1.052 | 1.049 |
| shooter skill | 9.41 | 0.932 | 0.923 | 9.97 | 0.875 | 0.871 |
| offence-team features | 5.41 | 1.520 | 1.520 | 5.49 | 1.484 | 1.500 |
| defence-team features | 5.83 | 1.422 | 1.410 | 6.27 | 1.319 | 1.320 |
| all team features | 2.30 | 0.949 | 1.030 | 2.47 | 1.256 | 1.338 |
| all + pace + shooter | 2.54 | 0.636 | 0.731 | 2.56 | 0.680 | 0.772 |

- **Removing possession_outcome's or fg_make's team rates** moves the slope to 0.95-0.98.
- **Removing the ratings** collapses the spread (SD 3.5-4.6). What remains is under-spread (slope 1.7-1.9).
- **Reading:** the ratings carry the signal; the rate features carry most of the excess.

### 2.2 L1, L2, L3 per stack

Values are k, with the game-bootstrap SE in brackets.

1 - slope is S0 0.088 (Y) / 0.096 (C); S1 0.131 / 0.133; R2 0.102 / 0.108.

| stack | ref | base (site, date, state) | PO rates | FG rates | RB rates | ratings | shooter | I1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | Y | +0.016 (0.004) | +0.042 (0.009) | +0.025 (0.008) | +0.011 (0.005) | -0.011 (0.021) | +0.005 (0.005) | +0.000 (0.001) |
| S0 | C | +0.015 (0.001) | +0.062 (0.003) | +0.032 (0.003) | +0.019 (0.002) | -0.040 (0.006) | +0.008 (0.001) | +0.000 (0.000) |
| S1 | Y | +0.014 (0.004) | +0.043 (0.010) | **+0.071 (0.010)** | +0.028 (0.008) | -0.028 (0.025) | +0.006 (0.002) | -0.004 (0.003) |
| S1 | C | +0.011 (0.001) | +0.060 (0.003) | **+0.075 (0.004)** | +0.036 (0.002) | -0.052 (0.007) | +0.005 (0.001) | -0.002 (0.001) |
| R2 | Y | +0.014 (0.004) | +0.042 (0.008) | +0.033 (0.008) | +0.009 (0.004) | +0.005 (0.023) | -0.002 (0.004) | -0.001 (0.001) |

**Realised response per unit and variance share (Y lens, beta / share):**

| component | S0 | S1 |
|---|---|---|
| PO rates | 0.57 / 0.10 | 0.64 / 0.12 |
| FG rates | 0.57 / 0.06 | **0.33 / 0.11** |
| RB rates | 0.70 / 0.04 | 0.65 / 0.08 |
| ratings | 1.02 / 0.72 | 1.05 / 0.58 |
| base (site, date, state) | 0.73 / 0.06 | 0.75 / 0.06 |

**L2 (offence team vs defence team):**
- S0: OFF +0.037 (0.024), DEF +0.043 (0.025), I2 -0.006 (0.002).
- S1: OFF +0.052 (0.023), DEF +0.061 (0.023), I2 -0.000 (0.001).
- Each side is about 45% of the variance, with beta 0.85-0.92. Neither side is identified at 2 floors.

**L3 (ratings by consumer, Y lens):**
- S0: ratings in PO -0.043 (0.033, beta 1.19); ratings elsewhere +0.043 (0.030, beta 0.91).
- S1: -0.035 / +0.045.
- These are inside 2 SE. Double counting of the ratings across sub-models is NOT identified.

### 2.3 S1 - S0 on the same games, against the retrain-seed floor (R2 - R)

| component | S1 - S0 (SE) | R2 - R (SE) | floors |
|---|---:|---:|---:|
| 1 - slope | +0.043 (0.006) | +0.014 (0.003) | 3.1 |
| base | -0.002 (0.002) | -0.002 (0.001) | |
| PO rates | +0.000 (0.008) | -0.000 (0.004) | 0.0 |
| **FG rates** | **+0.046 (0.011)** | +0.009 (0.006) | **4.1** |
| RB rates | +0.017 (0.006) | -0.001 (0.002) | 2.9 |
| ratings | -0.017 (0.020) | +0.017 (0.010) | -0.8 |
| shooter | +0.001 (0.005) | -0.007 (0.004) | 0.2 |
| I1 | -0.004 (0.003) | -0.001 (0.001) | -1.3 |
| OFF (L2) | +0.015 (0.017) | +0.023 (0.013) | 0.7 |
| DEF (L2) | +0.018 (0.016) | -0.010 (0.012) | 1.1 |
| I2 (L2) | +0.006 (0.002) | +0.002 (0.002) | 2.6 |

**Close lens:** FG +0.043, RB +0.017, PO -0.002, ratings -0.012, I1 -0.002, I2 +0.006.

**The served stack's own retrain under another seed moves the harness slope by 0.014.**
- `R` (retrain seed 0) reproduces the served harness exactly. `R2` (seed 1) is a spec-identical retrain.
- Per sub-model seed floors (addendum A): fg_make 0.009, possession_outcome 0.004, rebound 0.001.
- The box's "S1 worse by 15 floor-SD" used Monte Carlo floors only. Against the retrain-seed floor, the harness difference is 3.1 floors.
- A sim-level read of `R2` is requested (laneA_2).

### 2.4 Sim-anchored decomposition and closure

Here X is the sim's 200-seed mean margin, and `not_harnessed` = X - X_h is carried as its own component (Y lens).

**S0: 1 - slope 0.083.** It splits into base 0.014, PO 0.035, FG 0.021, RB 0.015, ratings -0.009, shooter 0.004, I1 -0.001, and not_harnessed 0.005 (SE 0.002; MC expectation 0.008).
- CLOSED: not_harnessed is within 1.9 SE of pure Monte Carlo.

**S1: 1 - slope 0.106.** It splits into base 0.013, PO 0.036, FG 0.058, RB 0.036, ratings -0.031, shooter 0.004, I1 -0.005, and not_harnessed -0.005 (SE 0.002; MC expectation 0.008).
- NOT CLOSED, by -0.012 (7 SE).
- The sim pulls S1's spread back relative to the harness: sim SD 9.83 vs harness 10.04.
- Rate by rate, the sim-on-harness slopes are the same in both stacks: makes 0.90-0.93, shares 0.92-0.95, OREB 0.87-0.88. The difference appears only at PPP (0.958 in S0 vs 0.947 in S1; `analysis_sim_vs_harness_v1.json`).
- So the residual sits in the harness's one-chance PPP aggregation, which amplifies make-rate spread a little more than the sim's possession loop does. It is not a separate sim mechanism that has been identified.
- This is also why the sim's S1 - S0 (+0.023) is about half the harness's (+0.043).

## 3. Addendum A: sub-model replacement factorial (harness, deterministic)

The three sub-models are possession_outcome (P), fg_make (F) and rebound (R). Each is either served or replaced by its Stage B `T` retrain.

1 - slope(Y on X_h) for the eight combinations:

| T sub-models | none (S0) | P | F | R | PF | PR | FR | PFR (S1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 - slope | 0.088 | 0.092 | 0.123 | 0.091 | 0.127 | 0.095 | 0.126 | 0.131 |

| effect | Y lens (SE) | close lens (SE) | own retrain-seed floor | floors |
|---|---:|---:|---:|---:|
| fg_make `T` main effect | **+0.035 (0.003)** | +0.032 (0.003) | 0.009 | **4.0** |
| possession_outcome `T` main effect | +0.004 (0.003) | +0.003 (0.002) | 0.004 | 1.0 |
| rebound `T` main effect | +0.003 (0.002) | +0.001 (0.002) | 0.001 | 1.5 |
| three-model interaction | +0.002 (0.001) | +0.002 (0.001) | | |
| total S1 - S0 | +0.043 (0.006) | +0.037 (0.004) | 0.014 (all three, R2) | 3.1 |

## 4. Part A: own vs cross terms of the exact channel identity

Data: the 200-seed full reads, compared with realised outcomes.

| | S0 | S1 | S1 - S0 (SE) |
|---|---:|---:|---:|
| 1 - slope | 0.084 | 0.107 | +0.023 (0.005) |
| OWN terms (each channel vs its own realisation) | 0.037 | 0.084 | **+0.047 (0.006)** |
| CROSS terms (channels moving together more than reality) | 0.047 | 0.023 | **-0.024 (0.008)** |

**S1 - S0 by channel** (total, with own in brackets where reported):
- make_rim +0.018 (0.003; own +0.011)
- make_jump +0.009 (own +0.003)
- make_3 +0.006 (own +0.005)
- mix_3 +0.012
- mix_jump -0.012
- mix_rim +0.003
- reb_chances -0.008
- tov -0.004
- oreb -0.002
- FT%, trips, pace and parity are each within +-0.001.

**Reading.** The make channels carry +0.034 of the net +0.023. The deterioration is in OWN terms: fg_make's game-level predictions are more over-spread against their own realised make rates, while the cross terms fall. This is "a sub-model is wrong at the game level". It is not "each sub-model is right and the product is over-spread".

**Dropped check.** The pre-registered in-sim expected-value non-linearity check is identically zero under the box identity, because ratios of seed-summed counts reproduce mean points exactly. It is uninformative and is not reported.

## 5. Inside fg_make

### 5.1 Addendum B: not same-season memorisation (REFUTED)

Every game's fg_make was scored by its pre-season refit, so no 2024-25 rows were in training.
- 1 - slope: S0 0.073, S1 0.124, X_F 0.116.
- X_F - S0 = +0.043, against +0.035 under the served schedule. S1 - S0 = +0.051, against +0.043.
- S1's rim make slopes for January-March are 0.63 / 0.64 / 0.65, against S0's 0.74 / 0.77 / 0.80 under the same refit.

### 5.2 Addendum C: the offence make-rate features

Each cell shows k, then (beta, variance share).

| component of fg_make's team response (Y lens) | S0 | S1 | S1 - S0 (SE) | close lens S1 - S0 (SE) |
|---|---|---|---:|---:|
| offence `off_make_c__*` | +0.013 (0.62, 0.034) | +0.051 (0.32, 0.074) | **+0.037 (0.008)** | **+0.030 (0.003)** |
| defence `def_allow_c__*` | +0.012 (0.52, 0.025) | +0.017 (0.60, 0.043) | +0.005 (0.006) | +0.010 (0.002) |
| off x def interaction | -0.000 | -0.005 | -0.005 (0.004) | -0.007 (0.001) |

- **What changes.** Under the E3 retrain, the margin variance driven by the offence's own make-rate features doubles, and the realised response per unit halves.
- **Feature spread.** The E3 offence make features are much tighter than the served ones:

  | class | E3 SD | served SD |
  |---|---:|---:|
  | rim | 0.030 | 0.060 |
  | jump | 0.018 | 0.061 |
  | three | 0.010 | 0.040 |

- **No reliability input.** fg_make has no season-phase or reliability feature: no `days_since_start` and no E3 posterior variance. One learned response therefore serves an estimate whose reliability changes through the season. This was the first working explanation. Sections 5.4-5.7 replace it: the larger cause is a train/serve skew in the shooter feature.

### 5.3 Per-rate team-game calibration

Each cell is the attempt-weighted slope of the realised team-game rate on the predicted rate (11,400 team-games), then the beta of the rate-feature part, then the SD of the rate-feature part.

| rate | S0 | S1 | R2 |
|---|---|---|---|
| TOV (PO) | 0.83 / 0.95 / 0.018 | 0.84 / 0.97 / 0.020 | 0.81 / 0.95 / 0.018 |
| FT trip (PO) | 1.92 / 1.99 / 0.010 | 2.08 / 2.21 / 0.011 | 1.92 / 1.83 / 0.011 |
| rim share (PO) | 0.94 / 0.96 / 0.044 | 0.91 / 0.93 / 0.050 | 0.93 / 0.95 / 0.043 |
| 3PA share (PO) | 0.91 / 0.97 / 0.052 | 0.88 / 0.91 / 0.057 | 0.90 / 0.97 / 0.052 |
| rim make (fg_make) | 0.84 / 0.73 / 0.023 | **0.75** / 0.75 / 0.030 | 0.83 / 0.77 / 0.023 |
| jump make (fg_make) | 0.71 / 0.69 / 0.015 | 0.67 / 0.68 / 0.023 | 0.67 / 0.54 / 0.016 |
| 3P make (fg_make) | 0.69 / 0.40 / 0.010 | 0.64 / 0.30 / 0.011 | 0.69 / 0.37 / 0.011 |
| OREB (rebound) | 0.78 / 0.86 / 0.033 | 0.78 / 0.85 / 0.039 | 0.78 / 0.87 / 0.033 |

**Make slopes by month, S0 -> S1:**

| month | rim | jump | three |
|---|---|---|---|
| Nov | 0.84 -> 0.87 | 0.79 -> 0.87 | 0.51 -> 0.59 |
| Dec | 0.92 -> 0.81 | 0.85 -> 0.78 | 0.74 -> 0.71 |
| Jan | 0.76 -> 0.62 | 0.74 -> 0.77 | 0.72 -> 0.58 |
| Feb | 0.81 -> 0.69 | 0.63 -> 0.57 | 0.74 -> 0.59 |
| Mar | 0.90 -> 0.71 | 0.63 -> 0.48 | 0.90 -> 0.69 |

- **S1 is better in November.** This is the prior-season carry.
- **From December on, the offence-feature response is too steep.**
- **Measurement caveat:** harness rates are taken at one reference state and as-of usage weights, which attenuates absolute slopes. Comparisons between stacks are paired and not affected.

**The FT-trip slope is near 2 in both stacks.** possession_outcome's trip class is too flat by team. That compression offsets part of the over-spread elsewhere; keep it in view when fg_make is fixed (Decision 11).

**Stage B could not see this.** Stage B graded fg_make only on attempt-level log loss and binned calibration (`team_rate_stageb_offline_2026-09-30.md`). It had no team-game slope or responsiveness line. `T`'s -8.4-floor log-loss win coexists with a rim make slope of 0.75 against 0.84 served.

### 5.4 Arm G2 (section 2.1, local, preliminary): dropping `off_make_c` does not fix it

| | rim log loss | jump log loss | three log loss | harness 1 - slope | Delta vs S0 |
|---|---:|---:|---:|---:|---:|
| R (served features) | 0.667252 | 0.666385 | 0.637738 | 0.088 | |
| T | 0.666884 | 0.666111 | 0.637701 | 0.123 | +0.035 |
| G2 seed 0 / 1 | 0.667258 / 0.667237 | 0.666265 / 0.666292 | 0.637828 / 0.637752 | 0.118 / 0.123 | +0.030 / +0.035 |

- G2 loses T's log-loss gains and keeps T's over-spread.
- The response moves onto other inputs. Under G2 the ratings component turns from k -0.048 (T) to +0.017.
- Reading: the defect is in what fg_make learned on the `T` design as a whole, not in one column.

### 5.5 Addendum E: not training-season overfitting (NOT SUPPORTED)

Each pre-season (2024-11-01) refit was scored on its own design rows: the training seasons (in sample) and the fold-2 rows (out of sample). Cells are the team-game make slope, in sample / out of sample.

| | rim | jump | three |
|---|---|---|---|
| T | 1.18 / 0.81 | 1.72 / 0.84 | 1.80 / 0.71 |
| R | 1.21 / 0.81 | 1.79 / 0.82 | 1.84 / 0.71 |

- T's gaps are no larger than R's.
- **Key observation:** on the design's own fold-2 shot rows, T and R have the SAME out-of-sample team-game slopes. Through the engine inputs, T is clearly worse. That points at the inputs the engine serves.

### 5.6 Addenda F and F2: a train/serve skew in Stage B's fg_make `T`

**Parity of each fold-2 (game, offence team, class) value, training design vs engine inputs:**

| input | T vs S1 | R vs S0 |
|---|---|---|
| `off_make_c` / `def_allow_c` | exact (corr 1.000, max difference 0) | exact |
| `shooter_shrunk_dev_c` | **corr 0.970 / 0.947 / 0.943** (rim / jump / three); engine SD 3-6% larger; 15% of rim rows differ by > 0.01 | exact |

**Cause** (checked on the fold-2 design rows):
- With `--team-rate-table`, `cbb_sim.team_rate_adapter.apply` replaces `off_make_c` / `def_allow_c` with E3 values.
- It leaves `off_make_raw` at the SERVED expanding-mean value: max |raw - (c_E3 + lg)| is 0.63, against 6e-8 on the served design.
- `train_fg_make_v4_par_v1.py` then computes `fit_m` and `build_extra` (and so `shooter_shrunk_dev_c`) from that stale raw rate.
- The engine builder derives the dev from c_E3 + lg, as documented in its header.
- So `T` was trained on a shooter feature measured against the served team rate and is served one measured against the E3 rate. Since dev = att (smc - c)/(m + att), the served value carries an extra team-level term att (c_served - c_E3)/(m + att). That term is team-level estimation noise the model treats as signal.
- The `--feature-table` path of the same trainer does re-derive raw. Only the `--team-rate-table` path, used for every Stage B `T` arm, skips it.

**F2 test.** The engine's S1 shooter devs were replaced with the training values (198,369 matched rows) and the harness rerun:

| | Y lens | close lens |
|---|---:|---:|
| X_F 1 - slope, before -> after | 0.123 -> 0.101 | 0.127 -> 0.107 |
| share of the X_F - S0 gap closed | **62% (SE 7%)** | **63% (SE 6%)** |

### 5.7 Addendum G: the skew-free retrain `Tfix`

`Tfix` re-derives raw as c + lg after the adapter. Everything else is T's spec.

**Parity:** shooter devs are exact against the `X_Tfix` engine inputs (corr 1.000, 0 rows off).

**Shot-level log loss** (seed 0 / seed 1):

| class | Tfix | T | R |
|---|---|---:|---:|
| rim | 0.667031 / 0.666990 | 0.666884 | 0.667252 |
| jump | 0.666049 / 0.666001 | 0.666111 | 0.666385 |
| three | 0.637736 / 0.637873 | 0.637701 | 0.637738 |

- Tfix still beats R on rim (about -5 R2-floors) and jump.
- Three is level with R.
- Calibration: all D8 checks pass.

**Harness 1 - slope (Y / close):**

| stack | 1 - slope (Y / close) | Delta vs S0, Y (SE) | Delta vs S0, close |
|---|---|---:|---:|
| X_F (S0 + fg_make T) | 0.123 / 0.127 | +0.035 (0.004) | +0.031 |
| **X_Tfix (S0 + fg_make Tfix), seed 0** | 0.100 / 0.105 | **+0.012 (0.003)** | +0.010 |
| X_Tfix, seed 1 | 0.092 / 0.098 | +0.004 (0.004) | +0.003 |
| S1 (PO T + RB T + fg_make T) | 0.131 / 0.133 | +0.043 (0.006) | +0.037 |
| **S1fix (PO T + RB T + fg_make Tfix)** | 0.109 / 0.111 | **+0.021 (0.006)** | +0.016 |
| X_PR (PO T + RB T, fg_make served) | 0.095 / 0.100 | +0.007 (0.004) | +0.004 |

- Tfix's seed-to-seed spread is 0.007.
- Tfix removes 67-88% of fg_make's harness damage. What remains is inside 2 x max(seed floor, SE).
- S1fix halves S1's harness loss. The remaining +0.021 is about 1.5 times the all-three-model retrain floor (0.014) in the harness. Its sim value is pending (section 8.3).

**Team-game make slopes, S0 / X_F / X_Tfix:**

| class | overall | Jan | Mar |
|---|---|---|---|
| rim | 0.84 / 0.75 / 0.80 | 0.76 / 0.62 / 0.68 | 0.90 / 0.71 / 0.75 |
| three | 0.69 / 0.64 / 0.67 | not reported | not reported |
| jump | 0.71 / 0.67 / 0.67 | not reported | not reported |

## 6. Multi-level evidence

Slope cells within a segment are range-restricted, as in the F diagnostic. They are compared between stacks, not read as calibration levels.

**By month (harness, Y lens).** Columns: n games; slope and k_FG for each stack.

| month | n | S0 slope | S1 slope | S0 k_FG | S1 k_FG |
|---|---:|---:|---:|---:|---:|
| Nov | 1,221 | 0.913 | 0.925 | +0.049 | +0.058 |
| Dec | 915 | 0.917 | 0.881 | +0.009 | +0.032 |
| Jan | 1,422 | 0.882 | 0.801 | +0.038 | +0.088 |
| Feb | 1,365 | 0.857 | 0.783 | -0.004 | +0.101 |
| Mar | 765 | 0.884 | 0.824 | +0.033 | +0.066 |

The per-cell SE of k is about 0.01-0.02.

**By team-strength quintile gap** (|q_home - q_away| of the as-of own-rating net). Columns: n games; slope and k_FG for each stack.

| gap | n | S0 slope | S1 slope | S0 k_FG | S1 k_FG |
|---:|---:|---:|---:|---:|---:|
| 0 | 1,854 | 0.767 | 0.699 | +0.016 | +0.065 |
| 1 | 2,201 | 0.824 | 0.800 | +0.050 | +0.065 |
| 2 | 1,044 | 0.932 | 0.870 | +0.001 | +0.139 |
| 3 | 463 | 0.967 | 0.925 | +0.066 | +0.086 |
| 4 | 143 | UNDERPOWERED | | | |

- k_FG rises at every gap.
- Nothing concentrates in mismatches (strong offence x weak defence), so tree non-additivity in matchup cells is not the mechanism.

**Home/away vs neutral.** Columns: n games; slope and k_FG for each stack.

| site | n | S0 slope | S1 slope | S0 k_FG | S1 k_FG |
|---|---:|---:|---:|---:|---:|
| home/away | 4,969 | 0.928 | 0.881 | +0.019 | +0.076 |
| neutral | 736 | 0.795 | 0.784 | +0.092 | +0.051 |

- The neutral cell swings by more than its SE; it is read as noise.
- The base (site/date/state) component is k 0.011-0.016 with beta 0.73-0.75 in both stacks. This is the home-advantage excess of the F diagnostic, and E3 does not change it.

**Per team** (team mean predicted margin, harness):

| | S0 | S1 | R2 | realised | close |
|---|---:|---:|---:|---:|---:|
| SD across teams | 6.00 | 6.33 | 6.08 | 6.51 | 5.75 |
| slope of realised team mean on predicted | 1.02 | 0.96 | 1.01 | | |
| slope of close on predicted | 0.92 | 0.87 | 0.91 | | |

**Per possession type:** section 5.3. **Per player:** the shooter component is k 0.005-0.006 in both stacks, with S1 - S0 +0.001.

## 7. The brief's six candidate channels

| # | channel | finding |
|---|---|---|
| 1 | offence and defence effects on both sides | OFF and DEF are each about 45% of the variance, beta 0.85-0.92 in both stacks; I2 is -0.006 (S0) and +0.000 (S1). Not the owner |
| 2 | correlated estimation error across sub-models | I1 is +0.000 / -0.004; Part A cross terms fall in S1 (-0.024). Not the owner |
| 3 | tree non-additivity in matchup cells | I2 is small; the quintile-gap cells show no concentration in mismatches. Not the owner |
| 4 | usage / rotation concentration | shooter component +0.005; S1 - S0 +0.001. Not the owner. Rotation and usage arrays were not swapped (stated in the pre-registration) |
| 5 | ratings double counting | ratings beta 1.02-1.05 (not over-spread); ratings in PO vs elsewhere are inside 2 SE. Not identified |
| 6 | possession count x efficiency | Part A pace / parity S1 - S0 is -0.001 / +0.001. The PACE swap was box tier 2 and NOT RUN. Not the owner of S1 - S0 |
| new | sub-model team-RATE responses, and a train/serve skew | S0 miss: possession_outcome (0.042) and fg_make (0.025) rate responses. S1 deterioration: fg_make's E3 retrain, through a skewed shooter feature (section 5.6). **Owner** |

## 8. Closed loop

### 8.1 Addendum D (box, 5,710 x 200, paired seeds 0-199, verified truth)

Floors follow Decision 12: four seed draws of 50, and a 200-rep game bootstrap.

| stack | slope Y | MC-corrected | slope C | SD(X) | Delta slope vs S0 (floor) | share of S1's drop (Y; close) |
|---|---:|---:|---:|---:|---:|---|
| S0 (served) | 0.9169 | 0.9245 | 0.9106 | 9.61 | | |
| S1 (all three `T`) | 0.8937 | 0.9007 | 0.8887 | 9.83 | -0.0232 (0.0054) | 100% |
| X_F (S0 + fg_make `T` only) | 0.8952 | 0.9023 | 0.8932 | 9.80 | **-0.0217 (0.0036), 6.0 floors** | **94% (SE 24%); 79%** |
| X_PR (S1 with fg_make served) | 0.9155 | 0.9231 | 0.9100 | 9.60 | -0.0014 (0.0048), 0.3 floors | 6%; 3% |

**CONFIRMED in the sim.** fg_make's E3 retrain alone reproduces S1's slope loss and S1's extra spread (+0.19 of +0.22 SD). Possession_outcome and rebound on E3 together leave the slope at S0's level.

The sim-level retrain-seed floor comes from laneA_2 (`R2`, the served features retrained under seed 1, 5,710 x 200):
- R2 slope 0.9112 (MC-corrected 0.9187); Delta vs S0 -0.0057 (boot 0.0036); close lens -0.0029.
- S1's -0.0232 is therefore 4.1 retrain floors (close 7.6), and X_F's -0.0217 is 3.8.
- The S1 loss is real, but the box read's "15 floor-SD" used Monte Carlo floors only and overstated its significance about fourfold.

### 8.2 Full-size swap arms (laneA_1, 5,710 x 48 seeds, paired with the box's FULL rows)

**MC-corrected slope of Y on each arm.** FULL here is the 48-seed subset.

| stack | FULL | PO swapped | FG swapped | RB swapped | ratings swapped | TEAM swapped | ALL swapped |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | 0.927 | 0.975 | 0.964 | 0.943 | 1.958 | 0.862 | 0.576 |
| S1 | 0.903 | 0.964 | **0.987** | 0.958 | 1.712 | 1.235 | 0.566 |

- **Agreement with the harness.** The per-arm pattern is the harness's: PO 0.982 / 0.948, FG 0.956 / 0.957, RB 0.926 / 0.914 for S0 / S1.
- **Removing fg_make's team rates.** In S1 this recovers the most (+0.084 vs +0.037 in S0).

**Component k's (IV on split seeds).** These are readable only where the pre-registered first-stage reliability is >= 0.5:
- Ratings (0.77-0.83): beta 0.98-1.16, not over-spread in either lens, consistent with the harness.
- OFF / DEF (0.62-0.70): S0 close lens OFF +0.061 (floor 0.013), DEF +0.039 (0.011). S1 close OFF +0.035 (0.016), DEF +0.079 (0.017). The Y lens is inside its floors (0.03-0.08).
- PO / FG / RB (reliability 0.23-0.32) and the interactions (0.03-0.09) are UNDERPOWERED at 48 seeds and not read.
- The seed-quarter IV floors for S1 are unstable: they reach 5-50 where 12-seed quarters make the instrument weak.
- Numbers are in `analysis_loop_s48_v1.json`.

**Not run:** tier 2 (RAT_PO, PACE, PLY), queued behind other lanes. The local 1,142 x 16 loop is UNDERPOWERED (component reliability 0.02-0.47) and is not read (`analysis_loop_s16_local5_v1.json`).

### 8.3 Closed loop of the skew-free fg_make (laneA_4)

Requested at 23:19 EDT: `docs/ops/box_queue/laneA_4.md` (X_Tfix, S1fix, X_Tfix1; 5,710 x 200). It had not returned when this section was committed; the result is appended here when it lands.

## 9. What this changes

**1. The owner of S1's G9 loss is a train/serve skew in how Stage B built fg_make `T`, not a property of the E3 estimator and not the cascade's aggregation.**
- Mechanism: under `--team-rate-table` the raw team make rate stayed stale, so `T`'s `shooter_shrunk_dev_c` was trained against the served team rate and is served against the E3 rate.
- Evidence, harness and box:
  - fg_make carries 82% of the harness loss and 94% of the sim loss.
  - Serving `T` its training shooter devs closes 62% of fg_make's harness gap.
  - A skew-free retrain (`Tfix`) leaves fg_make's harness effect inside its retrain floor while keeping most of the shot-level log-loss gain.
- This is a BUILD defect. The fix belongs in the trainer path or `cbb_sim.team_rate_adapter`: re-derive `off_make_raw` / `def_allow_raw` when the centred columns are replaced. The `--feature-table` path already does this.
- Lane A did not edit either shared file. The wrapper `train_fg_make_v4_par_rawfix_v1.py` shows the change.
- Every stack that serves fg_make `T` (S1 and its variants built with `--fg-artifacts .../round_stageb/T/...`) inherits the skew.
- **The PM ruling "E3 REFUTED as a sim improvement" rests on it, and should be re-read with S1fix** (section 8.3).

**2. The PM's "combination" statement is not supported.**
- Interactions are small, cross terms fall, and the over-spread is additive by sub-model.
- The served models' compressed levels were not compensating an aggregation over-spread. S1's extra spread was injected by one sub-model's skewed input.

**3. S0's own miss (0.083) is owned by the team-RATE responses of possession_outcome and fg_make** (harness beta about 0.57; close-lens k 0.062 and 0.032).
- Ratings are at scale (beta about 1.0).
- This refines the F diagnostic's "as-of team-rate features" owner: the excess enters through the sub-models' learned responses to rate-feature movement.
- possession_outcome's FT-trip class is under-spread by team (slope about 2) and partly offsets it.

**4. Every S1-vs-S0 sim comparison needs the retrain-seed floor.** It is 0.0057 on the sim slope and 0.014 on the harness slope. Spec-identical retrains under another seed move the G9 slope by a quarter of the effect being judged.

**5. Stage B's fg_make gate could not see either defect.**
- It graded attempt-level log loss and binned calibration only.
- Proposed for the PM: every scoring-stage Stage B also reports a train/serve parity check of every derived feature against the engine inputs, the team-game calibration slope by month, and the harness margin slope.

## 10. Next step and resume

**1. Decide on laneA_4 (section 8.3).** If S1fix's sim slope is within the retrain floor of S0, E3 is back on the table as a set (Decision 11). The Stage B `T` wins for possession_outcome and rebound, and Tfix's own log-loss win, become the candidate set.

**2. Fix the trainer path**, then rerun fg_make `T` (as `Tfix`) at fold 1 as well, under the experiments.md section 2 round.
- G1 / G3 are deprioritised. G2 was run (preliminary, fold 2): it does not help.
- The owner of that file is lane J's `train_fg_make_v4_par_v1.py` or `team_rate_adapter`. It is a small change and it needs a parity test.

**3. Audit the other Stage B arms for the same class of skew.** possession_outcome and rebound `T` have no derived team columns that we know of, and their harness effects sit inside their retrain floors. That is not a parity proof.

**Local resume:**
- Harness: `scripts/diag_aggregation_harness_v1.py --stack <S0|S1|R|R2|X_*|Z_*|X_Tfix|S1fix> [--arms ...] [--fg-first-refit] [--slot-dev-override ...]`.
- Analyses:
  - `scripts/diag_aggregation_overspread_v1.py --part identity --part harness --part rates --part factorial --part loop`.
  - `scripts/diag_aggregation_addendumD_v1.py` (S0, S1, X_F, X_PR, R2, and X_Tfix, S1fix, X_Tfix1 once synced).
  - `scripts/diag_aggregation_{parity,tfix,g2,overfit}_v1.py`.
- Stacks: builder commands in experiments.md 1.1, 1.8 and 2.5 and in `docs/ops/box_queue/laneA_{1..4}.md`.
