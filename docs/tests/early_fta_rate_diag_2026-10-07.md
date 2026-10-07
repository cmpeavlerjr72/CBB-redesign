# Early-season FTA/P deficit: owner and mechanism (2026-10-07)

Diagnostic only. Nothing fitted for adoption, no engine edit, nothing registered. 2025-26 not read.
Scripts (new, versioned, flat): `scripts/diag_early_fta_rate_v1.py` (levels, calendar, context, per-team, source split, offline foul check),
`_v2.py` (first-half team-foul state), `_v3.py` (team responsiveness by bucket, two-factor split of the bonus gap), `_v4.py` (seasons 2022-25,
calendar vs composition). Tables are written to `results/early_fta_diag/report*.txt` (gitignored).

Data. Truth = verified finals and hoopR team box; P = FGA - OREB + TOV + 0.44 FTA (box possessions, both teams, as in
`docs/tests/total_bias_decomp_2026-10-05.md`). Sims = the served-v2 200-seed reads (F1 `f1c_V2_full_s200_o0`, F2 `v3full_COMB9GCTKD_s200_o0`).
The FTA source split needs per-half foul counters, which exist only for F2 in the R9ao3 tap shards `r9tapfull_COMB9_s200_o0_off*_n25`
(8 x 25 seeds, 5,710 games; COMB9 = the served stack without G3/KD, which do not touch fouls; its window FTA/possession is 0.2675 against
0.2696 FTA/P in the served read). Actual sources come from `possessions_v2/chances_*.parquet` and `foul_accrual_poss_v2.parquet`.
Possession units differ between tap (possession rows) and box P, so tap gaps run about 25% smaller than box gaps (-0.0156 vs -0.0221 pooled).
Windows: d0-14 / d15-45 / d46+ days since the first game of the season (F2 n = 618 / 1,215 / 3,867 games; F1 696 / 1,181 / 3,755).

## Verdict

**Owner: the non-trip foul accrual layer (`lut_acc_A2_F2`, the foul-joint accrual LUT, inside `R9ao3`), with the shooting-foul trip layer a minor second owner.
Mechanism: a genuine calendar effect in the first half, not game context and not as-of thinness.**

1. **The real early FTA surplus is a first-half bonus effect.** In every season 2022-2025, in the first half the share of possessions with the offence in
   the bonus is 0.28-0.31 in week 0 against 0.196-0.21 from day 46 (2025: 0.296 / 0.275 / 0.241 / 0.222 / 0.196 by d0-6 / d7-13 / d14-20 / d21-45 / d46+).
   Bonus trips per first-half possession are 0.045-0.048 in week 0 against 0.028 late (+60%). Second-half bonus trips show no such pattern (0.084-0.090 week 0, 0.083-0.088 late).
   It is a within-team effect: re-weighting each team's own d46+ rate to its early game mix explains 0 of the early excess (composition part -0.0010 to +0.0015 in
   in-bonus share against an early excess of +0.054 to +0.088, all four seasons, defence and offence side). Non-trip fouls per first-half possession run +6% to +10% early and the bonus
   threshold amplifies it (highest engine-definition team-foul count seen per team first half 7.6 early vs 6.5 late, share of team-halves reaching a count of 6 or more 0.79 vs 0.66, F2).
2. **The sim has no calendar term in the layer that produces it.** A2 is a state-only LUT (period, clock, margin, defence and offence team fouls, site; `STATE_T`), no days-since-start and no team or as-of foul rate;
   the realised sim first-half in-bonus share is flat at 0.214 / 0.210 / 0.204 (d0-14 / d15-45 / d46+) against actual 0.284 / 0.227 / 0.196.
   Offline given the TRUE state (F2, first half): A2 mean p over mean y is -5.4% in d0-6, -1.6% in d7-13, +3.1%, +6.9%, +9.3%, +10.1% by d46+ (F1 whole-window +2.7% rising to +13.7%): a monotone calendar drift in the residual that no input can absorb.
3. **The trip layer is not the owner of the bonus gap.** Bonus trips per in-bonus possession in the window first half: sim 0.1590, actual 0.1572 (matches). The whole first-half bonus deficit is the in-bonus share (state), 100% of it.
4. **Context does not own it** (section 2). **As-of thinness is not the primary owner** (section 5): A2, T2c and AO3-state have no as-of feature, and in the one layer that has team as-of inputs (the possession-outcome shooting-trip base probability) the team response is weak early, a secondary effect worth about a fifth of the gap.

### How much of the window gap each piece explains (F2, tap units, per possession, half-weighted; box-scale gap -0.0221, tap-scale -0.0170)

| piece | FTA/poss contribution | share of gap | owner |
|---|---:|---:|---|
| bonus entry (in-bonus share), first half | -0.0099 | 58% | A2 accrual (no calendar term) |
| bonus entry, second half | -0.0089 | 52% | A2 accrual |
| bonus trips per in-bonus possession, second half (sim over) | +0.0062 | -36% | T2c / level, offsets the line above (compensation, see candidates) |
| bonus trips per in-bonus possession, first half | +0.0004 | -2% | none |
| shooting-foul trips (both halves, sim -3%) | -0.0035 | 20% | possession-outcome shooting-trip base + T2c, team response weak early |
| FTA per trip | -0.0008 | 4% | negligible |
| and-ones | -0.0001 | 1% | none |
| total | -0.0171 | 100% | |

Net: the bonus channel is about -0.0127 of -0.0171, **about 74% of the gap, about 0.016 of the box-scale -0.022; the first-half bonus state alone is 58%, about 0.013.**
Context (game type, site, mismatch, blowout, conference flag, late-game fouling) explains at most 0.002 and in the mix-standardised check it points the other way (section 2).

## 1. Overall and calendar

FTA/P, both teams, box possessions (se_act is the cross-game SE of the actual ratio).

| season | bucket | n games | actual | sim | gap | se_act |
|---|---|---:|---:|---:|---:|---:|
| 2023 | d0-14 / d15-45 / d46+ | 715 / 1289 / 3619 | 0.2668 / 0.2558 / 0.2689 | no sim | | |
| 2024 (F1) | d0-14 | 696 | 0.2847 | 0.2562 | -0.0285 | 0.0029 |
| | d15-45 | 1181 | 0.2782 | 0.2667 | -0.0116 | 0.0024 |
| | d46+ | 3755 | 0.2793 | 0.2763 | -0.0030 | 0.0013 |
| 2025 (F2) | d0-14 | 618 | 0.2916 | 0.2696 | -0.0221 | 0.0034 |
| | d15-45 | 1215 | 0.2790 | 0.2741 | -0.0049 | 0.0023 |
| | d46+ | 3867 | 0.2809 | 0.2821 | +0.0012 | 0.0013 |

Actual early minus d46+: 2023 -0.002, 2024 +0.005, 2025 +0.011 (the whole-game effect is not stable across seasons; the first-half bonus effect is, section 5b). Sim early minus d46+: -0.020 (F1), -0.013 (F2).
Week by week, F2 actual 0.294 / 0.291 / 0.285 / 0.277 / 0.276 / 0.271 from week 0, against sim 0.269 / 0.270 / 0.272 / 0.273 / 0.276 / 0.276: the sim climbs where the actual falls, and they cross at week 4 to 5.
The gap is 6.5 SE in the F2 window and 9.8 SE in F1; it is not noise.

## 2. Game context (hypothesis a): not the explanation

F2 window gap by cell (FTA/P sim - actual; w = share of window possessions; contribution = w x gap). F1 shows the same shape (all cells -0.013 to -0.043).

| variable | cell | n games | w | actual | sim | gap | contribution |
|---|---|---:|---:|---:|---:|---:|---:|
| site | home-court | 572 | 0.925 | 0.2914 | 0.2690 | -0.0224 | -0.0207 |
| site | neutral (MTE) | 46 (underpowered) | 0.075 | 0.2945 | 0.2766 | -0.0178 | -0.0013 |
| conference | non-conference | 618 | 1.000 | 0.2916 | 0.2696 | -0.0221 | -0.0221 |
| pregame \|sim mean margin\| | <5 | 142 (underpowered) | 0.231 | 0.3051 | 0.2746 | -0.0305 | -0.0070 |
| | 5-10 | 146 (underpowered) | 0.234 | 0.2954 | 0.2749 | -0.0205 | -0.0048 |
| | 10-15 | 123 (underpowered) | 0.199 | 0.2870 | 0.2705 | -0.0165 | -0.0033 |
| | 15+ | 207 | 0.337 | 0.2825 | 0.2617 | -0.0208 | -0.0070 |
| final \|margin\| (ex post; sim cut on its own realised margin) | 0-5 | 134 (underpowered) | 0.218 | 0.2981 | 0.2741 | -0.0239 | -0.0052 |
| | 6-12 | 164 | 0.265 | 0.3116 | 0.2849 | -0.0267 | -0.0071 |
| | 13-19 | 133 (underpowered) | 0.214 | 0.2933 | 0.2701 | -0.0232 | -0.0050 |
| | 20+ | 187 | 0.304 | 0.2685 | 0.2533 | -0.0151 | -0.0046 |

Every cell is negative and contributions are proportional to weight. Close games (pregame gap under 5) carry a larger gap (-0.031, 142 games, underpowered, about 0.002 of excess contribution).
The conference flag cannot own it: the window is 100% non-conference in both actual and sim (mix identical). Neutral/MTE games: -0.018 on 46 games, not separable.
Blowout and garbage time: sim and actual both depress FTA/P in 20+ games and the gap there is the smallest (-0.015); late-game intentional fouling (actual window: bonus-trip FTA in the last 2:00 with the offence ahead, 0.0189 FTA/poss, vs 0.0240 at d46+) is lower early, not higher, and it lives in H2 where the gap is a third of H1's.

Mix standardisation (window mix at each cell's own d46+ rate, F2; positive mix part would mean context predicts a higher early rate):

| variable | actual window - d46+ | mix part | within-cell part | sim window - d46+ | sim mix part |
|---|---:|---:|---:|---:|---:|
| site | +0.0107 | -0.0003 | +0.0110 | -0.0126 | -0.0001 |
| conference | +0.0107 | -0.0049 | +0.0156 | -0.0126 | -0.0096 |
| pregame mismatch | +0.0107 | -0.0040 | +0.0147 | -0.0126 | -0.0034 |
| final margin | +0.0107 | -0.0059 | +0.0166 | -0.0126 | -0.0032 |

Context makes the early window predict a lower rate (more non-conference and mismatched games), so the real early elevation is wholly within-cell (F1 conference: mix -0.0135 against within-cell +0.0188). The sim reproduces the (negative) mix part and misses the within-cell part.

## 3. Per team (hypothesis b, team side)

Window d0-14, teams with at least 2 window games, quintiles of the team's prior-season own FTA/P (grading only; about 72 teams per quintile, quintile SE 0.006-0.008, cells individually underpowered).

| F2 prior quintile | actual | sim | gap | | F1 prior quintile | actual | sim | gap |
|---|---:|---:|---:|---|---|---:|---:|---:|
| Q1 low | 0.2684 | 0.2632 | -0.0052 | | Q1 low | 0.2554 | 0.2499 | -0.0055 |
| Q2 | 0.2785 | 0.2642 | -0.0143 | | Q2 | 0.2748 | 0.2505 | -0.0242 |
| Q3 | 0.2918 | 0.2695 | -0.0223 | | Q3 | 0.2802 | 0.2575 | -0.0228 |
| Q4 | 0.3062 | 0.2727 | -0.0335 | | Q4 | 0.3031 | 0.2582 | -0.0449 |
| Q5 high | 0.3180 | 0.2748 | -0.0432 | | Q5 high | 0.3066 | 0.2613 | -0.0453 |

62.6% (F2) and 69.4% (F1) of teams have sim below actual; the gap is broad. It is strongly tilted to high-FT-rate teams (gap-vs-prior correlation -0.22 / -0.27; Q1 is within about 1 SE of zero).
Team responsiveness (OLS slope of the team's bucket FTA/P on its prior; teams with at least 3 games in the bucket):

| season | bucket | teams | slope actual | slope sim | ratio | gap slope |
|---|---|---:|---:|---:|---:|---:|
| F2 | d0-14 | 307 | 0.495 | 0.105 | 0.21 | -0.390 |
| F2 | d15-45 | 362 | 0.355 | 0.192 | 0.54 | -0.164 |
| F2 | d46+ | 362 | 0.381 | 0.237 | 0.62 | -0.144 |
| F1 | d0-14 | 331 | 0.588 | 0.154 | 0.26 | -0.434 |
| F1 | d15-45 | 361 | 0.452 | 0.162 | 0.36 | -0.290 |
| F1 | d46+ | 361 | 0.467 | 0.258 | 0.55 | -0.210 |

The sim passes through only about a fifth to a quarter of the team FT-rate prior early (0.21-0.26), against 0.55-0.62 late, and actual early response is steeper than late. The sim is under-responsive to team whistle style at every date and worst early (the standing matchup-specific rule is violated by this sub-model set, early most). This is the home of the second owner (shooting-foul trips, about a fifth of the gap) and is a candidate in its own right; it does not by itself create the calendar-wide first-half bonus deficit because the accrual and trip LUTs carry no team terms and the all-teams level gap exists in the low-prior teams as well (smaller).

## 4. Per possession type (source split, F2, per possession)

Actual (chances) vs sim (R9ao3 tap). FTA/trip is FTA divided by (shooting trips + bonus trips + and-ones). H1 = first half.

| bucket | half | metric | actual | sim | gap |
|---|---|---|---:|---:|---:|
| d0-14 | H1 | FTA/poss | 0.2290 | 0.2049 | -0.0241 |
| | | shooting trips/poss | 0.0647 | 0.0627 | -0.0019 |
| | | **bonus trips/poss** | **0.0447** | **0.0339** | **-0.0108** |
| | | and-ones/poss | 0.0169 | 0.0168 | -0.0001 |
| | | FTA/trip | 1.814 | 1.805 | -0.0085 |
| | H2 | FTA/poss | 0.3365 | 0.3265 | -0.0100 |
| | | shooting trips/poss | 0.0732 | 0.0714 | -0.0019 |
| | | bonus trips/poss | 0.0884 | 0.0852 | -0.0032 |
| d15-45 | H1 | FTA/poss | 0.2075 | 0.2056 | -0.0018 |
| | | bonus trips/poss | 0.0330 | 0.0326 | -0.0004 |
| | H2 | FTA/poss | 0.3322 | 0.3349 | +0.0027 |
| d46+ | H1 | FTA/poss | 0.2001 | 0.2080 | +0.0079 |
| | | bonus trips/poss | 0.0277 | 0.0309 | +0.0032 |
| | H2 | FTA/poss | 0.3425 | 0.3479 | +0.0054 |

Two-factor split of bonus trips per possession = in-bonus share x trips per in-bonus possession (F2):

| half | bucket | in-bonus actual | in-bonus sim | trips per in-bonus actual | sim | bonus gap from state | bonus gap from trip rate |
|---|---|---:|---:|---:|---:|---:|---:|
| H1 | d0-14 | 0.2839 | 0.2140 | 0.1572 | 0.1590 | -0.0110 | +0.0004 |
| H1 | d15-45 | 0.2266 | 0.2103 | 0.1455 | 0.1552 | -0.0024 | +0.0020 |
| H1 | d46+ | 0.1960 | 0.2042 | 0.1411 | 0.1511 | +0.0012 | +0.0020 |
| H2 | d0-14 | 0.4234 | 0.3765 | 0.2089 | 0.2268 | -0.0098 | +0.0068 |
| H2 | d15-45 | 0.3942 | 0.3764 | 0.2165 | 0.2310 | -0.0038 | +0.0055 |
| H2 | d46+ | 0.3940 | 0.3790 | 0.2221 | 0.2355 | -0.0033 | +0.0051 |

Late-season H2: the sim is short on in-bonus share by 4% and long on trips per in-bonus possession by 6%; the two offset, so the full-season bonus FTA matches by compensation (the compensation pattern CLAUDE.md warns about). Any fix to early in-bonus share must be read together with the H2 trip-rate level, otherwise window H2 overshoots.
Actual first-half foul state by finer bucket (2025, per possession): non-trip + trip defence fouls 0.186 (d0-6), 0.185, 0.176, 0.172, 0.168, 0.167 (d46+); bonus-trip rate given the defence's team-foul count is flat in the calendar (count 6: 0.138 early, 0.141 late), confirming that the surplus is how fast teams reach the bonus.
Sim off-ball: the tap shows `off_foul` = 0 throughout (the offensive-foul arm `O2`/CL3 is not adopted); actual offensive fouls charged per first-half possession are 0.0217 in d0-6 against 0.0163 late (+33%).

## 5. Offline foul layers given their inputs (mean p - mean y)

Fold-0 seed predictions, F2 tested on 2024-25, F1 on 2023-24 (`round9/preds_trip`, `round9/preds_ao`, `round7/preds_poss`; served arms T2c, AO3, A2). Rel = (p-y)/y; se_rel is binomial.

| fold | bucket | shooting trip (T2c) rel | bonus trip (T2c) rel | and-one (AO3) rel | non-trip accrual (A2) rel |
|---|---|---:|---:|---:|---:|
| F2 | d0-14 | -2.8% (se 1.3) | -4.5% (1.3) | -0.9% (2.4) | -1.1% (1.1) |
| F2 | d15-45 | +0.2% | -4.5% | +1.4% | +7.5% |
| F2 | d46+ | +1.2% | -6.7% | +1.1% | +9.6% |
| F1 | d0-14 | -4.5% | -8.3% | -3.5% | +2.7% |
| F1 | d15-45 | -4.6% | -9.0% | -2.4% | +12.0% |
| F1 | d46+ | -4.0% | -9.1% | -5.6% | +13.7% |

F2 first half only, finer buckets: A2 rel -5.4% / -1.6% / +3.1% / +6.9% / +9.3% / +10.1% (d0-6, d7-13, d14-20, d21-27, d28-45, d46+); T2c bonus rel -7.0% / -1.9% / +0.5% / +1.4% / +6.0% / +5.2%; T2c shooting rel -4.1% / -3.8% / +0.6% / +1.0% / +1.1% / +3.5%.
Reading: the trip and and-one layers are level-biased (F1 more than F2) but roughly flat across the calendar, so they do not carry the early gap. The accrual layer is the one whose bias swings with the date: it under-predicts non-trip fouls early and over-predicts them by about 10% late (a late-season defect of its own, hidden by the T2c under-prediction).

As-of sample size (min of the two teams' games played, hypothesis b). The window is almost entirely min-games 0-2 (F2 84,201 of 92,087 chances; F1 83,398 of 98,436), so sample size and calendar are confounded inside the window.
F2 window, min games 0-2: shooting trip -3.2%, bonus -4.6%, accrual -0.9%; the 3-5 cell (F2 7,886 chances) is underpowered. F1 window 3-5 (15,038): -5.2%, -14.9%, +4.8%.
In d15-45, the 0-2 cells are small (F2 8,349) and bonus -10.1% (F2) / -13.5% (F1, 3,840): underpowered, no monotone slope with games played in either fold. There is no evidence that the gap shrinks as the sample grows beyond what the calendar alone says, and A2/T2c/AO3-state use no as-of count or rate, so hypothesis (b) as posed (raw-count features scoring thin early samples at a low-volume level) does not apply to the layers that own the gap. It applies at most to the possession-outcome shooting-trip base (team FTR as-of), see the team-response table.

By game context in the window (trip model relative bias, F2): site home -2.5% shoot / -4.4% bonus, neutral -6.3% / -5.0% (6,463 chances, underpowered); mismatch <5: -7.6% / -8.3%, 15+: -0.8% / +2.4%; final margin 0-5: -7.2% / -10.7%, 20+: +4.1% / +9.9%. Close games are under-predicted, blowouts over-predicted: a state-conditional ex-post pattern (the model conditions on foul state, not on score), not a calendar effect, and small relative to the calendar drift.

### 5b. Repeats every season and is not composition

Actual first-half rates per possession, early (d0-6) vs late (d46+), all four seasons:

| season | in-bonus d0-6 | in-bonus d46+ | bonus trips d0-6 | bonus trips d46+ | non-trip fouls d0-6 | d46+ | team-std early minus late in-bonus (composition part) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 0.280 | 0.208 | 0.0394 | 0.0277 | 0.1106 | 0.1026 | +0.0015 of +0.054 |
| 2023 | 0.302 | 0.209 | 0.0457 | 0.0281 | 0.1086 | 0.1024 | +0.0002 of +0.066 |
| 2024 | 0.309 | 0.202 | 0.0483 | 0.0277 | 0.1019 | 0.0922 | -0.0010 of +0.085 |
| 2025 | 0.296 | 0.196 | 0.0478 | 0.0277 | 0.0982 | 0.0921 | -0.0008 of +0.088 |

The effect appears in every season, before and inside the fit seasons, and it is entirely within teams.
It is a calendar whistle effect (early-season games call and accumulate more fouls per possession in the first half; the cause is not identifiable from these data and none is claimed), not mismatch, not conference, not MTE, not garbage time, not composition.

## Underpowered cells (labelled)

F2 neutral window cell (46 games); every F2 mismatch and final-margin window cell under 150 games except the two largest; per-team quintile cells (about 72 teams, SE 0.006-0.008); the min-games 3-5 and 10+ cells; F1 has no half-level tap, so the F1 source split, the H1/H2 two-factor split and the finer-bucket sim state are F2 only (F1 evidence: window gap -0.0285, same context and per-team shape, actual first-half excess in 2024 as large as 2025).
The cross-season spread in "actual window minus d46+" (2023 -0.002 to 2025 +0.011) means a one-season check would mislead on the whole-game size of the effect.

## DRAFT candidate list for a pre-registered round (NOT registered; nothing here is a spec)

Owner: possession_outcome foul accrual (`A2`, round 7 spec) first, then the shooting-trip team response. Standing rules apply: bake-off on the F1/F2 folds, spec in `docs/models/possession_outcome/experiments.md` and committed before any fit, spec-identical retrain noise floor, ties to the simpler model, paired-seed closed loop before shipping.

Block A, accrual calendar term (primary).
- `A2` reference (state-only).
- `A2_cal`: `A2` + `days_since_start` (pregame-known, no leak test needed beyond the standard change-form check; calendar is a schedule quantity).
- `A2_calH1`: `A2` + calendar x half interaction (the effect is first-half only: H2 early is flat or lower).
- `A2_wk`: `A2` with a fitted week-of-season term on the odds, fitted on train seasons only (a fitted feature, not an output multiplier).
- `A2_O`: `A2_cal` + the offensive-foul channel (`O2`, CL3 round 7; actual offensive fouls +33% in d0-6 and absent from the sim's team-foul state).
Primary metric: log-loss of `y_nt` on the test fold plus a state metric, the first-half in-bonus share by week, p-minus-y by week on true state. Secondary: p-minus-y of T2c trips at the new state. Segments: half, week, margin, site. Closed loop (paired seeds, window games d0-14 plus the d46+ set): window FTA/P, first-half in-bonus share by week, bonus trips per in-bonus possession by half, and G1-G9 (including full-season total bias and G5) must not regress.
Compensation guard: A2 is +10% over-predicted late (offline) and T2c bonus is -5% to -9% under (offline) while the sim's H2 trips per in-bonus possession run +6% over; a calendar term will change the late offsets too, so every arm is graded on d46+ as well as the window. If an arm closes the window but opens a late-season bias it is a failure, not a win.

Block B, shooting-trip team response (secondary, about a fifth of the gap).
- `T2c` reference; `T2c_team`: `T2c` + a team-prior drawn-foul term (AO3-style prior) so the shooting-trip rate responds to the team's FT-rate prior; `PO_n`: the round-4 reliability features (`off_n_prior`, `def_n_prior`) restricted to the two FT-trip classes (round 4 G4/G2 results on the whole class set stand; this narrows the target).
Metric: log-loss of `y_shoot`; segment: team prior quintile slope (must move from 0.21-0.26 toward the late 0.55-0.62, standing responsiveness rule), weeks 0-3 versus d46+.

Block C, held for the PM: whether the H2 trips-per-in-bonus overshoot (+6% to +8% all season) is a T2c level defect or a state-definition mismatch (per-chance offline rel is -10% while per-in-bonus-possession in-sim is +6%: not the same quantity). Cheap diagnostic before any arm.

Not candidates: opponent-adjusted foul rates and a conference flag (Decision 9 closed; `A2_D9a/D9b` tied A2); the shared game-level whistle latent (round 8, not adopted, `R8bS` failed a gate line); FT make arms (owned elsewhere).

Gates for closing the item: window FTA/P gap inside 2 SE (about 0.007) in both folds, first-half in-bonus share within 0.02 of actual in each week, d46+ total and FTA/P bias not worse, no G1-G9 regression.
