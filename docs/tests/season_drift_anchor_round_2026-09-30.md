# Season-drift anchor, round 1: one design across rebound, shot_block and FT technicals (2026-09-30)

Lane C. Pre-registration: `docs/models/season_drift/experiments.md` section 1,
committed 5003ce7 BEFORE any fit. Fitter `scripts/exp_season_drift_anchor_v1.py`
(predictions only), one blind grader `scripts/grade_season_drift_anchor_v1.py`
(commit c09930e). Offline, `S0` static calendar, both folds, 2025-26 sealed.

**NOTHING IS ADOPTED. No served default is changed. No engine file is touched.**
All outputs are in `results/season_drift/round1/` (gitignored, HF bulk).

## 1. Verdict under the pre-registered rule

**No design wins.** Each sub-model's mechanical selection: rebound `R` (no arm
eligible), shot_block `F` (eligible at 1.24 floors, a 7e-06 log-loss gain),
ft_tech `R` (no arm eligible). No design is selected in two of three
sub-models, so the po control check is not reached.

The rule was not close to arbitrary. One design is close, and it is blocked
by exactly two checks, both flagged below as properties of the checks rather
than of the design:

**`O`, the target relative to the as-of league level (offset; day 0 = prior
season's end level), and its fitted-update twin `P`**, pass the drift-stops
check and improve level and likelihood across the four models: `P` in all four
models on both folds, `O` in all but shot_block F1 (where it is 2.4e-05 worse
and its level +0.180 vs `R`'s -0.178 pp):

| model | R level (F2) | O level | P level | O / P gain vs R (x floor) | F1 confirms |
|---|---:|---:|---:|---:|---|
| rebound (pp) | -1.137 | -0.084 | -0.115 | 4.0 / 4.1 | yes (level -0.65 -> +0.12 / +0.04) |
| shot_block (pp) | -0.167 | -0.083 | -0.054 | 0.9 / 0.96 | P yes, O no |
| ft_tech (rel %) | +32.5 | -10.1 | -8.5 | 1.4 / 1.7 | yes (+31.7 -> -7.7 / -2.9) |
| po control (max class pp) | 1.535 | 0.401 | 0.293 | 0.99 / 1.02 (published floor 8.04e-04) | yes (1.32 -> 0.22 / 0.12) |

What blocks them:

1. **rebound E6 (calibration gate).** `O` 2.195 pp and `P` 2.071 pp against the
   2.0 pp gate (`R` 1.958 PASS). The failing cell is the TOP OREB decile
   (+2.20 / +2.07 pp). `R`'s own top decile is +0.88 pp while its level shift is
   -1.02 pp: the ~1.9 pp top-decile shape defect was already there and was
   hidden by the level bias. Fixing the level unmasks it. This is the
   bottom-up rule at work, not a defect of the anchor; the shape belongs to the
   rebound model.
2. **ft_tech E5 (spread / slope must not fall below `R` by > 0.05).** `R`
   over-predicts the level by 32%, and in a log-link model a multiplicative level
   error inflates the team span and the team SD by the same factor. `R`'s slope
   1.54 and SD ratio 0.90 are the level error, not responsiveness: divided by
   the team-level ratio they are 1.17 / 0.68, and `O` / `P` / `F` sit at
   1.11 / 0.65-0.66 on that scale (unchanged spread). The pre-registered E5 is
   one-sided (written for rebound's under-responsive 0.685) and reads a level
   fix as spread compression here. A SUPPLEMENTARY symmetric, level-normalised
   check (reported, not the rule) passes `O`, `P`, `F` for ft_tech.

Counterfactual, stated only so the PM can see the structure (NOT the result):
with E6 and E5 read as in points 1-2, rebound would select `O` (tie with `P`
inside one floor, `O` simpler) and ft_tech would select `O` (tie with `P` and
`F` inside one floor); `O` would then be selected in 2 of 3 and passes the po
control (primary +7.95e-04, max class gap 1.54 -> 0.40 pp, worst class slope
drop 0.048 < 0.05; `P` fails that line by 0.003-0.005 on FGA_jump2 and
FT_trip_shooting). This needs a PM ruling on the two checks; the round does not
make it.

## 2. What each arm taught (multi-level evidence in section 6)

- `C0` (carry last season's level, no update) fixes rebound's level partly
  (-0.55 pp) but is the drift-stops failure the PM asked about: FT technicals'
  2023 spike carried into 2024 gives +78% on F1, and po TOV F2 -1.29 pp. Pooled
  drift-stops score +0.061: **ineligible everywhere (E4)**.
- `T` (trend extrapolation, cautionary) reproduces rebound round 3's `A5`
  exactly (0.645024, -0.337 pp: a cross-check of this harness), but FT F1 +114%
  and pooled drift-stops +0.092: **ineligible everywhere (E4)**. The
  extrapolation inverts on the first reversal it meets.
- `W` (recency weights, 365 d) moves levels by 0.1 pp or less, loses likelihood
  in rebound and po, fails E4 (+0.008, the ft 2023 spike). Not the fix.
- `F` (league level as a FEATURE, with the day-0 defect of rebound `A4` fixed)
  is **model-class dependent**: fine in the linear models (ft_tech +2.45
  floors, level +1.1%; shot_block selected), destructive in both trees (rebound
  -4.2 floors, team slope 0.67 -> 0.37; po -4.4 floors, level 3.0 pp). A tree
  splits on the anchor instead of learning a deviation from it. A design that
  works in one model class only cannot be the cross-model answer.
- `O` vs `P`: the fitted update weight is n0 = 10,000 live misses for rebound,
  10,000 missed FGA for shot_block, 31,600 team-chances for ft_tech, 10,000
  chances for po on F2 (F1: 10,000 / 100,000 / 31,600 / 31,600), i.e. a few
  days of league play (shot_block F1: ~7 weeks): the fit says "use the prior
  season for the first days, then the season to date". `P` is better
  than `O` early (rebound Nov-Dec -0.42 vs -0.33 pp is the exception; ft Nov
  -8.8% vs -13.7%) and the two are inside one floor of each other everywhere.

**shot_block's problem is not a league level.** The pooled block level is only
-0.17 pp off under `R`; rim is -0.83 pp and jump2 +0.29 pp. A scalar anchor
cannot move a composition-specific drift, which is why every arm is within ~1
floor of `R` there. EXPLORATORY, not pre-registered, not selectable
(`scripts/exp_season_drift_anchor_cell_v1.py`): the same `O` / `P` anchors built
PER SHOT TYPE take rim from -0.83 to +0.43 / +0.35 pp on F2 and -0.56 to
-0.09 / -0.16 pp on F1, pass the shot_block level gate (overall <= 0.25, every
type <= 0.50) on both folds (`R` fails it on rim), with a F2 gain of 3.0e-05 /
4.9e-05 (1.0 / 1.9 paired SE). The anchor cell should be the model's own
level-gate cell; that is the next pre-registration, not a result.

## 3. The segment lines the brief made mandatory (F2 unless stated)

| arm | model | x floor | level | Nov-Dec | drift-stops E4 (pooled, design-level) | team slope | SD ratio (nc) | F1 confirm |
|---|---|---:|---:|---:|---:|---:|---:|---|
| R | rebound | - | -1.137 | -1.376 | 0 | 0.666 | 0.681 | - |
| C0 | rebound | 4.1 | -0.554 | -0.802 | +0.061 FAIL | 0.672 | 0.690 | yes |
| O | rebound | 4.0 | -0.084 | -0.334 | -0.025 | 0.677 | 0.693 | yes |
| P | rebound | 4.1 | -0.115 | -0.422 | -0.033 | 0.669 | 0.692 | yes |
| W | rebound | -1.7 | -1.036 | -1.297 | +0.008 FAIL | 0.668 | 0.688 | no |
| F | rebound | -4.2 | -0.171 | -0.337 | -0.009 | 0.372 | 0.451 | no |
| T | rebound | 3.4 | -0.337 | -0.580 | +0.092 FAIL | 0.671 | 0.696 | yes |
| R | shot_block | - | -0.167 | -0.235 | 0 | 0.694 | 0.832 | - |
| C0 | shot_block | -0.2 | +0.217 | +0.145 | FAIL | 0.715 | 0.857 | no |
| O | shot_block | 0.9 | -0.083 | -0.134 | pass | 0.699 | 0.837 | no |
| P | shot_block | 0.96 | -0.054 | -0.103 | pass | 0.701 | 0.840 | yes |
| W | shot_block | 1.45 | -0.153 | -0.221 | FAIL | 0.686 | 0.826 | no |
| F | shot_block | 1.24 | -0.135 | -0.197 | pass | 0.696 | 0.834 | yes |
| T | shot_block | -0.1 | +0.189 | +0.115 | FAIL | 0.714 | 0.857 | no |
| R | ft_tech | - | +32.5% | +31.4% | 0 | 1.544 | 0.896 | - |
| C0 | ft_tech | 3.0 | +5.1% | +4.4% | FAIL | 1.211 | 0.697 | no (+78%) |
| O | ft_tech | 1.4 | -10.1% | -13.5% | pass | 0.998 | 0.588 | yes |
| P | ft_tech | 1.7 | -8.5% | -10.3% | pass | 1.015 | 0.598 | yes |
| W | ft_tech | 4.6 | +27.5% | +26.5% | FAIL | 1.486 | 0.848 | no |
| F | ft_tech | 2.45 | +1.1% | -1.9% | pass | 1.138 | 0.668 | yes |
| T | ft_tech | 5.2 | +30.7% | +29.7% | FAIL | 1.527 | 0.886 | no (+114%) |

Drift-stops cells: 10 of 28 rate x season cells are flat or reversed (shot_block
F1 jump2 and rim, F2 three; ft_tech F1; po F1 FGA_3 and FT_trip_bonus, F2 TOV,
FGA_rim, FGA_3, FT_trip_shooting). All 8 rebound miss-type cells continued the
trend, so rebound supplies no drift-stops evidence of its own; its arms are
judged on the other models' reversals, which is what E4's pooling was for. The
pooled score is dominated by the ft_tech F1 reversal (relative errors of
0.3-1.1 against pp-scale cells elsewhere); per-model scores are in section 6.

Noise floors: reseed of `R` (seed 1) rebound 3.9e-06 / level 0.039 pp, po
1.6e-05 / 0.038 pp; shot_block and ft_tech are deterministic, their floor is 2x
the paired game-block bootstrap SE (200 reps). Operative floor per cell =
max(reseed, published 6.7e-05 rebound / 8.04e-04 po, 2 x paired SE).

Harness checks: rebound `R` reproduces round 3 `A0B0C0` (0.645565, -1.137 pp);
`T` reproduces `A5` (0.645024, -0.337 pp); shot_block `R` (offset-capable
L-BFGS logistic) reproduces sklearn `K2` to 1.5e-06 (0.2632385 vs 0.263240).

## 4. What did NOT run

- Stage 2 (`S1_weekly`, the served refit calendar, ~3-4 min per rebound refit
  here x 23 refits) for any arm: out of scope today. The fitter has no
  `S1_weekly` path yet; it needs a v2 (below).
- Any closed loop: needs an engine-side offset feed for rebound / po (and the
  drawn-block feed from round 3); out of scope.
- A spec-identical reseed for shot_block / ft_tech: deterministic fitters, zero
  by construction; the paired bootstrap stands in, as pre-registered.
- The per-shot-type anchor is EXPLORATORY and has no F1-confirm, floor or
  drift-stops standing; it must be pre-registered before it can be selected.

## 5. Resume commands

    # re-run this round from scratch (cells skip if their .npy exists)
    .venv/Scripts/python.exe scripts/exp_season_drift_anchor_v1.py --light
    .venv/Scripts/python.exe scripts/exp_season_drift_anchor_v1.py --heavy --jobs 3
    .venv/Scripts/python.exe scripts/grade_season_drift_anchor_v1.py
    .venv/Scripts/python.exe scripts/exp_season_drift_anchor_cell_v1.py   # exploratory

    # stage 2 (NOT YET WRITTEN): exp_season_drift_anchor_v2.py = v1 + a
    # --scheme S1_weekly path reusing train_rebound_v3_round3.s1_weekly_cuts /
    # fit_predict_scheme; the anchor is already an as-of per-row column, so each
    # refit only re-fits the model. ~1.5 h per rebound cell at n_jobs=1 on this
    # box, ~3 h per po cell; run on the box.
    .venv/Scripts/python.exe scripts/exp_season_drift_anchor_v2.py --scheme S1_weekly \
        --cells rebound:F2:R:0,rebound:F2:O:0,rebound:F2:P:0,po:F2:R:0,po:F2:O:0
    # the round-3 stage-2 cells stay valid for R and T (= A5):
    .venv/Scripts/python.exe scripts/train_rebound_v3_round3.py --stage 2 --folds F2 --feeds --arms A0B0C0
    .venv/Scripts/python.exe scripts/train_rebound_v3_round3.py --stage 2 --folds F2 --feeds --arms A5

    # closed loop (after a PM ruling on E5/E6 and stage 2): wire the as-of
    # league-level offset DEFAULT-OFF into the rebound and po engine inputs,
    # parity against docs/ops/parity_reference_windows_v6.json, then
    # 500 games x 25 paired seeds as in round 3 section 7.

## 6. Full tables (generated by the grader, `results/season_drift/round1/tables_v1.md`)

Units: rebound / shot_block / po level gaps in pp; ft_tech in RELATIVE % of the realised rate. `sd_nc` = SD of team predictions over the noise-corrected SD of realised team rates. Slope = prior-season-quintile span ratio.
Reseed floors (R seed 1 vs 0, F2): `{"rebound": {"primary": 3.900000000056636e-06, "level": 0.03929999999999989}, "po": {"primary": 1.6000000000016e-05, "level": 0.0377}}`

## rebound

### rebound F2
| arm | primary | gain vs R | paired boot SE | level | Nov-Dec | team slope | slope Nov-Dec | sd_nc | sd_raw | slope/lvl (supp) | sd_nc/lvl (supp) | calib | per-game MAE pp | drift-stops (model) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R | 0.6455654 | 0.0 | 0.0 | -1.137 | -1.3764 | 0.6656 | 0.5712 | 0.681 | 0.6415 | 0.6904 | 0.7064 | P 1.958 | 4.9645 |  |
| C0 | 0.6449755 | 0.0005899 | 7.2e-05 | -0.5536 | -0.8015 | 0.6717 | 0.568 | 0.6902 | 0.6502 | 0.6829 | 0.7017 | P 1.615 | 4.9259 |  |
| O | 0.644914 | 0.0006514 | 8.12e-05 | -0.0836 | -0.3339 | 0.6768 | 0.5718 | 0.6925 | 0.6524 | 0.6772 | 0.6929 | F 2.195 | 4.9158 |  |
| P | 0.6448969 | 0.0006685 | 8.09e-05 | -0.1145 | -0.4217 | 0.6687 | 0.5741 | 0.6916 | 0.6515 | 0.6698 | 0.6927 | F 2.071 | 4.9238 |  |
| W | 0.6458781 | -0.0003127 | 9.11e-05 | -1.0363 | -1.2972 | 0.668 | 0.5696 | 0.6879 | 0.648 | 0.6905 | 0.7111 | F 2.093 | 4.9558 |  |
| F | 0.6472381 | -0.0016727 | 0.0001983 | -0.171 | -0.337 | 0.372 | 0.3265 | 0.4513 | 0.4251 | 0.3725 | 0.4519 | P 1.749 | 5.07 |  |
| T | 0.6450242 | 0.0005412 | 7.91e-05 | -0.337 | -0.58 | 0.6712 | 0.5688 | 0.6955 | 0.6551 | 0.6773 | 0.7019 | P 1.921 | 4.9151 |  |

level by sub-type and by site

| arm | ft | jump2 | rim | three | home | away | neutral |
|---|---|---|---|---|---|---|---|
| R | -0.8835 | -1.3169 | -0.5114 | -1.4487 | -1.2422 | -0.9905 | -1.2905 |
| C0 | -0.5318 | -0.7344 | 0.1549 | -0.8649 | -0.6529 | -0.4093 | -0.7197 |
| O | -0.281 | -0.2544 | 0.7171 | -0.4088 | -0.1693 | 0.0657 | -0.3123 |
| P | -0.3034 | -0.283 | 0.6892 | -0.4446 | -0.1909 | 0.0208 | -0.3258 |
| W | -0.7273 | -1.2001 | -0.454 | -1.3435 | -1.1878 | -0.8448 | -1.19 |
| F | 0.6939 | -0.5783 | -0.0308 | -0.1961 | -0.0851 | -0.0889 | -0.7412 |
| T | -0.4133 | -0.4986 | 0.4192 | -0.6667 | -0.4282 | -0.1996 | -0.5059 |

level by month

| arm | 1 | 2 | 3 | 4 | 11 | 12 |
|---|---|---|---|---|---|---|
| R | -1.6932 | -0.5451 | -0.5285 | 0.8139 | -1.427 | -1.3096 |
| C0 | -1.0899 | 0.0305 | 0.056 | 1.3506 | -0.8346 | -0.7576 |
| O | -0.5261 | 0.5247 | 0.3251 | 1.5158 | -0.3774 | -0.2764 |
| P | -0.5445 | 0.5287 | 0.3631 | 1.6543 | -0.5218 | -0.2893 |
| W | -1.5764 | -0.4059 | -0.4686 | 1.078 | -1.3215 | -1.2651 |
| F | -0.5589 | 0.4197 | -0.0682 | 1.5874 | -0.4332 | -0.2099 |
| T | -0.8702 | 0.2637 | 0.2227 | 1.6678 | -0.6117 | -0.5381 |

anchor: n0 fitted = 10000.0 (fit seasons [2023, 2024]); prior-season level for test = {'2025': [0.291553]}; trend to test = {'2025': [0.296061]}; Lbar = [0.287038]

### rebound F1
| arm | primary | gain vs R | paired boot SE | level | Nov-Dec | team slope | slope Nov-Dec | sd_nc | sd_raw | slope/lvl (supp) | sd_nc/lvl (supp) | calib | per-game MAE pp | drift-stops (model) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R | 0.6204458 | 0.0 | 0.0 | -0.6494 | -1.1987 | 0.6007 | 0.5665 | 0.7112 | 0.6674 | 0.6131 | 0.7259 | P 1.629 | 4.8003 |  |
| C0 | 0.6201706 | 0.0002752 | 7.47e-05 | -0.3242 | -0.8708 | 0.6087 | 0.5677 | 0.7183 | 0.6741 | 0.6142 | 0.7248 | P 1.849 | 4.7981 |  |
| O | 0.6201114 | 0.0003344 | 7.93e-05 | 0.1152 | -0.2332 | 0.6103 | 0.5658 | 0.7265 | 0.6818 | 0.6066 | 0.7221 | F 2.293 | 4.8021 |  |
| P | 0.620175 | 0.0002708 | 7.5e-05 | 0.044 | -0.369 | 0.6094 | 0.5746 | 0.7287 | 0.6838 | 0.6072 | 0.726 | F 2.159 | 4.7903 |  |
| W | 0.6206752 | -0.0002294 | 8.81e-05 | -0.6693 | -1.1776 | 0.6056 | 0.5691 | 0.7166 | 0.6725 | 0.6185 | 0.7319 | P 1.599 | 4.8063 |  |
| F | 0.6232948 | -0.002849 | 0.0001683 | -1.4163 | -2.047 | 0.3824 | 0.3753 | 0.5178 | 0.4859 | 0.4 | 0.5416 | F 3.635 | 5.0895 |  |
| T | 0.6200958 | 0.00035 | 7.39e-05 | -0.0752 | -0.6118 | 0.6074 | 0.5693 | 0.7214 | 0.677 | 0.6077 | 0.7217 | P 1.9 | 4.7872 |  |

level by sub-type and by site

| arm | ft | jump2 | rim | three | home | away | neutral |
|---|---|---|---|---|---|---|---|
| R | -0.2929 | -0.7657 | -0.6031 | -0.6809 | -0.4397 | -0.8341 | -0.7112 |
| C0 | -0.1255 | -0.4492 | -0.2393 | -0.3396 | -0.1221 | -0.508 | -0.363 |
| O | 0.1133 | -0.0199 | 0.2962 | 0.0903 | 0.3214 | -0.0652 | 0.0503 |
| P | 0.105 | -0.0773 | 0.1937 | 0.0161 | 0.2526 | -0.1556 | 0.0383 |
| W | -0.3292 | -0.7887 | -0.6481 | -0.6796 | -0.4904 | -0.8528 | -0.631 |
| F | -0.1591 | -1.6303 | -1.2268 | -1.6699 | 0.107 | -2.6929 | -2.0924 |
| T | 0.0182 | -0.1959 | 0.0646 | -0.1044 | 0.1337 | -0.2643 | -0.1186 |

level by month

| arm | 1 | 2 | 3 | 4 | 11 | 12 |
|---|---|---|---|---|---|---|
| R | -0.3671 | -0.2769 | -0.3017 | 1.2409 | -1.2092 | -1.187 |
| C0 | -0.0527 | 0.0368 | 0.0535 | 1.6833 | -0.8488 | -0.8952 |
| O | 0.4418 | 0.2868 | 0.1897 | 2.0834 | -0.2354 | -0.2307 |
| P | 0.3759 | 0.2741 | 0.1828 | 2.1532 | -0.4368 | -0.2938 |
| W | -0.4261 | -0.3182 | -0.3274 | 1.0252 | -1.1761 | -1.1792 |
| F | -1.1701 | -0.9284 | -0.9864 | 0.6681 | -2.0315 | -2.0642 |
| T | 0.1734 | 0.3161 | 0.2634 | 1.8879 | -0.6225 | -0.6 |

anchor: n0 fitted = 10000.0 (fit seasons [2023]); prior-season level for test = {'2024': [0.286827]}; trend to test = {'2024': [0.291226]}; Lbar = [0.28467]

## shot_block

### shot_block F2
| arm | primary | gain vs R | paired boot SE | level | Nov-Dec | team slope | slope Nov-Dec | sd_nc | sd_raw | slope/lvl (supp) | sd_nc/lvl (supp) | calib | per-game MAE pp | drift-stops (model) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R | 0.2632385 | 0.0 | 0.0 | -0.1672 | -0.2354 | 0.6937 | 0.6338 | 0.8315 | 0.7679 | 0.705 | 0.845 | P 1.071 | 3.1004 | 0.0 |
| C0 | 0.26325 | -1.15e-05 | 2.61e-05 | 0.2173 | 0.1448 | 0.7148 | 0.653 | 0.8572 | 0.7917 | 0.6987 | 0.8379 | P 0.707 | 3.1304 | 0.00019 |
| O | 0.2632244 | 1.41e-05 | 7.7e-06 | -0.0831 | -0.1337 | 0.6985 | 0.6378 | 0.8371 | 0.7731 | 0.7038 | 0.8434 | P 0.8 | 3.1033 | 0.00679 |
| P | 0.2632213 | 1.72e-05 | 9e-06 | -0.0544 | -0.1027 | 0.7005 | 0.64 | 0.8396 | 0.7754 | 0.7037 | 0.8434 | P 0.724 | 3.1052 | -0.00326 |
| W | 0.2632202 | 1.83e-05 | 6.3e-06 | -0.1532 | -0.2209 | 0.6859 | 0.6251 | 0.8259 | 0.7627 | 0.6958 | 0.8378 | P 0.893 | 3.1017 | 0.00112 |
| F | 0.2632313 | 7.2e-06 | 2.9e-06 | -0.1354 | -0.1971 | 0.6955 | 0.6353 | 0.8336 | 0.7699 | 0.7045 | 0.8444 | P 0.96 | 3.1011 | -1e-05 |
| T | 0.2632439 | -5.4e-06 | 2.43e-05 | 0.1893 | 0.1153 | 0.7144 | 0.6524 | 0.8566 | 0.7911 | 0.7003 | 0.8397 | P 0.622 | 3.1265 | 0.00464 |

level by sub-type and by site

| arm | jump2 | rim | three | home | away | neutral |
|---|---|---|---|---|---|---|
| R | 0.2886 | -0.8332 | -0.0458 | -0.2802 | -0.1078 | 0.0015 |
| C0 | 0.6709 | 0.0995 | 0.0239 | 0.0743 | 0.3068 | 0.3816 |
| O | 0.3783 | -0.6357 | -0.0301 | -0.2066 | -0.0222 | 0.1158 |
| P | 0.4066 | -0.5661 | -0.0249 | -0.1812 | 0.0095 | 0.1446 |
| W | 0.237 | -0.7547 | -0.0311 | -0.2327 | -0.1445 | 0.0805 |
| F | 0.3225 | -0.7587 | -0.0399 | -0.2524 | -0.0754 | 0.0446 |
| T | 0.6514 | 0.022 | 0.0196 | 0.0487 | 0.2751 | 0.3585 |

level by month

| arm | 1 | 2 | 3 | 4 | 11 | 12 |
|---|---|---|---|---|---|---|
| R | -0.2628 | -0.0261 | -0.0778 | 0.8615 | -0.3308 | -0.1101 |
| C0 | 0.1221 | 0.3612 | 0.3134 | 1.2215 | 0.0452 | 0.2755 |
| O | -0.2122 | 0.061 | 0.0143 | 0.9572 | -0.1489 | -0.1137 |
| P | -0.182 | 0.0866 | 0.0393 | 0.9784 | -0.1259 | -0.0722 |
| W | -0.2572 | -0.0149 | -0.0456 | 0.9083 | -0.3098 | -0.1042 |
| F | -0.2435 | 0.0068 | -0.043 | 0.8977 | -0.2626 | -0.1111 |
| T | 0.0921 | 0.338 | 0.2847 | 1.1974 | 0.0169 | 0.2446 |

anchor: n0 fitted = 10000.0 (fit seasons [2023, 2024]); prior-season level for test = {'2025': [0.100451]}; trend to test = {'2025': [0.101472]}; Lbar = [0.097283]

### shot_block F1
| arm | primary | gain vs R | paired boot SE | level | Nov-Dec | team slope | slope Nov-Dec | sd_nc | sd_raw | slope/lvl (supp) | sd_nc/lvl (supp) | calib | per-game MAE pp | drift-stops (model) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R | 0.2693475 | 0.0 | 0.0 | -0.1777 | -0.2447 | 0.785 | 0.7906 | 0.8259 | 0.7636 | 0.7992 | 0.8408 | P 1.104 | 3.1874 | 0.0 |
| C0 | 0.2693695 | -2.2e-05 | 5.5e-06 | -0.2573 | -0.3239 | 0.7799 | 0.7855 | 0.8205 | 0.7586 | 0.8005 | 0.8421 | P 1.256 | 3.186 | 0.00019 |
| O | 0.2693714 | -2.39e-05 | 2.47e-05 | 0.1797 | 0.1289 | 0.8082 | 0.8155 | 0.8494 | 0.7853 | 0.794 | 0.8345 | P 1.385 | 3.2127 | 0.00679 |
| P | 0.2693387 | 8.8e-06 | 1.39e-05 | 0.0048 | -0.1555 | 0.7968 | 0.7969 | 0.8384 | 0.7752 | 0.7965 | 0.8381 | P 0.927 | 3.1949 | -0.00326 |
| W | 0.2693711 | -2.36e-05 | 5.3e-06 | -0.2316 | -0.2964 | 0.7803 | 0.7888 | 0.8236 | 0.7614 | 0.7986 | 0.843 | P 1.219 | 3.1875 | 0.00112 |
| F | 0.2693369 | 1.06e-05 | 1.15e-05 | -0.0126 | -0.0722 | 0.7958 | 0.8022 | 0.8368 | 0.7737 | 0.7968 | 0.8379 | P 0.862 | 3.196 | -1e-05 |
| T | 0.2693993 | -5.18e-05 | 1.1e-05 | -0.3352 | -0.4013 | 0.7749 | 0.7807 | 0.8152 | 0.7537 | 0.8017 | 0.8434 | P 1.391 | 3.186 | 0.00464 |

level by sub-type and by site

| arm | jump2 | rim | three | home | away | neutral |
|---|---|---|---|---|---|---|
| R | 0.0042 | -0.5583 | -0.0595 | -0.3138 | -0.0012 | -0.3439 |
| C0 | -0.0726 | -0.7472 | -0.0733 | -0.3867 | -0.087 | -0.4237 |
| O | 0.3532 | 0.2848 | 0.0031 | 0.0122 | 0.3778 | 0.0427 |
| P | 0.1824 | -0.1272 | -0.0278 | -0.1442 | 0.1903 | -0.1497 |
| W | -0.079 | -0.6725 | -0.0572 | -0.3585 | -0.0672 | -0.3852 |
| F | 0.1651 | -0.168 | -0.0307 | -0.1632 | 0.1741 | -0.1653 |
| T | -0.1493 | -0.9301 | -0.0871 | -0.4592 | -0.1711 | -0.4978 |

level by month

| arm | 1 | 2 | 3 | 4 | 11 | 12 |
|---|---|---|---|---|---|---|
| R | -0.0561 | -0.1234 | -0.2804 | -2.2403 | -0.2283 | -0.2628 |
| C0 | -0.1356 | -0.2029 | -0.3612 | -2.3228 | -0.3071 | -0.3425 |
| O | 0.2858 | 0.2292 | 0.0678 | -1.8428 | 0.155 | 0.1001 |
| P | 0.1484 | 0.1303 | -0.0084 | -1.9279 | -0.1973 | -0.1093 |
| W | -0.115 | -0.1781 | -0.3297 | -2.3189 | -0.2775 | -0.3173 |
| F | 0.1021 | 0.0397 | -0.1194 | -2.0566 | -0.0516 | -0.095 |
| T | -0.2141 | -0.2814 | -0.4386 | -2.3983 | -0.3839 | -0.4205 |

anchor: n0 fitted = 100000.0 (fit seasons [2023]); prior-season level for test = {'2024': [0.095058]}; trend to test = {'2024': [0.093893]}; Lbar = [0.09563]

## ft_tech

### ft_tech F2
| arm | primary | gain vs R | paired boot SE | level | Nov-Dec | team slope | slope Nov-Dec | sd_nc | sd_raw | slope/lvl (supp) | sd_nc/lvl (supp) | calib | per-game MAE pp | drift-stops (model) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R | 0.4851456 | 0.0 | 0.0 | 32.4683 | 31.3914 | 1.5443 | 0.9928 | 0.8963 | 0.5877 | 1.1657 | 0.6766 | F 32.468 |  | 0.0 |
| C0 | 0.4765076 | 0.008638 | 0.0014294 | 5.1324 | 4.3699 | 1.2109 | 0.7816 | 0.6974 | 0.4573 | 1.1517 | 0.6633 | P 5.132 |  | 0.46655 |
| O | 0.4781739 | 0.0069717 | 0.0024657 | -10.1496 | -13.465 | 0.9975 | 0.6133 | 0.5881 | 0.3857 | 1.1098 | 0.6543 | F -10.15 |  | -0.2409 |
| P | 0.4773575 | 0.0077881 | 0.0023042 | -8.5133 | -10.34 | 1.0152 | 0.6364 | 0.5979 | 0.3921 | 1.1092 | 0.6533 | P -8.513 |  | -0.28888 |
| W | 0.4827303 | 0.0024153 | 0.0002604 | 27.546 | 26.5118 | 1.4862 | 0.9654 | 0.8479 | 0.556 | 1.1652 | 0.6647 | F 27.546 |  | 0.07736 |
| F | 0.4767423 | 0.0084033 | 0.001714 | 1.1303 | -1.8514 | 1.1381 | 0.708 | 0.6681 | 0.4381 | 1.1251 | 0.6605 | P 1.13 |  | -0.29213 |
| T | 0.4843001 | 0.0008455 | 8.13e-05 | 30.7356 | 29.6695 | 1.5268 | 0.9817 | 0.8861 | 0.5811 | 1.1678 | 0.6778 | F 30.736 |  | 0.82145 |

level by sub-type and by site

| arm | all | home | away | neutral |
|---|---|---|---|---|
| R | 32.4683 | 27.5014 | 30.9667 | 54.0754 |
| C0 | 5.1324 | 1.0433 | 3.8909 | 22.9418 |
| O | -10.1496 | -13.1771 | -10.8632 | 2.2243 |
| P | -8.5133 | -11.5928 | -9.2786 | 4.2288 |
| W | 27.546 | 25.2847 | 24.7096 | 45.8874 |
| F | 1.1303 | -2.415 | 0.2111 | 15.9501 |
| T | 30.7356 | 25.8349 | 29.2511 | 52.0664 |

level by month

| arm | 1 | 2 | 3 | 4 | 11 | 12 |
|---|---|---|---|---|---|---|
| R | 39.3758 | 22.0194 | 48.2763 | -42.5132 | 36.3244 | 25.3904 |
| C0 | 10.5023 | -3.2747 | 17.8203 | -54.0893 | 8.368 | -0.4938 |
| O | -4.0576 | -15.6717 | 2.4976 | -61.1159 | -13.6974 | -13.1822 |
| P | -3.2394 | -15.0734 | 3.3862 | -60.6809 | -8.8236 | -12.1846 |
| W | 34.5328 | 17.7526 | 41.6685 | -45.4246 | 31.1317 | 20.8918 |
| F | 7.567 | -5.5578 | 14.8604 | -56.1267 | -1.0638 | -2.8094 |
| T | 37.5526 | 20.4239 | 46.3449 | -43.2595 | 34.537 | 23.7483 |

anchor: n0 fitted = 31622.776601683792 (fit seasons [2023, 2024]); prior-season level for test = {'2025': [0.000708]}; trend to test = {'2025': [0.000832]}; Lbar = [0.000844]

### ft_tech F1
| arm | primary | gain vs R | paired boot SE | level | Nov-Dec | team slope | slope Nov-Dec | sd_nc | sd_raw | slope/lvl (supp) | sd_nc/lvl (supp) | calib | per-game MAE pp | drift-stops (model) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R | 0.5153666 | 0.0 | 0.0 | 31.7406 | 24.6818 | 1.2783 | 1.4608 | 0.8862 | 0.6321 | 0.9608 | 0.6661 | F 31.741 |  | 0.0 |
| C0 | 0.5522354 | -0.0368688 | 0.0021189 | 78.3959 | 68.737 | 1.8766 | 2.1755 | 1.2813 | 0.9139 | 1.0417 | 0.7113 | F 78.396 |  | 0.46655 |
| O | 0.5089109 | 0.0064557 | 0.0025809 | -7.6515 | -11.7222 | 0.7784 | 0.8707 | 0.555 | 0.3959 | 0.8342 | 0.5948 | P -7.651 |  | -0.2409 |
| P | 0.5075403 | 0.0078263 | 0.0022027 | -2.8525 | -4.3021 | 0.8181 | 0.9476 | 0.582 | 0.4151 | 0.8333 | 0.5928 | P -2.853 |  | -0.28888 |
| W | 0.5200332 | -0.0046666 | 0.0004094 | 39.4769 | 32.1672 | 1.272 | 1.4482 | 0.8847 | 0.631 | 0.9029 | 0.628 | F 39.477 |  | 0.07736 |
| F | 0.5075257 | 0.0078409 | 0.0018226 | 2.5282 | -2.336 | 0.8917 | 1.0015 | 0.6317 | 0.4506 | 0.8608 | 0.6098 | P 2.528 |  | -0.29213 |
| T | 0.5918116 | -0.076445 | 0.0033394 | 113.8858 | 102.5507 | 1.8468 | 2.0591 | 1.3185 | 0.9404 | 0.8548 | 0.6103 | F 113.886 |  | 0.82145 |

level by sub-type and by site

| arm | all | home | away | neutral |
|---|---|---|---|---|
| R | 31.7406 | 22.5916 | 38.0772 | 36.3084 |
| C0 | 78.3959 | 65.7797 | 86.9038 | 85.488 |
| O | -7.6515 | -13.3662 | -2.6568 | -8.3753 |
| P | -2.8525 | -9.0362 | 2.237 | -2.549 |
| W | 39.4769 | 30.008 | 46.0911 | 44.0109 |
| F | 2.5282 | -4.0403 | 7.8848 | 3.022 |
| T | 113.8858 | 99.4787 | 124.344 | 119.4222 |

level by month

| arm | 1 | 2 | 3 | 4 | 11 | 12 |
|---|---|---|---|---|---|---|
| R | 35.5098 | 29.589 | 49.5142 | nan | 27.0087 | 22.2011 |
| C0 | 83.2645 | 75.5339 | 103.0845 | nan | 71.8244 | 65.4454 |
| O | -4.7796 | -9.5901 | 2.5646 | nan | -9.4478 | -14.1468 |
| P | -1.3767 | -6.8987 | 5.696 | nan | 0.9089 | -9.8577 |
| W | 43.5569 | 37.0552 | 57.9053 | nan | 34.7264 | 29.4388 |
| F | 5.7006 | 0.5416 | 14.6297 | nan | -0.1274 | -4.6906 |
| T | 120.4464 | 110.3917 | 141.5966 | nan | 106.386 | 98.4618 |

anchor: n0 fitted = 31622.776601683792 (fit seasons [2023]); prior-season level for test = {'2024': [0.001102]}; trend to test = {'2024': [0.001484]}; Lbar = [0.000915]

## po

### po F2
| arm | primary | gain vs R | paired boot SE | level max|cls| | Nov-Dec max|cls| | team slope | slope Nov-Dec | sd_nc | sd_raw | slope/lvl (supp) | sd_nc/lvl (supp) | calib | per-game MAE pp | drift-stops (model) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R | 1.5167916 | 0.0 | 0.0 | 1.5354 | 1.3103 | 0.7054 | 0.7187 | 0.7441 | 0.6646 | 0.7123 | 0.7514 | F 2.777 |  | 0.0 |
| C0 | 1.5173391 | -0.0005475 | 0.000114 | 1.2942 | 1.7187 | 0.6748 | 0.6948 | 0.7058 | 0.6303 | 0.7363 | 0.7701 | P 1.758 |  | 0.0243 |
| O | 1.5159963 | 0.0007953 | 0.000106 | 0.4006 | 0.2439 | 0.6992 | 0.7311 | 0.7273 | 0.6495 | 0.7073 | 0.7357 | P 1.006 |  | -0.00548 |
| P | 1.5159676 | 0.000824 | 9.97e-05 | 0.2932 | 0.2344 | 0.702 | 0.7363 | 0.7323 | 0.654 | 0.7128 | 0.7436 | P 0.904 |  | -0.0055 |
| W | 1.5174631 | -0.0006715 | 9.3e-05 | 1.502 | 1.3348 | 0.7328 | 0.7279 | 0.7702 | 0.6878 | 0.7417 | 0.7795 | F 2.75 |  | -0.00015 |
| F | 1.5203572 | -0.0035656 | 0.0001407 | 3.0181 | 3.286 | 0.5944 | 0.6871 | 0.6547 | 0.5847 | 0.6147 | 0.677 | F 3.516 |  | 0.03385 |
| T | 1.516971 | -0.0001794 | 9.05e-05 | 1.2406 | 1.4417 | 0.675 | 0.7206 | 0.7103 | 0.6344 | 0.7155 | 0.753 | P 1.85 |  | 0.0144 |

per class: level pp, then team slope

| arm | lvl TOV | lvl FGA_rim | lvl FGA_jump2 | lvl FGA_3 | lvl FT_trip_shooting | lvl FT_trip_bonus | slope TOV | slope FGA_rim | slope FGA_jump2 | slope FGA_3 | slope FT_trip_shooting | slope FT_trip_bonus |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R | -0.1352 | -0.1132 | 1.5354 | -1.1354 | -0.0653 | -0.0863 | 0.7054 | 0.7213 | 0.7576 | 0.7633 | 0.3324 | 0.7521 |
| C0 | -1.287 | 1.2942 | 0.8229 | -1.1161 | 0.3567 | -0.0707 | 0.6748 | 0.7393 | 0.7367 | 0.7625 | 0.3225 | 0.7608 |
| O | -0.1619 | -0.4006 | 0.2491 | 0.212 | 0.0808 | 0.0206 | 0.6992 | 0.7268 | 0.7132 | 0.7812 | 0.2846 | 0.7811 |
| P | -0.2229 | -0.2932 | 0.2356 | 0.1678 | 0.0989 | 0.0138 | 0.702 | 0.7129 | 0.7046 | 0.7838 | 0.2777 | 0.7974 |
| W | -0.1745 | -0.0582 | 1.502 | -1.1278 | -0.0508 | -0.0907 | 0.7328 | 0.7324 | 0.7647 | 0.7662 | 0.3332 | 0.7653 |
| F | -0.4918 | 1.9756 | 1.9985 | -3.0181 | -0.151 | -0.3133 | 0.5944 | 0.7395 | 0.7408 | 0.6908 | 0.2401 | 0.7462 |
| T | -0.8682 | 0.8138 | 1.0432 | -1.2406 | 0.2541 | -0.0023 | 0.675 | 0.7468 | 0.7346 | 0.7577 | 0.3019 | 0.7611 |

level by month (class TOV)

| arm | 1 | 2 | 3 | 4 | 11 | 12 |
|---|---|---|---|---|---|---|
| R | -0.2272 | -0.344 | -0.0471 | 0.2444 | 0.2232 | -0.2614 |
| C0 | -1.3681 | -1.4638 | -1.219 | -0.859 | -0.945 | -1.4357 |
| O | -0.1823 | -0.2741 | 0.0517 | 0.0935 | -0.0347 | -0.3287 |
| P | -0.2253 | -0.3127 | -0.0474 | 0.0153 | -0.102 | -0.4071 |
| W | -0.2675 | -0.379 | -0.1708 | -0.3261 | 0.228 | -0.2816 |
| F | -0.6207 | -0.9495 | -0.667 | -0.1478 | 0.1591 | -0.3854 |
| T | -0.9345 | -1.0988 | -0.8449 | -0.7299 | -0.4918 | -0.9689 |

anchor: n0 fitted = 10000.0 (fit seasons [2023, 2024]); prior-season level for test = {'2025': [0.15362, 0.256923, 0.193988, 0.284106, 0.060836, 0.050527]}; trend to test = {'2025': [0.147718, 0.264668, 0.189719, 0.282302, 0.0637, 0.051893]}; Lbar = [0.161998, 0.246698, 0.199334, 0.284449, 0.057577, 0.049945]

### po F1
| arm | primary | gain vs R | paired boot SE | level max|cls| | Nov-Dec max|cls| | team slope | slope Nov-Dec | sd_nc | sd_raw | slope/lvl (supp) | sd_nc/lvl (supp) | calib | per-game MAE pp | drift-stops (model) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R | 1.5227132 | 0.0 | 0.0 | 1.3219 | 1.4692 | 0.78 | 0.7476 | 0.7901 | 0.7116 | 0.7274 | 0.7368 | F 2.344 |  | 0.0 |
| C0 | 1.5221779 | 0.0005353 | 9.86e-05 | 0.8812 | 1.023 | 0.7622 | 0.7366 | 0.7807 | 0.7031 | 0.7229 | 0.7404 | P 1.946 |  | 0.0243 |
| O | 1.5216312 | 0.001082 | 0.0001245 | 0.2171 | 0.1329 | 0.7126 | 0.7186 | 0.734 | 0.661 | 0.7179 | 0.7395 | P 0.968 |  | -0.00548 |
| P | 1.521568 | 0.0011452 | 0.0001206 | 0.1243 | 0.2818 | 0.7245 | 0.6964 | 0.7401 | 0.6665 | 0.725 | 0.7406 | P 0.812 |  | -0.0055 |
| W | 1.5229917 | -0.0002785 | 0.0001069 | 1.3204 | 1.4737 | 0.7833 | 0.7455 | 0.7973 | 0.718 | 0.7293 | 0.7423 | F 2.482 |  | -0.00015 |
| F | 1.5267888 | -0.0040756 | 0.0001826 | 2.079 | 2.1758 | 0.702 | 0.679 | 0.7465 | 0.6723 | 0.7183 | 0.7638 | F 3.418 |  | 0.03385 |
| T | 1.5221601 | 0.0005531 | 9.37e-05 | 0.8178 | 0.9963 | 0.7659 | 0.7569 | 0.7737 | 0.6967 | 0.7278 | 0.7352 | P 1.877 |  | 0.0144 |

per class: level pp, then team slope

| arm | lvl TOV | lvl FGA_rim | lvl FGA_jump2 | lvl FGA_3 | lvl FT_trip_shooting | lvl FT_trip_bonus | slope TOV | slope FGA_rim | slope FGA_jump2 | slope FGA_3 | slope FT_trip_shooting | slope FT_trip_bonus |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R | 1.1255 | -1.3219 | 0.7576 | 0.0168 | -0.4071 | -0.1709 | 0.78 | 0.6152 | 0.695 | 0.7795 | 0.3358 | 0.8146 |
| C0 | 0.8483 | -0.8812 | 0.5365 | -0.2993 | -0.1893 | -0.0149 | 0.7622 | 0.6235 | 0.6916 | 0.7674 | 0.3117 | 0.8564 |
| O | -0.1018 | 0.2171 | 0.04 | -0.0656 | -0.0137 | -0.0761 | 0.7126 | 0.6522 | 0.6798 | 0.7725 | 0.3556 | 0.8379 |
| P | 0.0001 | 0.0569 | 0.1243 | -0.0737 | -0.04 | -0.0677 | 0.7245 | 0.6361 | 0.6826 | 0.7662 | 0.3539 | 0.8222 |
| W | 1.1552 | -1.3204 | 0.731 | 0.0299 | -0.4205 | -0.1752 | 0.7833 | 0.6126 | 0.6925 | 0.7827 | 0.3308 | 0.8164 |
| F | -0.3237 | -2.0261 | 2.079 | 1.1559 | -1.0054 | 0.1203 | 0.702 | 0.5819 | 0.6786 | 0.7872 | 0.3473 | 0.8845 |
| T | 0.8178 | -0.8096 | 0.488 | -0.3284 | -0.1647 | -0.0031 | 0.7659 | 0.6361 | 0.6917 | 0.7715 | 0.3636 | 0.8569 |

level by month (class TOV)

| arm | 1 | 2 | 3 | 4 | 11 | 12 |
|---|---|---|---|---|---|---|
| R | 1.1355 | 1.0847 | 0.6994 | 0.5855 | 1.3994 | 1.2294 |
| C0 | 0.8535 | 0.8138 | 0.4684 | -0.0708 | 1.1453 | 0.8877 |
| O | -0.136 | -0.1759 | -0.4067 | -0.8299 | 0.2774 | -0.1131 |
| P | -0.0634 | -0.1108 | -0.4176 | -0.9269 | 0.5341 | 0.0027 |
| W | 1.1818 | 1.0678 | 0.7629 | 0.5375 | 1.436 | 1.2654 |
| F | -0.3204 | -0.1918 | -0.38 | -0.4038 | -0.384 | -0.3925 |
| T | 0.8171 | 0.7711 | 0.4577 | -0.158 | 1.0918 | 0.8906 |

anchor: n0 fitted = 31622.776601683792 (fit seasons [2023]); prior-season level for test = {'2024': [0.165576, 0.242854, 0.201131, 0.283091, 0.056732, 0.050616]}; trend to test = {'2024': [0.162702, 0.247436, 0.198299, 0.279877, 0.058976, 0.05271]}; Lbar = [0.166972, 0.240628, 0.202507, 0.284652, 0.055642, 0.0496]

## Drift-stops / reverses cells (relative gap per arm; FLAT_OR_REV cells score E4)

| model | fold | sub | k | class | levels (train..., test) | train slope | test - last | R | C0 | O | P | W | F | T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rebound | F1 | ft | 0 | TREND | [0.120971, 0.121916, 0.129408] | 0.00094 | 0.00749 | -0.02263 | -0.0097 | 0.00876 | 0.00811 | -0.02544 | -0.0123 | 0.00141 |
| rebound | F1 | jump2 | 0 | TREND | [0.275641, 0.279333, 0.282802] | 0.00369 | 0.00347 | -0.02708 | -0.01588 | -0.0007 | -0.00273 | -0.02789 | -0.05765 | -0.00693 |
| rebound | F1 | rim | 0 | TREND | [0.374285, 0.376715, 0.381221] | 0.00243 | 0.00451 | -0.01582 | -0.00628 | 0.00777 | 0.00508 | -0.017 | -0.03218 | 0.00169 |
| rebound | F1 | three | 0 | TREND | [0.266983, 0.272949, 0.276897] | 0.00597 | 0.00395 | -0.02459 | -0.01226 | 0.00326 | 0.00058 | -0.02454 | -0.06031 | -0.00377 |
| rebound | F2 | ft | 0 | TREND | [0.120971, 0.121916, 0.129408, 0.137658] | 0.00422 | 0.00825 | -0.06418 | -0.03863 | -0.02041 | -0.02204 | -0.05283 | 0.05041 | -0.03003 |
| rebound | F2 | jump2 | 0 | TREND | [0.275641, 0.279333, 0.282802, 0.292715] | 0.00358 | 0.00991 | -0.04499 | -0.02509 | -0.00869 | -0.00967 | -0.041 | -0.01976 | -0.01703 |
| rebound | F2 | rim | 0 | TREND | [0.374285, 0.376715, 0.381221, 0.383149] | 0.00347 | 0.00193 | -0.01335 | 0.00404 | 0.01872 | 0.01799 | -0.01185 | -0.0008 | 0.01094 |
| rebound | F2 | three | 0 | TREND | [0.266983, 0.272949, 0.276897, 0.287741] | 0.00496 | 0.01084 | -0.05035 | -0.03006 | -0.01421 | -0.01545 | -0.04669 | -0.00682 | -0.02317 |
| shot_block | F1 | jump2 | 0 | FLAT_OR_REV | [0.085745, 0.082087, 0.083708] | -0.00366 | 0.00162 | 0.0005 | -0.00867 | 0.04219 | 0.02179 | -0.00943 | 0.01972 | -0.01784 |
| shot_block | F1 | rim | 0 | FLAT_OR_REV | [0.253139, 0.245834, 0.256898] | -0.00731 | 0.01106 | -0.02173 | -0.02908 | 0.01109 | -0.00495 | -0.02618 | -0.00654 | -0.0362 |
| shot_block | F1 | three | 0 | TREND | [0.013914, 0.013931, 0.01453] | 2e-05 | 0.0006 | -0.04095 | -0.05046 | 0.00215 | -0.01913 | -0.03935 | -0.02112 | -0.05992 |
| shot_block | F2 | jump2 | 0 | TREND | [0.085745, 0.082087, 0.083708, 0.081685] | -0.00102 | -0.00202 | 0.03533 | 0.08213 | 0.04631 | 0.04978 | 0.02901 | 0.03948 | 0.07975 |
| shot_block | F2 | rim | 0 | TREND | [0.253139, 0.245834, 0.256898, 0.262171] | 0.00188 | 0.00527 | -0.03178 | 0.0038 | -0.02425 | -0.02159 | -0.02879 | -0.02894 | 0.00084 |
| shot_block | F2 | three | 0 | FLAT_OR_REV | [0.013914, 0.013931, 0.01453, 0.014615] | 0.00031 | 8e-05 | -0.03132 | 0.01637 | -0.02063 | -0.01702 | -0.02129 | -0.02727 | 0.01343 |
| ft_tech | F1 | all | 0 | FLAT_OR_REV | [0.000719, 0.001102, 0.000708] | 0.00038 | -0.00039 | 0.31741 | 0.78396 | -0.07651 | -0.02853 | 0.39477 | 0.02528 | 1.13886 |
| ft_tech | F2 | all | 0 | TREND | [0.000719, 0.001102, 0.000708, 0.000644] | -1e-05 | -6e-05 | 0.32468 | 0.05132 | -0.1015 | -0.08513 | 0.27546 | 0.0113 | 0.30736 |
| po | F1 | all | 0 | TREND | [0.16845, 0.165576, 0.15362] | -0.00287 | -0.01196 | 0.07327 | 0.05522 | -0.00663 | 1e-05 | 0.0752 | -0.02107 | 0.05324 |
| po | F1 | all | 1 | TREND | [0.238271, 0.242854, 0.256923] | 0.00458 | 0.01407 | -0.05145 | -0.0343 | 0.00845 | 0.00222 | -0.05139 | -0.07886 | -0.03151 |
| po | F1 | all | 2 | TREND | [0.203963, 0.201131, 0.193988] | -0.00283 | -0.00714 | 0.03905 | 0.02765 | 0.00206 | 0.00641 | 0.03768 | 0.10717 | 0.02516 |
| po | F1 | all | 3 | FLAT_OR_REV | [0.286305, 0.283091, 0.284106] | -0.00321 | 0.00102 | 0.00059 | -0.01054 | -0.00231 | -0.00259 | 0.00105 | 0.04069 | -0.01156 |
| po | F1 | all | 4 | TREND | [0.054488, 0.056732, 0.060836] | 0.00224 | 0.0041 | -0.06692 | -0.03112 | -0.00226 | -0.00657 | -0.06913 | -0.16527 | -0.02707 |
| po | F1 | all | 5 | FLAT_OR_REV | [0.048523, 0.050616, 0.050527] | 0.00209 | -9e-05 | -0.03382 | -0.00295 | -0.01505 | -0.0134 | -0.03466 | 0.02382 | -0.00062 |
| po | F2 | all | 0 | FLAT_OR_REV | [0.16845, 0.165576, 0.15362, 0.155378] | -0.00742 | 0.00176 | -0.0087 | -0.08283 | -0.01042 | -0.01435 | -0.01123 | -0.03165 | -0.05588 |
| po | F2 | all | 1 | FLAT_OR_REV | [0.238271, 0.242854, 0.256923, 0.256974] | 0.00933 | 5e-05 | -0.00441 | 0.05036 | -0.01559 | -0.01141 | -0.00227 | 0.07688 | 0.03167 |
| po | F2 | all | 2 | TREND | [0.203963, 0.201131, 0.193988, 0.178874] | -0.00499 | -0.01511 | 0.08584 | 0.046 | 0.01392 | 0.01317 | 0.08397 | 0.11173 | 0.05832 |
| po | F2 | all | 3 | FLAT_OR_REV | [0.286305, 0.283091, 0.284106, 0.295707] | -0.0011 | 0.0116 | -0.03839 | -0.03774 | 0.00717 | 0.00568 | -0.03814 | -0.10206 | -0.04195 |
| po | F2 | all | 4 | FLAT_OR_REV | [0.054488, 0.056732, 0.060836, 0.061638] | 0.00317 | 0.0008 | -0.01059 | 0.05788 | 0.01311 | 0.01605 | -0.00823 | -0.02451 | 0.04122 |
| po | F2 | all | 5 | TREND | [0.048523, 0.050616, 0.050527, 0.05143] | 0.001 | 0.0009 | -0.01679 | -0.01375 | 0.00401 | 0.00268 | -0.01764 | -0.06091 | -0.00044 |

pooled drift-stops score (mean |rel gap arm| - |rel gap R| over FLAT_OR_REV cells):

| arm | pooled | n cells | per model |
|---|---|---|---|
| R | 0.0 | 10 | {"shot_block": 0.0, "ft_tech": 0.0, "po": 0.0} |
| C0 | 0.06129 | 10 | {"shot_block": 0.00019, "ft_tech": 0.46655, "po": 0.0243} |
| O | -0.02534 | 10 | {"shot_block": 0.00679, "ft_tech": -0.2409, "po": -0.00548} |
| P | -0.03317 | 10 | {"shot_block": -0.00326, "ft_tech": -0.28888, "po": -0.0055} |
| W | 0.00798 | 10 | {"shot_block": 0.00112, "ft_tech": 0.07736, "po": -0.00015} |
| F | -0.0089 | 10 | {"shot_block": -1e-05, "ft_tech": -0.29213, "po": 0.03385} |
| T | 0.09218 | 10 | {"shot_block": 0.00464, "ft_tech": 0.82145, "po": 0.0144} |

## Decision rule (section 1.7), applied mechanically

### rebound: selected **R** (eligible [])

| arm | F2 gain | floor | x floor | F1 gain | E1 | E2 | E3 | E4 | E5 | E6 | E7 | eligible | E5-sym (supp) | eligible if E5-sym (supp) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 0.0005899 | 0.000144 | 4.1 | 0.0002752 | PASS | PASS | PASS | fail | PASS | PASS | PASS | False | True | False |
| O | 0.0006514 | 0.0001625 | 4.01 | 0.0003344 | PASS | PASS | PASS | PASS | PASS | fail | PASS | False | True | False |
| P | 0.0006685 | 0.0001618 | 4.13 | 0.0002708 | PASS | PASS | PASS | PASS | PASS | fail | PASS | False | True | False |
| W | -0.0003127 | 0.0001822 | -1.72 | -0.0002294 | fail | PASS | PASS | fail | PASS | fail | fail | False | True | False |
| F | -0.0016727 | 0.0003967 | -4.22 | -0.002849 | fail | PASS | PASS | PASS | fail | PASS | fail | False | False | False |
| T | 0.0005412 | 0.0001581 | 3.42 | 0.00035 | PASS | PASS | PASS | fail | PASS | PASS | PASS | False | True | False |

### shot_block: selected **F** (eligible ['F'])

| arm | F2 gain | floor | x floor | F1 gain | E1 | E2 | E3 | E4 | E5 | E6 | E7 | eligible | E5-sym (supp) | eligible if E5-sym (supp) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | -1.15e-05 | 5.22e-05 | -0.22 | -2.2e-05 | fail | fail | PASS | fail | PASS | PASS | fail | False | True | False |
| O | 1.41e-05 | 1.53e-05 | 0.92 | -2.39e-05 | fail | PASS | PASS | PASS | PASS | PASS | fail | False | True | False |
| P | 1.72e-05 | 1.79e-05 | 0.96 | 8.8e-06 | fail | PASS | PASS | PASS | PASS | PASS | PASS | False | True | False |
| W | 1.83e-05 | 1.26e-05 | 1.45 | -2.36e-05 | PASS | PASS | PASS | fail | PASS | PASS | fail | False | True | False |
| F | 7.2e-06 | 5.8e-06 | 1.24 | 1.06e-05 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | True | True | True |
| T | -5.4e-06 | 4.86e-05 | -0.11 | -5.18e-05 | fail | fail | PASS | fail | PASS | PASS | fail | False | True | False |

### ft_tech: selected **R** (eligible [])

| arm | F2 gain | floor | x floor | F1 gain | E1 | E2 | E3 | E4 | E5 | E6 | E7 | eligible | E5-sym (supp) | eligible if E5-sym (supp) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 0.008638 | 0.0028588 | 3.02 | -0.0368688 | PASS | PASS | PASS | fail | fail | PASS | fail | False | True | False |
| O | 0.0069717 | 0.0049314 | 1.41 | 0.0064557 | PASS | PASS | PASS | PASS | fail | PASS | PASS | False | True | True |
| P | 0.0077881 | 0.0046083 | 1.69 | 0.0078263 | PASS | PASS | PASS | PASS | fail | PASS | PASS | False | True | True |
| W | 0.0024153 | 0.0005207 | 4.64 | -0.0046666 | PASS | PASS | PASS | fail | fail | PASS | fail | False | True | False |
| F | 0.0084033 | 0.003428 | 2.45 | 0.0078409 | PASS | PASS | PASS | PASS | fail | PASS | PASS | False | True | True |
| T | 0.0008455 | 0.0001627 | 5.2 | -0.076445 | PASS | PASS | PASS | fail | PASS | PASS | fail | False | True | False |

Votes: `{"F": ["shot_block"]}`; po control: `{}`; **winning design: None**
