# L5 CLOCK-CONSUMPTION -- experiments log (append-only)

Format per `docs/models/DOCUMENTATION_STANDARD.md`. Section 1 is the
pre-registration, written by the PM on 2026-09-10 and committed BEFORE any of
the code in `src/cbb_sim/models/clock.py` or `scripts/train_clock_v1.py`
existed. Nothing in section 1 is edited after the fact; results are appended
as new sections.

---

## 1. Pre-registration (PM-authored, 2026-09-10)

Purpose: in the possession engine, pace is emergent (Decision 7). At each possession start the engine draws the possession's duration in seconds conditioned on state, THEN draws the terminal event (L3) given the duration bucket and state. This bake-off chooses the duration model.
Target: duration_s of a possession (end_period possessions are the censoring case: duration = seconds remaining, treated as right-censored at the period boundary).
Universe: D-I, non-truncated, CBBD-complete games, seasons 2022-2025; folds F1 train {2022, 2023} test 2024; F2 train {2022, 2023, 2024} test 2025 (selection). 2026 sealed (seal.assert_not_sealed).
State features (all known at possession start in the sim): previous possession's end type (DREB, TOV/steal, made FG, made FT, dead ball/OREB-continuation, period start), period, seconds remaining in period, score diff from offense view, bonus flag, chance number within possession, the two teams' as-of tempo priors (multiplicative formula from pace.py) and own ratings, site, season index. Feature sets: A_state (previous end type, period, seconds remaining, chance number), B_plus_teams (A + tempo priors + ratings + site), C_plus_score (B + score diff and its interaction with seconds remaining, bonus), D_plus_season (C + season index and days since season start).
Model classes and families: (1) empirical resampling from state cells (previous end type x seconds-remaining bucket x tempo-prior tercile; document the cell grid and minimum cell size with fallback); (2) parametric regression on the log-duration with Gamma and log-normal families, heteroscedastic; (3) LightGBM quantile regression (9 quantiles, sampled by inverse CDF with linear interpolation); (4) discrete-time hazard (per-second exit probability, logistic on state), which handles censoring at period end natively. Censoring at period end must be handled honestly by every arm (state the method per arm).
Metrics on F2: CRPS of the predictive distribution (primary); log score where defined; K-S of PIT by state cell (previous end type x seconds-remaining bucket) with cells n < 300 marked UNDERPOWERED; mean and SD of duration by terminal event class as a diagnostic (the duration model does not see the terminal event, so this checks whether the state carries the information); and the EMERGENT test: for each F2 game, chain draws from the model over two 1200-second halves using the real sequence of previous-end types as the state input (a fair, event-model-free test), and compare the resulting possessions per game to the actual per game: mean, SD, by-month G1, and K-S of the per-game count distribution; also end-of-half behaviour: the share of halves whose last possession starts with < 35 s remaining and its mean duration, sim vs actual.
Noise floor: seed-varied refit of the tree arm; bootstrap SE (game blocks) for the others.
Decision rules: winner = lowest F2 CRPS among arms that pass the emergent G1 (mean +/- 1.0, SD +/- 0.75, all powered months) and have no LEAK-sized PIT failure (K-S D > 0.05) in any powered cell; a tree arm must beat the best non-tree arm by more than the floor; ties go to the simpler arm (empirical < parametric < hazard < tree). Feature set chosen within the winning class by the same rule. If no arm passes the emergent G1, report the failure and the diagnosis; adopt nothing.

---

## 2. Run R1 -- the full pre-registered grid (2026-09-10)

`scripts/train_clock_v1.py`, seed 20260910. Artifacts in `data/processed/models/clock/`. Every number below is written by that script; none is typed in by hand.

### 2.1 Universe and exclusions

| item | value |
|---|---|
| raw_rows | 3029695 |
| raw_games | 21969 |
| games_in_universe | 21969 |
| games_cbbd_complete | 18902 |
| games_dropped_cbbd_incomplete | 3067 |
| rows_over_duration_cap | 163 |
| rows_over_duration_cap_pct | 0.00625 |
| censored_rows | 5600 |
| censored_pct | 0.2148 |
| final_rows | 2607192 |
| final_games | 18902 |

Rows per season: {2022: 574973, 2023: 603112, 2024: 697903, 2025: 731204}.

`chance number within possession` is a pre-registered A_state feature that is identically 1 at a POSSESSION's start (a possession begins on its first chance), so it is a zero-variance column in every feature set and is dropped by every arm rather than silently ignored: ['chance_number_at_start']. See `model.md` section 9.

### 2.2 F2 (selection fold: train 2022-2024, test 2025)

| arm | features | CRPS | log score | undef % | PIT fails | powered cells | underpowered | worst K-S D | G1 d-mean | G1 d-SD | months pass | months powered | emergent G1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm_quantile | D_plus_season | 4.8082 | 3.2812 | 7.6353 | 6 | 27 | 0 | 0.1985 | 2.4280 | -0.6290 | 0.0000 | 5.0000 | FAIL |
| lgbm_quantile | C_plus_score | 4.8102 | 3.2838 | 7.5621 | 6 | 27 | 0 | 0.1612 | 2.1900 | -0.6070 | 0.0000 | 5.0000 | FAIL |
| lgbm_quantile | B_plus_teams | 4.8779 | 3.3021 | 7.4877 | 7 | 27 | 0 | 0.1910 | 2.6870 | -0.6020 | 0.0000 | 5.0000 | FAIL |
| lgbm_quantile | A_state | 4.9179 | 3.3133 | 7.0470 | 6 | 27 | 0 | 0.1396 | 2.8160 | -1.8640 | 0.0000 | 5.0000 | FAIL |
| empirical | C_plus_score | 4.9307 | 3.5106 | 0.3920 | 5 | 27 | 0 | 0.0826 | 1.1240 | -0.7570 | 1.0000 | 5.0000 | FAIL |
| empirical | D_plus_season | 4.9329 | 3.5062 | 0.7131 | 5 | 27 | 0 | 0.0826 | 1.3700 | -0.7860 | 0.0000 | 5.0000 | FAIL |
| empirical | B_plus_teams | 4.9574 | 3.5234 | 0.0751 | 5 | 27 | 0 | 0.0826 | 1.1030 | -0.8200 | 1.0000 | 5.0000 | FAIL |
| empirical | A_state | 4.9755 | 3.5283 | 0.0097 | 5 | 27 | 0 | 0.0807 | 1.2800 | -1.7190 | 0.0000 | 5.0000 | FAIL |
| hazard | C_plus_score | 5.0294 | 3.5778 | 0.0000 | 23 | 27 | 0 | 0.3813 | 1.8100 | -0.3590 | 0.0000 | 5.0000 | FAIL |
| hazard | D_plus_season | 5.0322 | 3.5780 | 0.0000 | 22 | 27 | 0 | 0.3760 | 2.5110 | -0.3310 | 0.0000 | 5.0000 | FAIL |
| gamma | C_plus_score | 5.0394 | 3.6044 | 0.0001 | 23 | 27 | 0 | 0.3928 | 0.7200 | -0.2800 | 4.0000 | 5.0000 | FAIL |
| gamma | D_plus_season | 5.0431 | 3.6045 | 0.0001 | 23 | 27 | 0 | 0.3881 | 1.3460 | -0.2560 | 1.0000 | 5.0000 | FAIL |
| hazard | B_plus_teams | 5.0495 | 3.5819 | 0.0000 | 22 | 27 | 0 | 0.3849 | 1.9270 | -0.3690 | 0.0000 | 5.0000 | FAIL |
| gamma | B_plus_teams | 5.0580 | 3.6092 | 0.0001 | 23 | 27 | 0 | 0.4036 | 0.8870 | -0.2920 | 3.0000 | 5.0000 | FAIL |
| hazard | A_state | 5.0856 | 3.5877 | 0.0000 | 22 | 27 | 0 | 0.3887 | 2.0440 | -1.3710 | 0.0000 | 5.0000 | FAIL |
| gamma | A_state | 5.0968 | 3.6153 | 0.0001 | 23 | 27 | 0 | 0.4010 | 1.0430 | -1.3240 | 0.0000 | 5.0000 | FAIL |
| lognormal | C_plus_score | 5.2002 | 3.7142 | 0.0000 | 27 | 27 | 0 | 0.3343 | -2.0140 | 0.1800 | 0.0000 | 5.0000 | FAIL |
| lognormal | D_plus_season | 5.2003 | 3.7143 | 0.0000 | 27 | 27 | 0 | 0.3303 | -1.3490 | 0.2080 | 1.0000 | 5.0000 | FAIL |
| lognormal | B_plus_teams | 5.2201 | 3.7202 | 0.0000 | 27 | 27 | 0 | 0.3459 | -1.9050 | 0.1580 | 0.0000 | 5.0000 | FAIL |
| lognormal | A_state | 5.2574 | 3.7258 | 0.0000 | 27 | 27 | 0 | 0.3441 | -1.7520 | -0.5810 | 0.0000 | 5.0000 | FAIL |

### 2.3 F1 (train 2022-2023, test 2024) -- robustness only

| arm | features | CRPS | log score | undef % | PIT fails | powered cells | worst K-S D |
|---|---|---|---|---|---|---|---|
| lgbm_quantile | D_plus_season | 4.7656 | 3.2849 | 7.4560 | 6 | 27 | 0.1831 |
| lgbm_quantile | C_plus_score | 4.7676 | 3.2848 | 7.4610 | 6 | 27 | 0.1979 |
| lgbm_quantile | B_plus_teams | 4.8376 | 3.3025 | 7.4096 | 7 | 27 | 0.1923 |
| lgbm_quantile | A_state | 4.8759 | 3.3161 | 6.7739 | 6 | 27 | 0.1772 |
| empirical | C_plus_score | 4.8891 | 3.5029 | 0.4825 | 2 | 27 | 0.0820 |
| empirical | D_plus_season | 4.8907 | 3.5007 | 0.7109 | 2 | 27 | 0.0820 |
| empirical | B_plus_teams | 4.9138 | 3.5148 | 0.0974 | 2 | 27 | 0.0820 |
| empirical | A_state | 4.9335 | 3.5204 | 0.0162 | 2 | 27 | 0.0820 |
| hazard | D_plus_season | 4.9814 | 3.5690 | 0.0000 | 23 | 27 | 0.3960 |
| hazard | C_plus_score | 4.9818 | 3.5689 | 0.0000 | 23 | 27 | 0.3940 |
| gamma | D_plus_season | 4.9953 | 3.5988 | 0.0000 | 22 | 27 | 0.4055 |
| gamma | C_plus_score | 4.9972 | 3.5988 | 0.0000 | 23 | 27 | 0.4029 |
| hazard | B_plus_teams | 5.0049 | 3.5736 | 0.0000 | 23 | 27 | 0.4111 |
| gamma | B_plus_teams | 5.0190 | 3.6042 | 0.0000 | 23 | 27 | 0.4134 |
| hazard | A_state | 5.0369 | 3.5789 | 0.0000 | 23 | 27 | 0.4150 |
| gamma | A_state | 5.0510 | 3.6100 | 0.0000 | 22 | 27 | 0.4143 |
| lognormal | D_plus_season | 5.1622 | 3.7109 | 0.0000 | 26 | 27 | 0.3472 |
| lognormal | C_plus_score | 5.1641 | 3.7111 | 0.0000 | 27 | 27 | 0.3448 |
| lognormal | B_plus_teams | 5.1866 | 3.7176 | 0.0000 | 27 | 27 | 0.3558 |
| lognormal | A_state | 5.2160 | 3.7227 | 0.0000 | 26 | 27 | 0.3567 |

### 2.4 Noise floor

Tree seed-varied refit (seed 20260910 vs 20261910), worst |dCRPS| across feature sets: **0.000295**. Worst game-block bootstrap SE of the mean CRPS across all other arms: **0.006764**. Floor used by the decision rule: **0.006764**.

### 2.5 Verdict

```json
{
  "floor": 0.006764,
  "n_arms": 20,
  "n_eligible": 0,
  "n_pass_g1": 0,
  "n_pit_clean": 0,
  "winner": null,
  "verdict": "NO ARM ADOPTED",
  "reason": "no (arm, feature set) combination passed the pre-registered emergent G1 and PIT gates on F2",
  "best_crps_overall": 4.808153,
  "best_crps_arm": "lgbm_quantile",
  "best_crps_feature_set": "D_plus_season",
  "created_at": "2026-09-10T16:56:06.407714+00:00",
  "finished_at": "2026-09-10T17:42:38.378754+00:00",
  "seed": 20260910,
  "quick": false,
  "selection_fold": "F2"
}
```


## 3. Run R2 -- diagnosis of the R1 failure (2026-09-10)

`scripts/diag_clock_v1.py`, seed 20260910, feature set `C_plus_score` for every arm (the bake-off's own refit, same seed, so these rows are comparable to section 2 row by row).

### 3.1 D1 -- the conditional mean of duration by seconds remaining

| band_seconds_remaining | n | actual_mean | pct_clock_ran_out | empirical_mean | lognormal_mean | gamma_mean | hazard_mean | lgbm_quantile_mean |
|---|---|---|---|---|---|---|---|---|
| 0-9 | 4186 | 3.071 | 50.45 | 8.766 | 16.384 | 15.51 | 9.707 | 3.206 |
| 10-19 | 5288 | 7.371 | 19.03 | 9.199 | 16.977 | 16.076 | 11.243 | 7.265 |
| 20-24 | 2936 | 10.748 | 14.93 | 9.406 | 17.064 | 16.172 | 12.871 | 10.426 |
| 25-29 | 3136 | 13.081 | 14.43 | 9.492 | 17.063 | 16.179 | 13.68 | 12.79 |
| 30-34 | 3546 | 15.637 | 8.33 | 9.545 | 17.166 | 16.307 | 14.312 | 14.859 |
| 35-39 | 3804 | 15.55 | 3.46 | 15.545 | 17.168 | 16.316 | 14.798 | 15.317 |
| 40-44 | 3744 | 14.881 | 1.25 | 15.819 | 17.337 | 16.49 | 15.448 | 14.9 |
| 45-49 | 3702 | 14.249 | 0.49 | 15.823 | 17.424 | 16.58 | 15.984 | 14.342 |
| 50-59 | 6791 | 14.93 | 0.21 | 15.874 | 17.38 | 16.536 | 16.478 | 14.715 |
| 60-89 | 18879 | 16.615 | 0.04 | 16.056 | 17.481 | 16.644 | 17.07 | 16.321 |
| 90-1200 | 667164 | 17.883 | 0.0 | 17.877 | 18.394 | 17.656 | 17.488 | 17.691 |

Read: the actual conditional mean bends sharply inside the last 45 seconds of a period (17.9 s with more than 90 s left, down to about 3 s with fewer than 10). Every arm smooths that bend -- the empirical arm because its finest pre-registered seconds-remaining bucket is 0-34, the parametric and hazard arms because `seconds_remaining` enters them linearly. The emergent possession count is 2400 divided by the mean duration REALISED over the chain, so a conditional mean that is only wrong in the last 45 seconds of each half still moves the count, and it moves the end-of-half check directly.

### 3.2 D2 -- where the per-game count variance goes

```json
{
  "actual_mean": 68.1332,
  "actual_sd": 5.0882,
  "n_games": 5319,
  "arms": {
    "empirical": {
      "sim_mean": 69.2569,
      "sim_sd": 4.331,
      "sd_delta": -0.7571,
      "feature_implied_count_sd": 3.1158,
      "corr_sim_actual": 0.3901
    },
    "lognormal": {
      "sim_mean": 66.1197,
      "sim_sd": 5.2678,
      "sd_delta": 0.1796,
      "feature_implied_count_sd": 3.0665,
      "corr_sim_actual": 0.3583
    },
    "gamma": {
      "sim_mean": 68.8534,
      "sim_sd": 4.8086,
      "sd_delta": -0.2796,
      "feature_implied_count_sd": 3.3383,
      "corr_sim_actual": 0.4257
    },
    "hazard": {
      "sim_mean": 69.9428,
      "sim_sd": 4.7296,
      "sd_delta": -0.3586,
      "feature_implied_count_sd": 3.1839,
      "corr_sim_actual": 0.4109
    },
    "lgbm_quantile": {
      "sim_mean": 70.3232,
      "sim_sd": 4.4816,
      "sd_delta": -0.6066,
      "feature_implied_count_sd": 3.5875,
      "corr_sim_actual": 0.5189
    }
  },
  "iid_renewal_count_sd_reference": 3.2645,
  "note": "iid_renewal_count_sd_reference is sqrt(N) * sd(D) / mean(D) / 2 for N = 2400/mean(D) draws per game, i.e. the per-team count SD produced by identically-distributed possessions with no game-to-game pace variation at all."
}
```

Read: `iid_renewal_count_sd_reference` is the per-team count SD you get from identically-distributed possessions with NO game-to-game pace variation. `feature_implied_count_sd` is the extra between-game spread this model's own features carry. The two together are what any state-conditional i.i.d. duration model can produce, and the gap to `actual_sd` is the part of real pace dispersion that is neither in the features nor in possession-level noise -- the same shape as L10 one layer down: independent draws understate dispersion.

### 3.3 D3 -- responsiveness (CLAUDE.md standing rule)

| arm | tempo_quintile | n | sim_mean | actual_mean | delta | span_sim | span_actual | slope_ratio | steps_agreeing |
|---|---|---|---|---|---|---|---|---|---|
| empirical | 1 | 1064 | 66.183 | 63.969 | 2.214 | 6.528 | 8.291 | 0.7874 | 4 |
| empirical | 2 | 1064 | 66.738 | 66.569 | 0.17 | 6.528 | 8.291 | 0.7874 | 4 |
| empirical | 3 | 1063 | 69.632 | 68.066 | 1.566 | 6.528 | 8.291 | 0.7874 | 4 |
| empirical | 4 | 1064 | 71.02 | 69.803 | 1.217 | 6.528 | 8.291 | 0.7874 | 4 |
| empirical | 5 | 1064 | 72.711 | 72.26 | 0.452 | 6.528 | 8.291 | 0.7874 | 4 |
| lognormal | 1 | 1064 | 62.335 | 63.969 | -1.634 | 7.661 | 8.291 | 0.924 | 4 |
| lognormal | 2 | 1064 | 64.706 | 66.569 | -1.862 | 7.661 | 8.291 | 0.924 | 4 |
| lognormal | 3 | 1063 | 66.056 | 68.066 | -2.01 | 7.661 | 8.291 | 0.924 | 4 |
| lognormal | 4 | 1064 | 67.505 | 69.803 | -2.298 | 7.661 | 8.291 | 0.924 | 4 |
| lognormal | 5 | 1064 | 69.996 | 72.26 | -2.264 | 7.661 | 8.291 | 0.924 | 4 |
| gamma | 1 | 1064 | 64.727 | 63.969 | 0.758 | 8.373 | 8.291 | 1.0099 | 4 |
| gamma | 2 | 1064 | 67.27 | 66.569 | 0.702 | 8.373 | 8.291 | 1.0099 | 4 |
| gamma | 3 | 1063 | 68.817 | 68.066 | 0.752 | 8.373 | 8.291 | 1.0099 | 4 |
| gamma | 4 | 1064 | 70.352 | 69.803 | 0.55 | 8.373 | 8.291 | 1.0099 | 4 |
| gamma | 5 | 1064 | 73.1 | 72.26 | 0.84 | 8.373 | 8.291 | 1.0099 | 4 |
| hazard | 1 | 1064 | 65.953 | 63.969 | 1.984 | 8.091 | 8.291 | 0.9759 | 4 |
| hazard | 2 | 1064 | 68.451 | 66.569 | 1.882 | 8.091 | 8.291 | 0.9759 | 4 |
| hazard | 3 | 1063 | 69.913 | 68.066 | 1.847 | 8.091 | 8.291 | 0.9759 | 4 |
| hazard | 4 | 1064 | 71.354 | 69.803 | 1.552 | 8.091 | 8.291 | 0.9759 | 4 |
| hazard | 5 | 1064 | 74.044 | 72.26 | 1.784 | 8.091 | 8.291 | 0.9759 | 4 |
| lgbm_quantile | 1 | 1064 | 66.016 | 63.969 | 2.047 | 8.787 | 8.291 | 1.0598 | 4 |
| lgbm_quantile | 2 | 1064 | 68.564 | 66.569 | 1.996 | 8.787 | 8.291 | 1.0598 | 4 |
| lgbm_quantile | 3 | 1063 | 70.341 | 68.066 | 2.276 | 8.787 | 8.291 | 1.0598 | 4 |
| lgbm_quantile | 4 | 1064 | 71.891 | 69.803 | 2.089 | 8.787 | 8.291 | 1.0598 | 4 |
| lgbm_quantile | 5 | 1064 | 74.803 | 72.26 | 2.543 | 8.787 | 8.291 | 1.0598 | 4 |

Read: the emergent count must SLOPE with the pregame tempo prior, not sit flat at the league mean. `slope_ratio` near 1 with 4 of 4 agreeing steps says the model is matchup-specific; a ratio near 0 would say it is a league average wearing a feature vector.


---

## 4. Reading of R1 + R2 (2026-09-10)

Written after the numbers, appended, and never edited back into section 1.
The full interpretation, with every table restated, is
`docs/models/clock/model.md` sections 4 and 5; this is the short form for the
experiments log.

**Verdict: NO ARM ADOPTED.** `n_eligible = 0` of 20. The decision rule
eliminated every arm before CRPS was consulted, and `winner.pkl` was not
written.

1. **Emergent G1: 0 of 20 pass.** `gamma`/`C_plus_score` (+0.720 mean,
   -0.280 SD) and `gamma`/`B_plus_teams` (+0.887, -0.292) pass the overall
   reading and then fail the "all powered months" clause at 4/5 and 3/5.
   January is the month that breaks both. Every other arm misses the overall
   mean: `empirical` +1.10..+1.37, `hazard` +1.81..+2.51, `lgbm_quantile`
   +2.19..+2.82, `lognormal` -1.35..-2.01.
2. **PIT: 0 of 20 clean.** `empirical` 5 leak-sized cells of 27 powered (worst
   D 0.083, and all five are the smallest powered cells); `lgbm_quantile` 6-7
   (worst 0.198); `hazard` 22-23 and `gamma` 23 (worst ~0.40); `lognormal`
   26-27 (worst 0.35). The parametric and hazard failures reproduce IN-SAMPLE
   (pooled PIT D 0.056 for `gamma` on its own training rows against 0.0033 for
   `empirical`), so they are a distribution-family failure, not overfitting and
   not drift.
3. **CRPS ordering:** `lgbm_quantile` (4.808-4.918) < `empirical`
   (4.931-4.976) < `hazard` (5.029-5.086) ~ `gamma` (5.039-5.097) <
   `lognormal` (5.200-5.257). The tree beats the best non-tree arm by 0.1225,
   18x the 0.006764 floor -- it would have won outright had the gates passed.
   Within `lgbm_quantile`, `D_plus_season` (4.8082) and `C_plus_score` (4.8102)
   are 0.0021 apart, inside the floor, so the simpler feature set would take the
   tie-break. The same ordering holds on F1, so it is not a fold artifact.
4. **The tree arm's CRPS win is bought with a truncated tail.** 7.0-7.6% of
   test rows get ZERO predictive mass on the duration that occurred (every
   other arm is under 0.72%), because nine quantiles inverted with linear
   interpolation cannot reach past roughly q(0.9) + one segment. That costs
   0.19-0.26 s of mean duration and is most of its +2.2 to +2.8 possession
   overshoot.
5. **End-of-half check: 0 of 20 pass on shape.** Actual: 88.32% of halves start
   their last possession with under 35 s left, and it lasts 12.06 s. Sim:
   88.9% / 11.79 s (`lognormal`), 95.1% / 11.65 s (`gamma`), 98.3% / 9.50 s
   (`empirical`), 99.2% / 7.98 s (`hazard`), 99.7% / 5.83 s (`lgbm_quantile`).
6. **Responsiveness passes on every arm** (4/4 monotone quintile steps, slope
   ratios 0.79-1.06), and the count SD is nearly reachable (`gamma` 4.81 vs an
   actual 5.09 and an i.i.d.-plus-features ceiling around 4.7-4.9). The failure
   is a conditional-SHAPE failure near a period boundary, not a matchup-signal
   failure and not primarily a dispersion failure.

Blocking defect, in one sentence: **the conditional mean of possession duration
falls from 17.9 s to 3.1 s across the last 90 seconds of a period and only the
tree arm bends with it** (section 3.1), so every arm either squeezes an extra
possession into the end of each half or, in `lognormal`'s case, over-corrects
and loses two.

Followups are enumerated in `model.md` section 9. None of them is applied here:
no arm is adopted, and no multiplier, cap or calibration curve is fitted to
close the 1.6% gap in mean duration, per `docs/SIM_GUARDRAILS.md`.

---

## 5. Round 2 pre-registration (PM-authored, 2026-09-10)

Appended verbatim BEFORE any round-2 code was written or run. Section 1 (the
round-1 pre-registration) is untouched.

Round 1 diagnosis: every arm smooths the sharp bend in conditional duration inside the last 45 seconds of a period because seconds_remaining enters linearly or in a 0-34 s bucket. Round 2 changes the STATE REPRESENTATION of the period end and adds a two-regime arm; target, universe, folds, metrics, noise floor and decision rules are unchanged from round 1.
State changes (all arms): seconds_remaining enters as fine buckets {0-5, 5-10, 10-20, 20-30, 30-45, 45-60, 60-90, 90-150, 150-300, 300+}, crossed with period type {first half, second half/OT} and with offense trailing/tied/leading, so end-of-first-half and end-of-game behave differently and the late-trailing team's short possessions are representable; plus `last_shot_window` = seconds_remaining <= 30 (one shot clock) and `two_for_one_window` = 30 < seconds_remaining <= 45. Score diff x seconds_remaining interaction retained. The parametric and hazard arms take these as dummies/interactions; the tree arm takes the raw seconds_remaining plus the bucket id and is allowed a tree-parameter search (num_leaves, min_data_in_leaf) on F1 only, recorded.
Arms: (1) empirical resampling on the fine cell grid (prev end type x fine bucket x period type x trailing/leading x tempo tercile) with a documented minimum cell size and hierarchical fallback; (2) log-normal and Gamma heteroscedastic regressions with the new dummies; (3) discrete-time hazard with the new dummies; (4) LightGBM quantile with the new features; (5) NEW two-regime arm: the round-1 best-CRPS model (lgbm quantile, D_plus_season) for seconds_remaining > T and the fine empirical table for seconds_remaining <= T, with T chosen on F1 only from {30, 45, 60, 90} by emergent G1 + end-of-half error, recorded.
Gates: emergent G1 (mean +/- 1.0, SD +/- 0.75, all powered months), PIT cell K-S D <= 0.05 in all powered cells, and the end-of-half check (share of halves whose last possession starts with < 35 s and its mean duration, sim within the F1-derived noise floor of actual) is now a GATE, not a diagnostic. Responsiveness (emergent count slope vs tempo prior) reported. Decision rules as round 1; arm 5 counts as "tree" for the tree-must-beat-linear rule; if no arm passes, adopt nothing and report.

---

## 6. Run R3 -- the round-2 grid (2026-09-10)

`scripts/train_clock_v2.py`, seed 20260910. Artifacts are `v2_*` in `data/processed/models/clock/`; round 1's files are untouched. Every number is written by that script.

### 6.1 Choices made on F1 only

Tree-parameter search (allowed on F1 only by the pre-registration):

| num_leaves | min_child_samples | f1_crps | fit_seconds |
|---|---|---|---|
| 31.0000 | 1500.0000 | 4.7685 | 90.6000 |
| 63.0000 | 500.0000 | 4.7661 | 227.2000 |
| 127.0000 | 200.0000 | 4.7680 | 322.2000 |

Chosen: `{'num_leaves': 63, 'min_child_samples': 500}`.

Two-regime threshold search (F1 only). Score = |G1 mean delta| / 1.0 + |G1 SD delta| / 0.75 + |end-of-half share gap| / share floor + |end-of-half duration gap| / duration floor, i.e. how many tolerances of error, summed; the formula is stated here because the pre-registration says only "by emergent G1 + end-of-half error".

| T | f1_mean_delta | f1_sd_delta | f1_eoh_share_gap | f1_eoh_duration_gap | score |
|---|---|---|---|---|---|
| 30.0000 | 2.0000 | -0.5830 | 0.1186 | -5.6313 | 50.4845 |
| 45.0000 | 2.0230 | -0.5800 | 0.1112 | -5.4120 | 48.2006 |
| 60.0000 | 2.0740 | -0.5720 | 0.1093 | -5.5536 | 48.6717 |
| 90.0000 | 2.0820 | -0.5920 | 0.1108 | -5.4932 | 48.6305 |

**Chosen T = 45.**

End-of-half noise floor, derived on F1: the larger of the seed-to-seed SD of the SIM statistic (spec-identical re-chains at 5 seeds) and the game-block bootstrap SE of the ACTUAL statistic, times k = 2.0 (the multiplier is stated, not assumed). Sim seed SD: share [0.001336, 0.00244], duration [0.06324, 0.01688]. Actual block-bootstrap SE: share 0.003169, duration 0.09712. **Floor: share +/- 0.006338, duration +/- 0.19424 s.**

### 6.2 F2 (selection fold) -- the three gates

| arm | CRPS | log score | undef % | PIT fails | powered | worst D | G1 d-mean | G1 d-SD | months pass | G1 | EOH d-share | EOH d-dur | EOH |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm_quantile | 4.8091 | 3.2821 | 7.5467 | 6 | 27 | 0.1790 | 2.5250 | -0.5450 | 0.0000 | FAIL | 0.1137 | -6.4541 | FAIL |
| two_regime | 4.8192 | 3.2869 | 7.3983 | 6 | 27 | 0.1292 | 2.6500 | -0.5560 | 0.0000 | FAIL | 0.1073 | -5.2973 | FAIL |
| empirical | 4.8698 | 3.4973 | 0.1298 | 5 | 27 | 0.0999 | 1.6690 | -0.7800 | 0.0000 | FAIL | 0.1064 | -5.4340 | FAIL |
| gamma | 4.9035 | 3.5748 | 0.0000 | 19 | 27 | 0.2484 | 1.5310 | -0.1940 | 0.0000 | FAIL | 0.0675 | -3.9203 | FAIL |
| hazard | 4.9505 | 3.5575 | 0.0000 | 22 | 27 | 0.3553 | 2.5980 | -0.2480 | 0.0000 | FAIL | 0.0806 | -4.6460 | FAIL |
| lognormal | 5.0590 | 3.6846 | 0.0000 | 27 | 27 | 0.1961 | -1.3340 | 0.2590 | 1.0000 | FAIL | 0.0134 | -2.8268 | FAIL |

### 6.3 The end-of-half statistic is contaminated by feed truncation (measured, not assumed)

The CBBD event stream stops before the horn in a large minority of halves: only 44.6% of 2025 halves have a last logged possession ending at exactly 0:00, the median unaccounted time is 1 s, the 90th percentile 17 s and the mean 5.29 s. The 11.7% of halves whose last LOGGED possession starts with 35+ seconds left are mostly these -- their mean end clock is 20.4 s, and only 16.3% of them reach 0:00. A sim that always runs its clock to zero cannot reproduce a half that simply stops at 0:20, so the pre-registered actual value of 0.8832 is not a quantity any correct model can match.

Restricted to CLOCK-COMPLETE halves (last possession ending within 2 s of the horn, 63.5% of 2025 halves) the actual share is **0.9556** and the actual mean duration **12.41 s**. Both readings are reported below; the pre-registered all-halves number is the one the GATE uses, because a gate is not re-based after the fact.

| arm | actual share | sim share | d-share | actual dur | sim dur | d-dur | CC actual share | CC sim share | CC d-share | CC actual dur | CC sim dur | CC d-dur |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm_quantile | 0.8832 | 0.9969 | 0.1137 | 12.0630 | 5.6090 | -6.4541 | 0.9556 | 0.9957 | 0.0401 | 12.4060 | 6.1130 | -6.2937 |
| two_regime | 0.8832 | 0.9905 | 0.1073 | 12.0630 | 6.7660 | -5.2973 | 0.9556 | 0.9887 | 0.0332 | 12.4060 | 7.2040 | -5.2027 |
| empirical | 0.8832 | 0.9897 | 0.1064 | 12.0630 | 6.6290 | -5.4340 | 0.9556 | 0.9884 | 0.0329 | 12.4060 | 7.1260 | -5.2798 |
| gamma | 0.8832 | 0.9507 | 0.0675 | 12.0630 | 8.1430 | -3.9203 | 0.9556 | 0.9465 | -0.0090 | 12.4060 | 8.8530 | -3.5529 |
| hazard | 0.8832 | 0.9639 | 0.0806 | 12.0630 | 7.4170 | -4.6460 | 0.9556 | 0.9588 | 0.0033 | 12.4060 | 8.0590 | -4.3474 |
| lognormal | 0.8832 | 0.8967 | 0.0134 | 12.0630 | 9.2360 | -2.8268 | 0.9556 | 0.8885 | -0.0671 | 12.4060 | 9.8800 | -2.5260 |

`CC` = the clock-complete sub-universe. The same secondary read on the emergent count (clock-complete GAMES, both halves) is in the `cc_mean_delta` / `cc_sd_delta` columns of `v2_grid_results.csv`.

### 6.4 F1 (robustness only)

| arm | features | CRPS | log score | undef % | PIT fails | powered | worst D |
|---|---|---|---|---|---|---|---|
| lgbm_quantile | R2_tree | 4.7661 | 3.2838 | 7.4540 | 6 | 27 | 0.2187 |
| two_regime | R2_two_regime(T=45) | 4.7769 | 3.2901 | 7.2366 | 6 | 27 | 0.1658 |
| empirical | R2_dummy | 4.8279 | 3.4902 | 0.1624 | 5 | 27 | 0.1260 |
| gamma | R2_dummy | 4.8657 | 3.5703 | 0.0000 | 18 | 27 | 0.2927 |
| hazard | R2_dummy | 4.9048 | 3.5489 | 0.0000 | 22 | 27 | 0.3946 |
| lognormal | R2_dummy | 5.0276 | 3.6828 | 0.0000 | 27 | 27 | 0.2268 |

### 6.5 Noise floor and verdict

Tree seed-varied refit |dCRPS| = 2.7e-05. Worst game-block bootstrap SE = 0.006753. Floor used = **0.006753**.

```json
{
  "floor": 0.006753,
  "n_arms": 6,
  "n_eligible": 0,
  "n_pass_g1": 0,
  "n_pit_clean": 0,
  "n_pass_eoh": 0,
  "winner": null,
  "verdict": "NO ARM ADOPTED",
  "reason": "no arm passed all three pre-registered round-2 gates on F2 (emergent G1, PIT by state cell, end-of-half)",
  "best_crps_arm": "lgbm_quantile",
  "best_crps": 4.809089,
  "created_at": "2026-09-10T19:09:51.547386+00:00",
  "finished_at": "2026-09-10T20:04:51.539998+00:00",
  "seed": 20260910,
  "round": 2,
  "tree_params": {
    "num_leaves": 63,
    "min_child_samples": 500
  },
  "threshold_T": 45,
  "end_of_half_floor": {
    "share_se": 0.003169,
    "duration_se": 0.09712,
    "share": 0.006338,
    "duration": 0.19424,
    "k": 2.0,
    "sim_seed_sd_share": [
      0.001336,
      0.00244
    ],
    "sim_seed_sd_duration": [
      0.06324,
      0.01688
    ],
    "actual_block_bootstrap_se_share": 0.003169,
    "actual_block_bootstrap_se_duration": 0.09712,
    "seeds": [
      20260910,
      20260911,
      20260912,
      20260913,
      20260914
    ]
  }
}
```


---

## 7. Reading of the round-2 grid (2026-09-10)

Appended after the numbers. Interpretation in full: `model.md` section 10.

**Verdict: NO ARM ADOPTED.** 0 of 6 arms pass the emergent G1, 0 are PIT-clean,
0 pass the end-of-half gate. `v2_winner.pkl` was not written.

1. **The finer state representation works on the metric and BACKFIRES on the
   gate.** Round 2 improves F2 CRPS for every non-tree arm -- `empirical`
   4.9307 -> 4.8698, `gamma` 5.0394 -> 4.9035, `hazard` 5.0294 -> 4.9505,
   `lognormal` 5.2002 -> 5.0590, all 9-45x the 0.006753 floor -- and leaves the
   tree arm unmoved (4.8082 -> 4.8091, inside the floor). It also makes the
   EMERGENT COUNT WORSE on four of the five round-1 arms: `empirical` +1.124 ->
   +1.669, `gamma` +0.720 -> +1.531, `hazard` +1.810 -> +2.598,
   `lgbm_quantile` +2.190 -> +2.525; only `lognormal` improves (-2.014 ->
   -1.334). No arm passes a single powered month except `lognormal` in one.
2. **Why it backfires is the round-3 finding.** A model that knows the
   conditional law of duration given "7 seconds left" draws a SHORT duration
   there, because that law is short -- it is truncated by the horn in the
   training data. The sim then has 4 seconds left, draws shorter again, and
   subdivides the tail of the half into possessions the game never played. The
   conditional law has a large point mass AT the remaining clock (50.5% of
   possessions with under 10 s left consume all of it), and pooling even a
   5-second bucket destroys that mass. Round 1 hid this behind a coarse bucket;
   round 2 made the conditional mean right and the COUNT worse.
3. **The end-of-half gate is unpassable as pre-registered, for a data reason.**
   Section 6.3: the actual statistic is measured on the last LOGGED possession,
   and the CBBD stream stops before the horn in a third of halves. On the
   clock-complete sub-universe the actual share is 0.9556, and `gamma` hits
   0.9465 (gap **-0.009**) and `hazard` 0.9588 (gap **+0.003**) -- inside the
   0.00634 floor. So round 2 DID fix the end-of-half SHARE; the pre-registered
   number says otherwise only because it is contaminated.
4. **The end-of-half DURATION gap survives the correction and is the real
   defect**: -2.53 (`lognormal`) to -6.29 (`lgbm_quantile`) seconds against a
   0.194 s floor, on the clock-complete halves. Same mechanism as item 2.
5. **The tree-parameter search found nothing** (F1 CRPS 4.76850 / 4.76611 /
   4.76803 across the complexity ladder, a 0.0024 spread inside the floor), so
   round 1's fixed hyperparameters were not what held the tree arm back. The
   F1-chosen pair is round 1's own default.
6. **The two-regime arm is not a fix.** T is flat across {30, 45, 60, 90} on F1
   (score 48.2-50.5, selected T = 45), its F2 CRPS 4.8192 is 0.010 WORSE than
   the plain tree, and its emergent mean +2.650 is the worst in the grid. Its
   behaviour is dominated by the upper regime, which is the round-1 model.
7. **Responsiveness still passes on every arm** (4/4 monotone quintile steps,
   slope ratios 0.78-1.05), and the count SD is now inside tolerance on three
   arms (`gamma` -0.194, `hazard` -0.248, `lognormal` +0.259). Dispersion and
   matchup signal are not the problem.
8. **PIT: the parametric arms improved but remain refuted.** `gamma` 23 -> 19
   leak-sized cells, `hazard` 23 -> 22, `lognormal` 27 -> 27. `empirical` holds
   at 5 and `lgbm_quantile` at 6. The round-1 conclusion stands: a unimodal
   continuous family cannot represent this law, and the state representation was
   never the reason.

What round 3 must change, in one sentence: **model the duration the offence
INTENDED and let the engine truncate it at the horn** -- i.e. treat every
possession whose terminal event coincides with 0:00 as right-censored (about 50%
of possessions with under 10 s left, not the 0.2% currently flagged
`end_period`), so the conditional law stops being a law that always fits inside
the remaining clock. Nothing is adopted and no correction is applied.

---

## 8. Round 3 pre-registration (2026-09-10)

Appended VERBATIM BEFORE any round-3 code was written or run. Sections 1 and 5
(the round-1 and round-2 pre-registrations) are untouched. Written after the
descriptive censoring audit `docs/tests/clock_censoring_audit_2026-09-10.md`
(which fits no model and scores no arm) and before any arm exists.

Round-2 diagnosis, carried forward as a decided direction and not reopened
(`docs/LEARNINGS.md` L20): horn-ending possessions are TRUNCATED, not censored,
in the training data. Round 3 models the duration the offence INTENDED, flags
every horn-ending possession right-censored, and lets the engine truncate at the
horn. The audit measures the size of the defect: the flag rounds 1-2 used
(`terminal_event == "end_period"`) catches 0.2148% of rows; the correct flag
(`end_clock <= 0`, the possession consumed every second that was left) catches
0.6751%, 3.1x as many, and 61.1% / 36.4% of possessions starting with 0-5 s /
5-10 s left are in it.

### 8.1 Target, universe, folds

Target: the INTENDED duration T of a possession. Observation mechanism:
D = min(T, R) with R the seconds remaining at the possession's start; the row is
right-censored at D iff `end_clock <= 0`. The censoring flag comes from
`data/processed/clock_censoring/censoring_v1_{season}.parquet`, a SIDE TABLE
keyed on (game_id, period, poss_index) written by the audit script; neither
`data/processed/possessions/` nor `data/processed/possessions_v2/` is rewritten.
(`duration_s`, `start_clock` and `end_clock` are byte-identical between the two
possession layers, checked on 2025: 0 of 768,834 rows differ.)

Universe, folds, seal, `DURATION_CAP = 90` and the zero-variance drop rule are
UNCHANGED from rounds 1 and 2: D-I, hoopR feed not truncated, CBBD
points-complete, seasons 2022-2025; F1 trains {2022, 2023} and tests 2024, F2
trains {2022, 2023, 2024} and tests 2025 and is the SELECTION fold; 2026 sealed.

Clock-completeness, used only to define the gate universe, is the audit's
three-part definition: a period is clock-complete when its last logged
possession ends within 2 s of the horn (C1), its first starts within 2 s of the
period length (C2), and its summed durations are within 4 s of the period length
(C3). A GAME is clock-complete when both regulation halves are. The flag is
published on `data/processed/games_universe_v2.parquet`
(`clock_complete_reg`, `clock_complete_all_periods`, `points_complete`), a
VERSIONED SIBLING; `games_universe.parquet` is not rewritten, so the engine
worker's reader is untouched.

### 8.2 Arms

State representation is round 2's, unchanged, for every arm (`R2_dummy` for the
linear/cell arms, `R2_tree` for the tree arms), so round 3 differs from round 2
in the censoring treatment and nothing else. Nine arms:

| id | arm | censoring flag | what it is |
|---|---|---|---|
| A1 | `empirical_km3` | horn | round-2 fine cell grid, Kaplan-Meier, corrected flag AND the tail rule in 8.3 |
| A2 | `empirical_km3_srfloor` | horn | A1 with the seconds-remaining dimension FLOORED at the 45-60 s bucket: every row with fewer than 45 s left is served the 45-60 s cell's intended-duration law. The strongest form of L20 (all end-of-period effect is truncation, none is behaviour) and a falsification arm |
| A3 | `gamma_aft` | horn | heteroscedastic Gamma, censored MLE (unchanged code, corrected flag) |
| A4 | `lognormal_aft` | horn | heteroscedastic log-normal, censored MLE |
| A5 | `hazard3` | horn | discrete-time logistic hazard, censoring native |
| A6 | `xgb_aft` | horn | XGBoost `survival:aft` -- a censoring-aware TREE loss. Error distribution chosen ON F1 ONLY from {normal, logistic, extreme}, recorded. Predictive pmf from the fitted location and scale |
| B1 | `lgbm_quantile_r2` | old (excluded) | THE ROUND-2 REFERENCE, spec unchanged (censored rows excluded from training), refit inside the round-3 harness so the same code scores it on the same metrics |
| B2 | `empirical_r2` | old | round-2 empirical verbatim (old flag, old KM tail handling). PAIRED CONTROL: A1 vs B2 differ only in the flag and the tail rule |
| B3 | `gamma_r2` | old | round-2 gamma verbatim. Second paired control |

No hyperparameter search beyond A6's F1-only distribution choice; A6 otherwise
takes round 2's F1-chosen tree complexity (`num_leaves` 63,
`min_child_samples` 500).

### 8.3 The Kaplan-Meier tail rule (stated before it is used)

Round 2's `kaplan_meier_pmf` DROPPED the survival that never resolves inside a
cell and renormalised. With a 0.2% censoring flag that is immaterial; with the
correct 61%-censored low-clock cells it puts the dropped mass straight back onto
the SHORT durations and reproduces the truncated law -- it would silently undo
the fix. Round 3 fixes it explicitly:

> Within a cell, discrete-time Kaplan-Meier gives h(t), S(t) and
> pmf(t) = S(t-1) - S(t) on t = 0..90. Let t-star be the largest t carrying an
> uncensored exit in that cell and R = S(t-star) the unresolved survival. R is
> distributed over t > t-star in proportion to the PARENT level's pmf restricted
> to t > t-star and renormalised, recursing outward from the global cell, so a
> cell whose observations are nearly all censored inherits the next-coarser
> cell's TAIL rather than its own head. If the parent's restricted pmf is empty
> (t-star = 90) the mass is dropped, which is the `_normalise` convention every
> arm already uses.

Cell eligibility for A1/A2: at least 300 training rows (round 2's
`EMPIRICAL_MIN_CELL`) AND at least 100 uncensored exits; otherwise drop one
dimension from the right, as round 2 did. The event minimum is stated because
with the corrected flag a low-clock cell can hold 300 rows and 40 events.

### 8.4 Metrics

PRIMARY: **CRPS_trunc on uncensored test possessions.** A test row is uncensored
iff T < R, so it is drawn from T | T < R; scoring the unconditional predictive
law of T against it is improper and would reward exactly the truncated arms
round 3 exists to refute. Every arm's pmf is therefore renormalised onto
{0, ..., R-1} before CRPS is taken. For the 91.8% of rows with R > 90 the
renormalisation is a no-op, so the number stays broadly comparable to rounds 1-2;
the round-2 definition (all rows, censored rows scored as complete) is reported
alongside it as an explicitly labelled bridge column, never as the decision
metric.

SECONDARY: **censored log-likelihood**, the mean over ALL test rows of
log P(T = d) on uncensored rows and log P(T >= R) on censored rows. It is the
only metric the censored rows enter, and it is proper under censoring.

REPORTED: mean predicted survival beyond the censoring time on censored rows;
the round-2 log score with its undefined share; predicted vs actual mean
duration.

### 8.5 Gates (all three must pass; a gate is never softened to let an arm through)

- **G1-CC, the emergent gate and the selection gate.** `chain_halves` run over
  the REAL sequence of previous-end types, with only the clock-derived columns
  overridden by the simulated clock (no event model, no score model), restricted
  to CLOCK-COMPLETE GAMES. Tolerances are `docs/SIM_GUARDRAILS.md` G1: mean
  +/- 1.0, SD +/- 0.75, and EVERY powered month (>= 100 clock-complete games in
  that month). The all-games read is reported as a labelled secondary and is
  NOT the gate: the audit shows the all-games actual counts only LOGGED
  possessions in halves whose feed stops early, which no correct sim can
  reproduce. The re-basing is worth 0.00 to -0.20 possessions per team-game and
  makes the round-2 arms look very slightly WORSE, so it is a grading-truth fix
  and cannot be mistaken for a gap-closing one.
- **PIT.** K-S D <= 0.05 in every powered cell (n >= 300) of the pre-registered
  grid (previous end type x round-2 fine bucket), computed from the same
  truncated pmf as CRPS_trunc, on uncensored rows. Cells n < 300 are reported
  UNDERPOWERED and excluded from the decision.
- **End-of-half, on clock-complete halves.** Share of halves whose last
  possession starts with under 35 s left, and that possession's mean duration,
  each within the F1-derived noise floor (round 2's construction verbatim: the
  larger of the 5-seed spec-identical re-chain SD and the game-block bootstrap
  SE of the actual, times k = 2.0).

### 8.6 Segment breakdowns (reported for every arm)

By half (H1 / H2 / OT); by score-margin bucket at possession start (the five
`SCORE_BUCKET_LABELS`); by season. The pre-registration asks for a shot-clock-era
segment: the NCAA men's shot clock is 30 s in every season 2022-2025, so that
segment is DEGENERATE over this fold window and season is reported in its place,
with this reason stated rather than the segment silently dropped. Responsiveness
(CLAUDE.md standing rule) is the emergent count by team pace-prior QUINTILE with
the slope check: span ratio sim/actual and the number of monotone agreeing steps,
as in rounds 1 and 2.

### 8.7 Noise floor

Stochastic arms (A6, B1) refit spec-identically under a second seed;
|dCRPS_trunc| is their floor. Deterministic arms take the game-block bootstrap
SE of the mean CRPS_trunc. The floor used by the decision rule is the maximum of
the two, as in rounds 1 and 2. The chain and end-of-half floors come from the
5-seed re-chain described in 8.5.

### 8.8 Decision rule

Winner = lowest CRPS_trunc among arms passing ALL THREE gates on F2. A tree arm
(A6, B1) must beat the best non-tree arm by more than the floor. Ties -- a CRPS
gap inside the floor -- go to the simpler arm, ordered empirical < parametric <
hazard < tree; between A1 and A2, A2 is the simpler (strictly fewer distinct
cells). F1 is robustness only. If no arm passes, adopt nothing, report the
diagnosis, and do not soften a gate. No multiplier, cap, clip, offset or
calibration curve is fitted at any point (`docs/SIM_GUARDRAILS.md` core
principle).

### 8.9 Lookup-table export (deliverable, `docs/models/engine/RESUME.md` section 3 item 4)

For the WINNING arm only, export a binned lookup pmf over the round-2
discretised state grid (previous end type 6 x fine clock bucket 10 x period type
2 x score state 3 x tempo tercile 3 = 1,080 cells x 91 durations) and report the
binning error against the live model: mean and max |dCRPS_trunc| per row, the
maximum total-variation distance between the live and binned pmfs, the emergent
G1-CC mean/SD delta between them, and rows/s of each. If no arm wins, the export
is run on the best-CRPS arm and labelled NOT ADOPTED, exactly as rounds 1 and 2
labelled their reference pickles.

### 8.10 Execution

New trainer `scripts/train_clock_v3.py` (round 1's and round 2's scripts are
never overwritten); new round-3 arm/metric module `src/cbb_sim/models/clock_v3.py`
importing everything shared from `clock.py`, so no round-1/round-2 object or
pickle changes. Artifacts take a `v3_` prefix in `data/processed/models/clock/`.
Threads capped at 4 (`OMP_NUM_THREADS=4`, `n_jobs=4`); four other workers share
the machine. One blind grading path scores every arm.

---

## 9. Run R4 -- the round-3 grid (2026-09-10)

`scripts/train_clock_v3.py`, seed 20260910, module `src/cbb_sim/models/clock_v3.py`. Artifacts are `v3_*` in `data/processed/models/clock/`; rounds 1 and 2 files are untouched. Every number below is written by that script or by `scripts/diag_clock_v3_report.py` reading its output; none is typed by hand.

### 9.1 The censoring flag, as it lands on the design

| design rows | censored, rounds 1-2 flag (%) | censored, horn flag (%) | horn and not old | old and not horn | censored rows not consuming their clock |
|---|---|---|---|---|---|
| 2607192 | 0.2148 | 0.6751 | 12037 | 36 | 0 |

`old and not horn` are the handful of `end_period` rows whose feed stops a second or two before 0:00; they are reported, not reclassified.

### 9.2 Choices made on F1 only

| dist | scale | f1_censored_loglik | f1_crps_trunc | fit_seconds |
|---|---|---|---|---|
| normal | 0.69859 | -3.75829 | 5.05251 | 147.90000 |
| logistic | 0.34901 | -3.66928 | 5.03722 | 187.50000 |
| extreme | 0.52926 | -3.57625 | 4.89741 | 249.50000 |

**Chosen AFT error distribution: `extreme`** (highest F1 censored log-likelihood).

End-of-half noise floor, F1, on clock-complete halves: sim 5-seed re-chain SD share 0.00190 / duration 0.0457 s; actual game-block bootstrap SE share 0.00288 / duration 0.1269 s; k = 2.0. **Floor: share +/- 0.00576, duration +/- 0.2538 s** on 6,177 clock-complete halves.

### 9.3 F2 (selection fold) -- primary metric and the three gates

| id | arm | flag | crps_trunc | censored_loglik | crps_r2def | pred_mean_duration | pit_leak_failures | pit_worst_D | cc_mean_delta | cc_sd_delta | cc_months_pass | cc_n_powered_months | g1_pass | cc_eoh_share_gap | cc_eoh_duration_gap | eoh_pass | pit_pass | all_gates_pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | lgbm_quantile_r2 | old | 4.8386 | -5.1171 | 4.8039 | 17.3333 | 24 | 0.5266 | 2.2270 | -0.6040 | 0 | 5 | FAIL | 0.0401 | -6.3054 | FAIL | FAIL | FAIL |
| A1 | empirical_km3 | horn | 4.8785 | -3.5164 | 4.8794 | 17.5949 | 15 | 0.0949 | 1.3570 | -0.8280 | 0 | 5 | FAIL | 0.0227 | -3.4444 | FAIL | FAIL | FAIL |
| A2 | empirical_km3_srfloor | horn | 4.8880 | -3.5178 | 4.9171 | 17.6113 | 14 | 0.1665 | 0.8860 | -0.8840 | 1 | 5 | FAIL | 0.0281 | -1.4615 | FAIL | FAIL | FAIL |
| B2 | empirical_r2 | old | 4.8982 | -3.5245 | 4.8652 | 17.5343 | 20 | 0.5235 | 1.8050 | -0.7950 | 0 | 5 | FAIL | 0.0329 | -5.2798 | FAIL | FAIL | FAIL |
| A3 | gamma_aft | horn | 4.9128 | -3.5597 | 4.9069 | 17.5763 | 29 | 0.2349 | 1.1300 | -0.2960 | 1 | 5 | FAIL | -0.0155 | -1.8487 | FAIL | FAIL | FAIL |
| B3 | gamma_r2 | old | 4.9370 | -3.5693 | 4.8997 | 17.5341 | 35 | 0.5227 | 1.5310 | -0.2500 | 0 | 5 | FAIL | -0.0092 | -3.5129 | FAIL | FAIL | FAIL |
| A6 | xgb_aft | horn | 4.9484 | -3.5831 | 4.9374 | 17.3969 | 37 | 0.5637 | 1.2120 | -0.5550 | 1 | 5 | FAIL | 0.0055 | -2.2397 | FAIL | FAIL | FAIL |
| A5 | hazard3 | horn | 4.9572 | -3.5425 | 4.9522 | 17.3158 | 33 | 0.3164 | 2.1350 | -0.3200 | 0 | 5 | FAIL | -0.0024 | -2.9063 | FAIL | FAIL | FAIL |
| A4 | lognormal_aft | horn | 5.0685 | -3.6676 | 5.0675 | 18.3463 | 39 | 0.2108 | -1.7310 | 0.1820 | 0 | 5 | FAIL | -0.0714 | -1.1013 | FAIL | FAIL | FAIL |

`crps_trunc` is the primary metric (pre-registration 8.4): CRPS of the predictive law RENORMALISED onto {0..R-1}, on uncensored test rows only. `crps_r2def` is round 2's definition (all rows, censored rows scored as complete) and is a labelled bridge, not a decision metric. `cc_*` are read on CLOCK-COMPLETE games/halves, which is the round-3 gate universe.

### 9.4 The mechanism: predicted INTENDED duration by clock band

| band | n | horn_rate | actual_mean_observed | A1 | A2 | A3 | A4 | A5 | A6 | B1 | B2 | B3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [0, 5) | 2430 | 0.632 | 1.565 | 8.804 | 12.574 | 4.440 | 7.806 | 5.247 | 3.916 | 1.528 | 1.730 | 1.902 |
| [5, 10) | 3030 | 0.387 | 4.441 | 8.466 | 12.442 | 6.154 | 8.181 | 6.128 | 6.374 | 4.212 | 4.574 | 4.702 |
| [10, 20) | 5995 | 0.184 | 7.685 | 8.730 | 12.122 | 9.000 | 10.448 | 8.542 | 9.078 | 7.415 | 7.579 | 7.902 |
| [20, 30) | 6630 | 0.142 | 12.309 | 12.602 | 12.830 | 13.659 | 14.546 | 13.440 | 13.496 | 11.820 | 11.858 | 12.277 |
| [30, 45) | 11649 | 0.042 | 15.457 | 15.432 | 13.480 | 15.784 | 16.310 | 16.522 | 15.844 | 15.112 | 15.177 | 15.449 |
| [45, 60) | 10839 | 0.003 | 14.733 | 14.527 | 13.756 | 14.597 | 15.287 | 15.009 | 14.992 | 14.595 | 14.486 | 14.590 |
| [60, 90) | 19430 | 0.000 | 16.693 | 16.509 | 16.509 | 16.580 | 17.294 | 16.714 | 16.541 | 16.407 | 16.507 | 16.578 |
| [90, 1201) | 671201 | 0.000 | 17.894 | 17.915 | 17.915 | 17.899 | 18.654 | 17.595 | 17.699 | 17.702 | 17.915 | 17.898 |

This is where the fix has to show up and the one table round 2 could not produce. `actual_mean_observed` is the TRUNCATED mean the feed records; the arms trained on the corrected flag predict the INTENDED duration, which is longer wherever `horn_rate` is non-trivial, and the chain truncates it back at the horn. An arm whose column tracks `actual_mean_observed` inside the last 10 seconds has not changed.

### 9.5 Segment breakdowns (pre-registration 8.6)


**CRPS_trunc by half**

| level | n | A1 | A2 | A3 | A4 | A5 | A6 | B1 | B2 | B3 |
|---|---|---|---|---|---|---|---|---|---|---|
| H1 | 358430 | 4.8404 | 4.8506 | 4.8664 | 5.0083 | 4.9262 | 4.9426 | 4.8245 | 4.8563 | 4.8920 |
| H2 | 366371 | 4.9203 | 4.9291 | 4.9630 | 5.1318 | 4.9929 | 4.9610 | 4.8580 | 4.9420 | 4.9844 |
| OT | 6403 | 4.6101 | 4.6191 | 4.6266 | 4.8017 | 4.6345 | 4.5374 | 4.5115 | 4.7369 | 4.7441 |

**CRPS_trunc by score_bucket**

| level | n | A1 | A2 | A3 | A4 | A5 | A6 | B1 | B2 | B3 |
|---|---|---|---|---|---|---|---|---|---|---|
| -15..-6 | 148128 | 4.6704 | 4.6740 | 4.6888 | 4.8268 | 4.7640 | 4.7659 | 4.6488 | 4.6841 | 4.7082 |
| -5..5 | 346077 | 4.8811 | 4.8882 | 4.9227 | 5.0796 | 4.9633 | 4.9580 | 4.8469 | 4.9076 | 4.9523 |
| 6..15 | 130535 | 5.1152 | 5.1201 | 5.1710 | 5.3357 | 5.2009 | 5.1757 | 5.0669 | 5.1275 | 5.1893 |
| <=-16 | 57265 | 4.5938 | 4.6009 | 4.6011 | 4.7447 | 4.6722 | 4.6776 | 4.5624 | 4.6054 | 4.6177 |
| >=16 | 49199 | 5.1895 | 5.2482 | 5.1948 | 5.3845 | 5.1797 | 5.1406 | 5.0668 | 5.2090 | 5.2200 |

The shot-clock era is 30 s in every NCAA men's season 2022-2025, so that pre-registered segment is degenerate over this fold window and season stands in its place (`v3_segments_F2_*.csv`, `segment = season`), reported rather than silently dropped.


### 9.6 Responsiveness (CLAUDE.md standing rule)

| id | arm | span_sim | span_actual | slope_ratio | steps_agreeing | q1_delta | q5_delta |
|---|---|---|---|---|---|---|---|
| A1 | empirical_km3 | 6.4408 | 8.2909 | 0.7769 | 4 | 2.3661 | 0.5160 |
| A2 | empirical_km3_srfloor | 6.4215 | 8.2909 | 0.7745 | 4 | 1.9248 | 0.0555 |
| A3 | gamma_aft | 8.5056 | 8.2909 | 1.0259 | 4 | 1.1006 | 1.3153 |
| A4 | lognormal_aft | 7.7538 | 8.2909 | 0.9352 | 4 | -1.3825 | -1.9196 |
| A5 | hazard3 | 8.2138 | 8.2909 | 0.9907 | 4 | 2.2523 | 2.1753 |
| A6 | xgb_aft | 7.5822 | 8.2909 | 0.9145 | 4 | 1.9572 | 1.2486 |
| B1 | lgbm_quantile_r2 | 8.7195 | 8.2909 | 1.0517 | 4 | 2.3924 | 2.8210 |
| B2 | empirical_r2 | 6.4521 | 8.2909 | 0.7782 | 4 | 2.8134 | 0.9746 |
| B3 | gamma_r2 | 8.5672 | 8.2909 | 1.0333 | 4 | 1.4831 | 1.7594 |

Per-quintile tables: `v3_responsiveness_F2_{arm}.csv`.


### 9.7 By-month G1 on clock-complete games, for the arms that pass overall

| month | n_cc_games | A1 | A2 | A3 | A4 | A5 | A6 | B1 | B2 | B3 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 520 | 1.717 | 1.249 | 1.556 | -1.251 | 2.634 | 1.656 | 2.630 | 2.158 | 1.986 |
| 2 | 440 | 1.638 | 1.175 | 1.540 | -1.201 | 2.580 | 1.755 | 2.661 | 2.110 | 1.957 |
| 3 | 305 | 1.126 | 0.636 | 0.675 | -2.289 | 1.646 | 0.997 | 1.793 | 1.562 | 1.054 |
| 11 | 395 | 1.306 | 0.806 | 1.004 | -1.910 | 1.865 | 0.784 | 2.108 | 1.724 | 1.359 |
| 12 | 324 | 0.705 | 0.284 | 0.508 | -2.466 | 1.543 | 0.508 | 1.610 | 1.194 | 0.920 |

Powered months only (>= 100 clock-complete games). Tolerance +/- 1.0.


### 9.8 F1 (robustness only)

| id | arm | flag | crps_trunc | censored_loglik | crps_r2def | pit_leak_failures | pit_worst_D |
|---|---|---|---|---|---|---|---|
| B1 | lgbm_quantile_r2 | old | 4.7932 | -5.0973 | 4.7622 | 21 | 0.4763 |
| A1 | empirical_km3 | horn | 4.8354 | -3.5190 | 4.8372 | 12 | 0.1669 |
| A2 | empirical_km3_srfloor | horn | 4.8459 | -3.5207 | 4.8744 | 16 | 0.1577 |
| B2 | empirical_r2 | old | 4.8543 | -3.5259 | 4.8248 | 17 | 0.4735 |
| A3 | gamma_aft | horn | 4.8744 | -3.5566 | 4.8694 | 29 | 0.2921 |
| B3 | gamma_r2 | old | 4.8972 | -3.5653 | 4.8629 | 33 | 0.4735 |
| A6 | xgb_aft | horn | 4.8974 | -3.5763 | 4.8880 | 36 | 0.6256 |
| A5 | hazard3 | horn | 4.9110 | -3.5357 | 4.9070 | 35 | 0.3942 |
| A4 | lognormal_aft | horn | 5.0370 | -3.6674 | 5.0362 | 39 | 0.3016 |

### 9.9 Noise floor

| id | arm | kind | value |
|---|---|---|---|
| A1 | empirical_km3 | block_bootstrap_se | 0.006139 |
| A2 | empirical_km3_srfloor | block_bootstrap_se | 0.006157 |
| A3 | gamma_aft | block_bootstrap_se | 0.006026 |
| A4 | lognormal_aft | block_bootstrap_se | 0.005867 |
| A5 | hazard3 | block_bootstrap_se | 0.005948 |
| A6 | xgb_aft | seed_refit | 0.000523 |
| B1 | lgbm_quantile_r2 | seed_refit | 0.000021 |
| B2 | empirical_r2 | block_bootstrap_se | 0.006376 |
| B3 | gamma_r2 | block_bootstrap_se | 0.006257 |

Floor used by the decision rule: **0.006376** (the maximum).


### 9.10 Lookup-table export (deliverable, pre-registration 8.9)

| source arm | adopted | cells | empty cells | live CRPS_trunc | binned CRPS_trunc | dCRPS | TV mean | TV max | live G1-CC dmean | binned G1-CC dmean | dG1 | live rows/s | binned rows/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm_quantile | FAIL | 1080 | 187 | 4.838582 | 4.907261 | 0.068678 | 0.103182 | 0.925170 | 2.227000 | 2.686000 | 0.459000 | 13728.200000 | 183550.000000 |

Grid: prev_end_code 6 x r2_bucket_code 10 x r2_period_type 2 x r2_score_state 3 x tempo_tercile 3 = 1080 cells x 91 durations. Table at `v3_lookup_table.npz`.


### 9.11 Verdict

```json
{
  "round": 3,
  "seed": 20260910,
  "selection_fold": "F2",
  "floor": 0.006376068383219856,
  "end_of_half_floor": {
    "share": 0.005756463971823765,
    "duration": 0.2537743774655367,
    "k": 2.0,
    "sim_seed_sd_share": 0.0018997190107438357,
    "sim_seed_sd_duration": 0.04567575449004249,
    "actual_block_bootstrap_se_share": 0.0028782319859118827,
    "actual_block_bootstrap_se_duration": 0.12688718873276836,
    "seeds": [
      20260910,
      20260911,
      20260912,
      20260913,
      20260914
    ],
    "n_cc_halves": 6177
  },
  "xgb_dist_chosen": "extreme",
  "n_arms": 9,
  "n_pass_g1": 0,
  "n_pit_clean": 0,
  "n_pass_eoh": 0,
  "n_eligible": 0,
  "best_crps_arm": "lgbm_quantile_r2",
  "best_crps": 4.838582377520811,
  "created_at": "2026-09-10T23:02:08.539031+00:00",
  "finished_at": "2026-09-11T00:21:20.726295+00:00",
  "winner": null,
  "verdict": "NO ARM ADOPTED",
  "reason": "no arm passed all three pre-registered round-3 gates on F2 (emergent G1 on clock-complete games, PIT, end-of-half)"
}
```

---

## 9.12 Reading of the round-3 grid (2026-09-10)

Appended after the numbers. Interpretation in full: `model.md` section 11.

**Verdict: NO ARM ADOPTED.** 0 of 9 arms pass the emergent G1 on clock-complete
games, 0 are PIT-clean, 0 pass the end-of-half gate. `v3_winner.pkl` was not
written; the best-CRPS arm is persisted as
`v3_reference_not_adopted_lgbm_quantile_r2.pkl`.

1. **The censoring fix is real, measured, and in the right direction — and it is
   not big enough.** The pre-registered paired controls isolate it, because A1
   vs B2 and A3 vs B3 differ ONLY in the censoring flag and the tail rule:

   | pair | old flag (B) | horn flag (A) | change |
   |---|---:|---:|---:|
   | empirical: G1-CC mean delta | +1.805 | +1.357 | **-0.448** |
   | empirical: EOH-CC duration gap | -5.280 s | -3.444 s | **+1.836 s** |
   | gamma: G1-CC mean delta | +1.531 | +1.130 | **-0.401** |
   | gamma: EOH-CC duration gap | -3.513 s | -1.849 s | **+1.664 s** |

   Both moves are 60-70x the 0.0064 CRPS floor in effect size terms and far
   outside the 0.254 s end-of-half duration floor, so they are not noise. L20 is
   CONFIRMED as a real defect and the fix is CONFIRMED as a real improvement.
   It closes roughly a quarter of the possession overshoot and a third of the
   end-of-half duration gap. It does not close either one.

2. **The mechanism does exactly what it was designed to do.** Section 9.4: in
   the 0-5 s band (63.2% horn-ending) the round-2 arms predict a 1.7-1.9 s
   INTENDED duration — i.e. they reproduce the truncated observed mean of 1.57 s
   — while the corrected arms predict 8.8 s (`empirical_km3`), 12.6 s
   (`empirical_km3_srfloor`) and 4.4 s (`gamma_aft`). The engine then truncates
   that at the horn instead of subdividing the remaining seconds. This is the
   one table round 2 could not produce, and it is the direct evidence that the
   censoring flag changed the conditional law and not merely the fit.

3. **The single best G1 reading in three rounds is A2,
   `empirical_km3_srfloor`, at +0.886** — the first arm in any round to land
   INSIDE the +/- 1.0 mean tolerance on the gate universe. It fails on three
   other clauses: SD -0.884 against +/- 0.75, only 1 of 5 powered months, and an
   end-of-half duration gap of -1.46 s against a 0.254 s floor. A2 is the arm
   that throws away the most clock information (every row under 45 s is served
   the 45-59 s law), which says the remaining overshoot lives in how the model
   treats the last 45 seconds, not in the last 10.

4. **PIT is now the cleanest it has ever been on the corrected arms, and still
   fails.** Worst powered K-S D: `empirical_km3` 0.095, `empirical_km3_srfloor`
   0.166, `gamma_aft` 0.235 — against 0.52-0.56 for every old-flag arm
   (`empirical_r2` 0.524, `gamma_r2` 0.523, `lgbm_quantile_r2` 0.527). That gap
   is the truncation-aware metric working: scoring the old arms against the
   correctly TRUNCATED predictive law exposes a mis-specification the round-2
   metric could not see. But 14-39 leak-sized cells of 27-30 powered remain
   everywhere, so no arm is PIT-clean.

5. **The censored log-likelihood refutes the tree reference outright.**
   `lgbm_quantile_r2` scores -5.117 against -3.516 for `empirical_km3`: it wins
   CRPS_trunc (4.8386, the lowest in the grid) and is the worst arm in the grid
   on the only metric the censored rows enter, because 206 test rows get zero
   predictive mass below their own remaining clock and every censored row gets
   almost none above it. Its G1-CC is +2.227 and its EOH duration gap -6.31 s,
   both the worst in the grid. A model that cannot represent the censored tail
   cannot be the clock model, whatever its CRPS says — which is why the
   pre-registration gates before it ranks.

6. **`xgb_aft` is a genuinely censoring-aware tree and it does not rescue the
   tree class.** The F1-chosen error distribution is `extreme` (Weibull AFT),
   chosen on censored log-likelihood -3.576 against -3.669 (logistic) and -3.758
   (normal), with the scale fitted by profile likelihood at 0.5293. On F2 it
   lands CRPS_trunc 4.9484, G1-CC +1.212, EOH -2.24 s, PIT 37 failures: better
   than the quantile reference on every gate and worse on CRPS. The tree class
   is not where the remaining error is.

7. **Responsiveness still passes on every arm** (4 of 4 monotone quintile steps;
   slope ratios 0.775-1.052), and the SD story is unchanged from round 2: the
   parametric arms hold SD inside tolerance (`gamma_aft` -0.296, `hazard3`
   -0.320, `lognormal_aft` +0.182) while the empirical arms sit at -0.83 to
   -0.88. Matchup signal and dispersion are not the blocker; the conditional
   mean near a period boundary still is.

8. **What is now known that was not known before this round.** The horn
   censoring defect was one cause of the emergent overshoot, worth about 0.45
   possessions per team-game, and it is now fixed. Roughly +0.9 to +1.4
   possessions remain on the corrected arms. That residual is the size of the
   effect the engine worker independently attributed to `score_diff` feedback
   inside the clock model (L23 / Decision 10: freezing `score_diff` moves the
   engine's count from 71.6 to 67.84 against 67.88 actual). The two hypotheses
   are not competitors, they are additive, and round 3b tests the second with
   the first already in place.

Nothing is adopted. No multiplier, cap, clip or calibration is fitted to close
the remaining gap (`docs/SIM_GUARDRAILS.md` core principle).

---

## 10. Round 3b pre-registration (2026-09-10)

Appended AFTER round 3's verdict and BEFORE any round-3b code ran. Section 8
(the round-3 pre-registration) is STATIC-ONLY and is NOT edited; this is a
separate question asked separately, and it asks two of them.

Two independent findings arrived while round 3 was running:

- **L23 / Decision 10** (engine worker, commit db72e0d): a paired-stream
  ablation inside the engine shows that FREEZING the clock model's `score_diff`
  state feature moves simulated possessions per game from 71.6 to 67.84 against
  67.88 actual, while ablating `fg_make` or the event model leaves the count
  +3.6. `score_diff` is produced BY the simulation, so conditioning the clock on
  it closes a feedback loop that offline scoring cannot see.
- **L21**: S1 (in-season monthly walk-forward refit) is the project's default
  training scheme. Round 3 fitted statically.

Round 3 established that horn censoring is real and worth about 0.45
possessions. Round 3b asks whether the remaining +0.9 to +1.4 is the
`score_diff` loop, and whether S1 moves anything, with the censoring fix held in
place throughout.

### 10.1 Part A -- state parametrisation crossed with censoring

Base arms: the two round-3 arms that carried the fix furthest,
`empirical_km3` (A1) and `gamma_aft` (A3). Each is run under three state
parametrisations, all with the CORRECTED horn censoring flag:

| id | parametrisation | definition |
|---|---|---|
| P1 | as designed | round-2 state verbatim: `score_diff`, `x_score_diff__seconds_remaining`, and the fine-bucket x period-type x **score-state** cross |
| P2 | `score_diff` removed | all three dropped; the cross collapses to fine bucket x period type. The model then sees no simulation-produced score information at all |
| P3 | engine-safe end-game variables | `score_diff` replaced by three mutually exclusive indicators that are non-zero ONLY inside the last `ENDGAME_WINDOW_S = 120` seconds of the second half or an OT period: `eg_trailing_big` (offence down 4+, the intentional-foul regime), `eg_leading_big` (offence up 4+, the run-out-the-clock regime), `eg_close` (within 3). Outside the window the model sees no score. The margin still enters through a coarse bucket, so P3 is *safer*, not *safe*; the closed-loop gate in 10.3 is what decides whether it is safe enough |

Six arms (2 base x 3 parametrisations). The round-3 P1 rows are reused verbatim
as the P1 column rather than refitted, so the comparison is paired by
construction.

### 10.2 Part B -- S1 scheme confirmation

Whichever Part-A arm reads best on the closed-loop gate, refit under S1: a
refit at every month boundary of the test season on all prior seasons plus the
season to date, strictly before the refit date
(`possession_outcome.month_boundaries` REUSED, not re-implemented). S0 (static)
is the reference, refit inside the same harness so one code path scores both.
**S1 is adopted as the scheme unless a pre-registered gate regresses beyond the
round-3 floor.** Adopting S1 as a SCHEME does not adopt the underlying arm;
round 3's verdict governs that.

### 10.3 Gates

The three round-3 gates are carried over UNCHANGED (CRPS_trunc primary on
uncensored rows with the truncation renormalisation; PIT K-S D <= 0.05 in every
powered cell; emergent G1 on clock-complete games at mean +/- 1.0, SD +/- 0.75,
all powered months; end-of-half on clock-complete halves within the F1-derived
floor share +/- 0.005756, duration +/- 0.2538 s). The round-3 CRPS floor
0.006376 is carried over.

ADDED, and this is the point of round 3b -- **the Decision-10 closed-loop
gate**: each Part-A arm is run INSIDE the engine by
`scripts/diag_engine_multilevel.py` as a paired-stream ablation, 5 seeds, and
must (i) hold possessions per game inside the G1 tolerance against the same
games' actual, and (ii) not move margin SD beyond the paired-seed noise band.
An arm that passes the offline gates and fails the closed-loop gate is REFUTED,
because that is exactly the failure L23 was written about: offline scoring
cannot see a feedback loop.

### 10.4 Decision rule

Within Part A, the winner is the arm with the lowest CRPS_trunc among those
passing ALL FOUR gates (three offline + closed-loop); ties inside the floor go
to the simpler parametrisation, ordered P2 < P3 < P1 (fewer simulation-produced
inputs is simpler and safer), then to the simpler model class (empirical <
parametric). If no arm passes, adopt nothing, report the diagnosis, and do not
soften a gate. No multiplier, cap, clip, offset or calibration is fitted at any
point.

### 10.5 Artifacts and naming (engine-selectable, documented)

Offline artifacts take a `v3b_` prefix in `data/processed/models/clock/`. The
S1 schedule is persisted as per-month files the engine selects by game date:

    data/processed/models/clock/v3b_s1/
        {arm}_S1_{season}_{YYYY-MM}.pkl          the fit made at that month's boundary
        lookup_{arm}_S1_{season}_{YYYY-MM}.npz   its binned lookup table
        manifest.json                            refit_date / valid_from / valid_to /
                                                 model_file / lookup_file / n_train /
                                                 n_train_from_test_season / max_train_date

The engine picks the row whose [valid_from, valid_to] contains the game's own
date -- equivalently the latest refit_date at or before it -- so a game is never
served by a fit that has seen it. The binned lookup grid is round 3's: previous
end type 6 x fine clock bucket 10 x period type 2 x score state 3 x tempo
tercile 3 = 1,080 cells x 91 durations (P2 and P3 collapse or replace the score
dimension and their grids are reported with their own sizes).

New trainer `scripts/train_clock_v3b.py`; rounds 1-3 scripts and artifacts are
never overwritten. Threads capped at 4.

---

## 11. Run R5 -- the round-3b grid (2026-09-10)

`scripts/train_clock_v3b.py`, seed 20260910, arms in
`src/cbb_sim/models/clock_v3.py` section 8. Artifacts are `v3b_*`; rounds 1-3
files are untouched. Every number is written by that script.

### 11.1 Part A -- state parametrisation crossed with censoring (F2)

All six arms carry the round-3 horn censoring flag. P1 rows are round 3's,
reused verbatim, so the contrast is paired.

| arm | P | CRPS_trunc | censored loglik | PIT fails | PIT worst D | G1-CC dmean | G1-CC dSD | months | EOH d-share | EOH d-dur | gates |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| empirical_km3 | P1 | 4.8785 | -3.5164 | 15 | 0.0949 | +1.357 | -0.828 | 0/5 | +0.02275 | -3.444 | FAIL |
| empirical_km3 | P3 | 4.8880 | -3.5087 | 13 | 0.1090 | **+1.131** | -0.754 | 1/5 | +0.01978 | -3.398 | FAIL |
| empirical_km3 | P2 | 4.9058 | -3.5085 | 13 | 0.1270 | +1.276 | -0.806 | 0/5 | +0.01918 | -3.421 | FAIL |
| gamma_aft | P1 | 4.9128 | -3.5597 | 29 | 0.2349 | +1.130 | -0.296 | 1/5 | -0.01546 | -1.849 | FAIL |
| gamma_aft | P3 | 4.9327 | -3.5649 | 30 | 0.2290 | **+0.903** | **-0.277** | 2/5 | -0.01368 | -1.509 | FAIL |
| gamma_aft | P2 | 4.9435 | -3.5675 | 30 | 0.2289 | +1.092 | -0.287 | 1/5 | -0.01725 | -1.969 | FAIL |

CRPS block-bootstrap SEs 0.00602-0.00614; the round-3 floor 0.006376 is carried
over.

**Verdict on Part A: NO ARM ADOPTED offline.** 0 of 6 pass G1-CC, 0 are
PIT-clean, 0 pass end-of-half.

### 11.2 The result that matters, and it is a negative one

**P2 -- deleting `score_diff` outright -- does NOT reproduce the engine
worker's finding offline.** On both base arms it lands between P1 and P3 and
nowhere near the engine's 71.6 -> 67.84:

| base | P1 (as designed) | P2 (score removed) | P3 (engine-safe end-game) |
|---|---:|---:|---:|
| empirical_km3 G1-CC dmean | +1.357 | +1.276 | +1.131 |
| gamma_aft G1-CC dmean | +1.130 | +1.092 | +0.903 |

This is the expected answer once it is stated, and it is worth stating
precisely, because it is the difference between the two harnesses:
**`chain_halves` feeds the model the REAL score sequence.** The offline chain
overrides only the clock-derived columns; `score_diff` on every row is the score
the actual game had at that point. There is therefore NO feedback loop for P2 to
break offline -- the loop only exists when the score itself is simulated. The
offline grid can measure how much *information* `score_diff` carries (a little:
0.08-0.23 possessions) and is structurally incapable of measuring the *loop*
(which the engine measures at ~3.8 possessions).

That is precisely the failure mode Decision 10 exists for, and it is why the
closed-loop gate is not optional here: **Part A cannot decide between P1, P2 and
P3.** It can only say that none of them fixes G1 offline, and that P3 is the
best of the three on the emergent count on both base arms.

### 11.3 The best G1 reading the project has produced

`gamma_aft|P3`: G1-CC mean **+0.903** and SD **-0.277**, the first arm in any
round with BOTH inside the +/- 1.0 / +/- 0.75 tolerances. It fails the gate on
the all-powered-months clause (2 of 5) and fails end-of-half duration
(-1.51 s against the 0.254 s floor) and PIT (30 leak-sized cells of 30 powered).

Trajectory of the emergent overshoot on clock-complete games, one line per
decided change:

| state | best G1-CC mean delta | what changed |
|---|---:|---|
| round 2 (old flag, static) | +1.516 (gamma) | -- |
| round 3 (horn censoring) | +1.130 (gamma_aft) | L20 fix, -0.39 |
| round 3b P3 (engine-safe end-game state) | +0.903 | -0.23 |
| round 3b P3 + S1 | **+0.885** | -0.02 |

Cumulative: **+1.516 -> +0.885**, 58% of the round-2 overshoot removed by two
pre-registered, separately-evidenced changes and no tuning. The gate still
fails, on months and on end-of-half duration.

### 11.4 Part B -- S1 scheme confirmation on `gamma_aft|P3`

Six monthly refits over the 2025 test season, each on all prior seasons plus the
season to date, strictly before the refit date
(`possession_outcome.month_boundaries` reused).

| | CRPS_trunc | censored loglik | G1-CC dmean | G1-CC dSD | EOH d-share | EOH d-dur | PIT fails |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 static | 4.932682 | -3.56491 | +0.903 | -0.277 | -0.01368 | -1.5086 | 30 |
| S1 monthly | 4.931928 | -3.56475 | +0.885 | -0.269 | -0.01294 | -1.4756 | 30 |
| delta | **-0.000754** | +0.00016 | -0.018 | +0.008 | +0.00074 | +0.033 | 0 |

The CRPS gain is **0.12 of the floor** -- a wash, exactly as L21 predicts ("S1 is
a calibration fix, not a log-loss fix"). **No pre-registered gate regresses**,
so by the section-10.2 rule the scheme adopted is **S1**. Every gate moves in
the right direction, all inside the floor.

Adopting S1 as the SCHEME does not adopt the arm. Round 3 adopted nothing, Part
A adopted nothing, and the closed-loop gate has not run.

Per-month artifacts are persisted for engine selection by game date:
`data/processed/models/clock/v3b_s1/gamma_aft_P3_S1_2025_{YYYY-MM}.pkl`, the
matching `lookup_*.npz`, and `manifest.json` carrying refit_date / valid_from /
valid_to / model_file / lookup_file / n_train / n_train_from_test_season /
max_train_date. The engine picks the row whose [valid_from, valid_to] contains
the game's own date, so no game is served by a fit that has seen it.

### 11.5 The closed-loop gate has NOT run -- and nothing is adopted until it does

`scripts/diag_engine_multilevel.py` grades an existing engine results directory;
it does not run the engine with a swapped clock model. The Decision-10 gate
therefore needs an engine run with the round-3b arm wired in
(`ENGINE_CLOCK` pointing at `v3b_arm_gamma_aft_P3.pkl` or the S1 manifest),
which is the engine worker's harness and adapter. It is left as a handoff rather
than forced from here, per the worker-discipline rule about another worker's
files.

**Until that gate runs, round 3b adopts nothing.** The offline stage cannot
distinguish P1 from P2 from P3 on the question that matters (11.2), so the
closed-loop run is the deciding evidence, not a confirmation.

---

## 12. Round 3c pre-registration -- the Decision-10 closed-loop run (2026-09-10)

Appended VERBATIM BEFORE any round-3c code was written or run, and committed on
its own before the first engine process started. Sections 8 and 10 (the round-3
and round-3b pre-registrations) are STATIC and are NOT edited.

Round 3b ended with an explicit negative result (11.2): **the offline chain
cannot decide between P1, P2 and P3**, because `chain_halves` feeds the model
the REAL score sequence and therefore contains no feedback loop to break. The
offline grid measured the *information* in `score_diff` (0.08-0.23 possessions);
the engine measures the *loop* (about 3.8). Decision 10 says the loop is decided
inside the engine and nowhere else. Round 3c is that run.

This round decides ONE question: **which state parametrisation the clock model
ships with**. It does not reopen model class, censoring, or the training scheme;
those are settled (L20, L21, L26).

### 12.1 Arms

Every arm is the clock model LOADED INTO THE ENGINE and run over the same games,
the same seeds and the same RNG streams. Seven arms:

| id | `ENGINE_CLOCK` | base arm | state | scheme | what it is |
|---|---|---|---|---|---|
| I | `reference` | round-1 `lgbm_quantile` | round-1 `C_plus_score` | static | THE INCUMBENT. What the engine runs today and what every gate report so far was measured on |
| A1 | `v3c_gamma_P1_s1` | `gamma_aft` | P1 (as designed: `score_diff`, its clock interaction, the fine-bucket x period x score-state cross) | S1 | the round-3b best-CRPS state, with the loop fully live |
| A2 | `v3c_gamma_P2_s1` | `gamma_aft` | P2 (`score_diff` removed entirely) | S1 | no simulation-produced score information reaches the clock at all |
| A3 | `v3c_gamma_P3_s1` | `gamma_aft` | P3 (engine-safe end-game indicators only, last 120 s of H2/OT) | S1 | round 3b's best G1 arm (+0.885 / -0.269 offline) |
| B1 | `v3c_srfloor_P1_s1` | `empirical_km3_srfloor` | P1 | S1 | CELL-BASED, therefore a lookup table by construction with ZERO binning error (L26). Simplest arm in the round-3 simplicity order |
| B3 | `v3c_srfloor_P3_s1` | `empirical_km3_srfloor` | P3 | S1 | the same, engine-safe. P3's `eg_regime` is a cell dimension, so the state is supported |
| F | `v3c_gamma_P1_s1` with `ENGINE_CLOCK_FREEZE=1` | `gamma_aft` | P1 with the margin held at its PREGAME value (zero) for the clock only | S1 | THE DECISION-10 FROZEN ARM. Same fitted object as A1, same streams; the only difference is that no simulated margin reaches the clock. A1 vs F is the size of the loop, measured directly |

`empirical_km3_srfloor` is not fitted under P2: P2 deletes the score dimension
from the cell grid, and a `srfloor` grid without the score dimension is the same
object as `empirical_km3` under P2 with a floored clock bucket, which is a
different arm from the one round 3 scored. It is left out rather than silently
renamed.

Freeze semantics, stated before the run: `ENGINE_CLOCK_FREEZE=1` replaces the
clock adapter's view of the offence's score margin with zero on EVERY row, and
every margin-derived column is then computed from that zero -- the two round-2
score columns, the score-state leg of the R2 cross, and the P3 end-game
indicators alike. It is strictly stronger than L23's two-column ablation and it
is what "frozen at its pregame value" means for a model whose margin enters
through several derived columns. Under P2 the freeze is a NO-OP by construction,
which is the internal consistency check on the implementation.

### 12.2 Every other sub-model is held fixed and named explicitly

`ENGINE_EVENT=round2_s1`, `ENGINE_FG_MAKE=round2b_S_C_s1`,
`ENGINE_FG3=decision8`, `ENGINE_ROTATION=reference`. All four are passed as
explicit environment values on EVERY run, never left to a default, so that an
arm run at 03:00 and an arm run at 05:00 are the same comparison. Any run made
under a different value of any of them is discarded and re-run; a mixed design
is not a paired comparison.

### 12.3 Universe, subset selection rule, seeds, pairing

- Universe: the F2 2025 engine input slate, `data/processed/models/engine/games_F2_2025.parquet`, 5,710 games.
- **Subset selection rule, fixed here before any run:** sort the slate by `game_id` ASCENDING and take every 11th row (indices 0, 11, 22, ...), then keep the first 500. That is a deterministic, model-blind stride over the whole season; it is not the first 500 games, which would be November-only. The resulting subset spans 2024-11-04 to 2025-03-15 and holds 159 clock-complete games.
- Screening read: 5 seeds (0-4). Deciding read: 25 seeds (0-24) on the arms that survive screening, plus the incumbent.
- Pairing is by construction: the engine's RNG is seeded on (seed, game_id, family), so two arms run on the same games and seeds consume aligned streams and differ only where the clock model differs.

### 12.4 Metrics

Reported for every arm, at both seed counts:

| id | metric | level |
|---|---|---|
| M1 | possessions per game, mean and SD, on CLOCK-COMPLETE games | overall |
| M2 | possessions per game, mean and SD, on ALL 500 games | overall |
| M3 | end-of-half: share of period-ending possessions that start with under 35 s left, and their mean duration | overall, by period type |
| M4 | margin SD across all (game, seed) rows | overall |
| M5 | home/away score correlation across the same rows | overall |
| M6 | total bias against verified finals | overall, per game |
| M7 | points per possession | overall |
| M8 | possessions per game by team TEMPO-PRIOR quintile, with the slope ratio | per team quintile |
| M9 | possessions per game delta | per month |

M3 is accumulated inside the clock adapter itself: a possession whose INTENDED
duration reaches or exceeds the time left is the period's last possession by
construction, so the adapter records the seconds that were left at its start and
the truncated duration it actually consumed. That is the engine's analogue of
the offline end-of-half statistic and is labelled as such; the engine writes no
possession-level file, so it cannot be recomputed after the fact.

Tolerances (`docs/SIM_GUARDRAILS.md` G1, `docs/tests/gate_reference_2026-09-10.md`):
G1 mean +/- 1.0 and SD +/- 0.75 against the same games' actual; the 2025 season
figures are 67.875 possessions per game and 14.633 margin SD, but the SUBSET's
own actuals are what every delta is taken against, never the season figure. A
per-month or per-quintile cell with fewer than 100 games is reported
UNDERPOWERED and is excluded from the decision, never presented as signal or as
absence of signal. On the 500-game subset every clock-complete month cell is
underpowered by that rule and is reported as such.

### 12.5 Noise floor

The floor is the same arm re-run with `--seed-offset 1000` (seeds 1000-1004 for
the 5-seed read, 1000-1024 for the 25-seed read), which is a spec-identical
re-run under a different seed block -- the engine's form of the bake-off's
"spec-identical retrain under another seed". It is computed for the incumbent
and for the leading candidate, and the larger of the two absolute deltas per
metric is THE floor for that metric. A difference between two arms that is
inside the floor is a tie.

### 12.6 Decision rule

Adopt the SIMPLEST arm (cell-based before gamma; P3 before P2 before P1 on
engine-safety, i.e. fewer simulation-produced inputs first) that satisfies BOTH:

1. its G1 mean is within +/- 1.0 and its G1 SD within +/- 0.75 of the same
   games' actual, on BOTH the clock-complete set (M1) and all 500 games (M2); and
2. it does not move margin SD (M4) or the home/away correlation (M5) by more
   than the floor RELATIVE TO THE INCUMBENT.

Ordering among qualifying arms: `srfloor|P3` before `srfloor|P1` before
`gamma|P3` before `gamma|P2` before `gamma|P1`. If two qualifying arms differ on
M1 by less than the floor, the simpler one wins. If NO arm qualifies, adopt
nothing, report the diagnosis, and do not soften a gate. No multiplier, cap,
clip, offset, calibration curve or blend is fitted at any point
(`docs/SIM_GUARDRAILS.md` core principle).

The frozen arm F is NOT a candidate for adoption -- freezing a feature is an
ablation, not a model. It is the measuring stick: A1 minus F is the size of the
loop that P2 and P3 exist to remove, and an arm whose G1 is no better than A1's
has not removed it.

### 12.7 Artifacts, code and naming

- New module `src/cbb_sim/engine/clock_adapter_v3.py` holds ALL round-3c engine code. The hook in `adapters.py` is a few lines and is backward compatible: every existing `ENGINE_CLOCK` value keeps its exact current behaviour and still loads `ClockAdapter`.
- New S1 schedules for the arms round 3b did not fit (`gamma_aft` under P1 and P2, `srfloor` under P1 and P3) are written by `scripts/train_clock_v3c_s1.py` into `data/processed/models/clock/v3c_s1/`, each with its own manifest. `v3b_s1/` is READ, never written: `gamma_aft|P3` reuses round 3b's artifacts verbatim so that arm is byte-identical to the one round 3b scored.
- Artifact selection per game goes through `cbb_sim.engine.manifest.ArtifactManifest`, which enforces refit date at or before the game date AND last training date strictly before it, at load. The clock manifest format is translated into that object's format; the selection rule is not re-implemented.
- Engine results: `results/engine_v0/clock3c_<arm>/`. Tables: `docs/tests/clock_round3c_closed_loop_2026-09-10.md`. Results and the decision are appended to this file as section 13.

---

## 13. Run R6 -- the round-3c closed-loop grid (2026-09-11)

`scripts/run_clk3c_closed_loop.py` and `scripts/grade_clk3c_closed_loop.py`,
pre-registration section 12 committed a08635f BEFORE the harness existed
(1450faf). Full evidence, every table and the multi-level reads:
`docs/tests/clock_round3c_closed_loop_2026-09-10.md`. Artifacts `v3c_*`;
`v3b_s1/` was read and never written, so `gamma_aft|P3` is byte-identical to the
schedule round 3b scored.

Sub-models pinned by explicit environment value on EVERY run and recorded in
each `run_meta.json`: `ENGINE_EVENT=round2_s1`,
`ENGINE_FG_MAKE=round2b_S_C_s1`, `ENGINE_FG3=decision8`,
`ENGINE_ROTATION=reference`. Subset: the F2 2025 slate sorted by `game_id`
ascending, every 11th row, first 500 games; 159 clock-complete. Actual on that
subset: 68.530 possessions per team-game clock-complete, 68.328 all, margin SD
15.472, home/away score correlation 0.237, PPP 1.0705.

### 13.1 The deciding read (25 seeds, paired streams)

| id | arm | G1 cc mean | G1 cc SD | G1 all mean | G1 all SD | margin SD | corr(h,a) | total bias | PPP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| I | incumbent `reference` | +2.697 | -0.058 | +3.411 | -0.090 | 16.255 | 0.069 | +2.830 | 1.0391 |
| B3 | srfloor P3 + S1 | **+1.161** | -0.493 | **+1.666** | -0.582 | 15.878 | 0.018 | -0.934 | 1.0385 |
| B1 | srfloor P1 + S1 | +1.243 | -0.285 | +1.741 | -0.417 | 15.806 | 0.033 | -0.737 | 1.0388 |
| A3 | gamma P3 + S1 | +1.469 | +0.133 | +1.921 | +0.068 | 16.074 | 0.075 | -0.136 | 1.0402 |
| A2 | gamma P2 + S1 | +1.458 | +0.146 | +1.872 | +0.105 | 16.070 | 0.079 | -0.262 | 1.0400 |
| A1 | gamma P1 + S1 | +1.876 | +0.363 | +2.245 | +0.294 | 15.895 | 0.109 | +0.549 | 1.0402 |
| F | gamma P1, margin FROZEN | -2.592 | -0.329 | -2.176 | -0.352 | 15.724 | 0.045 | -9.381 | 1.0347 |

The 5-seed screening read agrees with every one of these to within 0.12
possessions. Noise floor (same arm, seeds +1000): G1 cc **0.089**, G1 all 0.055,
margin SD 0.158, correlation 0.008, total bias 0.261.

### 13.2 Verdict

**NO ARM ADOPTED. 0 of 6 pass criterion 1** (G1 mean inside +/- 1.0 on BOTH game
sets). The SD half of G1 passes for every arm; the MEAN half fails for every
arm, which is where G1 has been since round 1.

Criterion 2 (margin SD and correlation within the floor of the incumbent) is
reported in the evidence doc and is recorded as a **mis-specification in the
pre-registration**, not repaired: the floor measures seed-to-seed noise in margin
SD (0.158 on 12,500 rows) while any real clock change moves the possession count
and therefore the score variance, so a clock arm that satisfied it would be a
clock arm that did nothing. It changes no verdict -- criterion 1 already fails
everywhere -- so no gate was softened and nothing was adopted. The next
pre-registration states criterion 2 against the ACTUAL, or confines it to the
frozen-versus-live contrast it was borrowed from.

`ENGINE_CLOCK` stays `reference`; `provisional_clock` stays True. The best arm
the project has produced inside the engine is `empirical_km3_srfloor|P3|S1`, and
the pre-registered ordering would have chosen exactly it had it passed.

### 13.3 The question round 3b could not answer, answered

| contrast | offline (3b, G1-CC) | ENGINE (25 seeds, G1-CC) |
|---|---:|---:|
| gamma P1 -> P3 | -0.227 | **-0.407** |
| gamma P1 -> P2 | -0.038 | **-0.418** |
| gamma P2 -> P3 | -0.189 | +0.011 (tie, floor 0.089) |
| srfloor P1 -> P3 | -- | -0.082 (tie) |

Removing the simulation's own margin from the clock is worth about **0.42
possessions per team-game**, roughly double what the offline chain could see and
ten times what deleting it offline appeared to be worth. **P2 and P3 tie**: the
engine-safe end-game window buys nothing over deleting the margin outright, so
the late-game behaviour P3 exists to preserve is not reaching the possession
count. On the cell-based arm the P1/P3 gap is inside the floor, consistent with
L26's preference for cell-based arms.

### 13.4 The frozen arm over-corrects, and that revises how a Decision-10 ablation is read

Freezing the margin at zero swings the count by **4.468** possessions and the
game total by 9.9 points and lands 2.6 possessions and 9.4 points BELOW actual;
refitting the same family without the margin (P2) swings it by **0.418**. A
freeze feeds a model trained WITH the feature a value it barely saw in the states
that matter -- `score_diff = 0` with forty seconds left is a tie game, whose
fitted regime is "play normally" -- so the frozen model stops producing the short
intentional-foul possessions that end close games (its last-possession duration
goes to 15.34 s against an actual 11.89). **The freeze is a valid DETECTOR of a
loop and an invalid MEASUREMENT of one**; both it and a refit-without-the-feature
arm must run before a magnitude is quoted. L23 stands (fg_make's five refit state
arms agreed with its freeze), but the "+3.8 possessions of clock loop" quoted
from that ablation is an overstatement of what a refit recovers. The honest
engine figure is 0.42.

### 13.5 End of half, responsiveness, and what is left

- **End of half** (926 clock-complete regulation halves; truth from
  `actual_end_of_half_cc` + `eoh_stats`, the offline gate's own functions). The
  incumbent's mean last-possession duration is 5.88 s against an actual 11.89
  (**-6.01 s**); every round-3 arm cuts that to **-0.29 to -0.72 s**. That is the
  censoring fix (L20) arriving in the engine and it is the largest single
  improvement in the round. The SHARE is still high everywhere (0.94-0.99 vs
  0.876).
- **Responsiveness.** All six arms are monotone 4 of 4 across pregame
  tempo-prior quintiles, so nothing is flat. But the gamma family (1.376-1.400)
  and the incumbent (1.432) sit OUTSIDE Decision 8's [0.8, 1.2] band -- they
  spread fast and slow games 38-43% wider than reality -- while the two
  cell-based arms (1.061, 1.079) are inside it. A second axis, independent of
  G1, on which the cell arm is the better object; the offline read could not see
  it because the chain replays the real sequence of previous-end types.
- **The residual has an exact account.** Actual mean regulation possession
  duration on the subset is 17.555 s (clock-complete, 21,722 possessions). B3
  produces 17.216 s, -0.339 s = -1.93%, which implies +1.35 possessions against
  an observed +1.161; A3 -0.402 s implies +1.61 against +1.469; the incumbent
  -0.720 s implies +2.93 against +2.697. **The overshoot IS a uniform
  mean-duration shortfall of about a third of a second on ordinary possessions**
  -- not the horn (now within 0.3-0.7 s), not the margin loop (0.42, and removing
  it entirely leaves +1.46), not binning (cell arms are exact), not
  responsiveness.
- **Total bias inside +/- 1.0 is not a pass.** Every candidate runs +2.4 to +2.8%
  on possessions and -2.8 to -3.0% on PPP, and the two cancel. PPP is -0.03 on
  every arm INCLUDING the incumbent, so the per-possession shortfall is not the
  clock's and is not moved by any clock arm.

### 13.6 What round 4 must attack

A stated, measurable target: the conditional MEAN of the duration law is about
2% short across ordinary possessions. Candidates, in the order the evidence
supports: (a) time the engine never spends -- dead-ball and administrative
seconds the possession segmentation merges into ordinary possessions; (b) the
overtime gap, sim OT rate 2.6-4.2% against an actual 6.8%, which shortens
simulated games and is already an open defect; (c) NOT the `unknown`-terminal
rows, whose 6.48 s actual against a 19.31 s model-implied mean biases the count
the other way. Round 4 is a duration-LEVEL question, not a state-parametrisation
one, and the state question is now closed: P2 and P3 tie, both beat P1 by 0.42,
and the cell-based family is preferred on responsiveness as well as on binning.

---

## 14. Round 4 pre-registration -- the mean-duration shortfall (2026-09-11)

Appended VERBATIM BEFORE any round-4 arm was fitted or run, and committed on its
own together with the diagnosis it is written against. Sections 8, 10 and 12
(the round-3, 3b and 3c pre-registrations) are STATIC and are NOT edited.

Diagnosis: `docs/tests/clock_duration_shortfall_2026-09-11.md`. It fits no model
and scores no arm, and it revises L31's account of the residual:

- The train/serve quantity is CORRECT. The data's possessions tile each period
  exactly (`duration_s == start_clock - end_clock` on 768,834 of 768,834 rows;
  clock-complete regulation periods sum to 1199.58 s of 1200; 11 of 270,541
  clock-complete regulation possessions fall outside the design, worth -0.0036 s
  of the mean), and `loop.py` subtracts exactly `min(draw, seconds_remaining)`.
  There is no missing inbound component, no stoppage time, no OREB-split and no
  rounding term. Every one of those hypotheses is closed by the tiling identity.
- The shortfall is NOT uniform. On the 159 clock-complete subset games, 10
  seeds: total -0.2368 s = state COMPOSITION -0.1557, interaction -0.0580,
  the model's own LAW -0.0713, and +0.0482 given back by cells the engine
  reaches that these 159 real games never did.
- The composition term is ENTIRELY the `prev_end` mix (-0.158 of -0.156 on the
  joint grid): the engine starts 2.50 pp fewer possessions after a made field
  goal and 2.10 pp more after a defensive rebound, states whose durations differ
  by 7.5 s. **That is an upstream defect and round 4 does not touch it.**
- The clock's own share splits again: the `srfloor` floor in the 20-59 s band
  (2.9% of possessions, 37.5% of the offline law gap) and a SEASON-LEVEL drift
  the pooled cell fit under-tracks (2025 clock-complete mean 17.653 s against
  17.510 / 17.584 / 17.547 for 2024 / 2023 / 2022, and 17.374 in November
  against 17.836 in February). It is not cell sparsity: 99.0-100.0% of rows are
  served at the full five-dimension grid.

### 14.1 What round 4 decides, and what it does not

DECIDES: whether a cell-based clock arm that carries the CURRENT season's
duration level closes the law term without losing a gate. Family (cell-based),
censoring (L20/L26), state parametrisation (P3; closed by round 3c) and scheme
(S1; L21) are settled and are NOT reopened.

DOES NOT DECIDE, and is explicitly out of scope: the `prev_end` composition
term. Moving the clock to compensate for an upstream mix error is the pattern
`CLAUDE.md`'s bottom-up rule bans ("never accept a downstream stage that
compensates for a known upstream bias"). It is measured on every arm so that no
arm can pass by accidentally moving the mix, and it is logged for the event /
fg_make lane.

STATED IN ADVANCE: the law term is worth about 0.28 possessions per team-game in
the engine. An arm that removes ALL of it still lands near +0.6 to +0.9 on G1,
inside a gate that demands +/- 1.0 but only because the composition term does
not grow. A round-4 pass is therefore possible and a round-4 failure would not
be a surprise; neither outcome is allowed to move a gate.

### 14.2 Arms

All arms: cell-based, `P3` state, `S1` monthly walk-forward, censoring flag and
Kaplan-Meier tail rule exactly as round 3 (section 8.3). They differ only in how
the fit weights or partitions CALENDAR TIME, and in the clock-bucket floor.

| id | `ENGINE_CLOCK` | what it is |
|---|---|---|
| R | `v3c_srfloor_P3_s1` | **THE REFERENCE**, the arm served today. Re-run under round 4's own sub-model pinning, because round 3c ran `ENGINE_FG_MAKE=round2b_S_C_s1` and the interim served model is now `round3_shooter_S_C_s1` (L29); a table that mixed the two would not be a paired comparison |
| A1 | `v4_recency_P3_s1` | the same arm with EXPONENTIALLY RECENCY-WEIGHTED training rows: a row `d` days before the refit date carries weight `0.5 ** (d / H)` through a weighted discrete-time Kaplan-Meier. `H` is chosen on **F1 ONLY** from {120, 365, 730} days and recorded before F2 is touched. Fixes cross-season AND within-season level with one parameter |
| A2 | `v4_curseason_P3_s1` | the same arm with a two-level CALENDAR dimension appended LAST to the cell grid (0 = the season being simulated, 1 = prior seasons). The existing hierarchical fallback then serves a cell from the current season's own rows when it has >= 300 rows and >= 100 uncensored exits, and from the pooled cell otherwise. No new knob: `EMPIRICAL_MIN_CELL` and `EMPIRICAL_MIN_EVENTS` are round 3's |
| A3 | `v4_calpart_P3_s1` | the same arm with a three-level SEASON-PART dimension appended last, coded from the calendar month ({11,12} -> 0, {1} -> 1, {2,3,4} -> 2), pooling across seasons within a part. Fixes the within-season trend (L5) and not the cross-season level, so A1 vs A2 vs A3 separates the two causes |
| A4 | `v4_nofloor_P3_s1` | `empirical_km3` (NOT `srfloor`) under P3 + S1: the clock-bucket floor removed and nothing else changed. Prices the floor's own contribution to the law term inside the engine, which round 3 could only price offline |

A1's weight enters ONLY the fit. Nothing multiplies, scales, caps or offsets a
predicted duration or a simulated one; `docs/SIM_GUARDRAILS.md`'s core principle
and the standing no-hand-tuning rule forbid a multiplicative duration scaling,
and no arm here contains one.

### 14.3 Decision 10

No round-4 arm introduces a SIMULATION-PRODUCED state feature. A recency weight,
a season index and a calendar month are all known before tipoff and are
constants of the game, so a "frozen" arm and its "refit-without" counterpart are
the same object and the pair is VACUOUS. That is stated here rather than
silently skipped. The margin question -- the one Decision 10 exists for -- was
decided in round 3c with both arms run (frozen `gamma|P1` and the
refit-without-margin `gamma|P2`), and every round-4 arm inherits its answer, P3.
If any arm acquires an engine-produced state column before it is scored, BOTH
Decision-10 arms run for it and the arm is not read until they do.

### 14.4 Universe, folds, seal, subset, seeds, pairing

Unchanged from rounds 3 and 3c. D-I, hoopR not truncated, CBBD points-complete,
`DURATION_CAP = 90`; F1 trains {2022, 2023} and tests 2024, F2 trains
{2022, 2023, 2024} and tests 2025 and is the SELECTION fold; 2026 SEALED.
Closed loop: the F2 2025 slate sorted by `game_id` ascending, every 11th row,
first 500 games (159 clock-complete), 5 seeds screening and 25 seeds deciding,
paired by construction through the (seed, game_id, family) streams. Every other
sub-model pinned by explicit environment value on EVERY run and written into
`run_meta.json`: `ENGINE_EVENT=round2_s1`,
`ENGINE_FG_MAKE=round3_shooter_S_C_s1`, `ENGINE_FG3=decision8`,
`ENGINE_ROTATION=reference`. Each run records `git rev-parse HEAD` for `loop.py`
in `run_meta.json` (`loop_commit`).

### 14.5 Metrics

OFFLINE, exactly round 3's (section 8.4): CRPS_trunc on uncensored test
possessions with the pmf renormalised onto the reachable durations (PRIMARY
offline), censored log-likelihood, PIT K-S D per powered cell, the
`chain_halves` emergent G1 on clock-complete games, and the end-of-half pair.

CLOSED LOOP, the round-4 additions, on both game sets (the 159 clock-complete
games and all 500):

| id | metric |
|---|---|
| M1 | possessions per team-game, mean and SD, clock-complete games |
| M2 | possessions per team-game, mean and SD, all 500 games |
| M3 | margin SD across all (game, seed) rows |
| M4 | corr(home points, away points) |
| M5 | total bias against verified finals |
| M6 | points per possession |
| M7 | end-of-half: share of period-ending possessions starting inside 35 s, and their mean duration (the LAST-POSSESSION duration) |
| M8 | **mean regulation possession duration BY CELL** (previous end type x fine bucket x period x bonus x tempo tercile), and the law / composition / interaction decomposition against the same games' actual |
| M9 | possessions per team-game by pregame tempo quintile, with the Decision-8 slope ratio |

M8 is the round's own instrument and is reported for every arm including R. An
arm whose G1 improves through the COMPOSITION term rather than the LAW term has
not fixed the clock and is disqualified on that ground, stated here before any
arm exists.

### 14.6 Noise floor

The engine floor is the same arm re-run with `--seed-offset 1000`, computed for
R and for the leading candidate; the larger absolute delta per metric is the
floor for that metric. Round 3c's floors on the same subset and seed count are
carried as the prior expectation (G1 cc 0.089, G1 all 0.055, margin SD 0.158,
correlation 0.008, total bias 0.261) and are RE-MEASURED, not assumed. Offline,
the deterministic cell arms take the game-block bootstrap SE of mean CRPS_trunc.

### 14.7 Decision rule

Adopt the SIMPLEST arm satisfying BOTH:

1. G1 mean within +/- 1.0 and G1 SD within +/- 0.75 of the same games' actual on
   BOTH game sets (M1 and M2); and
2. margin SD (M3) within +/- 0.75 of the ACTUAL margin SD on the same games, and
   corr(home, away) reported. **This restates criterion 2 against the actual**,
   which section 13.2 recorded as a mis-specification in round 3c's
   pre-registration: a floor measured on seed-to-seed noise in margin SD would
   only be satisfiable by a clock arm that did nothing.

Simplicity order, fixed here: `R` (no calendar term at all) < `A4` (one fewer
cell dimension than R) < `A2` (one two-level dimension) < `A3` (one three-level
dimension) < `A1` (a fitted continuous weight). A tie is a gap inside the
measured floor and goes to the simpler arm. F1 is robustness only. If NO arm
qualifies, adopt nothing, report the diagnosis, and do not soften a gate. If the
leading arm fails only because the composition term did not move, say so and
leave `v3c_srfloor_P3_s1` served.

### 14.8 Execution and artifacts

New trainer `scripts/train_clock_v4.py`; new module
`src/cbb_sim/models/clock_v4.py` importing everything shared from `clock.py` and
`clock_v3.py`, so no round-3 object or pickle changes. Artifacts take a `v4_`
prefix in `data/processed/models/clock/` and are GITIGNORED and HF-synced.
Engine modes are added to the mode table in
`src/cbb_sim/engine/clock_adapter_v3.py`; `adapters.py`'s `_load_clock` hook
dispatches on prefix and is extended by one condition, nothing else. Engine
results: `results/engine_v0/clock4_<arm>/`, not committed. Threads capped at 4
and the engine pool at 8 workers; three other workers share the machine. Results
and the decision are appended to this file as section 15, `model.md` is updated
once, and a row goes to `docs/models/change_ledger.md`.

---

## 14. Correction to section 13 -- six arms were run on a pre-`6431772` engine loop and have been re-run (2026-09-11)

`experiments.md` is append-only, so section 13 stands as written and this section
supersedes its numbers. Nothing in its VERDICT or its ORDERING changes; the
numbers move in the third decimal to the second.

### 14.1 What happened

While round 3c was running, the rotation worker landed commit **`6431772`**
(2026-09-11 00:10:59 -0400) on `src/cbb_sim/engine/loop.py`: `push_lineups()`
moved after the period/halftime block, and a `rotation_sub` RNG family was added.
Both change which five are on the floor, hence usage, shot and foul draws, hence
every arm's output. A paired comparison whose arms straddle that change is not a
paired comparison.

Commit timestamps cannot settle this -- a working-tree edit precedes its commit by
an unknown interval -- so it was settled by **bit-identical reproduction**:
re-run a stored arm on today's code and compare `games.parquet` row for row.

| arm re-run | first produced | rows differing / 2500 | reading |
|---|---|---:|---|
| `clock3c_incumbent_s5` | 21:52 EDT (phase 1) | **1286** | OLD loop |
| `clkchk_gamma_P1_s5` | 22:49 EDT (phase 3) | 0 | new loop |
| `clock3c_srfloor_P3_s5` | 22:59 EDT (phase 3) | 0 | new loop |

The change landed inside the run sequence's own idle window -- phase 2, 22:23 to
22:45 EDT, while this worker was waiting on `train_clock_v3c_s1.py` with no
engine process running. That bounds it exactly: the six phase-1 runs are
old-loop, everything from phase 3 on is new-loop. The six (`incumbent_s5`,
`gamma_P3_s5`, both of their seed-offset floors, `incumbent_s25`, `gamma_P3_s25`)
were deleted and re-run. **All 19 runs now sit on the loop at `6431772`.**

### 14.2 The corrected deciding read (25 seeds, one loop)

| id | arm | G1 cc mean | G1 cc SD | G1 all mean | G1 all SD | margin SD | corr(h,a) | total bias | PPP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| I | incumbent `reference` | +2.743 | +0.096 | +3.443 | -0.037 | 16.459 | 0.047 | +3.184 | 1.0412 |
| B3 | srfloor P3 + S1 | **+1.161** | -0.493 | **+1.666** | -0.582 | 15.878 | 0.018 | -0.934 | 1.0385 |
| B1 | srfloor P1 + S1 | +1.243 | -0.285 | +1.741 | -0.417 | 15.806 | 0.033 | -0.737 | 1.0388 |
| A3 | gamma P3 + S1 | +1.446 | +0.109 | +1.899 | +0.114 | 16.275 | 0.062 | +0.394 | 1.0443 |
| A2 | gamma P2 + S1 | +1.458 | +0.146 | +1.872 | +0.105 | 16.070 | 0.079 | -0.262 | 1.0400 |
| A1 | gamma P1 + S1 | +1.876 | +0.363 | +2.245 | +0.294 | 15.895 | 0.109 | +0.549 | 1.0402 |
| F | gamma P1, margin FROZEN | -2.592 | -0.329 | -2.176 | -0.352 | 15.724 | 0.045 | -9.381 | 1.0347 |

Noise floor (same arm, seeds +1000): G1 cc **0.135**, G1 all 0.069, margin SD
0.276, correlation 0.030, total bias 0.658.

Only the two re-run arms moved: incumbent +2.697 -> **+2.743** cc and +3.411 ->
**+3.443** all; gamma P3 +1.469 -> **+1.446** cc and +1.921 -> **+1.899** all,
with its end-of-half duration gap improving -0.289 -> **-0.138 s** and its total
bias moving -0.136 -> **+0.394**. B3, B1, A2, A1 and F are unchanged because
they were already on the new loop.

### 14.3 What does and does not change

**Unchanged: the verdict.** 0 of 6 pass the G1 mean on either game set; all six
pass the G1 SD. Adopt nothing; `ENGINE_CLOCK` stays `reference`.

**Unchanged: the ordering.** B3 `srfloor|P3|S1` is still the best arm the project
has produced in the engine.

**Unchanged in substance, refined in value: the P-contrasts.** gamma P1 -> P3 is
now **-0.430** (was -0.407) and P1 -> P2 is -0.418, against a 0.135 floor;
P2 vs P3 is 0.012 apart, still a TIE. Removing the simulation's own margin from
the clock is worth **0.42 to 0.43** possessions per team-game.

**Changed: criterion 2 is satisfiable after all, and one arm satisfies it.** With
the corrected incumbent and the corrected floor (margin SD 0.276, correlation
0.030), A3 `gamma|P3` is inside BOTH halves (-0.184 and +0.015) and B3 and B1 are
inside the correlation half. Section 13.2 called criterion 2 "mis-specified"; that
reading was itself an artefact of the straddled incumbent row and is **withdrawn**.
The accurate statement is narrower: the margin-SD half is strict for the wrong
reason -- a clock model moves the possession count and the count drives score
variance, so an arm inside a seed-noise floor on margin SD would largely be an arm
that did nothing -- and the correlation half is the informative one, because that
is what L23's loop signature moves. The next pre-registration states the margin-SD
half against the ACTUAL, not against the incumbent. No gate was softened at any
point and nothing was adopted under either reading.

**Refined: the duration-level diagnosis.** Actual mean regulation possession
duration on the subset is 17.555 s (clock-complete, 21,722 possessions). B3
produces 17.216 s (-1.93%, implying +1.35 against an observed +1.161); A3
produces 17.166 s (-2.22%, implying +1.55 against +1.446); the incumbent 16.829 s
(-4.14%, implying +2.95 against +2.743). The conclusion stands and is if anything
tighter: **what is left is a uniform duration LEVEL bias of about a third of a
second on ordinary possessions**, not the horn, not the loop, not binning, not
responsiveness.

**Refined: end of half.** The incumbent's mean last-possession duration is 5.92 s
against an actual 11.89 (**-5.97 s**); every round-3 arm is within **-0.14 to
-0.72 s**, with `gamma|P3` best at -0.138. P3 beating P2 by 0.49 s there, while
tying it on the possession count, says the end-game behaviour P3 preserves is
real but is not what the count is missing.

### 14.4 The procedural lesson

Because the mixed set was found and replaced before any of it was read as
evidence, nothing downstream was ever graded on a straddled design -- but that
was luck of timing, not a control. Two cheap controls follow: **`run_meta.json`
should record the engine commit hash**, and **a long run sequence should re-run
its earliest arm against its latest code and assert bit-identity before the table
is read**. Both are cheaper than the four hours of engine time this cost.

---

## 15. Run R7 -- the round-4 grid (2026-09-11)

`scripts/train_clock_v4.py` (module `src/cbb_sim/models/clock_v4.py`) and
`scripts/run_clk4_closed_loop.py`, graded by the SAME blind path round 3c used
(`scripts/grade_clk3c_closed_loop.py --pattern "clock4_*"`). Pre-registration
section 14 committed **421b97b** BEFORE `clock_v4.py` or the trainer existed.
Diagnosis: `docs/tests/clock_duration_shortfall_2026-09-11.md`.

Every engine run: `ENGINE_EVENT=round2_s1`,
`ENGINE_FG_MAKE=round3_shooter_S_C_s1`, `ENGINE_FG3=decision8`,
`ENGINE_ROTATION=reference`, passed as explicit environment values and recorded
in every `run_meta.json`; `loop.py` at commit **4503c52** on all 17 round-4 runs
(verified from the metas, so the paired design holds). Subset: the F2 2025 slate
sorted by `game_id` ascending, every 11th row, first 500 games; 159
clock-complete. Actual on that subset: 68.530 possessions per team-game
clock-complete, 68.328 all, margin SD 15.472, corr(h,a) 0.237, PPP 1.0705.

Artifacts `v4_*` in `data/processed/models/clock/` (gitignored, HF-synced);
engine results `results/engine_v0/clock4_*` (not committed).

### 15.1 The half-life, chosen on F1 ONLY

| half-life (days) | F1 CRPS_trunc | F1 censored loglik | F1 pred mean duration |
|---:|---:|---:|---:|
| 120 | 4.85862 | -3.51668 | 17.469 |
| **365** | **4.85814** | -3.51095 | 17.514 |
| 730 | 4.85821 | -3.51035 | 17.528 |

**Chosen: 365 days**, by the pre-registered rule (lowest CRPS_trunc on F1). The
spread is 0.0005, well inside any floor, so the rule is doing the choosing and
not the data; recorded as such. F1 actual mean duration 17.419.

### 15.2 F2 offline

| id | arm | CRPS_trunc | censored loglik | pred mean dur | PIT worst D | PIT leak cells | G1-CC chain | eoh dur gap | slope ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R | `srfloor` (served) | 4.89946 | -3.50999 | 17.615 | 0.1553 | 16 | +0.662 | -1.065 | 0.786 |
| A1 | `recency` (H=365) | 4.89903 | -3.51086 | 17.591 | 0.1503 | 15 | +0.751 | -1.130 | 0.780 |
| A2 | `curseason` | 4.90023 | -3.54133 | 17.559 | 0.1474 | 16 | +0.884 | -1.047 | 0.756 |
| A3 | `calpart` | 4.89983 | -3.53240 | 17.619 | 0.1585 | 17 | +0.657 | -1.176 | 0.785 |
| A4 | `nofloor` | **4.88748** | -3.50713 | 17.609 | 0.1107 | 13 | +1.048 | **-3.216** | 0.787 |

Offline noise floor, game-block bootstrap SE of mean CRPS_trunc over 5,319 test
games, measured per arm: 0.00656-0.00684; **floor = 0.00684**. A4 beats R by
0.01198 = **1.75 floors**; A1 beats R by 0.00043 (0.06 floors, a tie); A3 and A2
lose by 0.04 and 0.08 floors (ties). PIT fails for every arm, as it has since
round 1, and the end-of-half duration floor is +/- 0.2538 s, which every arm
misses and A4 misses by 12.7 floors.

**The direct offline read of what round 4 exists to fix** -- the model's own
expected CONSUMED duration `E[min(T,R)]` against the actual, on all 270,530
clock-complete 2025 regulation possessions (actual mean 17.6526 s):

| arm | E[min(T,R)] | gap | implied possessions per team-game |
|---|---:|---:|---:|
| R `srfloor` | 17.4965 | -0.1561 | +0.606 |
| A1 `recency` | 17.4698 | **-0.1828** | +0.711 |
| A2 `curseason` | 17.4368 | **-0.2158** | +0.841 |
| A3 `calpart` | 17.5022 | -0.1505 | +0.584 |
| A4 `nofloor` | **17.5262** | **-0.1264** | +0.490 |

**Two of the three calendar arms make the law term WORSE, and the reason is
structural.** S1's "current season to date" is by construction the part of the
season BEFORE the game being served, and duration rises monotonically through
the season (2025 clock-complete: 17.374 s in November, 17.409 in December,
17.793 in January, 17.836 in February). Conditioning on the current season
(A2) or up-weighting recent rows (A1) therefore anchors the fit on the FASTEST
available part of the current season and pulls the prediction DOWN, while the
pooled multi-season fit the reference uses silently contains February and March
of three prior seasons. Only A3, which pools by SEASON PART across seasons, points
the right way, and it is worth 0.006 s. `v4_ref` and the served
`v3c_srfloor_P3_s1` produce identical E[min(T,R)] to four decimals on those
270,530 rows, which is the harness's own identity check.

### 15.3 The deciding closed-loop read (25 seeds, paired streams, 500 games)

| id | arm | G1 cc mean | G1 cc SD | G1 all mean | G1 all SD | margin SD | corr(h,a) | total bias | PPP | slope ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R | `v3c_srfloor_P3_s1` (served) | **+1.156** | -0.477 | **+1.704** | -0.599 | 15.941 | 0.003 | -0.931 | 1.0380 | 1.047 |
| A1 | `v4_recency_P3_s1` | +1.254 | -0.443 | +1.787 | -0.595 | 15.838 | 0.009 | -0.706 | 1.0384 | 1.040 |
| A2 | `v4_curseason_P3_s1` | +1.346 | -0.510 | +1.902 | -0.584 | 15.904 | 0.005 | -0.434 | 1.0386 | 1.027 |
| A3 | `v4_calpart_P3_s1` | **+1.127** | -0.468 | **+1.677** | -0.586 | 15.861 | 0.011 | -0.890 | 1.0387 | 1.049 |
| A4 | `v4_nofloor_P3_s1` | +1.561 | -0.468 | +2.108 | -0.567 | 15.947 | 0.013 | -0.198 | 1.0373 | 1.052 |

The 5-seed screening read agrees with every row to within 0.20 possessions and
preserves the ordering at the top (A3 < R < A1 < A2 < A4).

**Noise floor** (the same arm re-run with seeds 1000-1024, measured for R and
for A4): G1 cc **0.180**, G1 all **0.074**, margin SD 0.101, corr 0.013, total
bias 0.377. (R: +1.156 -> +0.999 cc, +1.704 -> +1.630 all. A4: +1.561 -> +1.381
cc, +2.108 -> +2.038 all.) The G1-cc floor is twice round 3c's 0.089 on the same
subset and seed count, which is itself worth recording: the floor is re-measured
per round, never assumed.

### 15.4 M8 -- law versus composition, per arm

`ENGINE_CLOCK_DIAG=1` on the 159 clock-complete games, 10 seeds, decomposed by
`scripts/diag_clk4_composition.py` against the same games' real possessions on
the round-2 cell grid (regulation; the identity closes to 3e-15):

| id | arm | mean consumed | total gap | LAW | COMPOSITION | interaction | sim-only cells |
|---|---|---:|---:|---:|---:|---:|---:|
| R | `srfloor` | 17.334 | -0.222 | **-0.071** | -0.148 | -0.052 | +0.049 |
| A1 | `recency` | 17.309 | -0.247 | -0.102 | -0.147 | -0.045 | +0.048 |
| A2 | `curseason` | 17.268 | -0.287 | -0.134 | -0.158 | -0.047 | +0.052 |
| A3 | `calpart` | 17.316 | -0.240 | -0.074 | -0.160 | -0.055 | +0.049 |
| A4 | `nofloor` | 17.221 | -0.334 | **-0.051** | **-0.291** | -0.040 | +0.048 |

This is the round's most useful table and it says three things no other read
does:

1. **A1 and A2 degrade the law term in the engine too** (-0.102 and -0.134
   against R's -0.071), confirming 15.2's structural explanation rather than an
   offline artefact.
2. **A4 is the only arm that improves the law term** (-0.051, a 28% reduction),
   exactly as the diagnosis predicted for removing the `srfloor` approximation
   -- and it is still the WORST arm on G1, because removing the floor doubles
   the COMPOSITION term (-0.291 against -0.148). Shorter end-of-period
   possessions (last-possession duration 9.12 s against an actual 11.89, R
   11.19) manufacture extra possessions at the horn, which dilutes the made-FG
   start share further. That is L20's mechanism arriving from the other
   direction, and it is why the floor stays.
3. **The composition term is untouched by every arm** (-0.147 to -0.160 for the
   four floored arms), as pre-registered. It is not the clock's.

### 15.5 Verdict

**NO ARM ADOPTED. 0 of 5 pass criterion 1** (G1 mean inside +/- 1.0 on BOTH game
sets). Criterion 2 passes for every arm (margin SD 15.84-15.95 against an actual
15.472, inside +/- 0.75; corr(h,a) 0.003-0.013 against 0.237 and reported, not
gated). Responsiveness passes for every arm (slope ratio 1.027-1.052, inside
Decision 8's [0.8, 1.2], monotone 4 of 4).

A3 `calpart` is the best arm at +1.127 / +1.677, and it **ties** the reference:
the gaps are 0.029 (cc, floor 0.180) and 0.027 (all, floor 0.074). The
pre-registered tie-break orders `R` before `A3`, so the simpler arm keeps the
slot. A1 and A2 are no better than R or worse; A4 is worse than R by 0.405
possessions, far outside the floor.

`ENGINE_CLOCK` therefore stays **`v3c_srfloor_P3_s1`** (the engine default since
commit 1a5acef), `provisional_clock` stays True, and nothing was hand-tuned,
capped, scaled or blended at any point.

### 15.6 What this round establishes, beyond the verdict

- **The train/serve quantity is not the defect** and every "missing component"
  hypothesis is closed by the tiling identity (section 14's diagnosis, four
  independent checks).
- **The clock's own share of the possession overshoot is about 0.28 possessions
  per team-game** (the -0.071 s law term), and the best any round-4 arm managed
  was 0.020 s of it -- at the cost of 0.144 s of composition.
- **The remaining +1.0 to +1.7 is the `prev_end` mix**, i.e. the event /
  fg_make lane: the engine starts 2.50 pp fewer possessions after a made field
  goal and 2.10 pp more after a defensive rebound, worth -0.148 s. Round 5 for
  the CLOCK has no target left that is worth a round; the target moved
  upstream, and that is logged for the event / fg_make lane rather than
  compensated for here.
- **In-season recency is anti-correlated with the within-season trend.** Any
  scheme that up-weights "what has happened so far this season" is up-weighting
  November when it is serving February. S1's calibration benefit (L21) comes
  from seeing the current season's LEVEL, not from a monotone trend, and for a
  target that trends monotonically within a season a recency weight moves the
  prediction the wrong way. That is a general result for every sub-model whose
  target drifts through a season, not a clock curiosity.

---

## 16. Round 5 pre-registration -- within-game possession-duration DISPERSION (2026-09-11)

Appended VERBATIM BEFORE any round-5 arm was fitted or scored, and committed on
its own together with the measurement it is written against. Sections 8, 10, 12
and 14 (the round-3, 3b, 3c and 4 pre-registrations) are STATIC and are NOT
edited. `experiments.md` is append-only.

**A NEW TARGET.** Rounds 3, 3b, 3c and 4 were all about the conditional MEAN
duration (and through it the possession COUNT mean). Round 4 closed that line:
section 15.6 records that the clock's own share of the mean overshoot is about
0.28 possessions per team-game and that the remaining +1.0 to +1.7 is the
upstream `prev_end` mix, so "round 5 for the CLOCK has no target left that is
worth a round". That sentence was written about the MEAN. It is wrong about the
VARIANCE, and this round exists because a different diagnostic found the
variance target three hours later.

Diagnosis this round is written against:
`docs/tests/engine_v1_variance_ot_diag_2026-09-11.md` (engine lane, commit
`4ea6237`), sections 2, 3.3 and 3.5:

- the engine's per-team-game possession SD is **3.743 produced against 4.972
  needed**, 25% short in SD and 43% short in variance;
- the shortfall is **WITHIN-game, not between-game**: the slope of actual
  possessions on the sim's per-game mean is 0.857, so the between-game pace
  draw is if anything 17% OVER-spread and widening it would make calibration
  worse;
- the possession channel carries **29.8% of the TOTAL's within-game variance**
  and 0.2% of the margin's, so G5's margin PASS is structurally blind to this
  and only the total line sees it;
- widening the possession draw to its own residual alone closes **40.8%** of
  the G5 total-SD gap (0.7985 -> 0.8862) and moves `corr(home, away)` from
  +0.027 to +0.111.

Measurement this round is written against (own lane, this session):
`scripts/diag_clk5_dispersion.py`, report
`data/processed/models/clock/v5_diag/v5_dispersion_report.json`, written up in
`docs/tests/clock_duration_dispersion_2026-09-11.md`. It fits nothing and scores
no arm. Its result is stated here because the arms are chosen against it:

| channel | share of the Var(Dbar) gap | measured |
|---|---:|---|
| (a) conditional law too narrow | **1.1%** | served-arm conditional SD 8.873 s against an actual residual SD 8.894 s, ratio **0.9977** |
| (b) missing within-game correlation | **98.9%** | per-game mean-residual variance 1.2349 s2 against an iid sampling floor of 0.5914 s2; game latent tau = **0.802 s**, CV **4.54%** |
| (c) `prev_end` composition (L34) | **-0.3%** | L34's engine mix shifts the marginal mixture variance by -0.258 s2, i.e. the WRONG WAY and negligibly |

and the bridge that converts them, an exact identity plus one delta-method step
checked at ratio 0.995 against the realised counts:

    P = 1200 / Dbar  on a clock-complete regulation game
    SD(P) ~= (Pbar / mu) * SD(Dbar) = 3.851 * SD(Dbar)

    produced 2.943   needed 4.279   ratio 0.688   (offline, 1,991 games)
    (the engine's own 3.743 / 4.972 = 0.753 on the same defect; the offline
    "produced" excludes the engine's across-seed state-composition feedback and
    is therefore a lower bound on it)

### 16.1 What round 5 decides, and what it does not

DECIDES: whether adding a WITHIN-GAME duration dependence -- as a random effect,
as an autoregressive residual, or as a wider conditional family -- closes the
per-game possession-count SD gap without moving the conditional mean or losing
the round-4 mean gate.

DOES NOT DECIDE, and is explicitly out of scope: the conditional mean and the
possession-count MEAN (round 4, closed); the `prev_end` composition term
(upstream, logged for the event / fg_make lane); the family, censoring, state
parametrisation and refit scheme (rounds 1-4, settled: cell-based,
`empirical_km3_srfloor`, `P3`, `S1`).

STATED IN ADVANCE, because it is arithmetic and it pre-commits the round to a
prediction that can fail:

1. **The implied average within-game pairwise residual correlation is 0.00807.**
   `Var(Dbar) = (sigma2/M)(1 + (M-1) rho_bar)`; needed/floor = 2.0883 at
   M = 135.88, so `rho_bar = 1.0883/134.88 = 0.00807`. A tiny per-pair number
   that is worth 45% of an SD in aggregate.
2. **An AR(1) residual CANNOT deliver it.** For AR(1) with lag-1 rho the
   variance of the mean inflates by about `(1+rho)/(1-rho)`; reaching 2.0883
   needs `rho ~= 0.35`. The measured lag-1 residual autocorrelation is
   **-0.0195** and the measured lag-2 is **+0.0350**. Arm A4 is therefore
   pre-registered with the PREDICTION THAT IT FAILS, and it is run anyway
   because the prediction is worth testing and because the alternative is
   asserting it.
3. **The measured autocorrelation does not decay like an AR process.** Even
   lags (the same offence's next possession) read +0.0350, +0.0326, +0.0317,
   +0.0304, +0.0274, +0.0193, +0.0114 at lags 2, 4, 6, 8, 12, 20, 40; odd lags
   (the opponent's next possession) read -0.0195, -0.0072. A near-flat positive
   floor at every even lag out to 40 possessions is the signature of a LEVEL,
   not of an autoregression.
4. **The level is the OFFENCE's, not the game's, and the two offences in a game
   are NEGATIVELY coupled.** The two offences' mean residuals inside one game
   correlate **-0.172** (covariance -0.514 s2); each offence's own latent
   variance is 2.361 (home) and 2.297 (away) s2. Netting to the game level,
   `(2.361 + 2.297)/4 - 0.514 = 0.651 s2`, which reproduces the measured
   tau2 = 0.651 exactly. So a SHARED per-game latent and an INDEPENDENT
   per-offence latent are observationally equivalent for the possession COUNT
   (which depends only on the average of the two) and are NOT equivalent for
   the per-offence duration structure. Both are run, and the bivariate arm that
   carries both parameters is run alongside them.
5. **CRPS and PIT are nearly blind to the winning mechanism.** A latent with
   CV 4.5% widens the marginal per-possession SD from 8.873 s to about 8.91 s,
   under 0.5%. Any round that selected on CRPS alone would select nothing. The
   primary metric is therefore the per-game possession-count SD, and
   CRPS_trunc / PIT are NO-REGRESSION lines. This is stated before any arm is
   scored precisely so that the metric cannot be chosen after the fact.

### 16.2 Arms

All arms: the round-4 reference artifact set (`empirical_km3_srfloor`, `P3`,
`S1` monthly walk-forward, censoring flag and Kaplan-Meier tail rule exactly as
round 3 section 8.3) unless the row says otherwise. They differ ONLY in the
dependence structure imposed on the draws, and in A5 in the conditional family.

| id | name | what it is |
|---|---|---|
| R | `v3c_srfloor_P3_s1` | **THE REFERENCE**, the arm served today: independent inverse-CDF draws, `clock.sample_from_pmf`, no within-game dependence of any kind |
| A1 | `v5_glat_shared` | ONE multiplicative log-normal pace realisation `A ~ LogN(-s2/2, s2)` per (seed, game), `E[A] = 1`, applied to BOTH offences' durations: `D = round(A * T)`. This is `CLAUDE.md`'s "one pace realisation per simulated game, both teams scaled by it", which the clock does not currently implement. `s` fitted walk-forward |
| A2 | `v5_glat_team` | the same latent drawn per (seed, game, offence team), the two offences INDEPENDENT. `s` fitted walk-forward |
| A3 | `v5_glat_biv` | the per-offence latent with a fitted CORRELATION `rho_t` between the two offences of a game (bivariate Gaussian copula on the two log-latents). Two fitted parameters; the only arm that can carry both the per-offence dispersion and the negative coupling |
| A4 | `v5_ar1` | Gaussian-copula AR(1) on the duration residual within (game, offence): `z_i = rho z_{i-1} + sqrt(1-rho2) e_i`, `u_i = Phi(z_i)`, then the SAME inverse-CDF draw. Marginals are preserved exactly, so CRPS and PIT are identical to R by construction. `rho` fitted walk-forward. PRE-REGISTERED PREDICTION: fails item 2 above |
| A5 | `v5_gamma_P3_s1` | round 3c's already-fitted `gamma_aft` under `P3` + `S1` -- a parametric, heavier-tailed conditional family in place of the empirical cell pmf, with NO within-game dependence. Prices channel (a) directly: if the conditional family is the defect, this arm moves and the latent arms are unnecessary |
| A6 | `v5_glat_tempo` | A1's shared latent with `s` a fitted linear function of the game's PREGAME tempo (`tempo_prior_game`, both teams), i.e. a dispersion function of state rather than a constant. Measured motivation: the latent CV rises 4.11 / 3.36 / 3.98 / 4.58 / 5.40 % across pregame-tempo quintiles Q1..Q5 |

**No arm contains a post-hoc multiplier, cap, clip, offset or blend.** A1, A2,
A3 and A6 are scale-mixture random-effects models with `E[A] = 1` by
construction, so the CONDITIONAL MEAN of every possession is unchanged and no
arm can move the round-4 mean gate in its own favour; A4 is a copula that
preserves every marginal exactly; A5 is a different fitted family. Every fitted
parameter (`s`, `rho_t`, `rho`, A6's two coefficients) is estimated on training
rows only, under the same `max_train_date < game_date` rule the artifacts
already pass, and is a MODEL PARAMETER, not an adjustment to engine output.
`docs/SIM_GUARDRAILS.md` core principle and section 5, and the standing
no-hand-tuning rule, are the tests this paragraph is written against.

**Fitting rule, fixed here.** Method of moments on the fold's TRAINING seasons
only, per S1 refit `k`, using only rows with `game_date <= max_train_date_k` and
artifact `k`'s own predictions:

    s2   solves   mean_g[ V_iid(g)(1 + s2) + s2 * mbar_g^2 ] = Var_g(rbar)      (A1)
    s2   solves   the same with the latent variance halved by averaging two
                  independent offences                                          (A2)
    (s2, rho_t)   match the per-offence latent variance AND the offence-pair
                  covariance                                                     (A3)
    rho           the lag-1 within-(game, offence) residual autocorrelation      (A4)
    A6            s2(tempo) = a + b * tempo_prior_game, by weighted least
                  squares of the per-game latent estimate on tempo

Nothing in the fitting touches the test fold. F1's fitted values are recorded
before F2 is touched and are reported as robustness.

### 16.3 Decision 10

No round-5 arm introduces a SIMULATION-PRODUCED state feature. A pace latent is
drawn from the `(seed, game_id, family)` stream before the game starts and is a
constant of the game; an AR(1) residual depends on the model's own previous
DRAW, not on any other sub-model's output, so no other sub-model can feed it.
A "frozen" arm and its "refit-without" counterpart are therefore the same
object and the Decision-10 pair is VACUOUS, exactly as in round 4. Stated rather
than silently skipped. The margin question Decision 10 exists for was decided in
round 3c (P3) and every round-5 arm inherits that answer.

### 16.4 Universe, folds, seal, subset, seeds, pairing

Unchanged from rounds 3, 3c and 4. D-I, hoopR not truncated, CBBD
points-complete, `DURATION_CAP = 90`. **F1 trains {2022, 2023} and tests 2024;
F2 trains {2022, 2023, 2024} and tests 2025 and is the SELECTION fold. The
2026 season is SEALED.** Offline universe: the 2025 clock-complete regulation
possessions, 270,530 rows over 1,991 games, the same set round 4 section 15.2
read `E[min(T,R)]` on. Closed loop, if it runs: the F2 2025 slate sorted by
`game_id` ascending, every 11th row, first 500 games (159 clock-complete),
5 seeds screening, paired by construction through the
`(seed, game_id, family)` streams, with every other sub-model pinned by explicit
environment value on EVERY run and written into `run_meta.json`
(`ENGINE_EVENT=round2_s1`, `ENGINE_FG_MAKE=round3_shooter_S_C_s1`,
`ENGINE_FG3=decision8`, `ENGINE_ROTATION=reference`).

### 16.5 Metrics

**PRIMARY (offline, F2):** the per-game possession-count SD the arm produces,
through the bridge of section 16, against the 4.279 the same games need:

    P_sd_produced(arm) = 3.851 * sqrt( Var_arm(Dbar) )
    ratio              = P_sd_produced / 4.279          target 1.00

reported as `|ratio - 1|`. `Var_arm(Dbar)` is computed from the arm's own law by
the closed form each arm admits, and is CROSS-CHECKED for the winning arm by a
direct Monte-Carlo resample of the real 2025 state sequences.

**NO-REGRESSION LINES (offline, F2), each with its own floor:**

| id | metric | round-4 reference value | rule |
|---|---|---:|---|
| N1 | `CRPS_trunc` on uncensored test possessions, pmf renormalised onto the reachable durations (round 3's PRIMARY offline metric, section 8.4) | R = 4.89946 | must not worsen by more than 1 floor (round 4's floor 0.00684) |
| N2 | `E[min(T,R)]` on all 270,530 clock-complete 2025 regulation possessions, against an actual 17.6526 s -- **THE ROUND-4 MEAN GATE** | R = 17.4965, gap -0.1561 | the gap must not worsen by more than 1 floor |
| N3 | PIT worst-cell K-S D and the count of leaking cells | R = 0.1553, 16 cells | reported; must not worsen materially |
| N4 | possessions per team-game MEAN implied by `E[min(T,R)]` | R = +0.606 | must not worsen by more than 1 floor |

**SECONDARY / STRUCTURAL (offline, F2), the discriminators between A1, A2 and
A3, all of which hit the primary by construction:**

| id | metric | actual |
|---|---|---:|
| S1 | per-offence mean-residual variance | 2.361 (home), 2.297 (away) s2 |
| S2 | correlation between the two offences' mean residuals in a game | -0.172 |
| S3 | within-(game, offence) residual autocorrelation at lags 2, 4, 8, 20, 40 | +0.0350, +0.0326, +0.0304, +0.0193, +0.0114 |

**MULTI-LEVEL EVIDENCE (the standing rule), reported for every arm:** overall;
by `prev_end` cell; by round-2 clock bucket; by period; by pregame-tempo
quintile (the responsiveness check -- the produced/needed SD ratio must SLOPE
with the quintile's own needed value, not sit flat); and the per-game
distribution of the mean residual. Cells with fewer than 300 possessions are
labelled UNDERPOWERED and are not read.

**CLOSED LOOP, if it runs (5 seeds, 500 games, paired):** G1 possessions per
team-game mean and SD on both game sets, G5 total SD ratio, `corr(home, away)`,
and the round-4 mean lines. A new `ENGINE_CLOCK` value behind the existing
dispatch; **the served default is NOT changed by this lane under any outcome.**

### 16.6 Noise floor

Offline, the arms are deterministic given the fitting window, so the floor is a
SPEC-IDENTICAL REFIT under a second seed: the fitting window is resampled by
GAME-BLOCK bootstrap (games drawn with replacement, all of a game's possessions
moving together) under `seed = 20260911` and again under `seed = 20260912`, each
refit is graded by the same blind path, and the larger absolute delta per metric
is that metric's floor. CRPS_trunc additionally carries round 4's measured
game-block bootstrap SE, 0.00684. Closed loop, if it runs: the same arm re-run
with `--seed-offset 1000`, round 4's measured floors carried as the prior
expectation (G1 cc 0.180, G1 all 0.074, margin SD 0.101, corr 0.013) and
RE-MEASURED, never assumed.

### 16.7 Decision rule

Adopt the SIMPLEST arm satisfying ALL of:

1. **PRIMARY**: `|P_sd_produced / 4.279 - 1| <= 0.10` on F2, and the improvement
   over R exceeds the measured floor;
2. **N1, N2, N4**: no no-regression line worse than R by more than 1 floor;
3. **responsiveness**: the produced/needed SD ratio slopes with the
   pregame-tempo quintile (monotone in at least 4 of 5, and no quintile outside
   [0.85, 1.15]);
4. **S1 and S2 reported for every arm**; where two arms tie on the primary
   inside the floor, the arm closer to the measured S1/S2 structure wins.

Simplicity order, fixed here: `R` (no dependence at all) < `A5` (a different
single family, no dependence) < `A4` (one fitted scalar) < `A1` (one fitted
scalar, one latent) < `A2` (one fitted scalar, two latents) < `A6` (two fitted
coefficients) < `A3` (two fitted parameters, a bivariate latent). A tie is a gap
inside the measured floor and goes to the simpler arm.

If NO arm qualifies, adopt nothing, report the decomposition, and do not soften
a gate. **An offline winner is a CANDIDATE, not an adoption**: `CLAUDE.md`
requires a paired-seed sim run showing no gate regressed before anything ships,
and this lane does not change the served default in any case -- the PM switches
it.

### 16.8 Execution and artifacts

New module `src/cbb_sim/models/clock_v5.py` (the latent / copula wrappers,
importing everything shared from `clock.py`; no round-3 or round-4 object or
pickle changes); new fitter+grader `scripts/exp_clk5_dispersion_bakeoff.py`,
which fits every arm and scores every arm through ONE code path so no arm gets
a bespoke scorer. Measurement script `scripts/diag_clk5_dispersion.py`
(committed with this pre-registration; it fits nothing). Artifacts take a `v5_`
prefix under `data/processed/models/clock/` and are gitignored and HF-synced.
Engine modes, if the closed loop runs, are added to the mode table in
`src/cbb_sim/engine/clock_adapter_v3.py`; `adapters.py` DEFAULTS ARE NOT
TOUCHED. Threads capped at 6 and the engine pool at 6 workers; four other lanes
share the machine. Results and the decision are appended to this file as
section 17, the evidence goes to
`docs/tests/clock_duration_dispersion_2026-09-11.md`, and a row goes to
`docs/models/change_ledger.md` only if something is adopted.

---

## 17. Run R8 -- the round-5 dispersion grid (2026-09-11)

`scripts/exp_clk5_dispersion_bakeoff.py` (module
`src/cbb_sim/models/clock_v5.py`), scored by the SAME blind path rounds 3, 3b,
3c and 4 used (`clock_v3.score_arm_v3`, unedited), with the primary metric
reduced by ONE piece of variance algebra applied identically to the reference
and to every arm. Pre-registration section 16 committed **28b5f17** BEFORE
`clock_v5.py`, the bake-off script or any fitted parameter existed.
Measurement: `scripts/diag_clk5_dispersion.py` (same commit; fits nothing).
Evidence, multi-level: `docs/tests/clock_duration_dispersion_2026-09-11.md`.

Universe: the 2025 clock-complete regulation possessions, **270,530 rows over
1,991 games**, the same set section 15.2 read `E[min(T,R)]` on. Needed
per-team-game possession SD on that set: **4.2795**
(`3.8509 * sqrt(Var(rbar)) = 3.8509 * sqrt(1.23494)`), the bridge checked
against the realised counts at ratio **0.9950**.

Artifacts `v5_*` under `data/processed/models/clock/` (gitignored, HF-synced).
**No engine run. No served default changed.**

### 17.1 The measurement the arms were chosen against

| channel | share of the `Var(Dbar)` gap | measured |
|---|---:|---|
| (a) conditional law too narrow | **+1.1%** | served conditional SD 8.8731 s against an actual residual SD 8.8940 s, **ratio 0.99765**; every powered `prev_end` cell inside 1.0% |
| (b) missing within-game correlation | **+98.9%** | `tau = 0.8022 s`, CV 4.545%; implied average pairwise residual correlation **0.00807** |
| (c) `prev_end` composition (L34) | **-0.3%** | L34's mix moves the mixture variance by **-0.258 s2**, the WRONG way |

Produced 2.9430 against needed 4.2795, **ratio 0.6877** (the engine's own
reading of the same defect is 3.743/4.972 = 0.753; the offline "produced"
excludes across-seed state-composition feedback and is a lower bound).

### 17.2 F2 offline -- the deciding table

| id | arm | **P SD produced** | **ratio** | CRPS_trunc | cens. loglik | PIT worst D | leak cells | `E[min(T,R)]` | mean gap | poss delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R | `v3c_srfloor_P3_s1` (served) | 2.9430 | **0.6877** | 4.92746 | -3.51919 | 0.4306 | 20 | 17.4965 | -0.1561 | +0.607 |
| **A1** | `v5_glat_shared` | **4.2980** | **1.0043** | 4.92777 | -3.50994 | 0.4307 | 20 | 17.5041 | -0.1485 | +0.577 |
| A2 | `v5_glat_team` | 4.3103 | 1.0072 | 4.92825 | -3.51112 | 0.4327 | 20 | 17.5196 | -0.1331 | +0.516 |
| A3 | `v5_glat_biv` | 4.3330 | 1.0125 | 4.92876 | -3.51106 | 0.4327 | 21 | 17.5159 | -0.1367 | +0.531 |
| A4 | `v5_ar1` | 3.0518 | 0.7131 | 4.92746 | -3.51919 | 0.4306 | 20 | 17.4965 | -0.1561 | +0.607 |
| A5 | `v5_gamma_P3_s1` | 3.3539 | 0.7837 | **4.97216** | **-3.57150** | **0.4712** | **34** | 17.4892 | -0.1635 | +0.635 |
| A6 | `v5_glat_tempo` | 4.3103 | 1.0072 | 4.92776 | -3.50984 | 0.4295 | 20 | 17.5044 | -0.1482 | +0.576 |

CRPS_trunc is read on the clock-complete subset, so its LEVEL is not comparable
to section 15.2's 4.89946 (the full 5,319-game F2 slice); the arm-to-arm
DIFFERENCES are, and those are what the no-regression line uses.

Fitted parameters, F2 train {2022, 2023, 2024} with F1 {2022, 2023} beside it:
`A1 sigma 0.047248 / 0.046959`; `A2 sigma 0.066741 / 0.066333`;
`A3 sigma 0.077680 / 0.077476, rho_t -0.2620 / -0.2680`;
`A4 rho 0.036823 / 0.034095`; `A6 b0 -0.002335 / -0.002749,
b1 6.70e-05 / 7.30e-05`. **F1 and F2 agree to the third decimal on every
parameter**: the dispersion is a property of the sport, not of a season.

### 17.3 Floor, and the Monte-Carlo cross-check

Two spec-identical refits on game-block bootstrap resamples of the F2 training
window (seeds 20260911 / 20260912): `sigma` 0.047493 / 0.047107, P SD produced
4.3217 / 4.2975. **Floor = 0.0242 possessions.** CRPS_trunc carries section
15.2's own measured game-block SE, 0.00684.

**A1 beats R by 1.3550 possessions = 56.0 floors.** A4 beats R by 4.5 floors
and is still 0.29 short of the target.

`scripts/diag_clk5_mc_check.py`, 25 replicate draws over the real 2025 state
sequences through `clock.sample_from_pmf`'s own inverse-CDF construction:
R closed form 2.9430 against MC 2.9551 (+0.41%), A1 4.2980 against 4.2684
(-0.69%), A2 4.3103 against 4.3434 (+0.77%). All inside the MC's own noise:
**the closed form is validated, the primary metric is not an artefact of the
algebra.**

### 17.4 Responsiveness -- and where every arm still fails

`produced / needed` per pregame-tempo quintile, all five powered (398-399 games
and 50,840-57,542 possessions each):

| arm | Q1 slow | Q2 | Q3 | Q4 | Q5 fast | worst |
|---|---:|---:|---:|---:|---:|---:|
| R | 0.711 | 0.793 | 0.732 | 0.684 | 0.631 | 0.369 |
| A1 | 1.035 | **1.168** | 1.067 | 0.997 | 0.919 | 0.168 |
| A2 | 1.039 | **1.172** | 1.070 | 0.999 | 0.921 | 0.172 |
| A3 | 1.044 | **1.178** | 1.076 | 1.005 | 0.926 | 0.178 |
| A4 | 0.737 | 0.823 | 0.759 | 0.709 | 0.654 | 0.346 |
| A5 | 0.826 | 0.900 | 0.838 | 0.774 | 0.706 | 0.294 |
| A6 | **1.004** | **1.157** | 1.071 | 1.012 | **0.958** | 0.157 |

The served arm is progressively worse the faster the game (0.711 -> 0.631): a
fast game has more possessions, so iid sampling averages more of the
conditional variance away and the missing level is a larger share of what is
left. A6's fitted tempo dependence is real -- it is the only arm inside 5% at
both ends -- but **every latent arm sits at 1.157-1.178 in Q2, outside the
pre-registered [0.85, 1.15] band.** Q2's needed dispersion is a genuine 4-SE
dip below the tempo trend, not noise, so the miss is not waived.

### 17.5 Verdict

**NO ARM ADOPTED. `ENGINE_CLOCK` stays `v3c_srfloor_P3_s1`,
`provisional_clock` stays True, and nothing was hand-tuned, capped, scaled or
blended at any point.** Two independent grounds:

1. **Criterion 3 fails for every arm that passes criterion 1.** A1, A2, A3 and
   A6 land the primary inside +/-1.3% against a +/-10% requirement and pass
   every no-regression line, and all four are outside the [0.85, 1.15]
   per-quintile band at Q2. A failing criterion is not softened.
2. **No closed-loop run exists**, and `CLAUDE.md` requires a paired-seed sim
   run showing no gate regressed before an offline winner ships (section 17.7).

**A1 `v5_glat_shared` is the leading CANDIDATE**: primary 0.688 -> **1.0043**,
56 floors, at the cost of ONE fitted scalar, with CRPS_trunc +0.00031 against a
0.00684 floor, PIT unchanged (worst D +0.0001, leak cells 20 -> 20), the
censored log-likelihood BETTER (-3.5099 against -3.5192), and round 4's mean
gate IMPROVED rather than merely held (`E[min(T,R)]` 17.4965 -> 17.5041, gap
-0.1561 -> -0.1485, implied possessions +0.607 -> +0.577; that is `round(A*T)`
interacting with the horn truncation, measured not assumed, and it is not why
the arm is preferred).

### 17.6 The two negative results, both pre-registered before they ran

- **A4 (AR(1)) fails exactly as predicted.** Section 16.1 item 2 put the
  required `rho` at ~0.35 against a measured lag-1 of -0.0195 and PREDICTED the
  failure in writing before the arm existed; it delivered 0.7131. The measured
  autocorrelation is a near-flat positive floor at every EVEN lag out to 40
  possessions (+0.0350, +0.0326, +0.0317, +0.0304, +0.0274, +0.0193, +0.0114)
  with odd lags at zero or slightly negative -- **a persistent LEVEL, not an
  autoregression, and not a shared-clock effect either.**
- **A5 (a heavier-tailed conditional family) fails on every line**: primary
  0.7837, CRPS_trunc 6.5 floors WORSE, PIT leak cells 20 -> 34, worst D
  0.4306 -> 0.4712, mean gap worse. **Channel (a) is confirmed not to be the
  defect by an arm built to exploit it**, which is stronger evidence than the
  0.99765 conditional-SD ratio on its own.

### 17.7 A pre-registration defect, RECORDED rather than applied

Criterion 4 of section 16.7 breaks a primary tie on the S1/S2 structure (the
per-offence latent variance 2.361/2.297 and the offence-pair correlation
-0.172). A1, A2 and A6 do tie on the primary inside the 0.0242 floor, so the
criterion fires, and it would select A2 over A1. It must not, and the reason is
arithmetic: on a clock-complete game the possessions ALTERNATE and tile 2400 s,
so `n_h*Dbar_h + n_a*Dbar_a = 2400` and the SUM of the two offences' mean
durations is a deterministic function of the possession count while the
DIFFERENCE is free. In those coordinates the variance splits

    (Dbar_h + Dbar_a)/2   ->  1.2349   the pace level: THE possession count
    (Dbar_h - Dbar_a)/2   ->  1.7487   which offence played slower: no gate reads it

so S1 and S2 are dominated by a component that is orthogonal to the target and
partly mechanical. `CLAUDE.md`'s standing rule -- ties go to the simpler model
-- is the one that applies, and it selects A1. This is the same class of error
section 13.2 recorded for round 3c's criterion 2; the criterion stands in the
append-only pre-registration and this section records that it was found
mis-specified, in the same commit as the result.

### 17.8 What this round establishes, beyond the verdict

- **The clock's conditional law is right and its JOINT law is missing.** The
  one-possession distribution is within 0.25% in SD and within 1% in every
  powered `prev_end` cell; the 135-possession aggregate is 31% short. An
  average pairwise correlation of **0.008** is the entire defect. A sub-model
  can be perfectly calibrated per row and badly wrong per game, and no
  per-row scoring rule will see it -- CRPS_trunc separates R from A1 by
  0.00031, which is 4% of its own floor. **Whenever a sub-model's draws are
  aggregated inside a game, the round that fits it must carry an aggregate
  metric; a per-row metric is structurally blind to the dependence.**
- **The mean defect and the variance defect are independent and point at
  different owners.** L34's `prev_end` mix owns two thirds of the MEAN
  shortfall (round 4) and -0.3% of the VARIANCE shortfall, with the wrong sign.
  Fixing the upstream mix would make the dispersion marginally worse. Round 4's
  "round 5 for the CLOCK has no target left" was true of the mean and false of
  the variance; a lane that closes one moment has not closed the others.
- **`CLAUDE.md`'s "one pace realisation per simulated game, both teams scaled
  by it" is not implemented for durations, and implementing it is worth 56
  floors.** The engine draws every possession independently; there is no
  game-level pace realisation anywhere in the clock path.
- **The dispersion is a stable property of the sport.** Every fitted parameter
  agrees to the third decimal between F1 and F2 -- unlike the round-4 calendar
  arms, whose whole difficulty was that the LEVEL drifts within a season.
- **The remaining defect is state-dependence of the dispersion.** The needed SD
  slopes +32% from the slowest to the fastest tempo quintile and a constant-CV
  latent reproduces +17% of that; A6's fitted tempo coefficient closes both
  ends and still misses Q2 by 16%. That is the target a round 6 would have, and
  it is a DISPERSION-function question, not a family question.

## 18. PROPOSED -- Round 5 closed-loop gate: the PACE-EFFICIENCY lines (written 2026-09-11 by the ENGINE lane; NOT YET RUN, NOT YET ACCEPTED by this lane)

**Status: PROPOSED. Nothing below has been run. No arm is selected here, no
served default is touched, and this section selects nothing -- round 5's
selection is section 17's and is already closed on fold 2.** This is a proposal
for the *closed-loop gate* that round 5's winner has to pass before it can ship,
covering a consequence of the same defect that section 16's pre-registration
does not measure. The clock lane owns it and may accept, amend or reject it.

Evidence and the full derivation: `docs/tests/pace_efficiency_sign_2026-09-11.md`.

### 18.1 Why the engine lane is proposing a clock gate

`docs/tests/engine_v1_variance_ot_diag_2026-09-11.md` section 4 found that the
engine's within-game `corr(possessions, eFG%)` is **-0.1976** where the season
reads +0.0454, priced it at 39% of the G5 total-SD gap, and ranked **fg_make**
as the responsible sub-model. Re-measured like for like (the season
matchup-residualised, which is the object the engine's across-seed read
actually corresponds to), the target is **+0.0050**, not +0.0454, and the
attribution comes out differently:

| slope of eFG on P, per possession | SIM (within game) | ACTUAL (matchup-resid.) | gap | share |
|---|---:|---:|---:|---:|
| total | -0.001399 | +0.000034 | -0.001433 | 100% |
| COMPOSITION channel (outcomes -> pace) | -0.001660 | -0.001384 | -0.000276 | 19.3% |
| **RESIDUAL-PACE channel (pace not caused by outcomes)** | **+0.000262** | **+0.001418** | **-0.001156** | **80.7%** |

`Var(P_resid)` within game is **40.17** against the season's **90.19**, and its
efficiency slope is 0.18x. That is section 17's missing game-level pace
realisation, read from the efficiency side instead of the dispersion side:
because durations are i.i.d., 73% of the engine's within-game pace variance is
clock draw noise and 27% is outcome composition, with no exogenous tempo
component at all, so the correlation is forced negative.

Three things the engine lane measured that bound this to the clock and rule out
the alternatives, all in the evidence doc:

- **fg_make is not responsible.** Its served transition lift is **1.63x its own
  design's**, not short: design `is_transition_f` lift +7.80 / +1.98 / +0.60 pp
  by class against a served +11.26 / +5.29 / +0.79 pp. The +32.5 pp figure that
  makes fg_make look short comes from the possessions table's POST-OUTCOME
  `is_transition` / `duration_s`, the column the change ledger bans at L5; on
  fg_make's own `chance_elapsed_s` the lift is +6.5 pp and the engine serves
  +9.4 pp.
- **The composition LEVER is right to 2.9%**: made_FG minus DREB duration is
  **7.066 s** in the engine against **7.274 s** in the season. The 19.3% row
  above is L34's `prev_end` MIX (made-FG starts -3.0 pp, DREB starts +2.1 pp),
  already owned, and no new round is proposed for it here.
- **The transition population and its pace responsiveness are right**:
  transition share 0.1587 against 0.1642, `corr(P, transition share)` +0.5793
  against +0.5035, `d(trans share)/dP` +0.002692 against +0.002440.

So the engine's own SERVED efficiency already rises with pace
(`corr(P, served eFG) = +0.2260`); the negative sign is entirely the
realisation channel, and the reason it is not offset is that the engine has no
pace variation the outcomes did not cause.

### 18.2 The proposed gate

1. **Candidates.** **A1 `v5_glat_shared`** (section 17's winner, `sigma` FROZEN
   at its fold-fitted 0.047248 -- no refit in the loop) against **R
   `v3c_srfloor_P3_s1`** (the served reference). **A2 `v5_glat_team`** as the
   per-team variant and **A4 `v5_ar1`** as the within-game-correlation
   alternative that adds no latent, both report-only, so the round can say
   whether a SHARED latent specifically is what moves the efficiency line.
   Paired streams, identical seeds, one engine commit.
2. **Primary metric.** Within-game (across-seed) **`corr(P, eFG%)`**, both teams
   pooled, regulation, against the matchup-residualised season target
   **0.000**, band **+-0.05**. Reported ALWAYS beside the served-eFG slope from
   the per-possession tap (`scripts/diag_pace_efficiency_poss_log_v1.py`), so
   the state channel and the realisation channel are never conflated again --
   that conflation is what produced the original mis-attribution.
3. **Folds and universe.** Selection is section 17's and is not re-opened. This
   is a **Decision-10 closed-loop gate** on the engine's F2/2025 slate, 500
   games x 25 seeds, paired, every `ENGINE_*` flag pinned to the values in
   `results/engine_v0/F2_2025_s200_rewire1/run_meta.json` and the engine commit
   recorded in the run report. Per L31, report BOTH the freeze and the
   refit-without instrument, and state whether the frozen value is inside the
   model's operating range.
4. **Floor.** (a) A spec-identical refit of A1 under a second fit seed --
   section 17.3 already has it at 0.0242 possessions; (b) **the seed-offset
   floor for `corr(P, eFG%)` itself, which has never been measured and MUST be
   measured before the primary metric is read** -- two disjoint seed windows of
   the SAME configuration, the method
   `docs/tests/gates_pair_seedfloor_20_2026-09-11.md` used for G1-G9. No
   movement is called a finding until that band exists.
5. **Decision rule.** A1 ships only if **all four** hold: (a) `corr(P, eFG%)`
   moves toward zero by more than its measured floor; (b) no G1-G9 gate line
   regresses beyond its own floor; (c) the per-team-tempo and per-team-scoring
   quintile tables stay flat -- the engine's are flat at -0.19 to -0.21 across
   all ten cells today, and a fix that moves the slate mean by concentrating the
   change in one tier is a FAIL; (d) the served-eFG slope stays positive. Ties
   go to R, per CLAUDE.md. **No multiplier, cap, clip, offset or calibration
   curve on eFG%, on any make rate, or on the possession count is an admissible
   response to any outcome of this round.**

**Expected movement, stated in advance so the round cannot be read
retrospectively.** Arithmetic on the measured decomposition, NOT a
re-simulation: taking within-game `Var(P)` from 54.92 to 98.9 at the engine's
own measured served slope (+0.000279/possession, a LOWER bound for a
multiplicative latent) moves `corr(P, eFG%)` from **-0.1976 to about -0.124**,
closing 37% of the distance to zero. **A1 is therefore expected to IMPROVE the
line and NOT to close it**, and an outcome near -0.12 is a pass on the
pre-registered rule, not a disappointment. If A1 lands at or beyond 0.00 the
arithmetic was wrong and the round must say so.

### 18.3 Report-only, not a gate

Re-measure the residual after A1: the arithmetic leaves `corr(P, eFG%)` at
about -0.071 once the pace latent and L34's `prev_end` mix are both corrected.
That residual corresponds to the engine's served efficiency response to pace
being +0.000279 where the season's non-composition channel is +0.001418. It is
**not attributable to any sub-model today**, because the engine has almost no
exogenous pace variation for its models to respond to and the served slope is
measured on a mixture that is 73% i.i.d. noise. Naming an owner before the
latent ships would repeat the mis-attribution this section corrects. Round 5's
closed loop should print the number and stop there.
