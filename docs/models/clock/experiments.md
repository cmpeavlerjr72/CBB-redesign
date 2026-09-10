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
