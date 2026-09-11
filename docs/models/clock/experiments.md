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
