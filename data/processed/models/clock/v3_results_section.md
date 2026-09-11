
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
