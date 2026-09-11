# Possession outcome, Stage A: is the surviving residual structure organised by the calendar or by the conference boundary?

Diagnostic for `ARCHITECTURE_DECISIONS.md` Decision 9, run 2026-09-10 by the possession-outcome worker BEFORE the round-3 pre-registration was written. No model is selected here and nothing is adopted.

Scored population: the **round-2 S1 winners** re-scored on fold 2 with the round-2 code path -- `lgbm` (first chances) and `cascade` (continuation chances), feature set `C_plus_state`, scheme S1 (calendar-monthly), seed 0, through `cbb_sim.models.possession_outcome.fit_predict_scheme`. Round 2 did not persist per-chance predictions, so they were regenerated; the run asserts the reproduced F2 log loss equals round 2's recorded value to 1e-6 and stops otherwise.

---

## Population `first` (lgbm, S1 monthly, n = 742,025 chances over 5,445 games)

Reproduced F2 log loss 1.515428 against round 2's recorded 1.515428 -- exact match, so this is round 2's model.
Games with no hoopR conference id on either side: 0 chances (treated as non-conference and reported, never imputed as conference).

### calendar week of the test season (`calendar_week`)

Bucket-mean residual, predicted minus actual, percentage points. Positive = the model over-predicts that class in that bucket.

| w_cal | n | n_games | underpowered | resid_TOV | resid_FGA_rim | resid_FGA_jump2 | resid_FGA_3 |
|---|---|---|---|---|---|---|---|
| 0 | 36499 | 258 |  | 0.166 | 0.769 | 1.169 | -1.852 |
| 1 | 40718 | 292 |  | 0.412 | 0.307 | 1.482 | -1.652 |
| 2 | 43658 | 317 |  | 0.138 | -0.029 | 1.715 | -1.547 |
| 3 | 43562 | 318 |  | 0.099 | 0.916 | 0.651 | -1.466 |
| 4 | 38103 | 279 |  | -0.306 | -0.445 | -0.076 | 0.745 |
| 5 | 23434 | 171 |  | -0.421 | -0.300 | 0.668 | -0.199 |
| 6 | 37840 | 277 |  | -0.291 | -0.287 | -0.343 | 0.996 |
| 7 | 11855 | 85 |  | -0.598 | -0.561 | 1.233 | -0.009 |
| 8 | 39059 | 286 |  | -0.266 | -0.273 | 0.483 | 0.049 |
| 9 | 43883 | 322 |  | -0.202 | -0.195 | 0.065 | 0.148 |
| 10 | 44008 | 324 |  | -0.078 | -0.163 | 0.215 | 0.191 |
| 11 | 45408 | 337 |  | 0.080 | -0.372 | 0.264 | -0.030 |
| 12 | 44086 | 328 |  | -0.095 | -0.135 | 0.268 | -0.127 |
| 13 | 43658 | 324 |  | 0.088 | -0.064 | 0.024 | -0.110 |
| 14 | 33263 | 247 |  | -0.292 | -0.626 | 0.643 | 0.216 |
| 15 | 41947 | 311 |  | -0.167 | -0.944 | 0.828 | 0.049 |
| 16 | 46972 | 347 |  | -0.324 | -0.779 | 0.604 | 0.364 |
| 17 | 42312 | 314 |  | 0.132 | -0.239 | 0.175 | -0.047 |
| 18 | 25213 | 187 |  | 0.219 | 0.924 | -0.640 | -0.552 |
| 19 | 10605 | 78 |  | 0.690 | 1.038 | -0.524 | -1.109 |
| 20 | 2993 | 22 | UNDERPOWERED | -0.367 | -1.654 | 1.552 | 0.777 |
| 21 | 2815 | 20 | UNDERPOWERED | 0.119 | 1.224 | 0.623 | -0.689 |
| 22 | 134 | 1 | UNDERPOWERED | -0.735 | 3.416 | 2.734 | -6.500 |

### week relative to the OFFENCE team's first conference game (`conf_rel_off`)

Bucket-mean residual, predicted minus actual, percentage points. Positive = the model over-predicts that class in that bucket.

| w_rel_off | n | n_games | underpowered | resid_TOV | resid_FGA_rim | resid_FGA_jump2 | resid_FGA_3 |
|---|---|---|---|---|---|---|---|
| -8.000 | 22388 | 263 |  | 0.267 | -0.144 | 2.318 | -2.112 |
| -7.000 | 25721 | 298 |  | 0.577 | -0.405 | 1.717 | -1.732 |
| -6.000 | 31605 | 379 |  | 0.038 | 1.037 | 1.159 | -1.748 |
| -5.000 | 30960 | 381 |  | 0.108 | 0.638 | 0.237 | -0.970 |
| -4.000 | 31536 | 381 |  | -0.038 | 0.690 | 0.562 | -0.836 |
| -3.000 | 34360 | 424 |  | -0.304 | -0.397 | 0.755 | 0.020 |
| -2.000 | 33065 | 410 |  | -0.300 | 0.050 | 0.520 | 0.059 |
| -1.000 | 18764 | 240 |  | -0.116 | -0.505 | 0.840 | -0.297 |
| 0.000 | 44688 | 361 |  | -0.102 | -0.582 | 0.049 | 0.497 |
| 1.000 | 36757 | 348 |  | -0.473 | -0.374 | 0.344 | 0.439 |
| 2.000 | 38187 | 351 |  | -0.325 | -0.004 | 0.126 | 0.064 |
| 3.000 | 39117 | 355 |  | 0.154 | -0.568 | 0.630 | -0.179 |
| 4.000 | 43395 | 365 |  | 0.172 | -0.493 | 0.305 | 0.012 |
| 5.000 | 44553 | 384 |  | -0.094 | 0.058 | 0.232 | -0.112 |
| 6.000 | 39161 | 342 |  | -0.207 | -0.915 | 0.472 | 0.516 |
| 7.000 | 42364 | 370 |  | -0.309 | -0.697 | 0.548 | 0.185 |
| 8.000 | 45009 | 380 |  | -0.245 | -0.146 | 0.038 | 0.252 |
| 9.000 | 41554 | 358 |  | 0.246 | 0.667 | -0.709 | -0.129 |
| 10.000 | 23835 | 230 |  | -0.195 | 0.049 | 0.588 | -0.382 |
| 11.000 | 21586 | 214 |  | 0.145 | -0.649 | 0.779 | -0.400 |
| 12.000 | 17825 | 167 |  | -0.170 | -0.273 | 0.394 | -0.417 |

### week relative to the DEFENCE team's first conference game (`conf_rel_def`)

Bucket-mean residual, predicted minus actual, percentage points. Positive = the model over-predicts that class in that bucket.

| w_rel_def | n | n_games | underpowered | resid_TOV | resid_FGA_rim | resid_FGA_jump2 | resid_FGA_3 |
|---|---|---|---|---|---|---|---|
| -8.000 | 22390 | 263 |  | 0.257 | 0.526 | 2.081 | -2.377 |
| -7.000 | 25751 | 298 |  | 0.403 | -0.076 | 1.589 | -1.479 |
| -6.000 | 31598 | 379 |  | 0.290 | 0.517 | 1.190 | -1.616 |
| -5.000 | 30972 | 381 |  | -0.194 | 0.222 | 0.533 | -0.566 |
| -4.000 | 31537 | 381 |  | -0.318 | 0.163 | 1.004 | -0.668 |
| -3.000 | 34363 | 424 |  | -0.052 | 0.366 | 0.232 | -0.346 |
| -2.000 | 33076 | 410 |  | -0.025 | 0.131 | 0.597 | -0.372 |
| -1.000 | 18738 | 240 |  | -0.326 | -0.393 | 1.083 | -0.241 |
| 0.000 | 44709 | 361 |  | -0.180 | -0.419 | 0.009 | 0.496 |
| 1.000 | 36748 | 348 |  | -0.253 | -0.549 | 0.251 | 0.409 |
| 2.000 | 38177 | 351 |  | -0.496 | 0.054 | 0.344 | 0.093 |
| 3.000 | 39134 | 355 |  | 0.173 | -0.597 | 0.383 | -0.016 |
| 4.000 | 43373 | 365 |  | 0.063 | -0.348 | 0.334 | 0.008 |
| 5.000 | 44547 | 384 |  | 0.043 | -0.066 | 0.260 | -0.363 |
| 6.000 | 39163 | 342 |  | -0.257 | -0.845 | 0.424 | 0.581 |
| 7.000 | 42349 | 370 |  | -0.347 | -0.661 | 0.754 | 0.026 |
| 8.000 | 45010 | 380 |  | -0.275 | -0.296 | 0.169 | 0.328 |
| 9.000 | 41538 | 358 |  | 0.165 | 0.603 | -0.632 | -0.175 |
| 10.000 | 23840 | 230 |  | 0.101 | -0.003 | 0.292 | -0.237 |
| 11.000 | 21596 | 214 |  | 0.229 | -0.658 | 0.653 | -0.290 |
| 12.000 | 17830 | 167 |  | 0.101 | -0.654 | 0.453 | -0.406 |

### conference vs non-conference game (`conf_flag`)

Bucket-mean residual, predicted minus actual, percentage points. Positive = the model over-predicts that class in that bucket.

| is_conf_game | n | n_games | underpowered | resid_TOV | resid_FGA_rim | resid_FGA_jump2 | resid_FGA_3 |
|---|---|---|---|---|---|---|---|
| no | 277908 | 2011 |  | 0.009 | 0.189 | 0.851 | -0.829 |
| yes | 464117 | 3434 |  | -0.092 | -0.303 | 0.253 | 0.062 |

### weeks since the monthly S1 refit boundary (`weeks_since_refit`)

Bucket-mean residual, predicted minus actual, percentage points. Positive = the model over-predicts that class in that bucket.

| w_since_refit | n | n_games | underpowered | resid_TOV | resid_FGA_rim | resid_FGA_jump2 | resid_FGA_3 |
|---|---|---|---|---|---|---|---|
| 0 | 186506 | 1369 |  | -0.003 | -0.150 | 0.140 | -0.013 |
| 1 | 186348 | 1364 |  | -0.085 | 0.064 | 0.490 | -0.425 |
| 2 | 167177 | 1225 |  | -0.146 | -0.276 | 0.628 | -0.147 |
| 3 | 155425 | 1146 |  | 0.020 | -0.119 | 0.558 | -0.427 |
| 4 | 46569 | 341 |  | -0.057 | -0.162 | 0.954 | -0.628 |

### Per-decile calibration gap, overall vs the first four weeks of conference play (offence-team alignment)

| segment | n | n_games | worst_gated_decile_gap_pp |
|---|---|---|---|
| overall | 742025 | 5445 | 0.980 |
| non-conference games | 277908 | 2011 | 2.492 |
| conference games | 464117 | 3434 | 0.979 |
| first 4 conf weeks (w_rel_off 0..3) | 158749 | 1334 | 1.166 |
| conf weeks 4+ (w_rel_off >= 4) | 344423 | 2601 | 1.016 |
| pre-boundary (w_rel_off < 0) | 238853 | 1846 | 2.945 |

### The Mississippi Valley State table: raw-centred style rates at the conference boundary, by own-rating quintile

One row per team-game on the offence side; quintiles are of the team's season-mean as-of `off_rating_off_c` (a DESCRIPTIVE stratifier, not a model feature). `shift` = conference mean minus non-conference mean, in the feature's own centred units (per-100-possession for 3PA and TOV, percentage-point share for rim and FTr).

| quintile | n_team_games | n_teams | own_rating_mean | n_nonconf | n_conf | off_3pa_c_nonconf | off_3pa_c_conf | off_3pa_c_shift | off_rim_c_nonconf | off_rim_c_conf | off_rim_c_shift | off_tov_c_nonconf | off_tov_c_conf | off_tov_c_shift | off_ftr_c_nonconf | off_ftr_c_conf | off_ftr_c_shift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.000 | 2064.000 | 73.000 | -8.850 | 750.000 | 1314.000 | -1.199 | -0.955 | 0.244 | -1.274 | -1.061 | 0.213 | 1.546 | 1.437 | -0.109 | -2.896 | -1.321 | 1.576 |
| 2.000 | 2080.000 | 73.000 | -4.260 | 746.000 | 1334.000 | -0.308 | -0.424 | -0.116 | -1.762 | -1.224 | 0.538 | 0.618 | 0.493 | -0.125 | -0.794 | -0.680 | 0.115 |
| 3.000 | 2118.000 | 72.000 | -0.560 | 786.000 | 1332.000 | -1.082 | -0.655 | 0.427 | 1.639 | 1.158 | -0.481 | 0.026 | 0.021 | -0.005 | 1.360 | 0.543 | -0.817 |
| 4.000 | 2220.000 | 73.000 | 3.400 | 811.000 | 1409.000 | 0.681 | 1.046 | 0.366 | 0.442 | 0.340 | -0.102 | -0.415 | -0.468 | -0.053 | 2.261 | 0.714 | -1.547 |
| 5.000 | 2408.000 | 73.000 | 10.350 | 929.000 | 1479.000 | 1.519 | 1.122 | -0.396 | 0.711 | 0.468 | -0.243 | -1.609 | -1.331 | 0.278 | 1.877 | 1.058 | -0.818 |

---

## Population `cont` (cascade, S1 monthly, n = 111,906 chances over 5,445 games)

Reproduced F2 log loss 1.499760 against round 2's recorded 1.499760 -- exact match, so this is round 2's model.
Games with no hoopR conference id on either side: 0 chances (treated as non-conference and reported, never imputed as conference).

### calendar week of the test season (`calendar_week`)

Bucket-mean residual, predicted minus actual, percentage points. Positive = the model over-predicts that class in that bucket.

| w_cal | n | n_games | underpowered | resid_TOV | resid_FGA_rim | resid_FGA_jump2 | resid_FGA_3 |
|---|---|---|---|---|---|---|---|
| 0 | 5569 | 258 |  | -0.334 | 1.346 | 1.040 | -1.803 |
| 1 | 6183 | 292 |  | -0.299 | 1.742 | -0.759 | -0.578 |
| 2 | 6805 | 317 |  | -0.158 | 0.588 | 0.704 | -1.278 |
| 3 | 6616 | 318 |  | -0.239 | 0.228 | 0.514 | -1.087 |
| 4 | 5805 | 279 |  | 0.172 | 0.244 | 1.150 | -1.159 |
| 5 | 3473 | 171 | UNDERPOWERED | 1.189 | -0.143 | -0.180 | -1.159 |
| 6 | 5725 | 277 |  | 0.250 | 0.271 | -0.085 | -0.824 |
| 7 | 1822 | 85 | UNDERPOWERED | 1.294 | -0.218 | -0.516 | -0.222 |
| 8 | 5935 | 286 |  | 0.119 | 1.631 | -0.918 | -0.434 |
| 9 | 6856 | 322 |  | -0.607 | 0.683 | 0.289 | -0.348 |
| 10 | 6693 | 324 |  | -0.668 | 0.969 | 0.049 | -0.672 |
| 11 | 6990 | 337 |  | -1.000 | 2.431 | -0.259 | -1.309 |
| 12 | 6642 | 328 |  | -0.419 | 0.990 | 0.251 | -1.294 |
| 13 | 6556 | 324 |  | 0.021 | 0.525 | 0.131 | -0.800 |
| 14 | 4774 | 247 | UNDERPOWERED | 0.996 | -0.707 | -0.671 | -0.099 |
| 15 | 6068 | 311 |  | 1.127 | -1.277 | 0.198 | -0.687 |
| 16 | 6714 | 347 |  | 0.505 | -1.557 | 0.252 | -0.143 |
| 17 | 6286 | 314 |  | 0.255 | 0.569 | 0.020 | -1.065 |
| 18 | 3881 | 187 | UNDERPOWERED | 0.044 | 0.680 | 0.099 | -0.395 |
| 19 | 1659 | 78 | UNDERPOWERED | 1.390 | 0.706 | -2.085 | -0.103 |
| 20 | 459 | 22 | UNDERPOWERED | 4.305 | -3.698 | 0.928 | -0.964 |
| 21 | 373 | 20 | UNDERPOWERED | 1.091 | -5.839 | 3.465 | -0.738 |
| 22 | 22 | 1 | UNDERPOWERED | -3.556 | 1.029 | -11.366 | 5.546 |

### week relative to the OFFENCE team's first conference game (`conf_rel_off`)

Bucket-mean residual, predicted minus actual, percentage points. Positive = the model over-predicts that class in that bucket.

| w_rel_off | n | n_games | underpowered | resid_TOV | resid_FGA_rim | resid_FGA_jump2 | resid_FGA_3 |
|---|---|---|---|---|---|---|---|
| -8.000 | 3384 | 262 | UNDERPOWERED | -0.413 | 0.861 | 1.287 | -1.224 |
| -7.000 | 3839 | 298 | UNDERPOWERED | 0.256 | 1.244 | 0.038 | -1.454 |
| -6.000 | 4960 | 379 | UNDERPOWERED | -0.459 | -0.115 | 1.380 | -1.677 |
| -5.000 | 4627 | 381 | UNDERPOWERED | 0.505 | 1.475 | 0.169 | -1.919 |
| -4.000 | 4838 | 381 | UNDERPOWERED | -0.080 | 0.789 | -0.383 | -0.524 |
| -3.000 | 5258 | 424 |  | -0.060 | -0.103 | -0.117 | -0.452 |
| -2.000 | 5062 | 410 |  | -0.656 | 1.807 | 1.066 | -1.917 |
| -1.000 | 2809 | 240 | UNDERPOWERED | 0.623 | -0.742 | -0.424 | 0.492 |
| 0.000 | 6785 | 361 |  | 0.134 | -0.196 | 0.296 | -0.300 |
| 1.000 | 5753 | 348 |  | -0.324 | 1.428 | 0.291 | -0.868 |
| 2.000 | 5854 | 351 |  | -0.258 | 0.481 | -0.415 | -0.395 |
| 3.000 | 6049 | 355 |  | -0.304 | 1.339 | 0.352 | -1.043 |
| 4.000 | 6530 | 365 |  | -0.160 | 1.309 | -0.591 | -0.852 |
| 5.000 | 6613 | 384 |  | -0.198 | 0.298 | 0.132 | -0.814 |
| 6.000 | 5650 | 342 |  | -0.488 | 0.235 | -0.050 | 0.249 |
| 7.000 | 6352 | 370 |  | 1.024 | 0.232 | -0.386 | -1.536 |
| 8.000 | 6458 | 380 |  | 0.089 | 0.506 | -0.095 | -0.424 |
| 9.000 | 6185 | 358 |  | 0.442 | 0.621 | -0.149 | -0.714 |
| 10.000 | 3654 | 230 | UNDERPOWERED | 0.568 | -0.456 | -0.281 | -0.048 |
| 11.000 | 3312 | 214 | UNDERPOWERED | 1.447 | -1.388 | -0.181 | -0.089 |
| 12.000 | 2649 | 167 | UNDERPOWERED | 0.090 | -2.233 | 0.347 | -0.645 |

### week relative to the DEFENCE team's first conference game (`conf_rel_def`)

Bucket-mean residual, predicted minus actual, percentage points. Positive = the model over-predicts that class in that bucket.

| w_rel_def | n | n_games | underpowered | resid_TOV | resid_FGA_rim | resid_FGA_jump2 | resid_FGA_3 |
|---|---|---|---|---|---|---|---|
| -8.000 | 3461 | 263 | UNDERPOWERED | -0.578 | 0.898 | 0.408 | 0.017 |
| -7.000 | 3984 | 298 | UNDERPOWERED | -0.865 | 0.703 | 0.058 | 0.167 |
| -6.000 | 4958 | 378 | UNDERPOWERED | -0.420 | 1.255 | 0.758 | -1.624 |
| -5.000 | 4856 | 381 | UNDERPOWERED | 0.568 | 0.964 | -0.172 | -1.703 |
| -4.000 | 4723 | 381 | UNDERPOWERED | -0.388 | 0.606 | 0.617 | -1.312 |
| -3.000 | 5115 | 424 |  | 0.863 | 0.083 | -0.015 | -0.921 |
| -2.000 | 4999 | 410 | UNDERPOWERED | 0.500 | 0.350 | -0.039 | -1.654 |
| -1.000 | 2753 | 240 | UNDERPOWERED | 1.143 | -0.130 | 0.443 | -0.882 |
| 0.000 | 6688 | 361 |  | 0.279 | 0.109 | 0.274 | -0.596 |
| 1.000 | 5820 | 348 |  | -0.807 | 1.153 | 0.610 | -1.059 |
| 2.000 | 5770 | 351 |  | -0.754 | 0.926 | 0.238 | -0.620 |
| 3.000 | 5963 | 355 |  | -0.447 | 1.083 | -0.260 | -0.315 |
| 4.000 | 6480 | 365 |  | -0.040 | 1.284 | -0.756 | -1.033 |
| 5.000 | 6636 | 384 |  | -0.491 | 0.472 | 0.480 | -0.884 |
| 6.000 | 5723 | 342 |  | -0.359 | -0.017 | -0.255 | 0.459 |
| 7.000 | 6371 | 370 |  | 0.660 | 0.621 | -0.292 | -1.368 |
| 8.000 | 6462 | 380 |  | 0.133 | 0.813 | 0.062 | -0.903 |
| 9.000 | 6166 | 358 |  | 0.496 | 0.438 | -0.399 | -0.666 |
| 10.000 | 3694 | 230 | UNDERPOWERED | 1.106 | -0.534 | -0.429 | -0.854 |
| 11.000 | 3264 | 214 | UNDERPOWERED | 0.670 | -0.551 | -0.316 | -0.241 |
| 12.000 | 2595 | 167 | UNDERPOWERED | 0.645 | -2.383 | 0.813 | -1.101 |

### conference vs non-conference game (`conf_flag`)

Bucket-mean residual, predicted minus actual, percentage points. Positive = the model over-predicts that class in that bucket.

| is_conf_game | n | n_games | underpowered | resid_TOV | resid_FGA_rim | resid_FGA_jump2 | resid_FGA_3 |
|---|---|---|---|---|---|---|---|
| no | 42282 | 2011 |  | 0.167 | 0.565 | 0.232 | -1.067 |
| yes | 69624 | 3434 |  | -0.010 | 0.456 | -0.002 | -0.680 |

### weeks since the monthly S1 refit boundary (`weeks_since_refit`)

Bucket-mean residual, predicted minus actual, percentage points. Positive = the model over-predicts that class in that bucket.

| w_since_refit | n | n_games | underpowered | resid_TOV | resid_FGA_rim | resid_FGA_jump2 | resid_FGA_3 |
|---|---|---|---|---|---|---|---|
| 0 | 28067 | 1369 |  | 0.190 | 0.239 | 0.297 | -0.917 |
| 1 | 28157 | 1364 |  | -0.216 | 0.661 | 0.106 | -0.488 |
| 2 | 25069 | 1225 |  | 0.061 | 0.784 | -0.234 | -0.695 |
| 3 | 23669 | 1146 |  | 0.207 | 0.213 | 0.084 | -1.116 |
| 4 | 6944 | 341 |  | 0.103 | 0.812 | 0.326 | -1.312 |

### Per-decile calibration gap, overall vs the first four weeks of conference play (offence-team alignment)

| segment | n | n_games | worst_gated_decile_gap_pp |
|---|---|---|---|
| overall | 111906 | 5445 | 1.859 |
| non-conference games | 42282 | 2011 | 2.609 |
| conference games | 69624 | 3434 | 2.156 |
| first 4 conf weeks (w_rel_off 0..3) | 24441 | 1334 | 2.787 |
| conf weeks 4+ (w_rel_off >= 4) | 51144 | 2601 | 2.010 |
| pre-boundary (w_rel_off < 0) | 36321 | 1846 | 2.666 |

---

## Verdict: which alignment carries the residual structure

`var` is the chance-weighted variance of the bucket-mean residual over POWERED buckets, in pp^2. Because the alignments have different bucket counts, the raw variance is not comparable across them; `null` is the same statistic under 200 shuffles of the bucket label BETWEEN GAMES (preserving bucket sizes), which is what makes them comparable, and `excess_rms` = sqrt(var - null_mean) is the reportable quantity in pp. `boot` is a 200-replicate game-block bootstrap CI on `var`.

### `first`

| alignment | class | n_buckets | n_powered | var_pp2 | null_mean_pp2 | null_p95_pp2 | excess_rms_pp | boot_lo | boot_hi | beats_null_p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| calendar_week | TOV | 23 | 20 | 0.060 | 0.041 | 0.064 | 0.139 | 0.049 | 0.151 | no |
| calendar_week | FGA_rim | 23 | 20 | 0.266 | 0.108 | 0.167 | 0.397 | 0.215 | 0.573 | yes |
| calendar_week | FGA_jump2 | 23 | 20 | 0.337 | 0.095 | 0.145 | 0.491 | 0.262 | 0.633 | yes |
| calendar_week | FGA_3 | 23 | 20 | 0.641 | 0.078 | 0.120 | 0.750 | 0.542 | 0.931 | yes |
| calendar_week | FT_trip_shooting | 23 | 20 | 0.022 | 0.014 | 0.022 | 0.091 | 0.020 | 0.050 | yes |
| calendar_week | FT_trip_bonus | 23 | 20 | 0.014 | 0.012 | 0.020 | 0.041 | 0.015 | 0.040 | no |
| conf_rel_off | TOV | 21 | 21 | 0.059 | 0.044 | 0.069 | 0.123 | 0.062 | 0.171 | no |
| conf_rel_off | FGA_rim | 21 | 21 | 0.266 | 0.116 | 0.182 | 0.389 | 0.202 | 0.484 | yes |
| conf_rel_off | FGA_jump2 | 21 | 21 | 0.319 | 0.110 | 0.165 | 0.457 | 0.290 | 0.644 | yes |
| conf_rel_off | FGA_3 | 21 | 21 | 0.470 | 0.087 | 0.130 | 0.619 | 0.351 | 0.727 | yes |
| conf_rel_off | FT_trip_shooting | 21 | 21 | 0.023 | 0.015 | 0.022 | 0.093 | 0.022 | 0.056 | yes |
| conf_rel_off | FT_trip_bonus | 21 | 21 | 0.010 | 0.013 | 0.020 | 0.000 | 0.012 | 0.038 | no |
| conf_rel_def | TOV | 21 | 21 | 0.057 | 0.043 | 0.068 | 0.117 | 0.058 | 0.180 | no |
| conf_rel_def | FGA_rim | 21 | 21 | 0.185 | 0.113 | 0.184 | 0.267 | 0.238 | 0.529 | yes |
| conf_rel_def | FGA_jump2 | 21 | 21 | 0.282 | 0.111 | 0.186 | 0.413 | 0.209 | 0.544 | yes |
| conf_rel_def | FGA_3 | 21 | 21 | 0.433 | 0.084 | 0.122 | 0.590 | 0.431 | 0.797 | yes |
| conf_rel_def | FT_trip_shooting | 21 | 21 | 0.024 | 0.014 | 0.022 | 0.096 | 0.022 | 0.055 | yes |
| conf_rel_def | FT_trip_bonus | 21 | 21 | 0.012 | 0.013 | 0.020 | 0.000 | 0.014 | 0.045 | no |
| conf_flag | TOV | 2 | 2 | 0.002 | 0.002 | 0.008 | 0.021 | 0.000 | 0.015 | no |
| conf_flag | FGA_rim | 2 | 2 | 0.057 | 0.005 | 0.020 | 0.227 | 0.007 | 0.145 | yes |
| conf_flag | FGA_jump2 | 2 | 2 | 0.084 | 0.005 | 0.020 | 0.280 | 0.022 | 0.197 | yes |
| conf_flag | FGA_3 | 2 | 2 | 0.186 | 0.004 | 0.014 | 0.427 | 0.102 | 0.311 | yes |
| conf_flag | FT_trip_shooting | 2 | 2 | 0.009 | 0.001 | 0.002 | 0.092 | 0.001 | 0.021 | yes |
| conf_flag | FT_trip_bonus | 2 | 2 | 0.003 | 0.001 | 0.003 | 0.043 | 0.000 | 0.010 | no |
| weeks_since_refit | TOV | 5 | 5 | 0.004 | 0.008 | 0.017 | 0.000 | 0.002 | 0.031 | no |
| weeks_since_refit | FGA_rim | 5 | 5 | 0.014 | 0.023 | 0.057 | 0.000 | 0.006 | 0.082 | no |
| weeks_since_refit | FGA_jump2 | 5 | 5 | 0.049 | 0.019 | 0.046 | 0.173 | 0.020 | 0.138 | yes |
| weeks_since_refit | FGA_3 | 5 | 5 | 0.039 | 0.017 | 0.039 | 0.150 | 0.018 | 0.128 | yes |
| weeks_since_refit | FT_trip_shooting | 5 | 5 | 0.003 | 0.003 | 0.006 | 0.024 | 0.001 | 0.015 | no |
| weeks_since_refit | FT_trip_bonus | 5 | 5 | 0.002 | 0.003 | 0.007 | 0.000 | 0.001 | 0.011 | no |

### `cont`

| alignment | class | n_buckets | n_powered | var_pp2 | null_mean_pp2 | null_p95_pp2 | excess_rms_pp | boot_lo | boot_hi | beats_null_p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| calendar_week | TOV | 23 | 15 | 0.260 | 0.154 | 0.264 | 0.326 | 0.168 | 0.688 | no |
| calendar_week | FGA_rim | 23 | 15 | 1.024 | 0.467 | 0.772 | 0.746 | 0.778 | 2.302 | yes |
| calendar_week | FGA_jump2 | 23 | 15 | 0.283 | 0.216 | 0.359 | 0.258 | 0.245 | 0.792 | no |
| calendar_week | FGA_3 | 23 | 15 | 0.179 | 0.299 | 0.504 | 0.000 | 0.225 | 0.814 | no |
| calendar_week | FT_trip_shooting | 23 | 15 | 0.098 | 0.101 | 0.179 | 0.000 | 0.095 | 0.325 | no |
| calendar_week | FT_trip_bonus | 23 | 15 | 0.049 | 0.071 | 0.116 | 0.000 | 0.059 | 0.194 | no |
| conf_rel_off | TOV | 21 | 12 | 0.185 | 0.164 | 0.277 | 0.143 | 0.176 | 0.636 | no |
| conf_rel_off | FGA_rim | 21 | 12 | 0.372 | 0.491 | 0.853 | 0.000 | 0.301 | 1.383 | no |
| conf_rel_off | FGA_jump2 | 21 | 12 | 0.168 | 0.237 | 0.409 | 0.000 | 0.143 | 0.720 | no |
| conf_rel_off | FGA_3 | 21 | 12 | 0.282 | 0.296 | 0.542 | 0.000 | 0.150 | 0.842 | no |
| conf_rel_off | FT_trip_shooting | 21 | 12 | 0.108 | 0.102 | 0.175 | 0.081 | 0.096 | 0.446 | no |
| conf_rel_off | FT_trip_bonus | 21 | 12 | 0.181 | 0.073 | 0.126 | 0.328 | 0.089 | 0.336 | yes |
| conf_rel_def | TOV | 21 | 11 | 0.286 | 0.162 | 0.297 | 0.353 | 0.120 | 0.563 | no |
| conf_rel_def | FGA_rim | 21 | 11 | 0.185 | 0.469 | 0.879 | 0.000 | 0.256 | 1.555 | no |
| conf_rel_def | FGA_jump2 | 21 | 11 | 0.156 | 0.235 | 0.419 | 0.000 | 0.155 | 0.726 | no |
| conf_rel_def | FGA_3 | 21 | 11 | 0.204 | 0.278 | 0.544 | 0.000 | 0.242 | 1.176 | no |
| conf_rel_def | FT_trip_shooting | 21 | 11 | 0.137 | 0.102 | 0.207 | 0.187 | 0.101 | 0.401 | no |
| conf_rel_def | FT_trip_bonus | 21 | 11 | 0.133 | 0.070 | 0.130 | 0.250 | 0.108 | 0.413 | yes |
| conf_flag | TOV | 2 | 2 | 0.007 | 0.008 | 0.025 | 0.000 | 0.000 | 0.059 | no |
| conf_flag | FGA_rim | 2 | 2 | 0.003 | 0.028 | 0.112 | 0.000 | 0.000 | 0.115 | no |
| conf_flag | FGA_jump2 | 2 | 2 | 0.013 | 0.015 | 0.057 | 0.000 | 0.000 | 0.121 | no |
| conf_flag | FGA_3 | 2 | 2 | 0.035 | 0.019 | 0.077 | 0.126 | 0.000 | 0.192 | no |
| conf_flag | FT_trip_shooting | 2 | 2 | 0.020 | 0.007 | 0.022 | 0.113 | 0.000 | 0.074 | no |
| conf_flag | FT_trip_bonus | 2 | 2 | 0.006 | 0.004 | 0.014 | 0.045 | 0.000 | 0.036 | no |
| weeks_since_refit | TOV | 5 | 5 | 0.028 | 0.039 | 0.098 | 0.000 | 0.010 | 0.189 | no |
| weeks_since_refit | FGA_rim | 5 | 5 | 0.065 | 0.107 | 0.244 | 0.000 | 0.022 | 0.480 | no |
| weeks_since_refit | FGA_jump2 | 5 | 5 | 0.038 | 0.054 | 0.128 | 0.000 | 0.011 | 0.192 | no |
| weeks_since_refit | FGA_3 | 5 | 5 | 0.067 | 0.068 | 0.169 | 0.000 | 0.019 | 0.307 | no |
| weeks_since_refit | FT_trip_shooting | 5 | 5 | 0.042 | 0.026 | 0.058 | 0.129 | 0.014 | 0.134 | no |
| weeks_since_refit | FT_trip_bonus | 5 | 5 | 0.022 | 0.017 | 0.044 | 0.072 | 0.006 | 0.093 | no |


---

## Reading (possession-outcome worker, 2026-09-10)

Appended by hand after the run. Re-running `scripts/diag_possession_outcome_conf_regime.py
--step analyse` regenerates the tables above; it does not regenerate this section.

### 1. The residual structure is a SEASON-START effect, not a conference-boundary effect

Excess RMS of the bucket-mean residual, in percentage points -- `sqrt(var - null mean)`,
where the null is 200 shuffles of the bucket label between games at the same bucket sizes,
which is what makes alignments with different bucket counts comparable. Population `first`
(the round-2 `lgbm` + S1 winner), fold 2:

| alignment | buckets | FGA_3 | FGA_jump2 | FGA_rim | TOV |
|---|---|---|---|---|---|
| calendar week | 23 (20 powered) | **0.750** | **0.491** | **0.398** | 0.139 |
| week rel. to OFFENCE team's first conference game | 21 | 0.619 | 0.457 | 0.389 | 0.123 |
| week rel. to DEFENCE team's first conference game | 21 | 0.590 | 0.414 | 0.267 | 0.117 |
| conference vs non-conference (one binary) | 2 | 0.427 | 0.280 | 0.227 | 0.021 |
| weeks since the monthly S1 refit | 5 | 0.150 | 0.173 | 0.000 | 0.000 |

**Calendar week explains MORE than either conference-relative alignment, on every class.**
Decision 9's diagnostic hypothesis -- that re-bucketing by each team's own conference
boundary would reveal structure the calendar hides -- is NOT supported. The two axes are
largely the same axis (conference play starts in a narrow calendar window for most teams),
and the conference-relative version is the blurrier of the two because the boundary date
varies by team while the season start does not.

The per-bucket tables say where it lives. On calendar week the residual sits almost
entirely in weeks 0-3: FGA_3 is over-predicted by -1.85, -1.65, -1.55, -1.47 pp and
FGA_jump2 under-predicted by +1.17, +1.48, +1.72, +0.65 pp, against |residual| mostly under
0.4 pp from week 8 on. On the offence-boundary axis the same games appear as weeks -8 to -4
(FGA_3 -2.11, -1.73, -1.75, -0.97, -0.84 pp). Those are the same November non-conference
games described two ways.

Two consequences worth stating plainly:

* **`weeks_since_refit` carries essentially nothing** (0.15 pp on FGA_3, 0.17 on jump2, and
  the rim and TOV cells sit inside the null). Refit STALENESS is not the mechanism. That is
  direct evidence against the part of Decision 9 that expects a cadence or alignment change
  to help, and it is the reason the round-3 stage order runs the feature ladder before the
  alignment ladder.
* **One binary conference flag reproduces over half of the 20-bucket calendar structure**
  (0.427 of 0.750 pp on FGA_3, 0.280 of 0.491 on jump2). That is direct support for
  Decision 9b, the conference-game FLAG as a feature, at one degree of freedom.

TOV is inside or barely outside the null on every alignment and is not read as signal.
Population `cont` is underpowered on this question: 11-15 of 21-23 buckets clear the
5,000-chance threshold and NO alignment beats its own null p95 on any class except calendar
week / FGA_rim. It is reported and labelled underpowered, not read either way.

### 2. Where the round-2 winner actually fails its gate

Worst gated decile calibration gap by segment, `first` population, fold 2, round-2 S1
winner (gate: 2.0 pp):

| segment | n chances | worst gated gap (pp) |
|---|---|---|
| overall | 742,025 | **0.98 PASS** |
| conference games | 468,689 | 0.98 PASS |
| first 4 conference weeks (offence boundary) | 158,749 | 1.17 PASS |
| conference weeks 4+ | 279,996 | 1.02 PASS |
| **non-conference games** | 273,336 | **2.49 FAIL** |
| **pre-boundary (before the offence team's first conference game)** | 228,399 | **2.95 FAIL** |

The headline 0.98 pp that earned round 2 its PASS is an average over a model that is well
calibrated in conference play and 2.5-2.9 pp out in non-conference play. The damage is
BEFORE the boundary, not in the first weeks after it, which is not where Decision 9 expected
it. Round 3 therefore carries the non-conference gap as a decision quantity alongside the
first-four-conference-weeks gap.

### 3. The style rates at the boundary, by own-rating quintile

One row per team-game on the offence side; quintiles of the team's season-mean as-of
`off_rating_off_c` (descriptive, not a feature). Values are the raw-centred style features
in the team's own units; `shift` = conference mean minus non-conference mean.

| quintile | teams | mean own rating | 3PA shift | rim shift | TOV shift | FTr shift | FTr level, non-conf |
|---|---|---|---|---|---|---|---|
| 1 (weakest) | 73 | -8.85 | +0.244 | +0.213 | -0.109 | **+1.576** | -2.896 |
| 2 | 73 | -4.26 | -0.116 | +0.538 | -0.125 | +0.115 | -0.794 |
| 3 | 72 | -0.56 | +0.427 | -0.481 | -0.005 | -0.817 | +1.360 |
| 4 | 73 | +3.40 | +0.366 | -0.102 | -0.053 | **-1.547** | +2.261 |
| 5 (strongest) | 73 | +10.35 | -0.396 | -0.243 | +0.278 | -0.818 | +1.877 |

The Mississippi Valley State effect is **present and ordered but small**. Free-throw rate
shows it most clearly and in the predicted direction: the shift at the boundary is opposite
in sign to the non-conference level and roughly proportional to it (Q1 -2.90 -> +1.58,
Q4 +2.26 -> -1.55, Q5 +1.88 -> -0.82), i.e. the as-of rate a team carries out of
non-conference play partly regresses once the schedule strengthens. Turnover rate shows the
same ordering an order of magnitude smaller (-0.11, -0.13, -0.01, -0.05, +0.28). 3PA and rim
show no monotone quintile pattern at all. On features whose cross-team SD is 2-4 units, a
1-1.6 unit shift is material for FTr and noise-sized for the other three -- so opponent
adjustment has something real to bite on, but only on one of the four rates, and the
diagnostic does not predict a large log-loss move.

### 4. What this sets up for round 3

The diagnostic supports Decision 9b (the conference flag) strongly, Decision 9a (opponent
adjustment) weakly and on one rate of four, and Decision 9c (refit alignment) not at all --
`weeks_since_refit` is the flattest axis measured and the conference-relative axes explain
less than the plain calendar. That ordering is what the round-3 stage order encodes, and all
three remain pre-registered ARMS, PENDING EVIDENCE: this is a diagnostic on round 2's
residuals, not a result about round 3's arms, and it cannot decide any of them.
