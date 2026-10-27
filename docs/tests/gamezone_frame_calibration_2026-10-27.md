# GameZone frame calibration and team band-profile reliability (2026-10-27)

Descriptive only: no recommendation. Private internal research data. Script `scripts/diag_gamezone_frame_v1.py`; follows `docs/tests/gamezone_vs_espn_agreement_2026-10-27.md`. Tables also in `docs/tests/gamezone_frame_calibration_2026-10-27.json`.

## 1. Frame calibration

Per-season least squares from GameZone (loc_y along the court from the baseline, loc_x lateral) to the CBBD folded frame in feet (u = xf/10 along, v = lateral). The CBBD fold mirrors the lateral axis at the far end (raw x > 470), so v is first aligned to GameZone's orientation (v = y/10 at the far end, 50 - y/10 otherwise; the correlation with loc_x is then positive at both ends). Fit set: joined jump2 + three attempts located by both vendors (305,250 of 475,554; rim attempts excluded because their pixels are canned). Two least-squares maps as asked: per-axis scale + offset (u = a*loc_y + b, v = c*loc_x + d) and a full 2x2 affine plus offset. Because the two vendors' coordinates for the same shot differ by several feet in either direction (residuals below), ordinary least squares is attenuated (slopes 0.6-0.8, pulling every shot toward the middle of the court) and is not a valid frame calibration; a third arm, per-axis total least squares (Deming, equal error variance in both vendors, so the slope is not shrunk), is added and labelled as such. Least squares with noise in both coordinates shrinks slopes below their true value (errors-in-variables), so the fitted scales are lower bounds on any true stretch. "identity" = GameZone raw coordinates read directly as the CBBD frame.

Fitted parameters:

| season | n_fit | axis_u_scale | axis_u_offset | axis_v_scale | axis_v_offset | full_u_from_along | full_u_from_lateral | full_u_offset | full_v_from_along | full_v_from_lateral | full_v_offset | tls_u_scale | tls_u_offset | tls_v_scale | tls_v_offset |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | 109273 | 0.802 | 3.562 | 0.639 | 9.472 | 0.803 | -0.021 | 4.037 | -0.009 | 0.639 | 9.639 | 1.226 | -4.236 | 1.034 | 0.018 |
| 2023 | 116684 | 0.815 | 3.176 | 0.586 | 10.922 | 0.816 | -0.021 | 3.646 | -0.007 | 0.586 | 11.052 | 1.325 | -6.321 | 1.030 | 0.556 |
| 2024 | 79293 | 0.826 | 3.245 | 0.694 | 8.214 | 0.826 | -0.015 | 3.602 | -0.003 | 0.694 | 8.260 | 1.195 | -3.413 | 1.028 | 0.281 |

Residual distance between the mapped GameZone point and the CBBD point (ft), and agreement of the distance to the basket (5.25, 25.0) after mapping, fit-set attempts:

| season | map | n | resid_p25 | resid_p50 | resid_p75 | resid_p90 | resid_p95 | resid_mean | abs_dbasket_p25 | abs_dbasket_p50 | abs_dbasket_p75 | abs_dbasket_p90 | abs_dbasket_p95 | abs_dbasket_mean | d_within_1ft_pct | d_within_3ft_pct | d_gt6ft_pct | u_resid_sd | v_resid_sd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | identity | 109273 | 2.918 | 5.461 | 11.324 | 23.619 | 32.898 | 9.220 | 0.693 | 1.582 | 3.087 | 5.381 | 7.407 | 2.351 | 34.602 | 74.037 | 8.061 | 6.341 | 11.791 |
| 2022 | axis | 109273 | 4.849 | 7.668 | 11.283 | 20.734 | 28.000 | 9.744 | 2.563 | 4.454 | 6.541 | 8.222 | 9.161 | 4.651 | 8.882 | 30.416 | 30.714 | 6.166 | 10.741 |
| 2022 | full | 109273 | 4.844 | 7.640 | 11.297 | 20.738 | 27.978 | 9.740 | 2.570 | 4.454 | 6.521 | 8.184 | 9.117 | 4.643 | 8.902 | 30.236 | 30.608 | 6.160 | 10.741 |
| 2022 | axis_tls | 109273 | 3.147 | 5.916 | 11.926 | 24.298 | 33.678 | 9.659 | 0.785 | 1.698 | 3.146 | 5.345 | 7.280 | 2.408 | 31.478 | 73.272 | 7.763 | 6.928 | 11.989 |
| 2023 | identity | 116684 | 3.037 | 5.791 | 12.348 | 25.712 | 35.099 | 9.835 | 0.703 | 1.576 | 3.061 | 5.308 | 7.187 | 2.317 | 34.437 | 74.282 | 7.684 | 6.478 | 12.599 |
| 2023 | axis | 116684 | 5.106 | 8.362 | 12.255 | 21.893 | 28.803 | 10.338 | 2.679 | 4.731 | 7.100 | 9.044 | 10.016 | 4.995 | 8.508 | 28.739 | 35.499 | 6.343 | 11.279 |
| 2023 | full | 116684 | 5.114 | 8.337 | 12.262 | 21.877 | 28.800 | 10.334 | 2.692 | 4.739 | 7.069 | 8.996 | 9.948 | 4.986 | 8.482 | 28.494 | 35.404 | 6.337 | 11.278 |
| 2023 | axis_tls | 116684 | 3.607 | 6.495 | 13.061 | 26.502 | 35.800 | 10.465 | 0.959 | 2.015 | 3.562 | 5.680 | 7.481 | 2.662 | 26.013 | 67.646 | 8.831 | 7.301 | 12.784 |
| 2024 | identity | 79293 | 2.658 | 4.861 | 9.651 | 21.416 | 31.519 | 8.367 | 0.666 | 1.496 | 2.936 | 5.228 | 7.149 | 2.269 | 36.074 | 75.732 | 7.611 | 6.040 | 10.948 |
| 2024 | axis | 79293 | 4.306 | 6.809 | 9.977 | 19.102 | 27.509 | 8.928 | 2.233 | 3.992 | 5.885 | 7.435 | 8.419 | 4.203 | 10.741 | 35.374 | 23.689 | 5.896 | 10.128 |
| 2024 | full | 79293 | 4.318 | 6.791 | 9.980 | 19.104 | 27.477 | 8.926 | 2.245 | 3.997 | 5.868 | 7.399 | 8.386 | 4.198 | 10.671 | 35.263 | 23.448 | 5.892 | 10.128 |
| 2024 | axis_tls | 79293 | 2.842 | 5.199 | 10.157 | 21.919 | 32.247 | 8.708 | 0.767 | 1.645 | 3.061 | 5.234 | 7.092 | 2.358 | 32.188 | 74.337 | 7.549 | 6.516 | 11.098 |

### (c) re-measured, before and after calibration

Joined attempts of the same class (attempt-level), co-located team-games with at least 5 attempts of the class under both vendors (team-game mean absolute band-share difference, points), team-seasons with at least 100 under both (team-season correlation), and GameZone pooled band shares (CBBD pooled: short2 41.3%, corner3 21.8%, wing3 44.3%, top3 33.9%):

| frame | jump2_attempt_band_differs_pct | three_attempt_band_differs_pct | mad_pp_short2 | mad_pp_corner3 | mad_pp_wing3 | mad_pp_top3 | corr_ts_short2 | corr_ts_corner3 | corr_ts_wing3 | corr_ts_top3 | share_short2 | share_corner3 | share_wing3 | share_top3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| before (textbook basket, raw GameZone frame) | 26.674 | 37.462 | 19.501 | 10.782 | 16.912 | 14.580 | 0.639 | 0.509 | 0.257 | 0.389 | 31.688 | 18.089 | 50.917 | 30.994 |
| after: per-axis scale+offset | 29.388 | 47.358 | 19.529 | 21.689 | 19.738 | 19.067 | 0.594 |  | 0.010 | 0.349 | 37.665 | 0 | 52.900 | 47.100 |
| after: full 2x2 affine | 29.302 | 47.391 | 19.551 | 21.689 | 19.697 | 19.071 | 0.595 |  | 0.005 | 0.351 | 37.733 | 0 | 52.787 | 47.213 |
| after: per-axis total least squares (Deming, equal error variances; added arm, not an LS fit) | 26.945 | 40.302 | 18.575 | 12.751 | 18.217 | 14.499 | 0.648 | 0.442 | 0.223 | 0.390 | 36.541 | 11.859 | 55.307 | 32.834 |

Verdict (c): Calibration by least squares does not bring the vendors closer; it makes the band comparison worse, and the reason is shot-level noise, not a global frame offset. Even with GameZone read as-is, the mapped point is a median 4.9-5.8 ft (p90 21-26 ft) from the CBBD point for jump2 + three attempts. Ordinary least squares shrinks every slope (along 0.80-0.83, lateral 0.59-0.69) toward the middle of the court, so corner threes disappear (GameZone corner3 share 18.1% before, 0.0% after) and the share of joined attempts whose band differs rises from 26.7% to 29.4% for jumpers and from 37.5% to 47.4% for threes; the full 2x2 affine is the same (off-diagonal terms are about 0.02). Team-game mean absolute band-share differences go from 19.5 / 10.8 / 16.9 / 14.6 points (short2 / corner3 / wing3 / top3) to 19.5 / 21.7 / 19.7 / 19.1; team-season correlations from 0.64 / 0.51 / 0.26 / 0.39 to 0.59 / undefined / 0.01 / 0.35. The added total-least-squares arm (no shrinkage; slopes along 1.20-1.33, lateral 1.03) is roughly neutral: band differs 27.0% (jumpers) and 40.3% (threes); team-game differences 18.6 / 12.8 / 18.2 / 14.5 points; team-season correlations 0.65 / 0.44 / 0.22 / 0.39; GameZone pooled shares 36.5 / 11.9 / 55.3 / 32.8 against CBBD 41.3 / 21.8 / 44.3 / 33.9. No affine map tried makes the two vendors agree better than the raw textbook frame on corner and wing threes.

### (d) re-reported under the calibrated frame

Text threes (vendor `desc`) inside the line and text jumpers outside it, all located GameZone attempts of the season, distance to the CBBD basket in the mapped frame; the CBBD rows are the joined text-threes in CBBD's own frame:

| frame | season | n_text_three | three_inside_22.15_pct | three_inside_21.65_pct | three_inside_20.65_pct | n_text_twojump | twojump_outside_22.15_pct | median_three_dist_ft | flag |
|---|---|---|---|---|---|---|---|---|---|
| identity | 2022 | 97889 | 2.697 | 0.457 | 0.036 | 66052 | 0 | 23.854 |  |
| identity | 2023 | 96451 | 2.230 | 0.339 | 0.003 | 65697 | 0 | 23.955 |  |
| identity | 2024 | 57761 | 3.328 | 0.738 | 0.220 | 37083 | 0 | 23.780 |  |
| axis | 2022 | 97889 | 89.976 | 82.137 | 63.215 | 66052 | 0 | 19.828 |  |
| axis | 2023 | 96451 | 91.084 | 84.828 | 68.958 | 65697 | 0 | 19.289 |  |
| axis | 2024 | 57761 | 83.435 | 74.396 | 56.218 | 37083 | 0 | 20.273 |  |
| full | 2022 | 97889 | 90.113 | 82.384 | 63.383 | 66052 | 0 | 19.830 |  |
| full | 2023 | 96451 | 91.237 | 85.025 | 69.241 | 65697 | 0 | 19.303 |  |
| full | 2024 | 57761 | 83.941 | 74.703 | 56.247 | 37083 | 0 | 20.281 |  |
| axis_tls | 2022 | 97889 | 3.383 | 0.987 | 0.052 | 66052 | 0.389 | 25.212 |  |
| axis_tls | 2023 | 96451 | 7.298 | 3.498 | 0.261 | 65697 | 0.473 | 25.499 |  |
| axis_tls | 2024 | 57761 | 4.583 | 1.735 | 0.253 | 37083 | 0.440 | 25.043 |  |
| CBBD (joined text-threes, own frame) | 2022 | 64536 | 1.774 | 0.273 | 0.107 |  |  | 24.668 |  |
| CBBD (joined text-threes, own frame) | 2023 | 69453 | 1.610 | 0.245 | 0.111 |  |  | 24.668 |  |
| CBBD (joined text-threes, own frame) | 2024 | 47471 | 1.784 | 0.238 | 0.099 |  |  | 24.615 |  |

Verdict (d): Under the least-squares maps 83-91% of text-threes fall inside 22.15 ft, which shows the map is invalid for this purpose (it compresses the court). Under the total-least-squares map 3.4% / 7.3% / 4.6% of text-threes (2022 / 2023 / 2024) are inside 22.15 ft and 0.4-0.5% of text two-point jumpers outside it, against 2.7% / 2.2% / 3.3% in the raw textbook frame and 1.8% / 1.6% / 1.8% for CBBD on its joined text-threes. Calibration did not improve the inside-the-line share.

## 2. Within-vendor reliability of team band profiles

Unit = team-season. Attempts: located jump2 (bands short2 / long2; short2 shown, long2 is its complement) and located three (corner3 / wing3 / top3), bands as in the spec (section 2 of the draft). Offence = the team's own attempts; defence = bands allowed (opponent attempts). Games are ordered by date and split odd / even; each half needs at least 20 attempts of the class. Shares are demeaned by season before correlating so that league drift does not inflate the correlation. `spearman_brown` = 2r/(1+r) on the split-half correlation r. `between_team_sd_pp` = SD across team-seasons of the full-season share; `binomial_sd_at_median_n_pp` = sqrt(p(1-p)/n) at the median attempt count; `sd_ratio` is their ratio; `signal_variance_share` = 1 - mean binomial variance / between-team variance. Team-seasons under 200 located attempts in the class are in a separate scope labelled UNDERPOWERED. Vendors: CBBD 2022-2025 (training source); GameZone 2022-2024 (matched games only, so fewer attempts per team-season); CBBD restricted to the same GameZone-matched games for a like-for-like attempt count.

Team-seasons with at least 200 attempts:

| vendor | side | class | band | team_seasons | median_attempts | split_half_r | spearman_brown | between_team_sd_pp | binomial_sd_at_median_n_pp | sd_ratio | signal_variance_share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CBBD 2022-2025 | offence | jump2 | short2 | 1391 | 421 | 0.802 | 0.890 | 9.851 | 2.425 | 4.062 | 0.939 |
| CBBD 2022-2025 | offence | three | corner3 | 1422 | 649 | 0.654 | 0.791 | 4.500 | 1.622 | 2.774 | 0.864 |
| CBBD 2022-2025 | offence | three | wing3 | 1422 | 649 | 0.499 | 0.666 | 3.860 | 1.951 | 1.978 | 0.730 |
| CBBD 2022-2025 | offence | three | top3 | 1422 | 649 | 0.661 | 0.796 | 4.741 | 1.855 | 2.556 | 0.839 |
| CBBD 2022-2025 | defence | jump2 | short2 | 1428 | 412 | 0.526 | 0.689 | 6.456 | 2.445 | 2.641 | 0.856 |
| CBBD 2022-2025 | defence | three | corner3 | 1447 | 630 | 0.614 | 0.761 | 4.103 | 1.638 | 2.505 | 0.834 |
| CBBD 2022-2025 | defence | three | wing3 | 1447 | 630 | 0.352 | 0.520 | 3.313 | 1.980 | 1.673 | 0.625 |
| CBBD 2022-2025 | defence | three | top3 | 1447 | 630 | 0.606 | 0.755 | 4.495 | 1.886 | 2.383 | 0.816 |
| CBBD 2022-2024, GameZone-matched games only | offence | jump2 | short2 | 248 | 332 | 0.748 | 0.856 | 9.114 | 2.709 | 3.365 | 0.911 |
| CBBD 2022-2024, GameZone-matched games only | offence | three | corner3 | 309 | 440 | 0.632 | 0.775 | 5.018 | 1.985 | 2.528 | 0.831 |
| CBBD 2022-2024, GameZone-matched games only | offence | three | wing3 | 309 | 440 | 0.403 | 0.575 | 4.259 | 2.365 | 1.801 | 0.666 |
| CBBD 2022-2024, GameZone-matched games only | offence | three | top3 | 309 | 440 | 0.616 | 0.762 | 5.182 | 2.257 | 2.295 | 0.795 |
| CBBD 2022-2024, GameZone-matched games only | defence | jump2 | short2 | 265 | 336 | 0.384 | 0.555 | 5.653 | 2.678 | 2.111 | 0.764 |
| CBBD 2022-2024, GameZone-matched games only | defence | three | corner3 | 316 | 420 | 0.525 | 0.689 | 4.264 | 1.988 | 2.145 | 0.771 |
| CBBD 2022-2024, GameZone-matched games only | defence | three | wing3 | 316 | 420 | 0.394 | 0.566 | 4.055 | 2.424 | 1.672 | 0.621 |
| CBBD 2022-2024, GameZone-matched games only | defence | three | top3 | 316 | 420 | 0.506 | 0.672 | 4.670 | 2.321 | 2.012 | 0.738 |
| GameZone 2022-2024 (textbook raw frame) | offence | jump2 | short2 | 295 | 469 | 0.568 | 0.725 | 7.417 | 2.166 | 3.424 | 0.898 |
| GameZone 2022-2024 (textbook raw frame) | offence | three | corner3 | 332 | 622.500 | 0.285 | 0.444 | 3.668 | 1.560 | 2.350 | 0.768 |
| GameZone 2022-2024 (textbook raw frame) | offence | three | wing3 | 332 | 622.500 | -0.017 | -0.034 | 3.887 | 2.004 | 1.940 | 0.665 |
| GameZone 2022-2024 (textbook raw frame) | offence | three | top3 | 332 | 622.500 | 0.138 | 0.243 | 4.256 | 1.856 | 2.293 | 0.760 |
| GameZone 2022-2024 (textbook raw frame) | defence | jump2 | short2 | 313 | 473 | 0.213 | 0.351 | 4.915 | 2.143 | 2.294 | 0.768 |
| GameZone 2022-2024 (textbook raw frame) | defence | three | corner3 | 337 | 646 | 0.222 | 0.364 | 3.230 | 1.518 | 2.127 | 0.701 |
| GameZone 2022-2024 (textbook raw frame) | defence | three | wing3 | 337 | 646 | -0.017 | -0.034 | 3.952 | 1.967 | 2.009 | 0.670 |
| GameZone 2022-2024 (textbook raw frame) | defence | three | top3 | 337 | 646 | 0.178 | 0.302 | 4.480 | 1.822 | 2.459 | 0.780 |
| GameZone 2022-2024 (full-affine calibrated) | offence | jump2 | short2 | 295 | 469 | 0.535 | 0.697 | 7.333 | 2.245 | 3.266 | 0.889 |
| GameZone 2022-2024 (full-affine calibrated) | offence | three | corner3 | 332 | 622.500 |  |  | 0 | 0 |  |  |
| GameZone 2022-2024 (full-affine calibrated) | offence | three | wing3 | 332 | 622.500 | 0.183 | 0.309 | 4.829 | 1.999 | 2.415 | 0.786 |
| GameZone 2022-2024 (full-affine calibrated) | offence | three | top3 | 332 | 622.500 | 0.183 | 0.309 | 4.829 | 1.999 | 2.415 | 0.786 |
| GameZone 2022-2024 (full-affine calibrated) | defence | jump2 | short2 | 313 | 473 | 0.182 | 0.307 | 5.109 | 2.227 | 2.294 | 0.770 |
| GameZone 2022-2024 (full-affine calibrated) | defence | three | corner3 | 337 | 646 |  |  | 0 | 0 |  |  |
| GameZone 2022-2024 (full-affine calibrated) | defence | three | wing3 | 337 | 646 | 0.203 | 0.338 | 4.874 | 1.963 | 2.483 | 0.786 |
| GameZone 2022-2024 (full-affine calibrated) | defence | three | top3 | 337 | 646 | 0.203 | 0.338 | 4.874 | 1.963 | 2.483 | 0.786 |

Team-seasons under 200 attempts (UNDERPOWERED):

| vendor | side | class | band | scope | team_seasons | median_attempts | split_half_r | spearman_brown | between_team_sd_pp | binomial_sd_at_median_n_pp | sd_ratio | signal_variance_share | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CBBD 2022-2025 | offence | jump2 | short2 | n<200 (UNDERPOWERED) | 127 | 78 | 0.444 | 0.615 | 13.979 | 5.550 | 2.519 | 0.851 | UNDERPOWERED |
| CBBD 2022-2025 | offence | three | corner3 | n<200 (UNDERPOWERED) | 173 | 59 | 0.081 | 0.150 | 7.316 | 5.184 | 1.411 | 0.532 | UNDERPOWERED |
| CBBD 2022-2025 | offence | three | wing3 | n<200 (UNDERPOWERED) | 173 | 59 | 0.098 | 0.179 | 7.602 | 6.504 | 1.169 | 0.314 | UNDERPOWERED |
| CBBD 2022-2025 | offence | three | top3 | n<200 (UNDERPOWERED) | 173 | 59 | 0.128 | 0.227 | 8.306 | 6.091 | 1.364 | 0.501 | UNDERPOWERED |
| CBBD 2022-2025 | defence | jump2 | short2 | n<200 (UNDERPOWERED) | 52 | 84 | -0.013 | -0.026 | 8.322 | 5.452 | 1.527 | 0.585 | UNDERPOWERED |
| CBBD 2022-2025 | defence | three | corner3 | n<200 (UNDERPOWERED) | 260 | 67.500 | 0.076 | 0.142 | 7.972 | 5.273 | 1.512 | 0.584 | UNDERPOWERED |
| CBBD 2022-2025 | defence | three | wing3 | n<200 (UNDERPOWERED) | 260 | 67.500 | -0.006 | -0.012 | 7.225 | 6.074 | 1.190 | 0.316 | UNDERPOWERED |
| CBBD 2022-2025 | defence | three | top3 | n<200 (UNDERPOWERED) | 260 | 67.500 | 0.140 | 0.246 | 7.460 | 5.471 | 1.363 | 0.482 | UNDERPOWERED |
| CBBD 2022-2024, GameZone-matched games only | offence | jump2 | short2 | n<200 (UNDERPOWERED) | 351 | 82 | 0.479 | 0.647 | 11.402 | 5.434 | 2.098 | 0.782 | UNDERPOWERED |
| CBBD 2022-2024, GameZone-matched games only | offence | three | corner3 | n<200 (UNDERPOWERED) | 480 | 84 | 0.236 | 0.381 | 6.719 | 4.452 | 1.509 | 0.548 | UNDERPOWERED |
| CBBD 2022-2024, GameZone-matched games only | offence | three | wing3 | n<200 (UNDERPOWERED) | 480 | 84 | 0.140 | 0.246 | 6.828 | 5.430 | 1.257 | 0.349 | UNDERPOWERED |
| CBBD 2022-2024, GameZone-matched games only | offence | three | top3 | n<200 (UNDERPOWERED) | 480 | 84 | 0.214 | 0.353 | 7.303 | 5.159 | 1.416 | 0.490 | UNDERPOWERED |
| CBBD 2022-2024, GameZone-matched games only | defence | jump2 | short2 | n<200 (UNDERPOWERED) | 303 | 74 | 0.032 | 0.062 | 8.657 | 5.763 | 1.502 | 0.573 | UNDERPOWERED |
| CBBD 2022-2024, GameZone-matched games only | defence | three | corner3 | n<200 (UNDERPOWERED) | 469 | 84 | 0.104 | 0.188 | 6.777 | 4.619 | 1.467 | 0.527 | UNDERPOWERED |
| CBBD 2022-2024, GameZone-matched games only | defence | three | wing3 | n<200 (UNDERPOWERED) | 469 | 84 | 0.049 | 0.094 | 6.311 | 5.413 | 1.166 | 0.247 | UNDERPOWERED |
| CBBD 2022-2024, GameZone-matched games only | defence | three | top3 | n<200 (UNDERPOWERED) | 469 | 84 | 0.197 | 0.330 | 7.344 | 5.124 | 1.433 | 0.505 | UNDERPOWERED |
| GameZone 2022-2024 (textbook raw frame) | offence | jump2 | short2 | n<200 (UNDERPOWERED) | 389 | 80 | 0.185 | 0.313 | 11.307 | 5.194 | 2.177 | 0.800 | UNDERPOWERED |
| GameZone 2022-2024 (textbook raw frame) | offence | three | corner3 | n<200 (UNDERPOWERED) | 531 | 88 | 0.102 | 0.186 | 6.983 | 4.107 | 1.701 | 0.648 | UNDERPOWERED |
| GameZone 2022-2024 (textbook raw frame) | offence | three | wing3 | n<200 (UNDERPOWERED) | 531 | 88 | -0.014 | -0.028 | 9.938 | 5.329 | 1.865 | 0.711 | UNDERPOWERED |
| GameZone 2022-2024 (textbook raw frame) | offence | three | top3 | n<200 (UNDERPOWERED) | 531 | 88 | -0.012 | -0.025 | 9.633 | 4.922 | 1.957 | 0.739 | UNDERPOWERED |
| GameZone 2022-2024 (textbook raw frame) | defence | jump2 | short2 | n<200 (UNDERPOWERED) | 335 | 73 | -0.006 | -0.013 | 9.428 | 5.488 | 1.718 | 0.680 | UNDERPOWERED |
| GameZone 2022-2024 (textbook raw frame) | defence | three | corner3 | n<200 (UNDERPOWERED) | 507 | 91 | 0.038 | 0.073 | 6.965 | 4.116 | 1.692 | 0.635 | UNDERPOWERED |
| GameZone 2022-2024 (textbook raw frame) | defence | three | wing3 | n<200 (UNDERPOWERED) | 507 | 91 | 0.009 | 0.018 | 9.717 | 5.241 | 1.854 | 0.700 | UNDERPOWERED |
| GameZone 2022-2024 (textbook raw frame) | defence | three | top3 | n<200 (UNDERPOWERED) | 507 | 91 | -0.020 | -0.041 | 8.947 | 4.830 | 1.853 | 0.699 | UNDERPOWERED |
| GameZone 2022-2024 (full-affine calibrated) | offence | jump2 | short2 | n<200 (UNDERPOWERED) | 389 | 80 | 0.175 | 0.298 | 11.446 | 5.408 | 2.117 | 0.787 | UNDERPOWERED |
| GameZone 2022-2024 (full-affine calibrated) | offence | three | corner3 | n<200 (UNDERPOWERED) | 531 | 88 |  |  | 0 | 0 |  |  | UNDERPOWERED |
| GameZone 2022-2024 (full-affine calibrated) | offence | three | wing3 | n<200 (UNDERPOWERED) | 531 | 88 | 0.014 | 0.029 | 10.612 | 5.318 | 1.995 | 0.751 | UNDERPOWERED |
| GameZone 2022-2024 (full-affine calibrated) | offence | three | top3 | n<200 (UNDERPOWERED) | 531 | 88 | 0.014 | 0.029 | 10.612 | 5.318 | 1.995 | 0.751 | UNDERPOWERED |
| GameZone 2022-2024 (full-affine calibrated) | defence | jump2 | short2 | n<200 (UNDERPOWERED) | 335 | 73 | -0.021 | -0.044 | 10.192 | 5.689 | 1.791 | 0.705 | UNDERPOWERED |
| GameZone 2022-2024 (full-affine calibrated) | defence | three | corner3 | n<200 (UNDERPOWERED) | 507 | 91 |  |  | 0 | 0 |  |  | UNDERPOWERED |
| GameZone 2022-2024 (full-affine calibrated) | defence | three | wing3 | n<200 (UNDERPOWERED) | 507 | 91 | 0.010 | 0.021 | 10.094 | 5.225 | 1.932 | 0.726 | UNDERPOWERED |
| GameZone 2022-2024 (full-affine calibrated) | defence | three | top3 | n<200 (UNDERPOWERED) | 507 | 91 | 0.010 | 0.021 | 10.094 | 5.225 | 1.932 | 0.726 | UNDERPOWERED |

Plain words, CBBD: CBBD (training source), team-seasons with at least 200 located attempts in the class: team band profiles are a stable trait at a season's attempt count, most clearly on offence and least for wing threes. Offence Spearman-Brown reliability is 0.89 for short2 (split-half r 0.80, median 421 attempts), 0.79 for corner3, 0.67 for wing3 and 0.80 for top3 (median 649 attempts); the between-team SD of the share (9.9 / 4.5 / 3.9 / 4.7 points) is 4.1 / 2.8 / 2.0 / 2.6 times the binomial SD at the median count (2.4 / 1.6 / 2.0 / 1.9 points). Defence (bands allowed) is weaker: 0.69 / 0.76 / 0.52 / 0.75, ratios 2.6 / 2.5 / 1.7 / 2.4. Restricted to the 40%-of-games GameZone-matched subset, reliabilities fall (offence 0.86 / 0.77 / 0.57 / 0.76; defence 0.55 / 0.69 / 0.57 / 0.67), which shows how fast it drops with fewer attempts. Team-seasons under 200 attempts (UNDERPOWERED) are mostly noise: offence threes 0.15-0.23, defence short2 and wing3 about 0.

Plain words, GameZone: GameZone alone, at a similar attempt count, the profiles are mostly noise except for short two-point jumpers. Offence reliability is 0.72 for short2, 0.44 for corner3, -0.03 for wing3 and 0.24 for top3; defence 0.35 / 0.36 / -0.03 / 0.30, with 295-337 team-seasons and medians of 470-650 attempts. The between-team SDs are close to CBBD's on the same games (short2 7.4 vs 9.1 points offence) but the split-half correlation is far lower (CBBD on the same games: 0.86 / 0.77 / 0.57 / 0.76 offence), so GameZone adds within-team variation that does not repeat across halves of a season; the full-affine calibrated version does not help (0.70 short2, 0.31 for wing3 and top3, corner3 empty). Under-200 team-seasons are UNDERPOWERED and near zero.
