# Shot-location round 2: evidence (2026-10-27)

Pre-registration: `docs/models/shot_location/experiments.md` section 3 (commit `5f6d6b0`, before any fit). Scripts committed before they ran: block V input assembler and grader (`c2a3504`), block P build (`18b8f69`; the k_s grid was widened once, `4f8e2e8`, before any arm), trainer (`84350bf`, before the first fit), grader block P (`f13137f`, before block P was graded); a post-hoc descriptive bootstrap was added to the grader at `cc1f4a1` (disclosure 21). Freeze ON. Nothing is adopted, wired or served. Results summary: `experiments.md` section 4. Worker: Sonnet.

## 0. Verdicts

### Block V (diagnostic): DEFECT CONFIRMED for the possession-outcome owner, per class

| class (share of chances, served `round2_s1` F2 2025) | over-dispersion ratio | null mean | floor | excess in floors | outcome | registered latent (logit-normal, method of moments) |
|---|---|---|---|---|---|---|
| `FGA_3` (3PA share) | **1.253** | 1.001 | 0.0088 | +28.8 | **DEFECT CONFIRMED** | s2 = 0.0369, **SD 0.192** |
| `FGA_rim` (rim share) | **1.360** | 1.001 | 0.0097 | +37.0 | **DEFECT CONFIRMED** | s2 = 0.0560, **SD 0.237** |
| `FGA_jump2` | 1.473 | 1.000 | 0.0119 | +39.8 | DEFECT CONFIRMED | s2 = 0.1034, SD 0.322 |
| `TOV` | 1.104 | 1.001 | 0.0073 | +14.2 | DEFECT CONFIRMED | s2 = 0.0221, SD 0.149 |
| FT trips (shooting + bonus) | 0.946 | 0.999 | 0.0068 | -7.8 | NOT CONFIRMED (UNDER-dispersed: the served FT-trip probabilities are wider than the realised team-game share) | s2 = -0.0140, SD 0 |

Band shares (B2 located attempts, team-games with >= 5 located attempts in the class; the F1 row is a BIASED SUBSET):

| band | ratio F2 (floor) | latent SD F2 | ratio F1 | latent SD F1 | outcome |
|---|---|---|---|---|---|
| short2 (long2 identical) | 1.258 (0.0090) | 0.455 | 1.255 | 0.441 | DEFECT CONFIRMED |
| corner3 | 1.200 (0.0090) | 0.353 | 1.181 | 0.351 | DEFECT CONFIRMED |
| wing3 | 1.097 (0.0077) | 0.200 | 1.085 | 0.195 | DEFECT CONFIRMED (smallest; wing3 is also the least reliable band) |
| top3 | 1.144 (0.0085) | 0.259 | 1.118 | 0.244 | DEFECT CONFIRMED |

**Are the band latent and the class latent one latent? No.**
- The residual 3PA share (of chances) is uncorrelated with every band residual (corner3, wing3, top3, short2, long2) once the multinomial mechanics are removed: raw correlations between -0.015 and +0.010 against null means within +/-0.001; disattenuated by the reliability of each residual (class 0.36, bands 0.17-0.37) the correlations are -0.06 to +0.03, boot SE 0.025-0.04 (table V-C). The over-dispersion in how often a team shoots threes and the over-dispersion in where its threes come from are separate.
- The only coupling is at the rim / short-jumper boundary. The rim-share residual and the short2 within-jumper band residual correlate -0.135 (null -0.0005, 12 null SDs; disattenuated -0.33, boot SE 0.03), and the jump2 class residual and short2 correlate +0.141. The rim-vs-jump2 class residuals are also more negatively correlated (-0.448) than the multinomial predicts (-0.281). This is the boundary where the possessions v2 2.27 ft rim override acts (round-1 diagnostic: the override fires on makes more than misses within game). It is a hypothesis for the PM, not a result here.
- The first principal component of the nine residual shares carries 18.2% of their variance against 18.5% (+/-0.1%) under the null: there is no single common shot-selection factor.
- **Where the fix belongs:** the class-level over-dispersion (3PA, rim, jump2, TOV) belongs in possession outcome (the per-game class latent with the registered SDs above is the quantity a variance arm must reproduce). The band-level over-dispersion is a separate, location-stage latent (SD 0.20-0.46 on the band logits) and would only matter if a location stage is ever served; it is not the same latent.

### Block P: NOT CANDIDATE under the literal section 3.2 rule (F2; F1 agrees)

| clause (F2) | result |
|---|---|
| P-C: player-game 3PM MAE improves beyond its floor | YES: MAE 0.6827 -> 0.6823, gain +0.000425 = **2.18 floors** (floor 0.000195) |
| P-C: no 3PA tier worse beyond its floor | YES: tiers 1-2 / 3-5 / 6+ gains +1.81 / +0.14 / +2.00 floors, none negative |
| **P-C: responsiveness by shooter-profile quintile, slope ratio in [0.8, 1.2]** | **NO: slope ratio 2.05 (F1 1.35).** FGM line passes (0.99; F1 1.15). |
| P-A or P-B beats its floor | YES: P-A jump2 +2.96 floors (gate held), P-A three -0.40; P-B +47.5 (jump2) and +46.7 (three) floors |
| **Verdict** | **NOT CANDIDATE** (clause 3 fails on the 3PM line, F2 and F1) |

Headline numbers (F2):

| read | quantity | jump2 | three |
|---|---|---|---|
| reliability (Spearman-Brown, shooter-seasons >= 100 located attempts; n = 764 jump2, 2,427 three) | band share | short2 0.934 | corner3 0.817, wing3 0.585, top3 0.837 |
| | make rate by band | short2 0.257, long2 0.302; contrast 0.173 | corner3 0.058, non-corner 0.135; contrast 0.044 |
| leak test | change form vs verified own-game margin, gate 0.15 | PASS; worst abs corr 0.041 (`sh_band_make_c`, 2025), every column and season inside the gate | |
| P-A (LL gain P1 vs P0, floor) | | +4.83e-4 (1.63e-4) = +2.96 floors, gate held, SIGNAL | -5.2e-5 (1.30e-4) = -0.40 floors, NO SIGNAL |
| P-A (P0 vs R, round-1 A1 replicated) | | +9.42e-4 = +5.77 floors | +1.53e-4 = +1.18 floors |
| P-B (LL BP vs B2, floor) | | 0.6713 -> 0.6236, +0.0477 (0.0010) = **+47.5 floors** | 1.0500 -> 1.0281, +0.0219 (0.00047) = **+46.7 floors** |
| P-C 3PM (71,159 player-games) | MAE (i) / (ii), gain, floor | 0.6827 / 0.6823, +0.000425, 0.000195 (+2.18) | |
| P-C FGM (95,038 player-games) | MAE (i) / (ii), gain, floor | 0.9621 / 0.9613, +0.000782, 0.000206 (+3.80) | |
| dispersion (SD of actual - implied over RMS implied Bernoulli SD), 3PM | (i) / (ii) | 1.005 / 1.003 | |
| player map | attempt join | 100.0% every season (`player_crosswalk_v2`); player-game to hoopR box 99.86% (2025), 99.78% (2024) | |

What the round shows in one line: the shooter's own band profile is highly predictable before tipoff (reliability 0.82-0.93 on corner3 / top3 / short2; B2 -> BP is a 47-floor gain with every responsiveness line at 0.92-1.03), and carrying it through to a player's expected 3PM lowers MAE by 0.0004 (2 floors) with no tier worse, but the registered profile-quintile line fails because its actual span is tiny; the band-specific MAKE skill (the P-A / P1 part) is measurement-limited (make-rate reliabilities 0.06-0.30) and adds only on jumpers.

**Reading note for the PM on clause 3 (post-hoc, descriptive, not a decision input).** The registered line is a ratio of two spans, and its denominator is small: the realised 3P rate moves 0.7 pp (F2) across the five shift quintiles. The bootstrap 95% interval of the 3PM slope ratio for arm (ii) is [1.10, 8.11] on F2 and [1.03, 2.35] on F1. The slope of (actual - R-implied) on the shift itself is 0.78 [0.63, 0.93] on F2 and 0.88 [0.75, 1.00] on F1, i.e. most of the predicted shift is realised; on FGM the registered line passes (0.98 [0.77, 1.36] F2). The literal rule still reads NOT CANDIDATE and I report it that way; whether the ratio form is the right test for a shift this small is a PM question.

## 1. Block V tables

Method conventions are in section 3. Class part is F2 / 2025 only (served artifacts); band part F2 and F1. Parametric-bootstrap null: 200 replicates redrawing every outcome from the model's own probabilities; floor = max(null SD, game-resampled bootstrap SE of the realised ratio); excess in floors = (ratio - null mean) / floor.

### V-A/V-B F2: over-dispersion ratio per class and band
| unit | name | team_games | mean_resid_pp | sd_implied_pp | sd_realised_pp | ratio | null_mean | floor | excess_over_floor | beyond_floor | z_sd |
|---|---|---|---|---|---|---|---|---|---|---|---|
| class | TOV | 10890 | 0.03316 | 4.028 | 4.449 | 1.104 | 1.001 | 0.007295 | 14.23 | True | 1.107 |
| class | rim | 10890 | 0.004469 | 4.982 | 6.774 | 1.36 | 1.001 | 0.009695 | 36.99 | True | 1.367 |
| class | jump2 | 10890 | -0.4121 | 4.268 | 6.289 | 1.473 | 0.9996 | 0.0119 | 39.82 | True | 1.475 |
| class | 3PA | 10890 | 0.3596 | 5.046 | 6.324 | 1.253 | 1.001 | 0.008764 | 28.81 | True | 1.254 |
| class | FT | 10890 | 0.01488 | 3.425 | 3.241 | 0.9463 | 0.9991 | 0.006797 | -7.763 | False | 0.9431 |
| band | short2 | 10358 | 0.8602 | 14.25 | 17.92 | 1.258 | 0.9997 | 0.008988 | 28.73 | True | 1.307 |
| band | long2 | 10358 | -0.8602 | 14.25 | 17.92 | 1.258 | 0.9997 | 0.008988 | 28.73 | True | 1.307 |
| band | corner3 | 10864 | 0.3707 | 8.96 | 10.75 | 1.2 | 0.9989 | 0.008988 | 22.37 | True | 1.223 |
| band | wing3 | 10864 | -0.3639 | 10.89 | 11.95 | 1.097 | 0.9994 | 0.007659 | 12.72 | True | 1.11 |
| band | top3 | 10864 | -0.006776 | 10.35 | 11.84 | 1.144 | 0.9992 | 0.008505 | 17.08 | True | 1.156 |

### V-B latent F2: logit-normal random effect by method of moments
| unit | name | latent_s2 | latent_sigma | sigma_boot_se_s2 | null_s2_sd | latent_beyond |
|---|---|---|---|---|---|---|
| class | TOV | 0.02209 | 0.1486 | 0.001614 | 0.001452 | True |
| class | rim | 0.05604 | 0.2367 | 0.001748 | 0.0008463 | True |
| class | jump2 | 0.1034 | 0.3215 | 0.00307 | 0.001194 | True |
| class | 3PA | 0.03686 | 0.192 | 0.001419 | 0.0008518 | True |
| class | FT | -0.01404 | 0 | 0.001684 | 0.001827 | False |
| band | short2 | 0.2075 | 0.4555 | 0.00799 | 0.004948 | True |
| band | long2 | 0.2075 | 0.4555 | 0.00799 | 0.004948 | True |
| band | corner3 | 0.1249 | 0.3534 | 0.006094 | 0.00394 | True |
| band | wing3 | 0.03981 | 0.1995 | 0.003272 | 0.002589 | True |
| band | top3 | 0.06698 | 0.2588 | 0.004211 | 0.002939 | True |

### V-A F2: within-team-game residual correlation, observed (null mean), n=10356; PC1 share 0.182 (null 0.185 +/- 0.001)
| resid | TOV | rim | jump2 | 3PA | FT | short2 | corner3 | wing3 | top3 |
|---|---|---|---|---|---|---|---|---|---|
| TOV | +1.000 (+1.000) | -0.212 (-0.263) | -0.202 (-0.199) | -0.216 (-0.268) | -0.129 (-0.151) | -0.007 (-0.000) | +0.029 (-0.000) | +0.010 (+0.000) | -0.036 (-0.000) |
| rim | -0.212 (-0.263) | +1.000 (+1.000) | -0.448 (-0.281) | -0.417 (-0.384) | -0.135 (-0.228) | -0.135 (-0.001) | -0.006 (+0.001) | +0.035 (-0.001) | -0.030 (-0.000) |
| jump2 | -0.202 (-0.199) | -0.448 (-0.281) | +1.000 (+1.000) | -0.308 (-0.314) | -0.092 (-0.148) | +0.141 (+0.001) | -0.028 (+0.000) | -0.024 (+0.001) | +0.049 (-0.001) |
| 3PA | -0.216 (-0.268) | -0.417 (-0.384) | -0.308 (-0.314) | +1.000 (+1.000) | -0.188 (-0.206) | +0.005 (+0.000) | +0.013 (-0.001) | -0.018 (+0.000) | +0.006 (+0.001) |
| FT | -0.129 (-0.151) | -0.135 (-0.228) | -0.092 (-0.148) | -0.188 (-0.206) | +1.000 (+1.000) | +0.013 (-0.000) | +0.002 (-0.001) | -0.008 (-0.000) | +0.006 (+0.001) |
| short2 | -0.007 (-0.000) | -0.135 (-0.001) | +0.141 (+0.001) | +0.005 (+0.000) | +0.013 (-0.000) | +1.000 (+1.000) | +0.057 (-0.001) | -0.020 (+0.000) | -0.031 (+0.001) |
| corner3 | +0.029 (-0.000) | -0.006 (+0.001) | -0.028 (+0.000) | +0.013 (-0.001) | +0.002 (-0.001) | +0.057 (-0.001) | +1.000 (+1.000) | -0.461 (-0.470) | -0.442 (-0.370) |
| wing3 | +0.010 (+0.000) | +0.035 (-0.001) | -0.024 (+0.001) | -0.018 (+0.000) | -0.008 (-0.000) | -0.020 (+0.000) | -0.461 (-0.470) | +1.000 (+1.000) | -0.592 (-0.646) |
| top3 | -0.036 (-0.000) | -0.030 (-0.000) | +0.049 (-0.001) | +0.006 (+0.001) | +0.006 (+0.001) | -0.031 (+0.001) | -0.442 (-0.370) | -0.592 (-0.646) | +1.000 (+1.000) |

### V-A F2: residual vs covariates
| resid | covariate | n | corr | floor_1_over_sqrt_n | beyond_floor |
|---|---|---|---|---|---|
| TOV | margin | 10890 | -0.2162 | 0.009583 | True |
| TOV | site_home | 9476 | -0.006875 | 0.01027 | False |
| TOV | n_chances | 10890 | 0.01611 | 0.009583 | False |
| TOV | opp_defence_profile:opp_def_tov_c | 10890 | 0.03136 | 0.009583 | True |
| rim | margin | 10890 | 0.07664 | 0.009583 | True |
| rim | site_home | 9476 | -0.005317 | 0.01027 | False |
| rim | n_chances | 10890 | 0.05461 | 0.009583 | True |
| rim | opp_defence_profile:opp_def_rim_c | 10890 | 0.007776 | 0.009583 | False |
| jump2 | margin | 10890 | -0.0187 | 0.009583 | False |
| jump2 | site_home | 9476 | 0.01817 | 0.01027 | False |
| jump2 | n_chances | 10890 | -0.02332 | 0.009583 | True |
| 3PA | margin | 10890 | 0.05774 | 0.009583 | True |
| 3PA | site_home | 9476 | -0.004771 | 0.01027 | False |
| 3PA | n_chances | 10890 | -0.02645 | 0.009583 | True |
| 3PA | opp_defence_profile:opp_def_3pa_c | 10890 | 0.02929 | 0.009583 | True |
| FT | margin | 10890 | 0.06016 | 0.009583 | True |
| FT | site_home | 9476 | -0.005512 | 0.01027 | False |
| FT | n_chances | 10890 | -0.03939 | 0.009583 | True |
| short2 | margin | 10358 | 0.02259 | 0.009826 | True |
| short2 | site_home | 9004 | 0.002646 | 0.01054 | False |
| short2 | n_chances | 10358 | 0.02296 | 0.009826 | True |
| short2 | opp_defence_profile:def_short2_c | 10358 | -0.02772 | 0.009826 | True |
| corner3 | margin | 10864 | 0.002952 | 0.009594 | False |
| corner3 | site_home | 9462 | -0.005161 | 0.01028 | False |
| corner3 | n_chances | 10864 | 0.009776 | 0.009594 | False |
| corner3 | opp_defence_profile:def_corner3_c | 10864 | 0.02226 | 0.009594 | True |
| wing3 | margin | 10864 | -0.006063 | 0.009594 | False |
| wing3 | site_home | 9462 | -0.001664 | 0.01028 | False |
| wing3 | n_chances | 10864 | -0.0007455 | 0.009594 | False |
| wing3 | opp_defence_profile:def_wing3_c | 10864 | 0.008175 | 0.009594 | False |
| top3 | margin | 10864 | 0.003436 | 0.009594 | False |
| top3 | site_home | 9462 | 0.006357 | 0.01028 | False |
| top3 | n_chances | 10864 | -0.008124 | 0.009594 | False |
| top3 | opp_defence_profile:def_top3_c | 10864 | 0.0223 | 0.009594 | True |

### V-B F2: by offence quintile
| name | offence_quintile | profile | team_games | ratio | z_sd | latent_s2 | mean_resid_pp |
|---|---|---|---|---|---|---|---|
| 3PA | 1 | off_3pa_c | 2178 | 1.243 | 1.238 | 0.03848 | 0.1024 |
| 3PA | 2 | off_3pa_c | 2178 | 1.2 | 1.199 | 0.02961 | 0.09538 |
| 3PA | 3 | off_3pa_c | 2178 | 1.272 | 1.278 | 0.04015 | 0.4654 |
| 3PA | 4 | off_3pa_c | 2178 | 1.264 | 1.269 | 0.03752 | 0.4949 |
| 3PA | 5 | off_3pa_c | 2178 | 1.276 | 1.28 | 0.03756 | 0.6401 |
| rim | 1 | off_rim_c | 2178 | 1.405 | 1.414 | 0.07028 | -0.06233 |
| rim | 2 | off_rim_c | 2178 | 1.383 | 1.386 | 0.06252 | -0.08223 |
| rim | 3 | off_rim_c | 2178 | 1.362 | 1.367 | 0.05649 | -0.04217 |
| rim | 4 | off_rim_c | 2178 | 1.349 | 1.358 | 0.05223 | 0.0394 |
| rim | 5 | off_rim_c | 2178 | 1.302 | 1.308 | 0.04279 | 0.1697 |
| TOV | 1 | off_tov_c | 2178 | 1.113 | 1.118 | 0.02549 | 0.02508 |
| TOV | 2 | off_tov_c | 2178 | 1.078 | 1.076 | 0.01709 | -0.0007229 |
| TOV | 3 | off_tov_c | 2178 | 1.096 | 1.099 | 0.02028 | 0.05517 |
| TOV | 4 | off_tov_c | 2178 | 1.093 | 1.096 | 0.01926 | 0.07671 |
| TOV | 5 | off_tov_c | 2178 | 1.138 | 1.144 | 0.02766 | 0.00954 |
| jump2 | 1 | off_3pa_c | 2178 | 1.499 | 1.5 | 0.1002 | -0.2081 |
| jump2 | 2 | off_3pa_c | 2178 | 1.469 | 1.473 | 0.09744 | -0.1609 |
| jump2 | 3 | off_3pa_c | 2178 | 1.477 | 1.487 | 0.1037 | -0.4322 |
| jump2 | 4 | off_3pa_c | 2178 | 1.406 | 1.413 | 0.09153 | -0.4797 |
| jump2 | 5 | off_3pa_c | 2178 | 1.509 | 1.495 | 0.1291 | -0.7798 |
| short2 | 1 | off_short2_c | 2151 | 1.267 | 1.311 | 0.1986 | 1.031 |
| short2 | 2 | off_short2_c | 2121 | 1.264 | 1.302 | 0.1982 | 0.7197 |
| short2 | 3 | off_short2_c | 2092 | 1.268 | 1.322 | 0.2106 | 0.8355 |
| short2 | 4 | off_short2_c | 2046 | 1.239 | 1.285 | 0.1981 | 1.228 |
| short2 | 5 | off_short2_c | 1948 | 1.255 | 1.314 | 0.232 | 0.4643 |
| corner3 | 1 | off_corner3_c | 2172 | 1.258 | 1.288 | 0.1931 | 0.2938 |
| corner3 | 2 | off_corner3_c | 2174 | 1.171 | 1.179 | 0.1129 | 0.2384 |
| corner3 | 3 | off_corner3_c | 2174 | 1.216 | 1.246 | 0.1353 | 0.6929 |
| corner3 | 4 | off_corner3_c | 2172 | 1.194 | 1.217 | 0.1135 | 0.1556 |
| corner3 | 5 | off_corner3_c | 2172 | 1.17 | 1.181 | 0.09601 | 0.4725 |
| wing3 | 1 | off_wing3_c | 2172 | 1.089 | 1.104 | 0.03669 | -0.6627 |
| wing3 | 2 | off_wing3_c | 2171 | 1.117 | 1.134 | 0.04885 | -0.3751 |
| wing3 | 3 | off_wing3_c | 2173 | 1.092 | 1.1 | 0.03711 | -0.08164 |
| wing3 | 4 | off_wing3_c | 2174 | 1.089 | 1.106 | 0.03642 | -0.4825 |
| wing3 | 5 | off_wing3_c | 2174 | 1.097 | 1.106 | 0.03966 | -0.2177 |
| top3 | 1 | off_top3_c | 2174 | 1.15 | 1.157 | 0.07707 | 0.1245 |
| top3 | 2 | off_top3_c | 2172 | 1.136 | 1.146 | 0.06533 | -0.06586 |
| top3 | 3 | off_top3_c | 2173 | 1.157 | 1.166 | 0.07177 | -0.4431 |
| top3 | 4 | off_top3_c | 2174 | 1.107 | 1.122 | 0.04662 | 0.06787 |
| top3 | 5 | off_top3_c | 2171 | 1.169 | 1.185 | 0.07486 | 0.2828 |

### V-C F2
| class_resid | band_resid | n | corr | null_mean | null_sd | boot_se | reliability_class | reliability_band | disattenuated_corr | disattenuated_boot_se |
|---|---|---|---|---|---|---|---|---|---|---|
| 3PA | corner3 | 10864 | 0.01019 | -0.0008001 | 0.00955 | 0.008484 | 0.3633 | 0.3055 | 0.03059 | 0.02547 |
| 3PA | wing3 | 10864 | -0.01542 | 0.000155 | 0.009685 | 0.009789 | 0.3633 | 0.1688 | -0.06226 | 0.03945 |
| 3PA | top3 | 10864 | 0.006302 | 0.0005258 | 0.009328 | 0.009906 | 0.3633 | 0.2364 | 0.0215 | 0.03398 |
| 3PA | short2 | 10358 | 0.004835 | 0.0002193 | 0.01023 | 0.009856 | 0.3633 | 0.368 | 0.01322 | 0.0275 |
| 3PA | long2 | 10358 | -0.004835 | -0.0002193 | 0.01023 | 0.009856 | 0.3633 | 0.368 | -0.01322 | 0.0275 |
| rim | corner3 | 10864 | -0.001006 | 0.001101 | 0.009439 | 0.00932 | 0.4591 | 0.3055 | -0.002686 | 0.02483 |
| rim | wing3 | 10864 | 0.02841 | -0.0008051 | 0.009836 | 0.009818 | 0.4591 | 0.1688 | 0.102 | 0.03502 |
| rim | top3 | 10864 | -0.02775 | -9.853e-05 | 0.009412 | 0.009636 | 0.4591 | 0.2364 | -0.08422 | 0.02927 |
| rim | short2 | 10358 | -0.1347 | -0.0005036 | 0.009556 | 0.01095 | 0.4591 | 0.368 | -0.3276 | 0.02611 |
| rim | long2 | 10358 | 0.1347 | 0.0005036 | 0.009556 | 0.01095 | 0.4591 | 0.368 | 0.3276 | 0.02611 |

### V-A/V-B F1: over-dispersion ratio per class and band (F1: located subset is BIASED SUBSET; band part only)
| unit | name | team_games | mean_resid_pp | sd_implied_pp | sd_realised_pp | ratio | null_mean | floor | excess_over_floor | beyond_floor | z_sd |
|---|---|---|---|---|---|---|---|---|---|---|---|
| band | short2 | 9105 | -0.7898 | 13.93 | 17.48 | 1.255 | 1 | 0.009314 | 27.32 | True | 1.305 |
| band | long2 | 9105 | 0.7898 | 13.93 | 17.48 | 1.255 | 1 | 0.009314 | 27.32 | True | 1.305 |
| band | corner3 | 9444 | 0.4219 | 9.353 | 11.04 | 1.181 | 0.9997 | 0.009194 | 19.67 | True | 1.213 |
| band | wing3 | 9444 | -0.5028 | 11.41 | 12.37 | 1.085 | 0.9995 | 0.008192 | 10.4 | True | 1.099 |
| band | top3 | 9444 | 0.08087 | 10.84 | 12.12 | 1.118 | 0.9995 | 0.00799 | 14.86 | True | 1.134 |

### V-B latent F1: logit-normal random effect by method of moments
| unit | name | latent_s2 | latent_sigma | sigma_boot_se_s2 | null_s2_sd | latent_beyond |
|---|---|---|---|---|---|---|
| band | short2 | 0.1943 | 0.4408 | 0.007959 | 0.004963 | True |
| band | long2 | 0.1943 | 0.4408 | 0.007959 | 0.004963 | True |
| band | corner3 | 0.123 | 0.3507 | 0.006695 | 0.005081 | True |
| band | wing3 | 0.0379 | 0.1947 | 0.003818 | 0.003445 | True |
| band | top3 | 0.05949 | 0.2439 | 0.004236 | 0.003777 | True |

### V-A F1: within-team-game residual correlation, observed (null mean), n=9048; PC1 share 0.399 (null 0.415 +/- 0.002)
| resid | short2 | corner3 | wing3 | top3 |
|---|---|---|---|---|
| short2 | +1.000 (+1.000) | +0.029 (+0.001) | -0.027 (-0.000) | +0.002 (-0.000) |
| corner3 | +0.029 (+0.001) | +1.000 (+1.000) | -0.469 (-0.470) | -0.435 (-0.369) |
| wing3 | -0.027 (-0.000) | -0.469 (-0.470) | +1.000 (+1.000) | -0.592 (-0.647) |
| top3 | +0.002 (-0.000) | -0.435 (-0.369) | -0.592 (-0.647) | +1.000 (+1.000) |

### V-A F1: residual vs covariates
| resid | covariate | n | corr | floor_1_over_sqrt_n | beyond_floor |
|---|---|---|---|---|---|
| short2 | margin | 9105 | 0.01038 | 0.01048 | False |
| short2 | site_home | 8282 | 0.01072 | 0.01099 | False |
| short2 | opp_defence_profile:def_short2_c | 9105 | -0.02067 | 0.01048 | False |
| corner3 | margin | 9444 | 0.006778 | 0.01029 | False |
| corner3 | site_home | 8546 | 0.009439 | 0.01082 | False |
| corner3 | opp_defence_profile:def_corner3_c | 9444 | -0.02194 | 0.01029 | True |
| wing3 | margin | 9444 | 0.009968 | 0.01029 | False |
| wing3 | site_home | 8546 | -0.00525 | 0.01082 | False |
| wing3 | opp_defence_profile:def_wing3_c | 9444 | -0.002515 | 0.01029 | False |
| top3 | margin | 9444 | -0.01636 | 0.01029 | False |
| top3 | site_home | 8546 | -0.003302 | 0.01082 | False |
| top3 | opp_defence_profile:def_top3_c | 9444 | 0.00207 | 0.01029 | False |

### V-B F1: by offence quintile
| name | offence_quintile | profile | team_games | ratio | z_sd | latent_s2 | mean_resid_pp |
|---|---|---|---|---|---|---|---|
| short2 | 1 | off_short2_c | 1877 | 1.272 | 1.321 | 0.1956 | 0.1951 |
| short2 | 2 | off_short2_c | 1841 | 1.277 | 1.329 | 0.1996 | -0.7347 |
| short2 | 3 | off_short2_c | 1841 | 1.23 | 1.272 | 0.1735 | -1.356 |
| short2 | 4 | off_short2_c | 1816 | 1.25 | 1.299 | 0.1947 | -1.044 |
| short2 | 5 | off_short2_c | 1730 | 1.248 | 1.3 | 0.2071 | -1.048 |
| corner3 | 1 | off_corner3_c | 1892 | 1.24 | 1.277 | 0.195 | 0.3697 |
| corner3 | 2 | off_corner3_c | 1887 | 1.19 | 1.21 | 0.1364 | 0.7465 |
| corner3 | 3 | off_corner3_c | 1874 | 1.145 | 1.181 | 0.0966 | 0.28 |
| corner3 | 4 | off_corner3_c | 1914 | 1.139 | 1.158 | 0.09019 | 0.09592 |
| corner3 | 5 | off_corner3_c | 1877 | 1.194 | 1.235 | 0.1197 | 0.6224 |
| wing3 | 1 | off_wing3_c | 1835 | 1.104 | 1.123 | 0.04759 | -0.5911 |
| wing3 | 2 | off_wing3_c | 1881 | 1.079 | 1.093 | 0.03577 | -0.1523 |
| wing3 | 3 | off_wing3_c | 1914 | 1.083 | 1.099 | 0.0366 | -0.4268 |
| wing3 | 4 | off_wing3_c | 1916 | 1.068 | 1.083 | 0.02942 | -0.4624 |
| wing3 | 5 | off_wing3_c | 1898 | 1.089 | 1.098 | 0.04032 | -0.8822 |
| top3 | 1 | off_top3_c | 1900 | 1.106 | 1.135 | 0.05888 | 0.2279 |
| top3 | 2 | off_top3_c | 1874 | 1.106 | 1.122 | 0.05459 | 0.14 |
| top3 | 3 | off_top3_c | 1909 | 1.107 | 1.119 | 0.05185 | -0.07758 |
| top3 | 4 | off_top3_c | 1892 | 1.13 | 1.142 | 0.06437 | 0.3283 |
| top3 | 5 | off_top3_c | 1869 | 1.139 | 1.151 | 0.0662 | -0.2166 |

## 2. Block P tables

Floors: P-A = max(|LL R - LL R_seed|, paired game-block bootstrap SE of the arm difference); P-B = paired game-block bootstrap SE (the GLM is deterministic, seed floor 0); P-C = max(player-game, game-block) bootstrap SE, 200 replicates. Cells under 5,000 attempts (P-A, P-B) and 2,000 player-games (P-C) are UNDERPOWERED; none occurs. F1 is a BIASED SUBSET (located share 87.5%).

### Reliability (split-half, Spearman-Brown); shooter-seasons >= 100 located attempts in the class (the 40-99 stratum is UNDERPOWERED and lives in build_report_v1.json)
| quantity | class | band | shooter_seasons | split_half_r | spearman_brown |
|---|---|---|---|---|---|
| band_share | FGA_jump2 | short2 | 764 | 0.8759 | 0.9338 |
| band_share | FGA_jump2 | long2 | 764 | 0.8759 | 0.9338 |
| make_rate | FGA_jump2 | short2 | 667 | 0.1472 | 0.2566 |
| make_rate | FGA_jump2 | long2 | 747 | 0.1781 | 0.3024 |
| make_rate_contrast | FGA_jump2 | short2 - long2 | 650 | 0.09456 | 0.1728 |
| band_share | FGA_3 | corner3 | 2427 | 0.6905 | 0.8169 |
| band_share | FGA_3 | wing3 | 2427 | 0.4139 | 0.5855 |
| band_share | FGA_3 | top3 | 2427 | 0.719 | 0.8365 |
| make_rate | FGA_3 | corner3 | 1400 | 0.02966 | 0.05762 |
| make_rate | FGA_3 | noncorner3 | 2427 | 0.0722 | 0.1347 |
| make_rate_contrast | FGA_3 | corner3 - noncorner3 | 1400 | 0.02227 | 0.04357 |

### P-A F1: log loss gain (positive = first arm better); floor = max(|LL R - LL R_seed|, game-block bootstrap SE) (F1: located subset is a BIASED SUBSET)
| cls | comparison | gain | seed_diff | boot_se | floor | gain_in_floors | gate_held | beats_floor |
|---|---|---|---|---|---|---|---|---|
| FGA_jump2 | P1_vs_P0 | 0.0002213 | 6.798e-05 | 0.0001533 | 0.0001533 | 1.444 | True | True |
| FGA_jump2 | P0_vs_R | 0.0005777 | 6.798e-05 | 0.0001604 | 0.0001604 | 3.602 | False | True |
| FGA_3 | P1_vs_P0 | 0.0001169 | 0.0001726 | 9.288e-05 | 0.0001726 | 0.6771 | True | False |
| FGA_3 | P0_vs_R | 0.0002907 | 0.0001726 | 9.433e-05 | 0.0001726 | 1.685 | True | True |

### P-A F1: decile battery as served (worst gap / level / shape, pp)
| cls | arm | log_loss | worst_decile_gap_pp | level_pp | shape_pp |
|---|---|---|---|---|---|
| FGA_jump2 | R | 0.6653 | 2.919 | -0.266 | 3.185 |
| FGA_jump2 | R_seed | 0.6652 | 2.993 | -0.287 | 3.28 |
| FGA_jump2 | P0 | 0.6647 | 3.065 | -0.263 | 2.802 |
| FGA_jump2 | P1 | 0.6645 | 2.645 | -0.223 | 2.461 |
| FGA_3 | R | 0.6382 | 2.03 | -0.133 | 2.163 |
| FGA_3 | R_seed | 0.6383 | 2.831 | -0.079 | 2.909 |
| FGA_3 | P0 | 0.6379 | 2.417 | -0.105 | 2.522 |
| FGA_3 | P1 | 0.6378 | 2.061 | -0.087 | 2.147 |

### P-A F1: make rate by quintile of the shooter's within-band skill smk_<band> (responsiveness)
| cls | arm | band | slope_ratio | steps | span_pred | span_actual | in_0_8_1_2 | d8_pass | pred | actual |
|---|---|---|---|---|---|---|---|---|---|---|
| FGA_jump2 | R | short2 | 0.6099 | 3 | 0.03172 | 0.05201 | False | False | [0.37666, 0.37533, 0.37606, 0.3878, 0.40838] | [0.38488, 0.38913, 0.39062, 0.41017, 0.43689] |
| FGA_jump2 | R | long2 | 0.698 | 3 | 0.03305 | 0.04734 | False | False | [0.37112, 0.3704, 0.37202, 0.38601, 0.40417] | [0.3543, 0.35761, 0.36428, 0.38039, 0.40164] |
| FGA_jump2 | P0 | short2 | 0.5934 | 4 | 0.03087 | 0.05201 | False | False | [0.39476, 0.39477, 0.39507, 0.40662, 0.42563] | [0.38488, 0.38913, 0.39062, 0.41017, 0.43689] |
| FGA_jump2 | P0 | long2 | 0.724 | 3 | 0.03428 | 0.04734 | False | False | [0.3564, 0.35468, 0.35726, 0.37153, 0.39068] | [0.3543, 0.35761, 0.36428, 0.38039, 0.40164] |
| FGA_jump2 | P1 | short2 | 0.783 | 2 | 0.04072 | 0.05201 | False | False | [0.39222, 0.39188, 0.38965, 0.40986, 0.43294] | [0.38488, 0.38913, 0.39062, 0.41017, 0.43689] |
| FGA_jump2 | P1 | long2 | 1.022 | 3 | 0.04837 | 0.04734 | True | False | [0.3527, 0.34864, 0.35318, 0.37878, 0.40107] | [0.3543, 0.35761, 0.36428, 0.38039, 0.40164] |
| FGA_3 | R | corner3 | 0.493 | 3 | 0.01875 | 0.03804 | False | False | [0.33522, 0.33063, 0.33182, 0.3415, 0.35397] | [0.34473, 0.34688, 0.3459, 0.37095, 0.38276] |
| FGA_3 | R | wing3 | 0.7261 | 3 | 0.02029 | 0.02795 | False | False | [0.3322, 0.32905, 0.33007, 0.34161, 0.35249] | [0.32809, 0.3242, 0.32737, 0.3488, 0.35604] |
| FGA_3 | R | top3 | 0.8236 | 3 | 0.01989 | 0.02415 | True | False | [0.32991, 0.32612, 0.32816, 0.33906, 0.3498] | [0.31871, 0.3188, 0.31842, 0.33229, 0.34286] |
| FGA_3 | P0 | corner3 | 0.6268 | 3 | 0.02384 | 0.03804 | False | False | [0.35119, 0.34668, 0.34804, 0.3607, 0.37504] | [0.34473, 0.34688, 0.3459, 0.37095, 0.38276] |
| FGA_3 | P0 | wing3 | 0.811 | 3 | 0.02266 | 0.02795 | True | False | [0.32967, 0.32677, 0.32788, 0.34047, 0.35233] | [0.32809, 0.3242, 0.32737, 0.3488, 0.35604] |
| FGA_3 | P0 | top3 | 0.5657 | 3 | 0.01366 | 0.02415 | False | False | [0.32455, 0.31873, 0.32018, 0.32962, 0.33821] | [0.31871, 0.3188, 0.31842, 0.33229, 0.34286] |
| FGA_3 | P1 | corner3 | 0.7864 | 3 | 0.02991 | 0.03804 | False | False | [0.34951, 0.34427, 0.34574, 0.36367, 0.37943] | [0.34473, 0.34688, 0.3459, 0.37095, 0.38276] |
| FGA_3 | P1 | wing3 | 0.9936 | 3 | 0.02777 | 0.02795 | True | False | [0.32722, 0.3242, 0.32702, 0.34336, 0.35499] | [0.32809, 0.3242, 0.32737, 0.3488, 0.35604] |
| FGA_3 | P1 | top3 | 0.7203 | 3 | 0.01739 | 0.02415 | False | False | [0.3239, 0.3163, 0.32011, 0.33229, 0.3413] | [0.31871, 0.3188, 0.31842, 0.33229, 0.34286] |

### P-A F1: P1 vs P0 by segment (cells < 5,000 attempts UNDERPOWERED)
| cls | segment | n | gain | boot_se | in_floors | underpowered |
|---|---|---|---|---|---|---|
| FGA_jump2 | site=away | 64475 | 0.0002209 | 0.0002197 | 1.006 | False |
| FGA_jump2 | site=home | 59433 | 0.0002877 | 0.0002435 | 1.182 | False |
| FGA_jump2 | site=neutral | 12607 | -8.934e-05 | 0.0004835 | -0.1848 | False |
| FGA_jump2 | month=Dec | 24536 | -3.169e-05 | 0.000367 | -0.08636 | False |
| FGA_jump2 | month=Feb | 34890 | -8.031e-05 | 0.0002406 | -0.3337 | False |
| FGA_jump2 | month=Jan | 32713 | 0.0002042 | 0.0003148 | 0.6487 | False |
| FGA_jump2 | month=Mar+Apr | 17692 | 0.0003561 | 0.0003767 | 0.9452 | False |
| FGA_jump2 | month=Nov | 26684 | 0.0007801 | 0.0003181 | 2.452 | False |
| FGA_jump2 | tier=low | 81878 | 0.0001042 | 0.0002032 | 0.5125 | False |
| FGA_jump2 | tier=mid | 22555 | 0.0003369 | 0.0003315 | 1.016 | False |
| FGA_jump2 | tier=power | 32082 | 0.0004391 | 0.0003481 | 1.261 | False |
| FGA_jump2 | locq=Q1 | 22116 | 0.000393 | 0.0003921 | 1.002 | False |
| FGA_jump2 | locq=Q2 | 40295 | 0.0001455 | 0.0002634 | 0.5526 | False |
| FGA_jump2 | locq=Q3 | 35836 | 0.0001355 | 0.0002938 | 0.4613 | False |
| FGA_jump2 | locq=Q4 | 38268 | 0.0002823 | 0.0003009 | 0.9382 | False |
| FGA_3 | site=away | 91659 | 0.0002554 | 0.0001428 | 1.789 | False |
| FGA_3 | site=home | 92824 | -2.182e-06 | 0.0001402 | -0.01556 | False |
| FGA_3 | site=neutral | 18137 | 2.606e-05 | 0.0003192 | 0.08162 | False |
| FGA_3 | month=Dec | 37526 | 4.691e-05 | 0.0002499 | 0.1877 | False |
| FGA_3 | month=Feb | 50806 | 0.0001072 | 0.0001926 | 0.5567 | False |
| FGA_3 | month=Jan | 48602 | -2.32e-05 | 0.0001857 | -0.1249 | False |
| FGA_3 | month=Mar+Apr | 25823 | 0.0001139 | 0.0002522 | 0.4515 | False |
| FGA_3 | month=Nov | 39863 | 0.0003677 | 0.000228 | 1.613 | False |
| FGA_3 | tier=low | 118631 | 0.0001518 | 0.000133 | 1.142 | False |
| FGA_3 | tier=mid | 37121 | 3.004e-05 | 0.0002035 | 0.1476 | False |
| FGA_3 | tier=power | 46868 | 9.711e-05 | 0.000206 | 0.4714 | False |
| FGA_3 | locq=Q1 | 32329 | -4.483e-05 | 0.000246 | -0.1822 | False |
| FGA_3 | locq=Q2 | 56897 | 0.0002387 | 0.0001866 | 1.279 | False |
| FGA_3 | locq=Q3 | 57363 | 5.056e-05 | 0.0001893 | 0.267 | False |
| FGA_3 | locq=Q4 | 56031 | 0.0001543 | 0.000169 | 0.9134 | False |

### P-B F1: multinomial log loss, B2 vs BP (floor = paired game-block bootstrap SE; seed floor 0) (F1: located subset is a BIASED SUBSET)
| cls | LL_B2 | LL_BP | gain | floor | gain_in_floors | beats_floor |
|---|---|---|---|---|---|---|
| FGA_jump2 | 0.6703 | 0.6198 | 0.05058 | 0.0008555 | 59.13 | True |
| FGA_3 | 1.051 | 1.028 | 0.02357 | 0.0005267 | 44.74 | True |

### P-B F1: realised band share by quintile of the shooter's own as-of share (wing3 WEAKER)
| cls | arm | band | slope_ratio | steps | in_0_8_1_2 | d8_pass | pred | actual |
|---|---|---|---|---|---|---|---|---|
| FGA_jump2 | B2 | short2 | 0.2236 | 4 | False | False | [0.385, 0.41809, 0.44483, 0.46811, 0.49313] | [0.23074, 0.33297, 0.41783, 0.51525, 0.71421] |
| FGA_jump2 | BP | short2 | 0.942 | 4 | True | True | [0.23344, 0.34297, 0.42621, 0.51656, 0.68887] | [0.23074, 0.33297, 0.41783, 0.51525, 0.71421] |
| FGA_3 | B2 | corner3 | 0.2029 | 4 | False | False | [0.1916, 0.2062, 0.21372, 0.22017, 0.23323] | [0.11951, 0.1752, 0.21298, 0.25427, 0.32465] |
| FGA_3 | B2 | wing3 | 0.2788 | 4 | False | False | [0.43308, 0.43976, 0.44708, 0.45369, 0.46313] | [0.38375, 0.42237, 0.44926, 0.46506, 0.49154] |
| FGA_3 | B2 | top3 | 0.1727 | 4 | False | False | [0.31731, 0.332, 0.34016, 0.34765, 0.36121] | [0.22537, 0.28793, 0.33062, 0.37797, 0.47952] |
| FGA_3 | BP | corner3 | 0.9471 | 4 | True | True | [0.12143, 0.17318, 0.20955, 0.24722, 0.31571] | [0.11951, 0.1752, 0.21298, 0.25427, 0.32465] |
| FGA_3 | BP | wing3 | 0.9049 | 4 | True | True | [0.39651, 0.42954, 0.44793, 0.46586, 0.49405] | [0.38375, 0.42237, 0.44926, 0.46506, 0.49154] |
| FGA_3 | BP | top3 | 0.9672 | 4 | True | True | [0.22535, 0.28995, 0.3336, 0.37895, 0.47117] | [0.22537, 0.28793, 0.33062, 0.37797, 0.47952] |

### P-B F1: BP vs B2 by segment
| cls | segment | n | gain | boot_se | in_floors | underpowered |
|---|---|---|---|---|---|---|
| FGA_jump2 | site=away | 64475 | 0.0491 | 0.001338 | 36.69 | False |
| FGA_jump2 | site=home | 59433 | 0.0519 | 0.001329 | 39.04 | False |
| FGA_jump2 | site=neutral | 12607 | 0.05197 | 0.003252 | 15.98 | False |
| FGA_jump2 | month=Dec | 24536 | 0.04618 | 0.002119 | 21.79 | False |
| FGA_jump2 | month=Feb | 34890 | 0.05656 | 0.00203 | 27.86 | False |
| FGA_jump2 | month=Jan | 32713 | 0.05494 | 0.001851 | 29.68 | False |
| FGA_jump2 | month=Mar+Apr | 17692 | 0.05696 | 0.002882 | 19.76 | False |
| FGA_jump2 | month=Nov | 26684 | 0.03725 | 0.001854 | 20.09 | False |
| FGA_jump2 | tier=low | 81878 | 0.04536 | 0.001149 | 39.49 | False |
| FGA_jump2 | tier=mid | 22555 | 0.05978 | 0.002336 | 25.59 | False |
| FGA_jump2 | tier=power | 32082 | 0.05744 | 0.002045 | 28.09 | False |
| FGA_jump2 | locq=Q1 | 22116 | 0.04993 | 0.002369 | 21.08 | False |
| FGA_jump2 | locq=Q2 | 40295 | 0.04651 | 0.001664 | 27.95 | False |
| FGA_jump2 | locq=Q3 | 35836 | 0.04509 | 0.0018 | 25.05 | False |
| FGA_jump2 | locq=Q4 | 38268 | 0.0604 | 0.001658 | 36.43 | False |
| FGA_3 | site=away | 91659 | 0.02304 | 0.0007612 | 30.27 | False |
| FGA_3 | site=home | 92824 | 0.02394 | 0.0007506 | 31.9 | False |
| FGA_3 | site=neutral | 18137 | 0.02429 | 0.001685 | 14.41 | False |
| FGA_3 | month=Dec | 37526 | 0.02164 | 0.00111 | 19.48 | False |
| FGA_3 | month=Feb | 50806 | 0.02641 | 0.001174 | 22.49 | False |
| FGA_3 | month=Jan | 48602 | 0.02623 | 0.001119 | 23.44 | False |
| FGA_3 | month=Mar+Apr | 25823 | 0.02842 | 0.001719 | 16.53 | False |
| FGA_3 | month=Nov | 39863 | 0.01538 | 0.0009549 | 16.1 | False |
| FGA_3 | tier=low | 118631 | 0.02178 | 0.0006481 | 33.6 | False |
| FGA_3 | tier=mid | 37121 | 0.02942 | 0.001193 | 24.67 | False |
| FGA_3 | tier=power | 46868 | 0.02346 | 0.001134 | 20.68 | False |
| FGA_3 | locq=Q1 | 32329 | 0.02332 | 0.001227 | 19 | False |
| FGA_3 | locq=Q2 | 56897 | 0.02185 | 0.000887 | 24.63 | False |
| FGA_3 | locq=Q3 | 57363 | 0.02116 | 0.001007 | 21.01 | False |
| FGA_3 | locq=Q4 | 56031 | 0.02792 | 0.001053 | 26.52 | False |

### P-C F1: player-game join counts (F1: located subset is a BIASED SUBSET)
```
{
 "player_games_in_design": 90950,
 "no_athlete_map": 0,
 "no_box_row": 201,
 "joined": 90749,
 "join_rate": 0.9977899945024739,
 "3pa_count_mismatch_dropped_from_3PM": 260,
 "fga_count_mismatch_dropped_from_FGM": 921,
 "3pm_design_equals_box_among_matched_attempts": 0.9999848810135769
}
```

### P-C F1: player-game 3PM, (i) served R vs (ii) P1 over the BP band mix (gain = MAE(i) - MAE(ii); floor = max(player-game, game-block bootstrap SE); dispersion ratio = SD(actual - implied) / RMS(implied Bernoulli SD))
| label | n_player_games | underpowered | mean_actual | mean_i | mean_ii | mae_i | mae_ii | gain | se_player_game | se_game_block | floor | gain_in_floors | dispersion_ratio_i | dispersion_ratio_ii |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 66142 | False | 1.148 | 1.141 | 1.142 | 0.6737 | 0.6728 | 0.0008728 | 0.0002441 | 0.0002471 | 0.0002471 | 3.533 | 0.9994 | 0.9976 |
| all attempts located | 58100 | False | 1.136 | 1.131 | 1.132 | 0.6714 | 0.6705 | 0.0008421 | 0.0002103 | 0.0002529 | 0.0002529 | 3.329 | 1.001 | 0.9987 |
| 3PA 1-2 | 30219 | False | 0.3727 | 0.4727 | 0.4709 | 0.4829 | 0.4823 | 0.0005797 | 0.0001392 | 0.0001536 | 0.0001536 | 3.775 | 0.9372 | 0.9378 |
| 3PA 3-5 | 24188 | False | 1.275 | 1.285 | 1.286 | 0.7291 | 0.7286 | 0.0004894 | 0.0003852 | 0.0003643 | 0.0003852 | 1.271 | 0.9933 | 0.9924 |
| 3PA 6+ | 11735 | False | 2.884 | 2.566 | 2.576 | 1.051 | 1.048 | 0.002418 | 0.0009674 | 0.001011 | 0.001011 | 2.391 | 0.9968 | 0.9951 |

### P-C F1: player-game FGM (rim attempts held at the pooled training-seasons rim make rate in both arms)
| label | n_player_games | underpowered | mean_actual | mean_i | mean_ii | mae_i | mae_ii | gain | se_player_game | se_game_block | floor | gain_in_floors | dispersion_ratio_i | dispersion_ratio_ii |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 89828 | False | 2.925 | 2.931 | 2.932 | 0.9683 | 0.9673 | 0.001013 | 0.0002353 | 0.0002369 | 0.0002369 | 4.275 | 1.02 | 1.019 |
| all attempts located | 80181 | False | 2.867 | 2.88 | 2.882 | 0.9591 | 0.958 | 0.001114 | 0.000241 | 0.0002474 | 0.0002474 | 4.501 | 1.02 | 1.019 |
| FGA 1-4 | 35174 | False | 0.9789 | 1.077 | 1.075 | 0.6166 | 0.6162 | 0.000416 | 0.000151 | 0.0001583 | 0.0001583 | 2.627 | 0.9984 | 0.9986 |
| FGA 5-9 | 32385 | False | 3.03 | 3.033 | 3.034 | 1.044 | 1.043 | 0.0009802 | 0.0003803 | 0.0003304 | 0.0003803 | 2.577 | 1.03 | 1.029 |
| FGA 10+ | 22269 | False | 5.847 | 5.711 | 5.718 | 1.414 | 1.412 | 0.002004 | 0.0007661 | 0.0007179 | 0.0007661 | 2.616 | 1.013 | 1.012 |

### P-C F1 3PM: responsiveness by quintile of the shooter's profile shift (per-attempt shift (ii)-(i); delta SD 0.01582, range -0.0907..0.1); slope of (actual - (i)) on shift = 0.8817218459893406; POST-HOC bootstrap 95% (not a decision input): slope ratio (ii) [1.031, 1.342, 2.349], excess slope [0.745, 0.883, 1.002]
| arm | pred | actual | span_pred | span_actual | slope_ratio | steps | raw_slope_in_0.8_1.2 |
|---|---|---|---|---|---|---|---|
| (i) R | [0.34872, 0.34032, 0.33632, 0.33277, 0.32575] | [0.32884, 0.33607, 0.34118, 0.34166, 0.34319] | -0.02297 | 0.01436 | -1.6 | 0 | False |
| (ii) P1 x BP | [0.32731, 0.33256, 0.33599, 0.33984, 0.34667] | [0.32884, 0.33607, 0.34118, 0.34166, 0.34319] | 0.01936 | 0.01436 | 1.349 | 4 | False |

quintile mean shift [-0.02141, -0.00776, -0.00033, 0.00707, 0.02093]; actual - (i) [-0.01988, -0.00425, 0.00486, 0.00889, 0.01745]

### P-C F1 FGM: responsiveness by quintile of the shooter's profile shift (per-attempt shift (ii)-(i); delta SD 0.01075, range -0.1..0.0933); slope of (actual - (i)) on shift = 0.8974025083029472; POST-HOC bootstrap 95% (not a decision input): slope ratio (ii) [0.843, 1.154, 1.656], excess slope [0.756, 0.898, 1.06]
| arm | pred | actual | span_pred | span_actual | slope_ratio | steps | raw_slope_in_0.8_1.2 |
|---|---|---|---|---|---|---|---|
| (i) R | [0.42361, 0.44656, 0.49239, 0.44547, 0.41208] | [0.4106, 0.44297, 0.49234, 0.44547, 0.42359] | -0.01153 | 0.01299 | -0.8872 | 2 | False |
| (ii) P1 x BP | [0.41045, 0.44279, 0.49228, 0.44891, 0.42538] | [0.4106, 0.44297, 0.49234, 0.44547, 0.42359] | 0.01493 | 0.01299 | 1.149 | 2 | True |

quintile mean shift [-0.01316, -0.00377, -0.00011, 0.00344, 0.0133]; actual - (i) [-0.01301, -0.0036, -4e-05, 0.0, 0.01151]

### Block-P decision F1 (literal section 3.2)
```
{
 "fold": "F1",
 "c1_3PM_MAE_improves_beyond_floor": true,
 "c1_gain": 0.0008728230342043881,
 "c1_floor": 0.0002470667070036575,
 "c2_no_tier_worse_beyond_floor": true,
 "c3_profile_quintile_slope_in_0.8_1.2": false,
 "c3_slope_ratio": 1.3488,
 "c3_reliability_sqrt": {
  "sqrt_rel_make_contrast_short2_long2": 0.41567299328255614,
  "sqrt_rel_make_contrast_corner3_noncorner3": 0.20873691750659795
 },
 "c4_P_A_or_P_B_beats_floor": true,
 "P_A_beats": true,
 "P_B_beats": true,
 "verdict": "NOT CANDIDATE"
}
```

### P-A F2: log loss gain (positive = first arm better); floor = max(|LL R - LL R_seed|, game-block bootstrap SE)
| cls | comparison | gain | seed_diff | boot_se | floor | gain_in_floors | gate_held | beats_floor |
|---|---|---|---|---|---|---|---|---|
| FGA_jump2 | P1_vs_P0 | 0.0004829 | 0.0001634 | 0.0001349 | 0.0001634 | 2.956 | True | True |
| FGA_jump2 | P0_vs_R | 0.0009421 | 0.0001634 | 0.0001551 | 0.0001634 | 5.766 | True | True |
| FGA_3 | P1_vs_P0 | -5.219e-05 | 0.0001296 | 7.964e-05 | 0.0001296 | -0.4026 | True | False |
| FGA_3 | P0_vs_R | 0.0001526 | 0.0001296 | 7.383e-05 | 0.0001296 | 1.177 | True | True |

### P-A F2: decile battery as served (worst gap / level / shape, pp)
| cls | arm | log_loss | worst_decile_gap_pp | level_pp | shape_pp |
|---|---|---|---|---|---|
| FGA_jump2 | R | 0.6655 | 2.228 | 0.043 | 2.271 |
| FGA_jump2 | R_seed | 0.6653 | 1.995 | 0.032 | 1.963 |
| FGA_jump2 | P0 | 0.6645 | 1.733 | 0.079 | 1.654 |
| FGA_jump2 | P1 | 0.6641 | 1.289 | 0.082 | 1.371 |
| FGA_3 | R | 0.6377 | 2.024 | -0.072 | 2.096 |
| FGA_3 | R_seed | 0.6378 | 1.82 | -0.076 | 1.811 |
| FGA_3 | P0 | 0.6375 | 1.98 | -0.045 | 2.024 |
| FGA_3 | P1 | 0.6376 | 2.097 | -0.024 | 2.12 |

### P-A F2: make rate by quintile of the shooter's within-band skill smk_<band> (responsiveness)
| cls | arm | band | slope_ratio | steps | span_pred | span_actual | in_0_8_1_2 | d8_pass | pred | actual |
|---|---|---|---|---|---|---|---|---|---|---|
| FGA_jump2 | R | short2 | 0.6077 | 4 | 0.03983 | 0.06553 | False | False | [0.37859, 0.38063, 0.38275, 0.39474, 0.41842] | [0.38487, 0.3979, 0.39577, 0.41965, 0.4504] |
| FGA_jump2 | R | long2 | 0.8287 | 4 | 0.04009 | 0.04838 | True | True | [0.37384, 0.3751, 0.37722, 0.39108, 0.41392] | [0.35437, 0.35101, 0.3585, 0.3819, 0.40274] |
| FGA_jump2 | P0 | short2 | 0.6009 | 4 | 0.03938 | 0.06553 | False | False | [0.39648, 0.39822, 0.39984, 0.41224, 0.43586] | [0.38487, 0.3979, 0.39577, 0.41965, 0.4504] |
| FGA_jump2 | P0 | long2 | 0.8814 | 4 | 0.04264 | 0.04838 | True | True | [0.35889, 0.35939, 0.3634, 0.37757, 0.40153] | [0.35437, 0.35101, 0.3585, 0.3819, 0.40274] |
| FGA_jump2 | P1 | short2 | 0.8206 | 4 | 0.05378 | 0.06553 | True | True | [0.39208, 0.39371, 0.3944, 0.4188, 0.44586] | [0.38487, 0.3979, 0.39577, 0.41965, 0.4504] |
| FGA_jump2 | P1 | long2 | 1.123 | 3 | 0.05432 | 0.04838 | True | False | [0.3548, 0.35374, 0.3588, 0.38274, 0.40912] | [0.35437, 0.35101, 0.3585, 0.3819, 0.40274] |
| FGA_3 | R | corner3 | 0.7076 | 3 | 0.02444 | 0.03454 | False | False | [0.3321, 0.33109, 0.33167, 0.34276, 0.35654] | [0.34429, 0.34357, 0.35056, 0.35705, 0.37883] |
| FGA_3 | R | wing3 | 0.9196 | 3 | 0.02628 | 0.02858 | True | False | [0.32917, 0.32855, 0.33049, 0.34313, 0.35545] | [0.32451, 0.32942, 0.3276, 0.3429, 0.35309] |
| FGA_3 | R | top3 | 0.7602 | 3 | 0.02561 | 0.03369 | False | False | [0.32699, 0.32602, 0.32844, 0.34058, 0.35261] | [0.31724, 0.324, 0.32274, 0.33106, 0.35093] |
| FGA_3 | P0 | corner3 | 0.8312 | 4 | 0.02871 | 0.03454 | True | True | [0.34765, 0.34769, 0.34853, 0.36105, 0.37636] | [0.34429, 0.34357, 0.35056, 0.35705, 0.37883] |
| FGA_3 | P0 | wing3 | 1.016 | 3 | 0.02904 | 0.02858 | True | False | [0.32626, 0.32623, 0.32817, 0.34135, 0.35529] | [0.32451, 0.32942, 0.3276, 0.3429, 0.35309] |
| FGA_3 | P0 | top3 | 0.6054 | 3 | 0.0204 | 0.03369 | False | False | [0.32213, 0.31899, 0.32059, 0.33098, 0.34253] | [0.31724, 0.324, 0.32274, 0.33106, 0.35093] |
| FGA_3 | P1 | corner3 | 0.9386 | 3 | 0.03242 | 0.03454 | True | False | [0.34597, 0.34569, 0.34695, 0.36362, 0.37839] | [0.34429, 0.34357, 0.35056, 0.35705, 0.37883] |
| FGA_3 | P1 | wing3 | 1.131 | 3 | 0.03233 | 0.02858 | True | False | [0.32503, 0.32474, 0.32699, 0.34484, 0.35737] | [0.32451, 0.32942, 0.3276, 0.3429, 0.35309] |
| FGA_3 | P1 | top3 | 0.6961 | 3 | 0.02345 | 0.03369 | False | False | [0.32095, 0.31709, 0.31931, 0.33477, 0.3444] | [0.31724, 0.324, 0.32274, 0.33106, 0.35093] |

### P-A F2: P1 vs P0 by segment (cells < 5,000 attempts UNDERPOWERED)
| cls | segment | n | gain | boot_se | in_floors | underpowered |
|---|---|---|---|---|---|---|
| FGA_jump2 | site=away | 65670 | 0.0004898 | 0.000193 | 2.537 | False |
| FGA_jump2 | site=home | 61404 | 0.0005536 | 0.0002011 | 2.753 | False |
| FGA_jump2 | site=neutral | 19853 | 0.0002413 | 0.0003519 | 0.6856 | False |
| FGA_jump2 | month=Dec | 24133 | -5.575e-05 | 0.0003495 | -0.1595 | False |
| FGA_jump2 | month=Feb | 32822 | 0.0005276 | 0.0002507 | 2.104 | False |
| FGA_jump2 | month=Jan | 36979 | 0.0005643 | 0.0002195 | 2.571 | False |
| FGA_jump2 | month=Mar+Apr | 21561 | 0.0003592 | 0.0003127 | 1.149 | False |
| FGA_jump2 | month=Nov | 31432 | 0.0008389 | 0.0002873 | 2.92 | False |
| FGA_jump2 | tier=low | 86023 | 0.0003441 | 0.0001563 | 2.202 | False |
| FGA_jump2 | tier=mid | 24877 | 0.0006904 | 0.0002976 | 2.32 | False |
| FGA_jump2 | tier=power | 36027 | 0.0006709 | 0.0002972 | 2.257 | False |
| FGA_jump2 | locq=Q1 | 38279 | 0.0004454 | 0.0002466 | 1.806 | False |
| FGA_jump2 | locq=Q2 | 35567 | 0.001019 | 0.0002419 | 4.211 | False |
| FGA_jump2 | locq=Q3 | 35884 | 0.0004135 | 0.0002485 | 1.664 | False |
| FGA_jump2 | locq=Q4 | 37197 | 7.602e-05 | 0.0002338 | 0.3251 | False |
| FGA_3 | site=away | 105486 | 1.946e-05 | 0.0001125 | 0.1729 | False |
| FGA_3 | site=home | 107741 | -9.649e-05 | 0.000112 | -0.8615 | False |
| FGA_3 | site=neutral | 31792 | -0.0001398 | 0.0002292 | -0.6099 | False |
| FGA_3 | month=Dec | 40577 | 0.0002471 | 0.0001838 | 1.344 | False |
| FGA_3 | month=Feb | 54697 | -0.0001432 | 0.0001423 | -1.007 | False |
| FGA_3 | month=Jan | 61205 | -7.798e-05 | 0.0001371 | -0.5688 | False |
| FGA_3 | month=Mar+Apr | 35061 | -0.0002315 | 0.0001968 | -1.177 | False |
| FGA_3 | month=Nov | 53479 | -3.907e-05 | 0.0001806 | -0.2163 | False |
| FGA_3 | tier=low | 142576 | -3.777e-05 | 0.0001009 | -0.3743 | False |
| FGA_3 | tier=mid | 43187 | -0.0001489 | 0.0001635 | -0.9112 | False |
| FGA_3 | tier=power | 59256 | -1.636e-05 | 0.0001672 | -0.09786 | False |
| FGA_3 | locq=Q1 | 59744 | 9.652e-05 | 0.0001553 | 0.6213 | False |
| FGA_3 | locq=Q2 | 63032 | -0.0002012 | 0.0001499 | -1.342 | False |
| FGA_3 | locq=Q3 | 61308 | -7.526e-05 | 0.0001487 | -0.506 | False |
| FGA_3 | locq=Q4 | 60935 | -2.064e-05 | 0.0001577 | -0.1309 | False |

### P-B F2: multinomial log loss, B2 vs BP (floor = paired game-block bootstrap SE; seed floor 0)
| cls | LL_B2 | LL_BP | gain | floor | gain_in_floors | beats_floor |
|---|---|---|---|---|---|---|
| FGA_jump2 | 0.6713 | 0.6236 | 0.04772 | 0.001004 | 47.53 | True |
| FGA_3 | 1.05 | 1.028 | 0.02187 | 0.0004679 | 46.73 | True |

### P-B F2: realised band share by quintile of the shooter's own as-of share (wing3 WEAKER)
| cls | arm | band | slope_ratio | steps | in_0_8_1_2 | d8_pass | pred | actual |
|---|---|---|---|---|---|---|---|---|
| FGA_jump2 | B2 | short2 | 0.2613 | 4 | False | False | [0.3753, 0.41849, 0.44653, 0.46976, 0.50087] | [0.24209, 0.33827, 0.43566, 0.54616, 0.72259] |
| FGA_jump2 | BP | short2 | 0.9899 | 4 | True | True | [0.22262, 0.33032, 0.41866, 0.52753, 0.69829] | [0.24209, 0.33827, 0.43566, 0.54616, 0.72259] |
| FGA_3 | B2 | corner3 | 0.2066 | 4 | False | False | [0.19263, 0.20639, 0.21378, 0.22283, 0.23504] | [0.12272, 0.17613, 0.2086, 0.25455, 0.32797] |
| FGA_3 | B2 | wing3 | 0.2901 | 4 | False | False | [0.42964, 0.43887, 0.44487, 0.45043, 0.46186] | [0.38093, 0.42654, 0.44403, 0.46155, 0.492] |
| FGA_3 | B2 | top3 | 0.1807 | 4 | False | False | [0.31846, 0.33373, 0.34137, 0.34753, 0.36257] | [0.23219, 0.28812, 0.32929, 0.37915, 0.47623] |
| FGA_3 | BP | corner3 | 0.9893 | 4 | True | True | [0.11967, 0.17269, 0.21003, 0.25014, 0.32273] | [0.12272, 0.17613, 0.2086, 0.25455, 0.32797] |
| FGA_3 | BP | wing3 | 0.9223 | 4 | True | True | [0.39151, 0.42687, 0.44469, 0.46238, 0.49396] | [0.38093, 0.42654, 0.44403, 0.46155, 0.492] |
| FGA_3 | BP | top3 | 1.028 | 4 | True | True | [0.22342, 0.28907, 0.33429, 0.38411, 0.47443] | [0.23219, 0.28812, 0.32929, 0.37915, 0.47623] |

### P-B F2: BP vs B2 by segment
| cls | segment | n | gain | boot_se | in_floors | underpowered |
|---|---|---|---|---|---|---|
| FGA_jump2 | site=away | 65670 | 0.04498 | 0.001295 | 34.74 | False |
| FGA_jump2 | site=home | 61404 | 0.04826 | 0.001446 | 33.38 | False |
| FGA_jump2 | site=neutral | 19853 | 0.05515 | 0.002926 | 18.85 | False |
| FGA_jump2 | month=Dec | 24133 | 0.04352 | 0.00231 | 18.84 | False |
| FGA_jump2 | month=Feb | 32822 | 0.05427 | 0.001894 | 28.66 | False |
| FGA_jump2 | month=Jan | 36979 | 0.04693 | 0.001703 | 27.55 | False |
| FGA_jump2 | month=Mar+Apr | 21561 | 0.05824 | 0.002516 | 23.15 | False |
| FGA_jump2 | month=Nov | 31432 | 0.03783 | 0.001809 | 20.92 | False |
| FGA_jump2 | tier=low | 86023 | 0.04301 | 0.001045 | 41.17 | False |
| FGA_jump2 | tier=mid | 24877 | 0.05271 | 0.002394 | 22.01 | False |
| FGA_jump2 | tier=power | 36027 | 0.05553 | 0.002016 | 27.54 | False |
| FGA_jump2 | locq=Q1 | 38279 | 0.04376 | 0.001805 | 24.25 | False |
| FGA_jump2 | locq=Q2 | 35567 | 0.04869 | 0.001879 | 25.91 | False |
| FGA_jump2 | locq=Q3 | 35884 | 0.04568 | 0.00181 | 25.24 | False |
| FGA_jump2 | locq=Q4 | 37197 | 0.05284 | 0.002029 | 26.04 | False |
| FGA_3 | site=away | 105486 | 0.02076 | 0.0006109 | 33.98 | False |
| FGA_3 | site=home | 107741 | 0.02169 | 0.0007078 | 30.65 | False |
| FGA_3 | site=neutral | 31792 | 0.02611 | 0.001537 | 16.98 | False |
| FGA_3 | month=Dec | 40577 | 0.01797 | 0.001128 | 15.93 | False |
| FGA_3 | month=Feb | 54697 | 0.02639 | 0.001157 | 22.82 | False |
| FGA_3 | month=Jan | 61205 | 0.02289 | 0.0009501 | 24.09 | False |
| FGA_3 | month=Mar+Apr | 35061 | 0.02872 | 0.001313 | 21.88 | False |
| FGA_3 | month=Nov | 53479 | 0.01453 | 0.001042 | 13.94 | False |
| FGA_3 | tier=low | 142576 | 0.02106 | 0.0006466 | 32.57 | False |
| FGA_3 | tier=mid | 43187 | 0.02232 | 0.001166 | 19.14 | False |
| FGA_3 | tier=power | 59256 | 0.02347 | 0.001075 | 21.84 | False |
| FGA_3 | locq=Q1 | 59744 | 0.01996 | 0.001033 | 19.31 | False |
| FGA_3 | locq=Q2 | 63032 | 0.02189 | 0.001144 | 19.13 | False |
| FGA_3 | locq=Q3 | 61308 | 0.02137 | 0.0009972 | 21.43 | False |
| FGA_3 | locq=Q4 | 60935 | 0.02421 | 0.001001 | 24.19 | False |

### P-C F2: player-game join counts
```
{
 "player_games_in_design": 96037,
 "no_athlete_map": 0,
 "no_box_row": 137,
 "joined": 95900,
 "join_rate": 0.9985734664764622,
 "3pa_count_mismatch_dropped_from_3PM": 269,
 "fga_count_mismatch_dropped_from_FGM": 862,
 "3pm_design_equals_box_among_matched_attempts": 1.0
}
```

### P-C F2: player-game 3PM, (i) served R vs (ii) P1 over the BP band mix (gain = MAE(i) - MAE(ii); floor = max(player-game, game-block bootstrap SE); dispersion ratio = SD(actual - implied) / RMS(implied Bernoulli SD))
| label | n_player_games | underpowered | mean_actual | mean_i | mean_ii | mae_i | mae_ii | gain | se_player_game | se_game_block | floor | gain_in_floors | dispersion_ratio_i | dispersion_ratio_ii |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 71159 | False | 1.164 | 1.161 | 1.162 | 0.6827 | 0.6823 | 0.0004251 | 0.0001841 | 0.0001954 | 0.0001954 | 2.176 | 1.005 | 1.003 |
| all attempts located | 69684 | False | 1.16 | 1.157 | 1.159 | 0.6814 | 0.681 | 0.0003962 | 0.0001946 | 0.0001896 | 0.0001946 | 2.036 | 1.004 | 1.003 |
| 3PA 1-2 | 31903 | False | 0.3734 | 0.4771 | 0.4762 | 0.4855 | 0.4853 | 0.0002332 | 0.000129 | 0.0001237 | 0.000129 | 1.808 | 0.9365 | 0.9368 |
| 3PA 3-5 | 26256 | False | 1.274 | 1.289 | 1.29 | 0.7321 | 0.7321 | 4.57e-05 | 0.0002985 | 0.000327 | 0.000327 | 0.1398 | 0.9936 | 0.9928 |
| 3PA 6+ | 13000 | False | 2.881 | 2.579 | 2.588 | 1.067 | 1.065 | 0.001662 | 0.0008247 | 0.0008316 | 0.0008316 | 1.999 | 1.011 | 1.01 |

### P-C F2: player-game FGM (rim attempts held at the pooled training-seasons rim make rate in both arms)
| label | n_player_games | underpowered | mean_actual | mean_i | mean_ii | mae_i | mae_ii | gain | se_player_game | se_game_block | floor | gain_in_floors | dispersion_ratio_i | dispersion_ratio_ii |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 95038 | False | 2.9 | 2.898 | 2.899 | 0.9621 | 0.9613 | 0.0007815 | 0.0001992 | 0.0002059 | 0.0002059 | 3.796 | 1.019 | 1.018 |
| all attempts located | 93250 | False | 2.876 | 2.881 | 2.882 | 0.9587 | 0.9579 | 0.0007708 | 0.0002007 | 0.0001989 | 0.0002007 | 3.84 | 1.019 | 1.018 |
| FGA 1-4 | 37300 | False | 0.9832 | 1.076 | 1.075 | 0.613 | 0.6126 | 0.0003797 | 0.0001222 | 0.0001415 | 0.0001415 | 2.684 | 0.993 | 0.9929 |
| FGA 5-9 | 34902 | False | 3.04 | 3.03 | 3.031 | 1.039 | 1.038 | 0.0006224 | 0.0002886 | 0.0002924 | 0.0002924 | 2.129 | 1.025 | 1.024 |
| FGA 10+ | 22836 | False | 5.815 | 5.672 | 5.677 | 1.415 | 1.414 | 0.001681 | 0.0007022 | 0.0006537 | 0.0007022 | 2.394 | 1.016 | 1.015 |

### P-C F2 3PM: responsiveness by quintile of the shooter's profile shift (per-attempt shift (ii)-(i); delta SD 0.01319, range -0.0857..0.0827); slope of (actual - (i)) on shift = 0.7808152761970419; POST-HOC bootstrap 95% (not a decision input): slope ratio (ii) [1.096, 2.017, 8.113], excess slope [0.631, 0.78, 0.929]
| arm | pred | actual | span_pred | span_actual | slope_ratio | steps | raw_slope_in_0.8_1.2 |
|---|---|---|---|---|---|---|---|
| (i) R | [0.34701, 0.34063, 0.33723, 0.33393, 0.32666] | [0.32932, 0.33887, 0.34153, 0.3416, 0.33656] | -0.02035 | 0.00725 | -2.808 | 0 | False |
| (ii) P1 x BP | [0.32953, 0.3345, 0.33718, 0.33997, 0.34439] | [0.32932, 0.33887, 0.34153, 0.3416, 0.33656] | 0.01485 | 0.00725 | 2.05 | 4 | False |

quintile mean shift [-0.01748, -0.00613, -5e-05, 0.00603, 0.01772]; actual - (i) [-0.01769, -0.00176, 0.0043, 0.00767, 0.0099]

### P-C F2 FGM: responsiveness by quintile of the shooter's profile shift (per-attempt shift (ii)-(i); delta SD 0.00935, range -0.103..0.0801); slope of (actual - (i)) on shift = 0.9785961919211855; POST-HOC bootstrap 95% (not a decision input): slope ratio (ii) [0.768, 0.982, 1.357], excess slope [0.838, 0.983, 1.146]
| arm | pred | actual | span_pred | span_actual | slope_ratio | steps | raw_slope_in_0.8_1.2 |
|---|---|---|---|---|---|---|---|
| (i) R | [0.42381, 0.44586, 0.4923, 0.44448, 0.41449] | [0.41396, 0.44159, 0.49629, 0.44345, 0.42767] | -0.00932 | 0.01372 | -0.6794 | 2 | False |
| (ii) P1 x BP | [0.41244, 0.44269, 0.49222, 0.44736, 0.426] | [0.41396, 0.44159, 0.49629, 0.44345, 0.42767] | 0.01356 | 0.01372 | 0.9885 | 2 | True |

quintile mean shift [-0.01137, -0.00317, -8e-05, 0.00288, 0.01151]; actual - (i) [-0.00985, -0.00427, 0.00399, -0.00103, 0.01319]

### Block-P decision F2 (literal section 3.2)
```
{
 "fold": "F2",
 "c1_3PM_MAE_improves_beyond_floor": true,
 "c1_gain": 0.0004251046607886843,
 "c1_floor": 0.00019535718014080273,
 "c2_no_tier_worse_beyond_floor": true,
 "c3_profile_quintile_slope_in_0.8_1.2": false,
 "c3_slope_ratio": 2.0495,
 "c3_reliability_sqrt": {
  "sqrt_rel_make_contrast_short2_long2": 0.41567299328255614,
  "sqrt_rel_make_contrast_corner3_noncorner3": 0.20873691750659795
 },
 "c4_P_A_or_P_B_beats_floor": true,
 "P_A_beats": true,
 "P_B_beats": true,
 "verdict": "NOT CANDIDATE"
}
```


## 3. Worker disclosures (every choice the spec left open)

**Block V**
1. **Which "served artifact".** The monthly S1 refits in `data/processed/models/engine/event_round2_s1_F2_2025/` (the winners: `lgbm` on `first`, `cascade` on `cont`), scored on the round-2 design's own `C_plus_state` columns for every 2025 chance (`first` 742,025 + `cont` 111,906 = 853,931 chances, 10,890 team-games). Each chance is scored by the most recent refit at or before its game date (no refit here). Log loss reproduces the recorded winners exactly (first 1.51543, cont 1.49976), which verifies the scoring path. This is the artifact the bake-off graded, not the engine's inputs-v3 team block.
2. **F1 class part NOT RUN.** The engine holds no F1 artifact set; a replay would be a refit. "By season" in V-B is therefore 2025 only. The band part has both folds.
3. **Classes.** Shares are of chances (first + continuation). TOV, rim, jump2, 3PA are the four registered; FT is the sum of the two FT-trip classes (binary marginal), added because it is the remaining mass.
4. **Residual and ratio definitions.** r = (count - sum p) / n per team-game; implied SD of the share = sqrt(sum p (1 - p)) / n (multinomial marginal). Ratio = SD(r) / sqrt(mean implied variance of the share), unweighted over team-games; the z SD beside it is the n-free version (within 0.01 of the ratio for the classes, 0.01-0.05 above it for the bands). The band part uses located attempts only, team-games with >= 5 located attempts in the class, B2 probabilities (round-1 carried arm, F1 / F2 test predictions).
5. **Floor.** Parametric bootstrap SD of the ratio (200 replicates, outcomes redrawn from the model's own per-chance / per-attempt probabilities; the null mean is 0.999-1.001, so the estimator is unbiased) combined with the game-resampled bootstrap SE of the realised ratio by `max`. "BEYOND" = ratio - null mean > floor. Excess is also reported in floors so any stricter threshold can be read off.
6. **Why the multinomial null is the right null.** Chance probabilities condition on the realised game state (score, bonus, chance number), which is a martingale difference structure, so the sum of Bernoulli variances is the correct variance under a calibrated model and the independent redraw is the matching null. The ratio is therefore not inflated by state dependence to first order.
7. **Latent.** First-order logit shift u ~ N(0, s2) applied to every chance of a game for one class against the rest: s2 = (mean r_dm^2 - mean implied share variance) / mean((sum p (1 - p) / n)^2), with r_dm demeaned (the mean residual is reported separately: 3PA +0.36 pp, jump2 -0.41 pp, i.e. the known class-level shifts, not dispersion). SD = sqrt(max(s2, 0)). Each class is treated as binary vs rest, so the class latents are marginal and not a joint softmax latent. Bootstrap SEs are on s2.
8. **Correlation floors.** The residual correlation matrix reports the parametric-bootstrap null mean beside each entry (multinomial mechanics make the within-class entries non-zero under the null, e.g. -0.47 between corner3 and wing3). Covariate lines use the Fisher floor 1 / sqrt(n) and flag beyond 2 / sqrt(n) (not a parametric null).
9. **Covariates.** Margin = verified finals (`load_actual_games`, offence perspective); site excludes neutral games; possession count = the offence's chance count; opponent defence profile = `opp_def_{tov,rim,3pa}_c` for classes and `def_<band>_c` for bands (none for jump2 and FT, none for the other class).
10. **Quintiles for V-B** are of the offence's as-of profile: `off_3pa_c` (3PA and jump2), `off_rim_c`, `off_tov_c`, and `off_<band>_c` for bands, quintiles of the team-game set.
11. **V-C disattenuation.** The reliability of a residual = 1 - (mean implied variance / variance of the residual), floored at 0; the disattenuated correlation divides by the geometric mean. "One latent" is read as a disattenuated correlation near 1 (>= 0.8); the registered pairs are far below that (|r| <= 0.06 for 3PA with every band; -0.33 for rim with short2).

**Block P**
12. **Player map.** `data/processed/player_crosswalk_v2.parquet`, key (season, `cbbd_player_id` = the design's corrected `shot_shooter_id`) -> `espn_athlete_id` = hoopR `player_box` `athlete_id`. Attempt join rate 100.0% in 2022-2025 (1,420,545 attempts `source_id_verified`, 1,934 `source_id_unverified`); player-game to hoopR box join 99.86% (2025: 137 of 96,037 player-games have no box row) and 99.78% (2024: 201 of 90,950). The design's own `espn_athlete_id` column is null for 2022-23, so v2 is used; it equals the design column wherever both exist. Player-games without a box row are dropped and counted, never imputed.
13. **Shooter feature recipe (left open by the spec).**
    - Counts: the shooter's located attempts over games strictly before this one in the season plus RHO = 0.5 times his previous season's final totals (RHO is the round-1 team value, not re-chosen).
    - `sh_<band>_c` = logit(p) - logit(league as-of share), p = (k_s T_b + S_b) / (k_s + N), T_b the round-1 team shrunk share on that date.
    - `smk_<band>` = (k_m prior + D_b) / (k_m + S_b), prior = `shooter_shrunk_dev_c` recomputed from the design (m = 200 jumper / 300 three, as served), D_b the sum of (made - league as-of make rate in the band). The league band make rate is (500 prior-season-final pseudo-attempts + cumulative) as-of. The prior is a deviation from the shooter's TEAM class rate while D_b is a deviation from the LEAGUE band rate (mixed baselines, a consequence of the spec's choice of prior).
    - `sh_band_make_c` = `smk` of the attempt's own band; the P1 counterfactual sets it to each band's `smk` in turn.
14. **k.** Chosen per class on fold-1 training seasons (2022-23) by located-attempt-weighted squared error: k_s = 5 (jump2), 20 (three); k_m = 200 (jump2), 400 (three). The first build put k_s at the edge of its grid (10); the grid was widened downward to (1, 2, 5, ...) and the build re-run BEFORE any arm; the committed history shows both builds.
15. **Reliability rules.** Split by alternating games in date order within a shooter-season; band shares need >= 20 located attempts in each half and >= 100 in the season; make rates need >= 10 attempts in the band cell per half; values demeaned by season; Spearman-Brown. The 40-99 stratum is reported (build report) and labelled UNDERPOWERED.
16. **Leak test.** Team-game mean over the team's attempts of each shooter column (class-specific columns over that class's attempts; `sh_band_make_c` over located attempts), change form against the verified own-game margin, per season and pooled.
17. **P-A.** R, R_seed (seed 1), P0 and P1 are refit on the LOCATED population exactly as round 1 (R reproduces round-1 R; P0 reproduces round-1 A1: F2 jump2 LL 0.664539, three 0.637527, identical). Floor for P1 vs P0 = max(|LL R - LL R_seed|, paired game-block bootstrap SE); "gate held" = P1 worst decile gap <= P0's + max(|gap R - gap R_seed|, 0.10 pp) (the round-1 convention). Responsiveness lines are quintiles of the shooter's within-band skill `smk_<band>` of the attempt's own band, within each band separately.
18. **P-B.** BP = B2 + `sh_<band>_c` (jumpers: short2; threes: corner3, wing3, top3) in the same GLM, predictions for all test attempts. B2 re-fit equals round-1 B2 to 0.0 (max abs difference). The "slope ratio in [0.8, 1.2]" is the raw predicted-span / realised-span ratio for every band line of the class (as round 1, disclosure 8).
19. **P-C construction.**
    - (i) = R's counterfactual make probability for every test-season jump2 / three attempt (located or not); (ii) = sum over bands of BP's band probability times P1's band-conditional make probability (P1 evaluated with that band's one-hots and that shooter's `smk`).
    - Attempt counts are held at the actual: the player-game's design attempts. The 3PM line keeps player-games whose design 3PA equals the box 3PA (dropped for mismatch: 269 in F2, 260 in F1); the FGM line keeps those whose design FGA equals the box FGA (862, 921). The design's 3PM equals the box 3PM on 100.0% of the matched player-games, which verifies the truth join.
    - Rim attempts of the test season are held at the pooled rim make rate of the fold's training seasons in BOTH arms (the spec's "served class level" was read this way: no rim model is fitted in this round).
    - Implied SD for the dispersion ratio = sqrt of the summed Bernoulli variances of the implied probabilities.
    - Floor = max(player-game bootstrap SE, game-block bootstrap SE), 200 replicates each (3.2 says player-game, 3.5 says block; the larger is used). Underpowered = fewer than 2,000 player-games; no cell is.
    - Located-only side by side = player-games all of whose jump2 / three attempts are located.
20. **P-C responsiveness driver.** Not specified. I used the per-attempt shift (ii) - (i) of the expected make probability (the part of the prediction the shooter's profile and band skill add), quintiles of its rank weighted by attempts, slope ratio = predicted span (arm ii) / realised span of the 3P rate. The 3PM line is the decisive line (the clause is about 3PM); the FGM line is reported and passes. "Read against sqrt(reliability)": the gate is the raw ratio (round-1 convention); sqrt of the make-rate-contrast reliabilities (0.42 jumper, 0.21 three) is reported in the decision JSON and not used as a gate.
21. **Post-hoc addition.** After the first P-C read I added a bootstrap of the slope ratio and of the excess-vs-shift slope (labelled POST-HOC in the tables, not a decision input). The grade was re-run; decisions are identical (first run saved as `results/shot_location_v2/grade_P_firstrun.json`).
22. **Tier artefact (descriptive).** Tiers are defined by the ACTUAL 3PA (the spec holds attempts at actual), which selects on outcome: games with 1-2 3PA have actual mean 0.373 3PM against 0.477 implied, games with 6+ have 2.88 against 2.58. Both arms share this; it does not affect the paired gains but it makes the per-tier absolute levels uninformative.
23. **Blinding caveat.** The wrapped trainers print their own log loss to their logs (as in every round); I read those logs before the grade. No arm, population, setting or k was changed after any log, grade or table output. P-A and P-B arm codes were hidden in the grader until `tables_P_blind.md` was written.
24. **Concurrency and wall time.** Start 12:07 ET (this session's clock). Block V scored by about 12:25 (diag 2 min, grader 41 s). Build 14 s, 8 P-A fits plus 4 P-B fits 3 min wall clock (4 concurrent at 3 threads each, beside another worker's processes; nothing of anyone else's was signalled), grader 13 s. Docs after.

## 4. Suite, parity, artifacts

- **Suite.** `pytest -q`: **880 passed, 1 skipped, 0 failed** (306.9 s; log `results/shot_location_v2/log_pytest.txt`). The known pre-existing `tests/test_shot_block_live_lut.py` failure did not fire in this run (it reads a live table the scheduler maintains, and another worker's diagnostic `docs/ops/suite_failure_shot_block_live_lut_2026-10-27.md` traces it to a stale cache); no file under `src/` or `tests/` changed in this round.
- **Parity v10.** `scripts/run_parity_smoke_v1.py`: PASS, bit-identical digest (60 games x 5 seeds, served defaults, `results/engine_v0/parity_v10_shotloc_r2_20261027`). No file under `src/` changed.
- **Artifacts** (gitignored, HF `model_artifacts` under `data/processed/models/shot_location/round2/`): `design_props_cols_v1.parquet` (row-aligned to the fg design: `sh_*_c`, `smk_*`, `sh_band_make_c`), `build_report_v1.json` (reliability, k grids, leak test, player map), `V/` (chance-level served probabilities, band probabilities, margins), `make/<arm>_s<seed>_<fold>/` (preds, rows, counterfactual predictions), `readB/{B2,BP}_<fold>/`. Grades and logs: `results/shot_location_v2/` (`grade_V.json`, `grade_P.json`, `tables_V.md`, `tables_P.md`, `tables_P_blind.md`, logs).
- **Reproduce:**

```
.venv/Scripts/python.exe scripts/diag_shot_selection_dispersion_v1.py
.venv/Scripts/python.exe scripts/grade_shot_location_v2.py --block V
.venv/Scripts/python.exe scripts/build_shooter_band_v1.py
.venv/Scripts/python.exe scripts/train_shot_location_v2_props_v1.py --arm {R,P0,P1} --fold {F1,F2} --seed {0, 1 for R} --arms B1 --no-floor --no-leak --n-jobs 3 --out-dir data/processed/models/shot_location/round2/make/<arm>_s<seed>_<fold>
.venv/Scripts/python.exe scripts/train_shot_location_v2_props_v1.py --arm {B2,BP} --fold {F1,F2}
.venv/Scripts/python.exe scripts/grade_shot_location_v2.py --block P
```

## 5. NOT RUN

- Block V class part on F1 (no F1 served artifacts; a replay is a refit), hence no by-season class table beyond 2025.
- A P-A / P-B located-only vs all-games side-by-side (both are located by construction); P-C has it.
- A (k_s, k_m) sensitivity fit under the fold-1 second-best values (not registered for this round).
- Serving, usage-model integration of the shooter band mix, any paired-seed sim run (all outside this round).

