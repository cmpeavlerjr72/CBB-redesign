# Shot-location stage (L3, between possession outcome and make / foul / rebound)

Status: ROUND 1 RUN 2026-10-27 (`experiments.md` sections 1-2): **NOT CANDIDATE in either class** under the registered rule. Nothing served, freeze ON.
Companion docs: `features.md`, `experiments.md`. Origin and evidence trail: `docs/tests/coordinate_audit_2026-10-27.md`,
`gamezone_*_2026-10-27.md`, `sportradar_*_2026-10-27.md`, `docs/research/shot_coordinate_sources_2026-10-27.md`.

## 1. Purpose
Given offence, defence, site, game state and the possession-outcome class (`FGA_jump2` or `FGA_3`; the rim class is
ONE band by PM ruling 11 of 2026-10-27), sample the location band of the attempt so that make, and-one draw and
offensive rebound can be conditioned on it. Candidate stage; exists only if round 1's Reads B and C pass.

## 2. Target variable
Band within class (section 1.2 of `experiments.md`): jump2 -> short2 / long2; three -> corner3 / wing3 / top3.
Population: fg_make design population, CBBD located attempts, seasons 2022-2025, folded frame.

## 3. Methodology at a glance
Walk-forward folds (F1 train 2022-23 test 2024; F2 train 2022-24 test 2025, selection). S1 monthly schedule for the
Read A wrappers. Three reads: A ceiling (attempt level), B predictability (multinomial band models B0-B3), C implied
game aggregates. Primary decision on B + C. Registered measurement ceiling: split-half reliability of team band
profiles 0.89 / 0.79 / 0.67 / 0.80 offence, 0.69 / 0.76 / 0.52 / 0.75 defence.

## 4. Winner
No winner.

- **Read B carried arm: `B2`** (multinomial logit on the offence's and the defence's as-of band log-odds, plus site and state). It is MATCHUP in both classes on F2.
  - B2 beats the league-average null B0 by 25.2 floors (jumpers) and 27.9 floors (threes) on attempt log loss.
  - The defence profile adds 8.9 / 16.8 floors over the offence profile alone.
  - Team-game band-share MAE falls from 16.30 to 14.37 pp (short2) and by 0.25-0.72 pp across the three-point bands.
  - Every B2 responsiveness line passes, with slope ratios 0.93-1.05.
- **Read C does not carry the stage.** Marginalising the band-aware make, and-one and rebound models over B2's predicted mix moves team-game MAE by at most about one bootstrap SE against the class-level pair.
  - Jumpers: no target improves.
  - Threes: 3P make rate +1.17 floors and OREB +1.07 floors, but and-one -1.95 floors, and the responsiveness lines fail.
- **Read A (ceiling)** is SIGNAL for make (both classes), rebound (both classes) and and-one on made jumpers. It is NO SIGNAL for and-one on made threes.
- Full numbers: `experiments.md` section 2 and `docs/tests/shot_location_round1_2026-10-27.md`.

## 5. Robustness check
- **Segments.** Site, month, offence tier and located-share-of-game quartile, per read (evidence doc tables T5 and T8 segment blocks). Cells under 5,000 attempts (2,000 rebound opportunities) are labelled UNDERPOWERED.
- **Read B's defence-profile gain is strongly site-asymmetric.**
  - Jumpers: +5.6e-3 log loss when the offence is away against -8.8e-4 when it is at home.
  - Threes: +6.1e-3 away against +1.5e-3 home.
  - A venue-charting component, where the home crew charts the home defence's allowed profile, is plausible and not separated.
- **Fold 1 confirms the Read B verdicts.** B3's edge over B2 on F2 does not reproduce on F1.
- **Team-game band shares are over-dispersed against the multinomial**: z SD 1.31 short2, 1.11-1.22 threes. A served stage would need a per-game shot-selection latent.

## 6. Decisions log
- Rim excluded (ruling 11): outcome-dependent rim placement in ESPN 2022-25 and Sportradar 2013-18.
- CBBD 2022-2025 only; no vendor union (rulings 3, 13).
- Foul channel read as and-one rate only (feed logs no FGA on missed fouled shots).

## 7. Consumption from the sim
None. No engine column, no flag. Serving needs its own paired-seed sim round.

## 8. Artifacts
All under `data/processed/models/shot_location/round1/` (gitignored, HF `model_artifacts`), plus `results/shot_location_v1/`.

| Path | What it is |
|---|---|
| `design_loc_cols_v1.parquet` | row-aligned to `fg_make/design_v2_shotshooter.parquet`: located flag, folded xf / y, d, theta, band, band one-hots, `cell_p`, as-of `off_/def_<band>_c` (and `_alt` under the second-best k / rho), as-of shares, prior counts |
| `rebound_loc_cols_v1.parquet` | row-aligned to `rebound/round3/design_round3.parquet`: fg design row of a jump2 / three miss, its band, d, theta, located |
| `team_band_profiles_v1.parquet` | per (side, season, game, team) located band counts and as-of profiles |
| `build_report_v1.json`, `diag_report_v1.json` | join, cell check, k / rho grid, leak test; registered diagnostics 1.7 |
| `readB/F{1,2}/preds.parquet`, `run_report.json` | B0-B3 and B2alt band probabilities on every test-season attempt of the class; GLM coefficients |
| `make/`, `andone/`, `rebound/<arm>_s<seed>_<fold>/` | Read A wrapper runs: preds, scored-row index, counterfactual per-band predictions (`cf_*.parquet`, R and A1) |
| `results/shot_location_v1/grade_v1.json` | the blind grade: all reads, both folds, section 1.6 per class |

## 9. Known gaps / followups
`experiments.md` section 1.10.

**Round 2 finding for the variance queue (block V, 2026-10-27; `experiments.md` section 4, evidence `docs/tests/shot_location_round2_2026-10-27.md`).** The served possession-outcome class mix is over-dispersed per team-game: the SD of the residual share is 1.25x (3PA), 1.36x (rim), 1.47x (jump2) and 1.10x (TOV) the SD implied by the model's own per-chance Bernoulli variances (floors 0.007-0.012, parametric null 1.00); FT trips are slightly under-dispersed (0.95). The registered logit-normal per-game latent a future variance arm must reproduce is SD 0.192 (3PA), 0.237 (rim), 0.322 (jump2), 0.149 (TOV). Shot-location band shares carry their own over-dispersion (1.10-1.26x; latent SD 0.20-0.46 on the band logits) and it is NOT the same latent as the class latent (3PA residual vs band residuals: disattenuated |r| <= 0.06); the one coupling is rim vs short2 (-0.33 disattenuated), at the rim-override boundary. The class latent belongs to the possession-outcome owner; a location stage is not where to fix it. Round 2 block P: the shooter's band profile is very predictable (reliability 0.82-0.93, B2 -> BP +47 floors) but moves player-game 3PM MAE by only 0.0004 and fails its responsiveness line; NOT CANDIDATE.
