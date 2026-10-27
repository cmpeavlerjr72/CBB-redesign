# Shot-location stage: feature inventory

Status: ROUND 1 RUN 2026-10-27 (NOT CANDIDATE). Every column is built by `scripts/build_shot_location_bands_v1.py --part build` into `data/processed/models/shot_location/round1/design_loc_cols_v1.parquet` unless stated; detail below the registered table.

| Feature | Dtype | Source | Computation | Fallback | Notes |
|---|---|---|---|---|---|
| `band` | category | CBBD `shot_location_x/y`, folded frame | `experiments.md` 1.2 thresholds; d in ft from basket centre (52.5, 250 tenths), theta per `build_pae_shot_rating_v1.dist_angle` | none: unlocated attempts are never imputed | target of Read B; feature of Read A arms A1 |
| `cell_p` | float | prior-season `da15` cell make rate | as fg_make `experiments.md` 31.2 | class mean | Read A arm A2 |
| `off_band_c` | float per band | own located attempts before tipoff | centred shrunk log-odds, k and rho chosen on fold-1 training (1.4) | 0 (league) | leak-tested before any arm |
| `def_band_c` | float per band | located attempts faced before tipoff | same recipe | 0 | leak-tested before any arm |
| site, `C_plus_state` columns | as served | possession_outcome `features.md` | unchanged | as served | |

## Provenance detail (round 1)

| Feature | Dtype | Source | Computation | Fallback | Notes |
|---|---|---|---|---|---|
| `located` | bool | CBBD pbp `shot_location_x/y` joined to the fg design | `build_pae_shot_rating_v1.design_join` (key game, period, seconds, shooter, class, made + ordinal; 100.0% matched) | n/a | 79.6 / 84.5 / 89.3 / 98.7% of FGA, 2022-25 |
| `xf`, `y`, `d`, `theta` | float | same | `fold_xy` (x > 470 -> 940 - x); `dist_angle`: d = distance to (52.5, 250) / 10 ft, theta = atan2(y - 250, xf - 52.5) deg | NaN when unlocated | theta 0 = toward half court; plot `docs/tests/shot_location_round1_bands_2026-10-27.png` |
| `b_short2` .. `b_top3` | float 0/1 | `band` | one-hot | 0 | Read A arm A1 (fg_make, and-one, rebound) |
| `cell_p` (as built) | float | prior-season da15 cell make rate | `PAE.baselines_and_residuals(des, "da15")` (2022 baseline = 2023) | NaN when unlocated (never used: located population) | reproduces `pae_v1/design_pae_cols_v1.parquet` to 3e-8 |
| `off_<band>_c` | float | team's own located attempts before the game date, same season | p = (k (lg + rho prev_c) + S) / (k + N); feature = logit(p) - logit(lg), p clipped to [1e-3, 1 - 1e-3]; k = 100, rho = 0.5 | 0 when S = N = prev_c = 0 | lg = as-of league share (games strictly before the date; season opener = previous season final); prev_c = previous-season raw share minus league |
| `def_<band>_c` | float | located attempts faced before the game date | same recipe, k = 200, rho = 0.5 | 0 | |
| `*_c_alt` | float | same | second-best (k, rho): offence 100 / 0.75, defence 400 / 0.5 | 0 | Read B sensitivity arm B2alt |
| `off_n_j`, `off_n_t`, `def_n_*` | float | same | located attempts in the class before the game | 0 | |
| `loc_d`, `loc_theta` (rebound) | float | `rebound_loc_cols_v1.parquet` | d and theta of the missed jump2 / three, joined to its fg design row (88.0% of miss rows; 99.3% within games in both tables) | NaN (row outside the population) | rebound arm A2 |

## Feature sets tested (round 1)

| Set | Included | Rationale |
|---|---|---|
| Read A `R` | served fg_make B1 (lane-A) / served rebound `C_plus_state` | control |
| Read A `A1` | R + band one-hots | band-level location |
| Read A `A2` | R + `cell_p` (fg) / R + d, theta (rebound) | fine location |
| Read B `B0` | none (training band shares) | null = today's engine |
| Read B `B1` | `off_<band>_c` + `site_home`, `site_away` + `period`, `seconds_remaining`, `score_diff`, `in_bonus`, `is_transition_f`, `chance_number` | team profile |
| Read B `B2` / `B3` | B1 + `def_<band>_c` (GLM / LightGBM) | matchup form |

Jumpers use only the short2 column per side (long2 is exactly its negative in log-odds).

## Rejected features

- **Rim location of any kind.** Ruling 11: rim placement is outcome-dependent.
- **Imputed location for unlocated attempts.** Never built, by spec 1.1.
- **hoopR or other vendor coordinates.** No union, by rulings 3 and 13.
