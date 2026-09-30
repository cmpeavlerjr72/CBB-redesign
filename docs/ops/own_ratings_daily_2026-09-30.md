# Daily own-ratings entry point (Lane K part 2, 2026-09-30)

Closes audit gap 3 (mechanics only). New files: `scripts/build_own_ratings_asof_v1.py` (the entry point),
`scripts/diag_own_ratings_asof_parity_v1.py` (parity vs the stored files), `tests/test_own_ratings_asof.py` (2 tests),
parity results `docs/ops/own_ratings_asof_parity_*.json`. Nothing under `data/processed/ratings` or `src/cbb_sim` was touched.

## 1. How the stored ratings are built

`scripts/build_own_ratings.py` -> `cbb_sim.ratings.own_ratings`: per season, walk-forward ridge (efficiency: intercept, home, away,
off dummy per team, def dummy per opponent, on points per 100 possessions; tempo: intercept, neutral, tempo dummy per team, on possessions).
The row `as_of_date == D` is the fit on the season's games strictly before D, ridge toward a PRIOR: `lambda * ||b - b_prior||^2`, with
`b_prior = w * (previous season's FINAL fit)` for team effects and the previous final intercept / home / away / neutral for the fixed terms.
Seasons are CHAINED (2022 has no prior; 2023's prior is 2022's final fit, and so on). lambda = 5 and w = 0.8 for both models, selected once on the
fold-1 training seasons (2022-23, one season-to-season transition) and stored in `own_ratings_manifest.json`. Every rating is re-centred on the mean over the
season's team set, so the team set is part of the definition. The stored file carries no coefficients and no final fit, only rows for dates with games.
On day 1 of a season nothing in the batch job exists for that season: no file, no entry point (`ratings_for_slate` raises `FileNotFoundError`).

## 2. The entry point

```
.venv/Scripts/python.exe scripts/build_own_ratings_asof_v1.py --season S --as-of D[,D2] --out-dir <dir> [--root .] [--teams-source tg|schedule|file:P|cbbd:GAMES|XWALK]
```

1. Refits the prior chain from 2022 to S-1 (full seasons, same `fit_season`, same hyperparameters from the manifest).
2. Fits season S on games with `game_date < D` only (asserted) plus one STUB game at D per team pair: the fit for D is taken before D's own rows are folded in,
   so stub content cannot reach the output (unit test `test_ratings_at_D_ignore_D_and_later`); the stubs only put D and every team of the season on the grid.
3. Writes `<out-dir>/own_ratings_{S}.parquet` (batch schema + `created_at`) and `own_ratings_{S}_provenance.json` (hyperparameters, team source, number of source games,
   latest source game date, prior-chain seasons, prior-carry description). Refuses `data/processed/ratings` as output. About 11 s per date after a ~20 s chain/load.
4. Chaining through 2026 needs `CBB_UNSEAL=1` (the script calls `assert_not_sealed`); any 2027 date needs the 2026 final fit.

## 3. Parity with the stored ratings (2025 and 2024 seasons, real data, nothing from 2026)

Every as-of row compared: all 10 numeric columns (`off_c, def_c, tempo_rel, n_games, league_off_mean, league_tempo_mean, home_off_eff, away_off_eff, neutral_tempo_eff, n_games_window`) and the
team set, 364 / 362 teams each.

| season | as-of date | season games used | teams | max abs difference over all columns |
|---|---|---:|---:|---:|
| 2025 | 2024-11-04 (day 1: no season game, prior chain only) | 0 | 364 | 2.8e-14 |
| 2025 | 2024-11-12 | 327 | 364 | 1.1e-14 |
| 2025 | 2025-01-16 | 2,822 | 364 | 1.6e-14 |
| 2025 | 2025-03-04 | 5,101 | 364 | 1.6e-14 |
| 2024 | 2023-11-06 (day 1) | 0 | 362 | 1.2e-14 |
| 2024 | 2024-02-10 | 3,875 | 362 | 1.9e-14 |

Equal to floating-point noise (4 to 6 of 10 columns bit-identical; the rest differ at the 1e-14 level, from summation order). End-to-end: the ratings for 2025-01-16 were
produced by the entry point from the INGESTION REPLAY root tables (`docs/ops/daily_ingestion_2026-09-30.md`, tables through 2025-01-15 built by the ingestion command, 2,822 games) with the stored team set,
put in place of the stored file in that root, and the live inputs builder for the 62-game 2025-01-16 slate was run: 12 arrays, 85,188 cells, games table, names/rules and event block bit-identical to the builder on the on-disk root.

## 4. What day 1 of 2026-27 (season 2027, 2026-11-02) needs

Mechanical (no choice involved):
1. `CBB_UNSEAL=1`. The 2027 prior chain ends in the 2026 FINAL fit, i.e. the sealed 2025-26 results enter a fit for the first time. The seal lift is the PM's decision.
2. `own_ratings_manifest.json` (present); hoopR team box and universe rows for 2022-2026 (present) and, once games are played, for 2027 (from the daily ingestion).
3. A team list for the season. The hoopR 2027 schedule gives only 319 D-I teams (1,629 thin games); CBBD games 2027 with the team crosswalk v2 (West Florida added) give 365. Use
   `--teams-source "cbbd:data/raw/preseason/2027_v2_20260930/games_2027.parquet|data/reference/team_crosswalk_v2.parquet"` until hoopR catches up (audit gap 5). For 2025 the schedule rule and the game-based team set both give 364 teams and identical ratings.
4. The consumer reads `own_ratings_2027.parquet` from `ctx.ratings_dir` (rows with `as_of_date == slate date`); point it at the entry point's `--out-dir` (one file per run).

Modelling choices the code currently makes by default and that the PM should decide (NOT chosen or invented here):
1. **Prior carry weight w = 0.8 (eff and tempo)**, one value for every team, fitted on a single 2022-to-2023 transition. Options to bake off: re-select with the 2023-24 and 2024-25 transitions (2025-26 stays sealed until chosen), separate weights for offence, defence, tempo.
2. **Uniform carry regardless of roster turnover.** A team that lost 70% of its minutes gets the same 0.8 as one that returned everyone. Candidates: weight by returning-minutes share (`roster_continuity_2027` exists as an upper bound), coach-change flag, KenPom preseason arm (no 2027 KenPom snapshot on disk).
3. **Teams with no previous rating** (new D-I, West Florida) get 0 = league mean by rule. Candidate: conference-level prior.
4. **Fixed-term priors**: intercept, home, away and neutral-tempo priors are last season's final values (league scoring level and pace drift, e.g. rule changes, carry through).
5. **lambda = 5 constant all season.** The prior loses weight only as data accumulate; no schedule such as game-count-dependent lambda or w.
6. **Team set used for re-centring** (section 4 item 3): the set changes the centring of every rating, so it is part of the definition; `tg` (played games) vs schedule vs CBBD.
7. **Early-season D-I membership**: ratings use games with `is_d1_game`, which the ingestion command sets with the provisional rule (strict >= 5 conference games UNION prior-season D-I teams); an alternative rule changes which early games count.
8. Only games with two usable box rows and both sides D-I enter (exhibitions vs non-D-I never do); whether to use them at a reduced weight is a choice.

Runtime for the day-1 build: chain load plus fits about 1 minute once 2026 is unsealed (not executed, sealed).
