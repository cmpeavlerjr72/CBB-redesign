# control_engine

Status: BUILDING (spec pre-registered 2026-09-10, before any code).

## 1. Purpose

The Control engine is the yardstick, not a candidate. It is last year's counts x percentages decomposition done correctly, so that the possession engine (Decision 1) has a cheap, honest baseline to beat on every game-level gate. It also produces the project's first walk-forward scorecard against real lines (2022-23 to 2024-25 closes), which sets the accuracy bar the possession engine must clear before any player-prop work starts. It is deliberately simple; its components are fixed by this spec and are not bake-off winners. Where it shares a component with a real sub-model (the own-ratings arm of the L1 anchor bake-off), that component is built once and reused.

## 2. Target variables

Per team-game, D-I non-truncated games only: possessions (FGA - OREB + TOV + 0.44 FTA, averaged over both teams), and per 100 possessions: 3PA, 2PA (rim and jumper pooled at this level), FTA, TOV; and the make rates 3P%, 2P%, FT%. Home margin and total are outputs, never targets.

## 3. Methodology at a glance

- Data: hoopR team_box and schedules 2021-22 to 2024-25 via `data/processed/games_universe.parquet`. 2025-26 is sealed.
- Temporal walk-forward, no random split. Fold 1: train 2021-22 and 2022-23, test 2023-24. Fold 2: train through 2023-24, test 2024-25 (reported metric).
- Features, all pregame and leak-tested: own ridge ratings (offense and defense efficiency per possession, tempo) fitted on games strictly before the game date within the season, with a prior-season shrinkage prior; centred KenPom snapshot as-of the day before, as a second feature set so the two anchors are compared on identical folds (this is L1 arm (a) vs arm (b), run here for free); home/away/neutral; days since season start.
- Component models: pace, Gaussian GLM with fitted residual SD (candidate families are baked off in L2 later; the Control uses the Gaussian). Rates per 100 possessions, Poisson GLMs with log link and a fitted dispersion (NegBin if Poisson is overdispersed on the training residuals; the choice is recorded, not tuned). Make rates, Binomial GLMs weighted by trials, with a fitted Beta-Binomial overdispersion.
- Simulation: for each game and sim draw, one shared possessions draw; each team's attempts from its rate model scaled by the shared possessions; makes from Beta-Binomial; points = 3 x 3PM + 2 x 2PM + FTM. Ties resolve by re-simulating a 5-minute period at the same per-possession rates with possessions drawn as 5/40 of a game's draw (an explicit stub, flagged, replaced by the L5 overtime model). RNG seeded on (seed, game_id, "control").
- Primary metrics on fold 2: margin MAE and bias, total MAE and bias, win-probability Brier, calibration slope; G1, G5, G6, G9 as defined in `docs/SIM_GUARDRAILS.md`; and G10 vs the CBBD ESPN BET closes for 2023-24 and 2024-25.

## 4. Winner

Not applicable. The Control is fixed by this spec. What IS decided from its output: which anchor feature set (own ratings vs centred KenPom vs both) gives the lower fold-2 MAE, and whether Poisson or NegBin dispersion is needed on each rate. Both are recorded in `experiments.md` with the numbers.

## 5. Robustness check

Fold-2 metrics by month, by conference tier, by home/neutral, by predicted-total tercile (last year's bias tripled in the top tercile). Responsiveness: predicted margin bucketed by rating-difference quintile must slope with actual margin.

## 6. Decisions log

1. No hand-set dispersion anywhere; every SD is a fitted parameter with its estimate reported.
2. No tuning against 2025-26; sealed guard active.
3. Overtime stub is explicit and listed as a known gap.

## 7. Consumption from the sim

`scripts/run_control.py --season 2025 --seeds 200` writes `results/control/<season>/games.parquet` (one row per game x seed: game_id, seed, home_pts, away_pts, possessions) and `summary.parquet` (per game: mean/SD margin, mean/SD total, p_home, created_at, tipoff). Artifact writer enforces created_at < tipoff for live runs and stamps `backtest=True` otherwise.

## 8. Artifacts

`data/processed/models/control_engine/{pace,rates,pcts}_{fold}.pkl`, `data/processed/ratings/own_ratings_{season}.parquet` (as-of daily table), `results/control/`.

## 9. Known gaps / followups

Overtime stub. No player layer. No rim/jumper split. Pooled 2PA hides shot-mix drift; L3 replaces it.
