# Change ledger (living doc — update statuses in place, never delete rows)

Purpose: the running list of every investigated change, its status, and the
evidence — so nothing goes dark. Statuses: SHIPPED / TESTING / BUILDING /
VALIDATED-PENDING-SHIP-ACTION / REFUTED / CONFIRMED-INCUMBENT / OPEN /
DEFERRED / BANKED. Rule: every status change updates this doc in the same
commit series as the change itself.

## A. Model changes (the cascade)

| change | status | evidence / waiting on |
|---|---|---|
| Control engine built to the pre-registered spec (counts x percentages, shared pace draw, fitted dispersion everywhere, OT stub) | BUILDING -> BUILT AND GATED 2026-09-10 | `docs/tests/control_engine_F2_2026-09-10.md`; F2 margin MAE 9.146, total MAE 13.204, Brier 0.1857 |
| Attempt counts drawn as four INDEPENDENT overdispersed marginals | REFUTED (as a distributional model) 2026-09-10 | G5 margin SD ratio 1.678, total 1.318, home/away score corr 0.084 vs 0.229, PIT p=2e-79. Residual corr of the counts is -0.56 / -0.36 / -0.29; independent components imply team-points SD 13.87 vs actual 9.22. Waiting on the L3 possession-outcome model. Report section 10, D1 |
| Pooled-season training with no season-level feature | CONFIRMED-DEFECT 2026-09-10 | G9 total bias -2.83 on F2 (-4.08 on F1) = 77% of the +3.69 season drift; close is unbiased at -0.56 on the same games. Fix is pre-registration of a season-aware level term at L2/L3. Report section 10, D2 |
| Overtime stub (5/40 of the game's own possession draw, re-simulated until untied) | OPEN (known gap, flagged in output) | Sim OT rate 0.0183 vs actual 0.0558; the stub also double-counts ~+0.16 poss/game because the pace target already includes OT. Waiting on the L5 overtime model. Report section 10, D5 |
| L2 pace bake-off: model class (multiplicative / ridge / Gaussian GLM / LightGBM) x feature set (A_tempo..D_plus_state) | DECIDED: `multiplicative`/`A_tempo` 2026-09-10 | F2 RMSE 4.8805 vs best raw arm 4.8177 (`lightgbm`/`B_plus_season`), inside the measured noise floor (0.157); every richer feature bundle and LightGBM land inside the same floor, so the zero-fitted-parameter formula wins the simplicity tie-break. `docs/models/pace/experiments.md` R2-R4 |
| L2 pace target definition: T_box vs T_pbp | DECIDED: T_pbp 2026-09-10 | Pooled 2022-2025 corr 0.9717, mean abs diff 0.854 poss (possession SD ~5.6); reliable, pre-registered default applies. `docs/models/pace/experiments.md` R1 |
| L2 pace distribution family: Gaussian fixed SD / heteroscedastic Gaussian / NegBin-Poisson count | DECIDED: `gaussian_hetero` (best-of-three, none pass PIT) 2026-09-10 | Every one of the 32 stage-1 arms fails the pre-registered PIT gate (F2 p < 1e-19 everywhere); traced to overtime games (~5.6% of games, +8.2 poss residual) plus residual excess kurtosis (~1.8) that persists even in regulation-only games. `docs/models/pace/experiments.md` R7-R8 |

## B. Features & training specs

| change | status | evidence / waiting on |
|---|---|---|
| Own ridge as-of ratings (off/def per 100 poss + tempo, prior-season shrinkage prior) | SHIPPED 2026-09-10 | `data/processed/ratings/own_ratings_{season}.parquet`; fitted lambda 5.0 / prior weight 0.8 for both models, selected on fold-1 training seasons only; walk-forward RMSE 13.706 (eff) and 4.732 (tempo). `experiments.md` R1 |
| Own ratings pass the INV-45 change-form leak test | VALIDATED 2026-09-10 | as-joined corr +0.054 / -0.062 / +0.002 vs a 0.15 gate and a 0.04-0.08 honest baseline; `data/processed/ratings/own_ratings_leak_test.csv` |
| Anchor feature set for the Control: A_own vs B_kp vs C_both | DECIDED: A_own 2026-09-10 | F2 margin MAE 9.146 / 9.075 / 9.059 against a measured seed noise floor of 0.047-0.077; the pre-registered tie clause falls through to A_own. Paired (MC-cancelled) SEs show B_kp and C_both hold small real edges (t = 2.4, 3.5) that the unpaired floor cannot resolve. `experiments.md` R4-R6 |
| Rate dispersion family per target (deviance/df > 1.2 rule) | DECIDED 2026-09-10 | NegBin for 3PA (dev/df 1.60, alpha 0.0268), 2PA (1.27, 0.0070), FTA (2.59, 0.0832); Poisson for TOV (1.06). Same decision on both folds. `experiments.md` R2-R3 |
| Site feature = home / away / neutral only | CONFIRMED-DEFECT (neutral bucket) 2026-09-10 | G6 neutral sim +2.23 vs actual +3.29 (-1.06, tol +/-1.0) while non-neutral passes at +0.33. Needs a designated-home or venue-distance term. Report section 10, D4 |

## C. Engine & config

| change | status | evidence / waiting on |
|---|---|---|
| Counter-based RNG keyed on (seed, game_id, family) | SHIPPED 2026-09-10 | `src/cbb_sim/control/rng.py`; `tests/test_control.py` re-simulates a 37-game subset and requires bit-identical draws, and checks chunk-independence |
| Feature preflight before any draw | SHIPPED 2026-09-10 | `cbb_sim.control.models.preflight` raises `FeaturePreflightError`; every persisted model carries its feature list and train-time medians |
| Artifact writer stamps created_at / tipoff / backtest | SHIPPED 2026-09-10 | `scripts/run_control.py`; `--live` asserts created_at < tipoff, backtests are stamped rather than faked |
| Decision rules that need to resolve < ~0.05 margin MAE must use the paired per-game SE, not the unpaired seed-offset floor | OPEN | `experiments.md` R5-R6; MC SE of a game's predicted margin is 1.39 points at 200 seeds |
| `pace.py` predictive sampler reuses `cbb_sim.control.rng` verbatim, keyed on (seed, game_id, "pace") | SHIPPED 2026-09-10 | `tests/test_pace.py` bit-identical/seed-independence checks, same contract as the Control |

## D. Data repairs & data defects

| item | status | evidence |
|---|---|---|
| | | |
