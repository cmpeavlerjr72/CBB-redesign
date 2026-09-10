# pace (L2)

Status: BAKE-OFF COMPLETE (spec pre-registered 2026-09-10 before any code; run 2026-09-10). Full grid and
decisions in `experiments.md`; feature inventory in `features.md`; status history `docs/models/change_ledger.md`.

## 1. Purpose

Predicts possessions per game for a matchup -- one shared quantity both teams play at, matching CLAUDE.md's
modelling rule "one pace realisation per simulated game, both teams scaled by it." It feeds the possession
engine (L3 onward): every possession-outcome draw is scaled by this game's pace realisation, exactly the role
the Control's own (fixed, non-bake-off) Gaussian GLM plays today (`docs/models/control_engine/model.md`
section 3). This bake-off is what decides which model class, feature set, and predictive-distribution family
*replaces* that incumbent for the possession engine, per the standing rule "no model class ... is adopted
because it is convenient or familiar" (`CLAUDE.md`).

## 2. Target variable

Two candidate definitions were pre-registered and compared before any model was fit (`experiments.md` R1):

- **T_box**: mean over both teams of `FGA - OREB + TOV + 0.44*FTA` (hoopR `team_box`; identical to the
  Control's own `game_poss` and to `scripts/build_gate_reference.py`'s possession estimate).
- **T_pbp**: possessions counted directly from hoopR play-by-play events (`scripts/build_possessions_pbp.py`;
  the exact rule -- an explicit event-role mapping and a forward state machine -- is documented in that
  script's docstring in full, and summarized in `features.md` section 4).

Population: D-I, non-truncated games, 2022-2025 (`data/processed/games_universe.parquet`,
`is_d1_game & ~pbp_truncated`); T_pbp additionally requires `has_pbp` (about 1.3-2.0% of the universe per
season has no pbp file coverage at all, distinct from the truncation flag). Pooled 2022-2025 (21,965 games with
both definitions available): T_box mean 68.05 (SD 5.51), T_pbp mean 68.26 (SD 5.67), corr 0.9717, mean
difference (pbp - box) +0.219, SD of the per-game difference 1.339.

**Chosen: T_pbp.** The pre-registration's own rule ("T_pbp, unless it is unreliable") applies as written: the
two definitions agree tightly (corr 0.97, mean absolute difference 0.85 possessions against a possession SD of
~5.6), and T_pbp is the more direct measurement of an actual possession (a play-by-play state machine) rather
than a box-score formula's estimate of one. See `experiments.md` R1 for the full per-season table and the
reliability threshold applied.

## 3. Methodology at a glance

- Data window: 2022-2025 (hoopR/CBBD "season" convention -- ending year). 2025-26 (season 2026) is sealed;
  every fold/build call in `scripts/train_pace_v1.py` and `scripts/build_possessions_pbp.py` is guarded by
  `cbb_sim.data.seal.assert_not_sealed`.
- Temporal walk-forward, no random split: F1 trains {2022, 2023}, tests 2024; F2 trains {2022, 2023, 2024},
  tests 2025 (the selection fold).
- Four feature sets of increasing richness (A_tempo through D_plus_state, `features.md` section 2), every
  quantity an as-of / strictly-pregame join or a `shift(1)+expanding().mean()` rolling average.
- Four model classes (multiplicative formula, ridge, Gaussian GLM -- the Control's own incumbent mechanism
  reused verbatim, LightGBM) x three predictive-distribution families (Gaussian fixed SD, heteroscedastic
  Gaussian, NegBin/Poisson on the rounded count).
- Primary metric: F2 RMSE, gated by PIT K-S p > 0.05 and a by-month G1 check run through the actual
  deterministic sampler (`cbb_sim.models.pace.sample_pace`), not just a point-estimate comparison.

## 4. Winner

**`multiplicative` / `A_tempo`, target T_pbp.** F2 RMSE 4.8805, MAE 3.7024 (`experiments.md` R2, R4).

This is the zero-fitted-parameter KenPom-style formula `home_tempo_rel * away_tempo_rel *
league_tempo_mean_asof`, using the project's own self-contained ridge tempo ratings (not KenPom) per L9's
finding that they tie centred KenPom within seed noise.

What beat the competition: **nothing did, cleanly.** Every one of the 32 (model class x feature set x fold)
arms failed the pre-registered PIT gate on F2 (every F2 arm's PIT K-S p is below 1e-19 -- traced in
`experiments.md` R8 to overtime games, which run ~8.2 possessions hotter than any OT-blind point estimate and
right-skew/fat-tail every arm's residuals identically, a defect in the shared target, not a reason to prefer
one model over another). The decision rule's fallback -- relax only the gate that fails universally, still
require the one that individual arms CAN pass (by-month G1) -- leaves 13 arms within the measured noise floor
(0.157 RMSE points) of the best raw arm (`lightgbm`/`B_plus_season`, RMSE 4.8177). Simplest of those 13 by the
pre-registered tie-break (fewest features, linear before tree) is the zero-parameter multiplicative formula,
which is why it wins: **every fitted feature this bake-off tried -- season term, style rates, rest/back-to-back
state, and a full LightGBM -- buys at most 0.16 RMSE points of pace accuracy, inside the noise floor.** This
mirrors the Control's own L1 anchor finding (`docs/models/control_engine/experiments.md` R6: A_own and B_kp
tie within seed noise) one level further out: pace itself is close to as good as a two-number formula gets on
this metric.

## 5. Robustness check

- **By month (F2, winner):** 5 of 6 months scored PASS on both mean (+/-1.0 poss) and SD (+/-0.75 poss)
  tolerance sampled through the real predictive-distribution sampler at 50 seeds; April is UNDERPOWERED
  (n = 17). `experiments.md` R6.
- **Responsiveness:** the pre-registered check (predicted pace by tempo-DIFFERENCE quintile) is, as expected,
  flat -- pace is structurally a function of the two teams' *combined* tempo, not their difference. The
  substantive check, tempo-SUM quintile, slopes monotonically with actual pace across all 5 quintiles (64.0 to
  71.9 actual poss/game, delta 0.32-1.12 throughout). `experiments.md` R5.
- **Distribution shape (F2, winner):** none of the three candidate families is well-calibrated in an absolute
  sense (R7/R8); `gaussian_hetero` is best-of-three by PIT p (still << 0.05) and is what the sim consumes, on
  the pre-registered rule that distribution family is chosen by PIT/coverage, never RMSE, even when none of the
  candidates clears the bar.
- **Noise floor:** LightGBM 5-seed refit spread 0.0020 RMSE points (a tight, well-behaved tree fit); linear
  arms' bootstrap SE x1.96 ranges 0.150-0.157. The floor used for the decision (0.157) is the larger of the two.

## 6. Decisions log

1. **Feature columns are per-team (`home_`/`away_`), not pre-summed**, unlike the Control's own pace GLM. This
   bake-off's ridge/GLM/LightGBM classes can all learn a symmetric or asymmetric combination on their own from
   two columns; pre-summing would only throw away information a fresh bake-off doesn't need to discard.
   `experiments.md` implementation note 2.
2. **`multiplicative` uses the project's own tempo ratings, not KenPom**, per L9 and CLAUDE.md's compliant-data
   preference; it is algebraically identical to the classic `AdjT_A * AdjT_B / league_AdjT` form, reconstructed
   from already-centred, as-of pieces so no raw rating level is ever read (CLAUDE.md modelling rule 1).
   `experiments.md` implementation note 3.
3. **Distribution family fit in two stages** (every arm gets the cheap Gaussian-fixed wrapper for gating;
   only the WINNING arm gets the two alternative families), matching the decision rule's own separation of
   RMSE (point estimate) from PIT/coverage (distribution shape) and keeping the grid's cost bounded.
   `experiments.md` implementation note 4.
4. **The universal PIT failure is reported, not engineered around.** No post-hoc widening, no truncated-normal
   patch, no arm rejected or favoured because of it beyond the pre-registered fallback (relax the failing gate,
   keep the achievable one). `docs/SIM_GUARDRAILS.md`'s core principle: fix the sub-model (here: build an
   overtime-aware target or a heavy-tailed error family, both flagged as follow-ups) rather than adjust the
   output.
5. **`fit_count_glm`'s `FittedModel.linpred()` returns the log-link linear predictor, not the mean** (exactly
   how the Control's own rate models are consumed in `cbb_sim.control.simulate._draw_counts`) -- caught during
   this build via a coverage/RMSE sanity check (an early run showed 0% interval coverage and RMSE 64 for the
   Poisson arm before the `np.exp(...)` fix). Recorded here because it is the kind of link-function bug this
   project's multi-level-evidence rule exists to catch.
6. **The heteroscedastic-Gaussian bias correction.** A naive OLS of `log(residual^2)` on the features
   understates sigma by a factor of ~1.89 (the `E[log(chi2_1)] != 0` bias); `pace.hetero_sd` adds back
   `ln(2) + Euler-Mascheroni gamma` before exponentiating. Without it, `gaussian_hetero` would have reported a
   spuriously tight, "better-looking" SD that was simply miscalibrated in the opposite direction.

## 7. Consumption from the sim

```python
import pickle
from cbb_sim.models import pace as P

with open("data/processed/models/pace/pace_v1_winner.pkl", "rb") as fh:
    art = pickle.load(fh)

model = art["point_model"]                 # MultiplicativeModel; .predict(df) -> mu per game
dist = art["dist_gaussian_hetero"]          # chosen family; pace.dist_sd(dist, df) -> per-row SD
mu = model.predict(games_df)                # games_df must carry home_/away_ tempo_rel + league_tempo_mean_asof
draw = P.sample_pace(mu, games_df["game_id"].to_numpy(), seed, dist, family="pace", df=games_df)
```

- **Artifact path:** `data/processed/models/pace/pace_v1_winner.pkl` (dict: `point_model`,
  `dist_gaussian_fixed`, `dist_gaussian_hetero`, `count_model`, `chosen_distribution_family`, `noise_floor`,
  `target_name`).
- **Feature order:** `MultiplicativeModel.predict` only reads `home_tempo_rel`, `away_tempo_rel`,
  `league_tempo_mean_asof` (cross-reference `features.md` section 1); any future re-run that picks a different
  winner will need whatever `winner["features"]` lists in `pace_v1_report.json`.
- **NaN handling:** every model class fills a missing feature with its own train-time median
  (`pace._fillna`/`_fit_medians`), the same fallback contract as the Control's `FittedModel.design`.
- **Output shape:** `sample_pace` returns one float (possessions) per game per call; call once per (seed, game)
  pair exactly like `cbb_sim.control.simulate.simulate`'s own `IDX_POSS` draw, keyed on
  `(seed, game_id, "pace")` so paired arms/seeds reproduce bit-identically (`tests/test_pace.py`).

## 8. Artifacts

| Path | What it is |
|---|---|
| `data/processed/possessions_pbp.parquet` | T_pbp per game, `scripts/build_possessions_pbp.py` |
| `data/processed/possessions_pbp_compare.json` | T_box vs T_pbp mean/SD/corr, per season and pooled |
| `data/processed/models/pace/pace_v1_winner.pkl` | winning point model + all three fitted distribution families |
| `data/processed/models/pace/pace_v1_grid.csv` | every (fold, feature set, model class) row of the stage-1 grid |
| `data/processed/models/pace/pace_v1_report.json` | target decision, noise floor, winner, decision notes |

## 9. Known gaps / followups

1. **No arm passes the pre-registered PIT gate**, traced to overtime games (~5.6% of games, ~8.2 possessions
   hotter than any OT-blind point estimate) plus a residual excess kurtosis of ~1.8 that persists even in
   regulation-only games. The real fix is the L5 overtime model (already a known Control gap) and/or a
   heavy-tailed error family (Student-t) as a future bake-off arm -- not a patch here. `experiments.md` R8.
2. **No player layer, no rim/jumper split** -- out of scope for a game-level pace model.
3. **`multiplicative` winning by the simplicity tie-break is itself informative**, not a failure of the richer
   arms: every added feature bundle and the tree model all landed inside one shared noise floor. A larger
   seed/bootstrap study (or a genuinely different feature -- e.g. injury/roster availability) would be needed
   before concluding richer features can't help pace specifically.
4. **Rest-days/back-to-back/conference-game features (`D_plus_state`) never separated from noise** in this
   grid; they may still matter for margin/efficiency models even though they didn't move pace's RMSE outside
   the floor.
