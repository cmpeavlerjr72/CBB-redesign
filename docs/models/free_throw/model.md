# L3 FREE THROW — model

Status: **BAKE-OFF RUN 2026-09-10.** Winner on F2: **`lgbm`** on the shooter
feature set, with **`eb_shrink`** (empirical-Bayes shrinkage of the shooter's
own as-of rate toward a POSITION prior, strength m = 30 pseudo-attempts) as the
simplest arm that also passes both gates. Absolute numbers, the full grid and
the noise floor live in [`experiments.md`](experiments.md); this file gives the
relative picture and how the sim consumes it.

Companion docs: [`features.md`](features.md), [`experiments.md`](experiments.md).
Pre-registration: `experiments.md` section 1 (PM, 2026-09-10, written before
any modelling).

---

## 1. Purpose

Two things the possession engine needs once the L3 possession-outcome model has
produced a free-throw trip:

- **FT-1, trip structure** — how many attempts the trip contains. This is a
  rule, not a fit, and this model's job is to state the rule, verify it against
  the data, and hand the engine the per-season bonus thresholds.
- **FT-2, make probability per attempt** — the per-attempt Bernoulli the engine
  draws from. This is where the points come from: free-throw rate rose from
  0.305 to 0.352 FTA/FGA across five seasons (L4), so a mis-specified free-throw
  layer shows up directly in gate G2 (points per possession) and G4 (FT rate).

The model sits between the possession-outcome model (which decides that a trip
happens and whether it is a shooting or a bonus trip) and the rebound model
(which takes over when the LAST attempt of a trip misses).

---

## 2. Target variable

**FT-1.** The number of attempts in a trip, against `TRIP_RULES`:

| Foul class | Attempts the rule allows | Identified from context alone? |
|---|---|---|
| and-one | 1 | yes |
| shooting foul (below the bonus) | 2 on a two, 3 on a three | yes, as "at least 2" — the feed does not log the field-goal attempt on a shooting foul that misses, so 2-vs-3 is not observable |
| bonus one-and-one | 1 if the front end misses, 2 if it makes | only in the sharp direction: a ONE-attempt trip here must have MISSED |
| double bonus | 2 (3 on a three) | a one-attempt trip is a violation; a two-attempt trip is ambiguous with a shooting foul |
| technical | 1 or 2 by infraction | no — the feed does not name the infraction |

The foul class is derived from CONTEXT only (a technical row, the and-one
signature, the fouling team's running period foul count), never from the
attempt count, because deriving the class from the count and then checking the
count against the class would verify nothing.

**FT-2.** One row per free-throw attempt: made (1) or missed (0). Technical
attempts are excluded — the shooter is chosen by the coach, not by who was
fouled, so they are drawn from a different shooter distribution. Population,
class balance and the excluded counts: `experiments.md` section 2.

---

## 3. Methodology at a glance

- **Data window.** Seasons 2022-2025 for the fit. The FT-1 rule check also
  reports 2026 DESCRIPTIVELY, because a rule check is data rather than a fit —
  the same standing `scripts/build_possessions.py` gives the 2026 possession
  tables. No fold ever sees 2026; `fold_slices` calls `assert_not_sealed` on
  both slices.
- **Split.** Temporal walk-forward, no random split. F1 trains {2022, 2023} and
  tests 2024; F2 trains {2022, 2023, 2024} and tests 2025 and is the selection
  fold.
- **Primary metric.** Attempt-level log loss on F2. Log loss rather than
  accuracy because the engine draws from the probability, so being right about
  *how* likely a make is matters more than being right about which way it went.
- **Model families tested.** The team-level floor; empirical-Bayes shrinkage
  with the prior and the strength both fitted; logistic ridge; LightGBM.
- **Gates.** Calibration (worst decile gap <= 2 pp on classes with a >= 5%
  share) and responsiveness (predicted make rate by shooter as-of FT% quintile
  must slope with actual). Then: lowest log loss among arms passing both; a tree
  must beat the best simpler passing arm by more than the noise floor; ties to
  the simpler.

---

## 4. Winner

**FT-1: the rule holds, and there is no era boundary.** The two NCAA thresholds
re-derived from each season's own data land at the same place — the 7th team
foul of the half for the one-and-one, the 10th for the double bonus — in
2022, 2023, 2024, 2025 AND 2026. The pre-registration flagged the 2024-25 bonus
structure as a "known era boundary"; the data does not show one. The bonus
structure of NCAA men's basketball is unchanged across the whole five-season
window, and the free-throw growth L4 measured is volume (more trips), not a
different rule. The per-season table is written to
`data/processed/models/free_throw/bonus_era.json` and **the engine reads it into
GameState**, so if a future season does move the thresholds, the engine changes
by a data file and no fitted object changes at all.

**FT-2: `lgbm` wins on F2**, with the lowest log loss of any arm and a clean
pass on both gates. The ordering is the interesting part, and it is exactly what
L15 predicts:

- The **team-level floor** (`team_asof`) is the worst arm by a wide margin and
  fails calibration badly. A team rate cannot tell a 90% shooter from a 55%
  shooter, so its extreme deciles are far off even though it slopes the right
  way. **Free throws are not a team quantity**, and the engine must carry
  shooter identity into them.
- **`eb_shrink`** — the shooter's own as-of rate shrunk toward a POSITION prior
  — beats the floor decisively, passes both gates, and beats the logistic ridge
  on the same information. A two-parameter arm out-scoring a nine-feature
  regression says the signal really is "who is shooting", not "who is shooting
  interacted with the state".
- **`ridge`** fails calibration. It has the shooter's rate as one linear term
  among nine and cannot reproduce the tails.
- **`lgbm`** finds the shrinkage curve itself and edges `eb_shrink` by
  comfortably more than the noise floor, so the pre-registered simplicity
  tie-break does NOT rescue the simpler arm.

**The fitted shrinkage.** Prior = position group, strength m = 30
pseudo-attempts, on BOTH folds independently. A shooter's own rate therefore
carries half the weight at 30 attempts on the season, three quarters at 90 and
nine tenths at 270. The exact share of F2 attempts already past that point is in
`experiments.md` section 5. Two things follow for the engine: early-season free
throws are mostly prior, and a "position mean" is a better prior than either the
league mean or the player's own prior season — which is the same
transfer-attenuation shape L15 measured, arrived at independently here.

---

## 5. Robustness check

- **By bonus vs shooting trip** — reported per arm in `experiments.md` section 4
  as `bonus_gap_pp` / `shooting_gap_pp`.
- **By decile of predicted probability** — the calibration gate itself, split
  into a level component (the overall rate is wrong) and a shape component (the
  ordering is wrong), because those point at different fixes.
- **Transfer subset** — `experiments.md` section 5.1 scores every arm
  separately on players whose modal team changed since the prior season, on
  continuing players, and on players with no prior season at all. This is the
  natural experiment L15 used, and it is the subset where a prior-season-based
  prior is least trustworthy; it is also why the fitted prior being `position`
  rather than `prior_season` is a substantive result and not a shrug.
- **Noise floor** — game-level block bootstrap for the non-tree arms (attempts
  inside one game share officials, lineups and pressure, so the resampling unit
  is the game) and seed-varied refits for the tree.

---

## 6. Decisions log

- **The model key is the CBBD player id, not the ESPN athlete id.** The
  pre-registration prefers the ESPN id via the crosswalk "if present"; it is
  present only for 2024-2026, because CBBD rosters were pulled for those
  seasons, so it cannot span the 2022-2025 fit window. The CBBD id can, and it
  maps to the SAME ESPN athlete id on 100.0% of the 9,421 players the crosswalk
  covers in both 2024 and 2025 and the 3,730 covered in both 2025 and 2026 — so
  it is a stable cross-season identity, not a per-season surrogate. The ESPN id
  is attached wherever it resolves (that is what the L4 layer joins on) and the
  coverage is reported.
- **Technical free throws are excluded from FT-2.** The shooter is chosen by the
  coach. Pooling them would bias both populations. Their count and make rate are
  reported and the engine needs its own rule for them (section 9).
- **`period` is in the feature list although the pre-registration lists only
  "seconds remaining, score diff, bonus vs shooting".** `secondsRemaining` is
  per period in the feed, so on its own it does not identify late-game at all:
  30 seconds left in period 1 and in period 2 are different situations with the
  same value. Recorded as the one addition, in `features.md` section 1.
- **The bonus thresholds are re-derived per season rather than taken from
  `possessions.py`'s constants.** Only a detector that could find a moved
  threshold can answer the era question; a check that assumes the thresholds
  would report "no violations" no matter what the rulebook did.
- **The shrinkage strength is selected on the TRAINING fold's own log loss.**
  That is honest here rather than in-sample, because every quantity the arm uses
  is an as-of feature: a shooter's shrunk rate on 12 January is built from games
  strictly before 12 January, so the training log loss is already a walk-forward
  number.
- **No site feature.** `CLAUDE.md` makes home/away/neutral first-class in every
  scoring-stage model and this one does not carry it, because the
  pre-registered feature list does not. This is a real gap, listed in section 9,
  not closed by an unregistered addition after seeing the grid.

---

## 7. Consumption from the sim

```python
from cbb_sim.models import free_throw as FT

# --- FT-1: trip structure. A RULE, read out of GameState, not a model. ------
era = FT.load_bonus_era()                    # {season: (bonus_prior, double_prior)}
bonus_prior, double_prior = era[game_state.season]
# GameState tracks each team's period foul count; the engine asks:
#   in_bonus        = defence_period_fouls >= bonus_prior
#   in_double_bonus = defence_period_fouls >= double_prior
# and the attempt count follows TRIP_RULES:
#   and-one                 -> 1
#   shooting foul on a 2/3  -> 2 / 3
#   one-and-one             -> 1, then a 2nd IF AND ONLY IF the 1st is made
#   double bonus            -> 2
n_attempts = FT.TRIP_RULES[foul_class]["expected"]

# --- FT-2: make probability. One Bernoulli per attempt. --------------------
# Feature order is FT.FT_FEATURES; see features.md for every column's source
# and fallback. NaN never reaches the matrix: a shooter with no prior game
# sits at a centred 0.0, which IS the league mean.
p_make = model.predict_proba(FT.design_matrix(attempt_rows))[:, FT.CLASS_INDEX["MAKE"]]
```

Sim-loop rule (`CLAUDE.md`): the loop uses lookup tables and vectorised NumPy,
never live model calls. The winning arm is exported as a per-shooter lookup
(shrunk rate by shooter x attempts-to-date bucket) plus the state adjustment,
built once per simulated slate. RNG is keyed on `(seed, game_id, "free_throw")`
through `cbb_sim.control.rng`, the same contract the Control and the pace
sampler use, so paired bake-off arms difference game by game.

Attempts within one trip are drawn independently EXCEPT for the one-and-one,
where the second attempt exists only if the first is made — that dependence is
the rule, and it lives in the trip structure, not in the make model.

---

## 8. Artifacts

| Path | What it is |
|---|---|
| `data/processed/models/free_throw/trips_v1_era.parquet` | one row per free-throw trip, 2022-2026, with the context-derived foul class |
| `data/processed/models/free_throw/attempts_v1_era.parquet` | one row per free-throw attempt, same seasons |
| `data/processed/models/free_throw/bonus_era.json` | **the per-season bonus thresholds the ENGINE reads into GameState** |
| `data/processed/models/free_throw/grid_results.csv` | every (arm, fold) row of the bake-off |
| `data/processed/models/free_throw/run_report.json` | everything `experiments.md` sections 2-5 are rendered from, including the full shrinkage grid |

Trainer: `scripts/train_free_throw_v1.py`. Module:
`src/cbb_sim/models/free_throw.py`. Shared event layer:
`src/cbb_sim/models/event_stream.py`. Shared metrics:
`src/cbb_sim/models/prob_metrics.py`.

---

## 9. Known gaps / followups

1. **No site feature**, against the `CLAUDE.md` standing rule (section 6). It
   was not in the pre-registered list. Pre-register it next round rather than
   add it now.
2. **`trip_pos` (first vs second attempt of a trip) is not a feature.** A
   well-known effect, deliberately not added after the fact. Same treatment.
3. **Technical free throws have no model.** They are ~0.8% of trips; the engine
   currently needs a rule for them. Their make rate is reported in
   `experiments.md` section 2.
4. **The bonus/shooting split is ambiguous on ~40% of trips at the source**
   (data-defect row D5 of `docs/models/change_ledger.md`): once the bonus is in
   force, a two-shot shooting foul and a bonus trip are indistinguishable in the
   feed. `in_bonus` is therefore a partly-noisy feature and the ambiguous mass
   is reported rather than reassigned.
5. **The winner is a tree, so the engine needs a lookup export.** Until that
   export exists and is gated, this model is selected but not shipped.
6. **No paired-seed sim run yet.** Per `CLAUDE.md`, an offline winner ships only
   after a paired-seed sim run shows no gate regressed. G2 (points per
   possession) and G4 (FT rate) are the gates this model moves.
