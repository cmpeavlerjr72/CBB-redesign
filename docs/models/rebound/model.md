# L3 REBOUND — model

Status: **BAKE-OFF RUN 2026-09-10.** Winner on F2: **`lgbm` / `C_plus_state`**,
the first L3 arm in this project to pass BOTH pre-registered gates. Rebounding
is a **TEAM-level** sub-model, not a lineup-level one, and dead-ball rebounds do
NOT need a model of their own. **S1 scheme confirmation (2026-09-10,
`experiments.md` sections 7-8): refit cadence is `S1_weekly`**, beating the L21
monthly default and the conference-aligned calendar on log loss beyond a
(corrected) noise floor, though no calendar tested clears the conference-window
calibration gate on this sub-model -- see section 10 below. Absolute numbers,
the full grid and the noise floor live in [`experiments.md`](experiments.md);
this file gives the relative picture and how the sim consumes it.

Companion docs: [`features.md`](features.md), [`experiments.md`](experiments.md).
Pre-registration: `experiments.md` section 1 (PM, 2026-09-10, written before any
modelling).

---

## 1. Purpose

Given that a shot or a last free throw has missed, who ends up with the ball.
This is the branch that decides whether a possession continues (a second chance,
a second shot at the same defence) or changes hands, so it drives gate G4's
OREB% directly and gate G2's points per possession through the extra chances it
creates.

It sits immediately after the possession-outcome model (which produces the miss)
and the free-throw model (which produces the missed last attempt), and it feeds
back into the possession-outcome model when the offence retains.

---

## 2. Target variable

One row per **rebound opportunity**: a missed field-goal attempt, or the missed
LAST free throw of a non-technical trip. Three classes:

| Class | What it is |
|---|---|
| `OREB` | the offence secured it |
| `DREB` | the defence secured it |
| `DEAD` | a dead-ball rebound — the ball went out of bounds and was awarded |

Plus one outcome that is deliberately NOT modelled: `unresolved`, where the feed
logs a loose-ball foul, a shot or a turnover next and the rebound row never
appears. Those rows are dropped with their count reported, the same treatment
`unknown` gets in L3 round 1.

Two things about the target are worth stating because they are not obvious:

- **A missed first free throw of a trip is not an opportunity.** The ball is
  dead and the same shooter shoots again. ESPN logs an administrative
  "Offensive Rebound" between the two attempts, which
  `cbb_sim.models.event_stream` drops exactly as `possessions._collect_trip`
  identifies it. Getting this wrong would add roughly 25,000 phantom
  opportunities a season.
- **The target cannot be read off the possession tables.** An offensive rebound
  is visible there only as "a further chance exists in this possession", a
  defensive one only as "the possession ended", and a dead-ball rebound is
  invisible entirely, because `possessions.py` treats `DeadBallReb` as a no-op
  by design. So the model reads its target off the event stream. What it does
  NOT re-derive is the possession segmentation.

Class shares overall and by season, by miss type, and the `unresolved` share:
`experiments.md` sections 2.1 and 2.2.

---

## 3. Methodology at a glance

- **Data window.** Seasons 2022-2025.
- **Split.** Temporal walk-forward, no random split. F1 trains {2022, 2023} and
  tests 2024; F2 trains {2022, 2023, 2024} and tests 2025 and is the selection
  fold. Season 2026 is sealed; `fold_slices` calls `assert_not_sealed` on both
  slices.
- **The lineup bundle gets its own fold, L2** (train 2024, test 2025), because
  CBBD `onFloor` is empty at the source before 2024 (L13) and the feature
  therefore cannot exist in an F1 or F2 training window.
- **Primary metric.** Three-class log loss on F2.
- **Model families tested.** A league-share baseline BY MISS TYPE (the floor),
  multinomial ridge logit, LightGBM multiclass.
- **Gates.** Calibration (worst decile gap <= 2 pp on classes with a >= 5%
  share) and responsiveness (predicted OREB share by quintile of the offence's
  as-of OREB% must slope with actual, monotone in 4 of 4 steps — and the same
  for the defence's as-of DREB%; the pre-registration says 4 of 4 explicitly,
  which is STRICTER than the L3 round-1 reading of 3 of 4).

---

## 4. Winner

**`lgbm` / `C_plus_state`.** It has the lowest F2 log loss of any arm and is the
only arm that passes both gates. This is the first L3 bake-off in this project
where anything passes: round 1 of the possession-outcome model had 0 of 11 arms
clear the same two gates.

The shape of the result:

- **The miss type is by far the biggest block.** Going from `A_team` to
  `B_plus_miss` is worth roughly eight noise floors on the linear arm and nine
  on the tree. That is what the raw shares predict: a missed rim attempt is
  rebounded by the offence 37.7% of the time and a missed free throw 12.7%, so a
  model that does not know which kind of miss it is looking at is averaging over
  a 25-point spread.
- **Game state is a genuinely marginal block, and only the tree can use it.**
  `B_plus_miss` -> `C_plus_state` is worth about one noise floor on
  `ridge_logit` — i.e. nothing you could defend — but about three on `lgbm`.
  This is the opposite of L3 round 1, where the same block was worth 30-140
  floors in every model class. Where the ball goes after a miss is mostly
  physics and personnel, not clock and score.
- **No LINEAR arm passes calibration on F2.** The best of them,
  `ridge_logit`/`C_plus_state`, misses at a gap driven by a genuine SHAPE
  component, not only by the season-drift level miss. So the pre-registered
  clause "a tree arm must beat the best linear arm by more than the floor" has
  no passing linear arm to bind against; `experiments.md` section 6 reports the
  raw tree-minus-best-linear gap in floors anyway, and it clears the floor by
  about 1.4x — real but not large — so the stricter reading selects the same arm,
  on a thinner margin than the gate outcome itself.
- **The matchup-naive baseline passes calibration and fails responsiveness**, as
  it should: knowing the league average by miss type gets the level right and
  cannot tell a good offensive rebounding team from a bad one. That is the
  `CLAUDE.md` "matchup-specific, not league-average" rule doing its job.
- **Season drift is visible and unfixed.** OREB% rose from 28.0% (2022) to 29.6%
  (2025) and every arm carries a level miss on F2 in the same direction. The
  pre-registration's four bundles contain no season term, so nothing here
  addresses it; it is the same L11 finding, at a third sub-model.

### Lineup vs team — the question the PM asked to be reported separately

**TEAM-LEVEL, on a margin of about one noise floor.** On the L2 fold, restricted
to the 95.25% of opportunities that carry all ten on-floor ids so that C and D
are scored on exactly the same rows, adding the on-floor five's as-of individual
OREB/DREB rates is worth **0.88 floors on the linear arm and 1.11 floors on the
tree**. The two straddle the bar. Report it as it is rather than rounding it to
a clean answer: the lineup aggregate adds at most one floor's worth of log loss,
which is not enough to justify making rebounding a lineup-level sub-model in the
engine, and it is not zero either. The exact gains, the fold's own bootstrap SE
and the fitted shrinkage rung are in `experiments.md` section 5.

Two things make the reading safe in the team-level direction. Neither `C` nor
`D` passes the calibration gate on this fold at all, so nothing here is
adoptable regardless of which wins. And the fold is two seasons long (L13), so
a one-floor effect is exactly the size this fold cannot resolve.

Read it this way: individual rebounding ability is real (L15 puts player-beyond-
both at 35.9% of the variance in per-minute rebound rate), but by the time five
of those players are aggregated and compared against the two teams' own as-of
rebounding rates, the lineup aggregate is nearly redundant with the team rate it
was built from. **The engine can draw rebounds at the team level.** Individual
rebound ATTRIBUTION — which of the five players gets credited, which is what the
player-props layer needs — is a separate question this bake-off does not answer,
and the per-player as-of rates built here (`rebound.player_rebound_rates`) are
the right input to it.

The verdict string in `run_report.json` is computed from the LINEAR arm's gain
alone; both arms' gains are printed next to it in `experiments.md` section 5, and
the straddle above is the honest summary of the pair.

### Dead balls — the other question the pre-registration asked

**They do not need their own model.** Replacing the three-class model with a
LIVE-only binary model composed with a fixed dead-ball share per miss type costs
less log loss than the noise floor, and the composed model's dead-ball decile
calibration is well inside the gate. The class still has to EXIST — it is 0.5-1%
of opportunities, it is invisible in the possession tables, and measured on 2024
only 58.1% of dead-ball rebounds are followed by the next real action from the
shooting team, so it is not a deterministic hand-back to either side. But a
deterministic per-miss-type share is sufficient, which simplifies the engine.

---

## 5. Robustness check

- **By miss type** — `missgap_rim` / `missgap_jump2` / `missgap_three` /
  `missgap_ft` in every grid row of `experiments.md` section 3: the worst
  predicted-minus-actual class-share gap inside each miss type. This is the
  pre-registration's by-miss-type calibration.
- **By decile of predicted probability**, split into a level component (the
  class's overall rate is wrong) and a shape component (the model orders
  opportunities wrongly), because those point at different fixes.
- **By responsiveness quintile on BOTH drivers** — the offence's as-of OREB% and
  the defence's as-of DREB% — with the predicted-vs-actual slope ratio reported
  alongside the monotone step count, so "slopes the right way but too flat" is
  distinguishable from "slopes the wrong way".
- **By season** — class shares and the `unresolved` share per season, section
  2.1, which is where the OREB% drift is visible.
- **Noise floor** — a 200-replicate GAME-level block bootstrap for the linear
  arms (opportunities inside one game share lineups, officials and pace, so the
  resampling unit is the game) and 5 seed-varied refits for the tree.
- **Underpowered cells are labelled.** The `DEAD` class is 1% of rows, so it is
  below the 5% share threshold and is NOT gated on calibration; its numbers are
  reported, never presented as a pass or a failure.

---

## 6. Decisions log

- **The target is read off the event stream, not the possession tables.**
  Justified in section 2: the dead-ball class does not exist in the possession
  tables at all, so a three-class target cannot be built from them.
- **Team rates come from FIRST-CHANCE opportunities only.** The
  pre-registration's "first-chance-safe sources" clause. It closes the exact
  channel by which L16's 2025 labelling defect reached a first-chance model's
  predictors in L3 round 1, and it stops "as-of OREB%" from being partly a
  function of how many second chances a team happened to get.
- **The `blocked` flag is a flag, not an event.** ESPN logs "Block Shot" as its
  own row, usually between the missed attempt and the rebound. Left in the
  stream it would break "the next event is the rebound"; dropped without being
  recorded it would throw away a real predictor (10.1% of missed FGAs in 2024).
  So it is folded onto the attempt and then the row is dropped.
- **`chance_index` is NOT a feature**, although it is a real predictor (in 2024, OREB%
  is 28.7% on a possession's first miss and 31.1% on the second). The
  pre-registration's four bundles do not contain it and this round does not add
  features to a pre-registered grid. Recorded in `features.md` section 5 so the
  next round can pre-register it deliberately.
- **The lineup shrinkage strength is fitted, not assumed** (L13), on the L2
  fold's TRAIN season over a stated grid, and the chosen rung is reported.
- **The k = 0 rung of that grid falls back to the league per-player rate rather
  than to NaN.** Without it, a player with no prior opportunities would drop the
  whole opportunity out of the lineup subset, so the k = 0 rung would be scored
  on a different and smaller set of rows than every other rung and the grid
  would not be comparing shrinkage strengths at all.
- **`unresolved` opportunities are dropped, never imputed**, with the count
  reported per season.

---

## 7. Consumption from the sim

```python
from cbb_sim.models import rebound as RB

# Feature order is RB.feature_set("C_plus_state"); see features.md for every
# column's source and fallback. NaN never reaches the matrix: a team with no
# prior game sits at a centred 0.0, which IS the league mean.
features = RB.feature_set("C_plus_state")
p = RB.predict_arm("lgbm", model, opportunity_rows, features)
# columns are RB.CLASSES == ("OREB", "DREB", "DEAD"), in that fixed order
oreb, dreb, dead = p[:, 0], p[:, 1], p[:, 2]
```

Because the dead-ball question came back "a fixed share is enough", the engine
may equivalently draw the live/dead split deterministically per miss type and
then draw OREB vs DREB from a binary model —
`RB.compose_binary_plus_fixed_dead` is that composition, and
`experiments.md` section 4 is the measurement that says it costs nothing.

Sim-loop rule (`CLAUDE.md`): the loop uses lookup tables and vectorised NumPy,
never live model calls. The winning arm is exported as a lookup over the
matchup features x miss type x the state bins. RNG is keyed on
`(seed, game_id, "rebound")` through `cbb_sim.control.rng`, the same contract
the Control and the pace sampler use, so paired bake-off arms difference game by
game.

The engine draws the rebound at the TEAM level (section 4). Attributing it to
one of the five on-floor players is the player layer's job, and
`rebound.player_rebound_rates` is the as-of input it should use.

---

## 8. Artifacts

| Path | What it is |
|---|---|
| `data/processed/models/rebound/events_v1.parquet` | one row per rebound opportunity, 2022-2025, with outcome, miss type, blocked flag, chance index, on-floor ids and the rebounder id. `events_v2.parquet` when run with `--version v2` |
| `data/processed/models/rebound/grid_results.csv` | every (arm, feature set, fold) row of the bake-off |
| `data/processed/models/rebound/run_report.json` | everything `experiments.md` sections 2-6 are rendered from, including the full decile calibration and the lineup block |

Trainer: `scripts/train_rebound_v1.py`. Module:
`src/cbb_sim/models/rebound.py`. Shared event layer:
`src/cbb_sim/models/event_stream.py`. Shared metrics:
`src/cbb_sim/models/prob_metrics.py`.

---

## 9. Known gaps / followups

1. **No season term in any bundle**, so every arm carries the OREB%-drift level
   miss on F2. The pre-registration did not include one. This is L11 at a third
   sub-model and should be pre-registered next round together with the
   recency-weighting question the possession-outcome model left open.
2. **`chance_index` is not a feature** (section 6). It is the largest known
   omitted predictor.
3. **The winner is a tree, so the engine needs a lookup export.** Until that
   export exists and is gated, this model is selected but not shipped.
4. **The lineup verdict is a one-floor straddle on two seasons of data**
   (L13: `onFloor` starts in 2024), on the 95.25% of opportunities that carry
   all ten ids. It should be re-read when 2026 unseals and a third season
   exists, and the individual-rate definition itself (share of on-floor
   opportunities) is only one of several reasonable ones.
5. **`unresolved` opportunities are a real if small hole** (~0.8%). If they are
   not missing at random with respect to OREB — a loose-ball foul is more likely
   on a contested rebound — the level is slightly biased. Not yet measured.
6. **No paired-seed sim run yet.** Per `CLAUDE.md`, an offline winner ships only
   after a paired-seed sim run shows no gate regressed. G4's OREB% is the gate
   this model moves.

---

## 10. S1 scheme confirmation: refit cadence (2026-09-10)

Pre-registration and full results: `experiments.md` sections 7-8. Model class and feature set held
fixed (`lgbm` / `C_plus_state`); only the refit CALENDAR was in question. Four schemes tested on F2
(2025): `S0` static (reference, reproduces the adopted F2 log loss 0.645565 exactly), `S1_monthly`
(the L21 default, 6 refits), `S1_conf_aligned` (29 refits), `S1_weekly` (23 refits).

**Winner: `S1_weekly`.** All three S1 schemes beat the `S0` reference on log loss, and by a wide
margin against the CORRECTED noise floor: the pre-registered second-seed gap came back
implausibly small (4e-06) because seeds 0 and 1 happen to sit close together, so the floor used is
round 1's own already-published 5-seed SD for this arm (6.7e-05, the larger and more robust number,
per the standing "use the larger floor" convention -- `experiments.md` section 8.2). Under that
floor, `S1_weekly` (log loss 0.644522, 15.6x the floor) is the only scheme within the floor of
itself as the best beater; `S1_monthly` (9.2x) and `S1_conf_aligned` (10.3x) both clear the
reference but not `S1_weekly`'s own number.

**A caveat this project does not smooth over.** Unlike the free-throw S1 confirmation (same date),
NO scheme here clears the 2.00 pp conf4 gate: `S0` 2.462 pp, `S1_monthly` 2.660, `S1_conf_aligned`
2.367 (the best of the four, but not the log-loss winner), `S1_weekly` 2.565 -- a slight WORSENING
relative to `S0` (-0.103 pp). `S1_weekly` wins on the primary metric alone, which the pre-registered
OR decision rule permits, but Decision 9's predicted damage segment (the first four weeks of
conference play) is not resolved for rebound by any calendar tested here. This is reported as an
open item, not folded into the winner's headline.

F1 (2024) evidence exists for `S0` (cited from section 3, not refit: 0.620446) and `S1_monthly`
(refit: 0.619726) only; `S1_conf_aligned` and `S1_weekly` were not run on F1 by the pre-registered
budget/drop order.

**Engine consequence.** The FT-2-style manifest the adapter needs is
`data/processed/models/rebound/s1_confirm/S1_weekly/F2/manifest.json` (season 2025;
`manifest.py` format, `max_train_date` on every entry, 23 dated artifacts). As with free throw, this
is the SELECTION-fold schedule; a live 2026 deployment manifest is a separate, later step.
