# L7 LATE-GAME REGIME -- model.md

**STATUS: PROPOSED. Nothing is fitted, nothing is served, no engine default is
changed. This folder exists so that the pre-registration in `experiments.md`
can be committed BEFORE any modelling, as CLAUDE.md's bake-off rule requires.**

Companion files: `features.md` (exactly what would go in) and `experiments.md`
(the pre-registered bake-off, section 1). The evidence that motivates the whole
folder is `docs/tests/late_game_regime_2026-09-11.md`; no number is copied from
it here beyond the three headlines, per the documentation standard.

## 1. Purpose

The engine currently plays the final two minutes of a close regulation game
with the same conditional laws it uses everywhere else, conditioned on the
clock but not on the ROLE (who is ahead). Real basketball in that window is a
different game: the trailing team manufactures possessions by fouling and
shoots threes, the leading team holds the ball and shoots free throws. The
consequence the gates see is G7: the sim produces 0.030 regulation ties against
an actual 0.0557, and the regulation-margin distribution has 47% too few ties
and 50% too many one-point margins with a correct SD -- a SHAPE defect that no
amount of widening can reach.

This model, if a bake-off selects one, would own the conditional law of a
possession inside a pre-declared END-GAME REGION, feeding the same L3 cascade
slot as `possession_outcome` (terminal-event class) and `clock` (duration), but
under a regime gate rather than replacing them. It does not add a probability
to anything, and it never touches engine output: a post-hoc adjustment that
raised the tie rate directly is banned by CLAUDE.md's no-hand-tuning rule and
is explicitly out of scope.

## 2. Target variable

Two targets, one per half of the regime, both already defined by the models
this layer would gate:

- **terminal event class**, the six `possession_outcome.CLASSES`
  (`TOV`, `FGA_rim`, `FGA_jump2`, `FGA_3`, `FT_trip_shooting`,
  `FT_trip_bonus`), from `data/processed/possessions_v2/chances_*.parquet`.
- **possession duration**, the clock model's own target on the same rows.

**Population filter (the regime gate, pre-declared and frozen before any fit):**
`period == 2` and `seconds_remaining <= 120` and `|score_diff| <= 6`. That is
2.6% of regulation possessions (21,327 of 334,069 in 2024-25 on the close set).
The bounds are pre-registered candidates, not tuned: `experiments.md` section 1
carries a sensitivity arm over the 120 s / 6 point cut points, and the cut used
for SELECTION is fixed before any arm is fitted.

## 3. Methodology at a glance

- **Data window.** Seasons 2022-2025 from `possessions_v2`.
- **Split.** Temporal walk-forward, no random split. Fold 1 trains through
  2022-23 and tests 2023-24; fold 2 trains through 2023-24 and tests 2024-25.
  **Fold 2 is the selection metric.** 2025-26 is sealed.
- **Primary metric.** Offline: multiclass log loss on window possessions only.
  Closed-loop primary: the regulation-margin density near zero (`P(0)`, `P(1)`,
  and their ratio), with the tie rate and the margin excess kurtosis as named
  secondaries.
- **Families tested.** Four arms: the served cascade unchanged (A); state
  enrichment inside the existing sub-models (B); a regime-conditioned refit of
  the same classes (C); a dedicated end-game model (D). Full grid in
  `experiments.md` section 1.

## 4. Winner

**NONE. Not run.** No arm has been fitted, graded or adopted. This section
stays empty until `experiments.md` records a fold-2 result that beats its own
noise floor and a Decision 10 closed loop in which no gate regressed.

## 5. Robustness check

Pre-registered, not run. Three cuts are mandatory before any arm can be called
a winner and each has a pre-declared direction, so that a flat result is a
failure rather than a silence:

- **role** (trailing / leading / tied at the window start): the arm must
  reproduce the SIGN of the trailing-minus-leading split on the three-point
  share (negative) and the bonus-free-throw rate (positive).
- **seconds-remaining bucket** ((90,120], (60,90], (30,60], (10,30], (0,10]).
- **team quintile** by season PPG: the fouling channel must SLOPE, per
  CLAUDE.md's matchup-specific rule.

Cells below 200 possessions are reported as UNDERPOWERED and never read as
signal or as absence of signal.

## 6. Decisions log

1. **Built as a regime LAYER rather than as new columns in
   `possession_outcome`** -- because that model's served arm already carries
   `score_diff`, `seconds_remaining` and `in_bonus` and still reproduces about
   a tenth of the role asymmetry. The defect is a conditional law a pooled fit
   averages away, not a missing feature. Arm B nevertheless tests the
   feature-addition hypothesis head to head, because the standing rule is that
   no structure is adopted for being plausible.
2. **The regime gate is a hard, pre-declared region, not a learned one.** A
   learned boundary would be a free parameter fitted to the gate it is graded
   on, which is the hand-tuning pattern CLAUDE.md bans.
3. **The tie rate is a SECONDARY, never the primary.** Optimising the tie rate
   directly is the banned pattern in another costume. The primary is the shape
   of the margin density near zero, which a wrong model cannot hit by accident.
4. **A regime-only fix is pre-registered to land near 0.044, not 0.0557**, and
   the closed-loop gate is a two-sided band around that. About 27% of the tie
   deficit is the margin-at-2:00 distribution, which this layer does not own.
5. **Home/away/neutral is a first-class feature of every arm**, per CLAUDE.md.
6. **Rule-era flags stay in GameState.** The regime gate reads the live state
   block; no era constant is baked into a fitted object.

## 7. Consumption from the sim

Not written. No artifact exists to load. When and if an arm is adopted it is
served behind a flag over the shared core (`ENGINE_LATE_GAME`, default `off`),
never as a fork, per CLAUDE.md's one-engine rule, and the gate is evaluated on
the live state block already assembled in `src/cbb_sim/engine/loop.py`
(`period`, `seconds_remaining`, `score_diff`, `in_bonus` are all present).

## 8. Artifacts

| Path | What it is |
|---|---|
| -- | none; nothing has been trained |

## 9. Known gaps / followups

- The motivating measurement is on run A's flag set, not on the current served
  `ENGINE_CLOCK=v5b_glat_pmean`. The default-clock re-read is a precondition of
  running the round, recorded in `experiments.md` section 1.
- The sim tap behind the evidence is 1,000 simulations; the per-cell kernel is
  underpowered and a 200-seed re-read is pre-registered as part of the round.
- The layer owns about 70% of G7 by the measured attribution. The rest is the
  margin-at-2:00 distribution, which belongs to the clock and efficiency lanes
  and is explicitly out of this folder's scope.
- Overtime periods are OUT of the declared region. The variance/OT diagnostic
  established that the OT module is fine once a game is tied; extending the
  regime into OT is a separate, later question.
