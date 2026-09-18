# L?: SHOT BLOCK -- experiments

Append-only. Status changes go to `docs/models/change_ledger.md` in the same
commit (`CLAUDE.md`, standing rule "bake-off before any choice").

---

## 1. Pre-registration (written 2026-09-18 by the rebound round-3 lane, BEFORE any modelling; PROPOSED, NOT RUN at the time of writing, NOTHING ADOPTED)

### 1.1 Why this model exists

`docs/tests/g4_oreb_fta_diagnostic_2026-09-18.md` section 1.2 measured that the
engine hard-sets `blocked_f = 0.0` on every rebound opportunity
(`src/cbb_sim/engine/loop.py`, the rebound block) **because the cascade has no
shot-block model**: there is no event class, no rate table and no `blocked` draw
anywhere in the engine. The rebound model nevertheless carries `blocked_f` in its
served `C_plus_state` bundle and uses it correctly -- on blocked rows, with the
true feature, it predicts 0.4225 / 0.3995 / 0.3997 P(OREB) against actuals of
0.4237 / 0.4113 / 0.4274 by miss type. Feeding it a false zero costs
**-0.740 pp of pooled OREB%, 47.4% of gate G4's -1.561 pp miss**.

Blocked misses are not rare:

| miss type | blocked share of live opportunities | actual P(OREB) unblocked | actual P(OREB) blocked |
|---|---:|---:|---:|
| rim | 0.2617 | 0.3688 | 0.4237 |
| jump2 | 0.0816 | 0.2822 | 0.4113 |
| three | 0.0143 | 0.2857 | 0.4274 |
| ft | 0.0000 | 0.1377 | (n = 1) |

This is an **engine-feed defect**, and closing it honestly means the cascade has
to be able to say whether a given miss was blocked. That is this sub-model.

### 1.2 Target, and where it sits in the cascade

**Target: `P(blocked | the attempt MISSED, context)`** -- a binary outcome on the
population of missed field-goal attempts. `blocked` is `cbb_sim.models.
event_stream`'s own folded `Block Shot` flag, the identical column the rebound
opportunity table already carries.

**Cascade placement:**

    shot selection  ->  make / miss  ->  [if miss] BLOCK draw  ->  rebound

and **not** `shot selection -> block -> make / miss`. The reason is a property of
the model immediately upstream, verified in code rather than assumed:
`cbb_sim.models.fg_make` lists `blocked` in `BANNED_FEATURES` with the note "a
`Block Shot` row exists only because the attempt missed; it is a post-outcome
field, not a pre-release one", does not build the column, and its test suite
asserts no bundle contains it. `fg_make` therefore estimates
`P(make | shot context)` **marginally over block status**: blocked attempts are
already inside its miss population, at their natural rate.

Consequently **a block draw placed before the make draw would double count** --
it would divert attempts away from a make model that has already priced them as
misses, and the engine's eFG% would move. Drawing the block *after* the miss is
realised changes no field-goal aggregate at all: it only labels a miss that has
already happened, and the label is consumed by exactly one downstream consumer,
the rebound model's `blocked_f` column. **`fg_make` is not edited, not refit and
not double counted by this model, and the round reports that as a checked fact.**

Free throws are excluded: a missed free throw is never blocked (n = 1 in
2022-2025, a feed artefact).

### 1.3 Universe and folds

D-I, non-truncated, seasons 2022-2025, one row per **missed FGA** taken from the
same `data/processed/models/rebound/events_v1.parquet` event layer the rebound
round-3 design is built on, so the two models see the identical rows and the
`B3` feed is a drop-in.

Folds are the project's: **F1 trains {2022, 2023} and tests 2024; F2 trains
{2022, 2023, 2024} and tests 2025; F2 selects.** Season 2026 is SEALED and
`assert_not_sealed` is called on both slices of both folds.

### 1.4 Candidate arms (pre-registered, exhaustive)

| arm | what it is | Decision-9 status |
|---|---|---|
| `K0` | **league rate by shot type**, from the most recent TRAINING season (the honest baseline; the test season's own shares are reported separately and labelled unattainable ORACLE) | reference |
| `K1` | `K0` refined by the **defence's as-of block rate**, league-centred, expanding within season, strictly before the game -- a shot-type x defence cell rate | -- |
| `K2` | logistic ridge on the full pre-registered feature bundle below | -- |
| `K3` | LightGBM on the same bundle | -- |
| `K1_oa` | `K1` with the defence as-of rate **opponent-adjusted** (`cbb_sim.features.opponent_adjust`, `one_pass`) | Decision 9a arm |
| `K3_conf` | `K3` + a **conference-game flag** | Decision 9b arm |
| `K3_prior` | `K3` + **prior-season carry** of the defence (and shooter) as-of rate, the same EB shrinkage `possession_outcome` round 4 uses | -- |

Refit cadence (Decision 9c) is **NOT** an arm of this round: this model has never
been fitted at all, and a cadence bake-off on top of a model-class bake-off would
confound the two. The round runs on the static calendar and records cadence as an
explicit open item for a round 2, exactly as `free_throw` and `rebound` did.

### 1.5 Features

Every rate feature is an expanding mean over games STRICTLY BEFORE the current
one, within season, centred on the league's own as-of mean on the same date
(`CLAUDE.md`: "every rating feature is expressed relative to its own snapshot's
league mean"). No raw level enters any bundle.

| feature | source | why |
|---|---|---|
| `miss_rim`, `miss_jump2`, `miss_three` | the miss type the possessions build labels (L16 rim override applied) | the block rate is 18x higher at the rim than on a three |
| `def_block_c` | defence's as-of blocks / missed FGA faced, league-centred | the matchup-specific driver; the quintile slope is checked on it |
| `off_blocked_c` | offence's as-of blocked / missed FGA taken, league-centred | some offences get blocked much more than others |
| `shooter_blocked_c` | the SHOOTER's own as-of blocked share, league-centred, EB-shrunk toward the league per-player rate with a FITTED pseudo-count (grid `(0, 50, 100, 200, 400)` opportunities, fitted on the train fold, never assumed) | the PM condition names the shooter explicitly |
| `site_home`, `site_away` | neutral is the reference level | `CLAUDE.md`: home/away/neutral is a first-class feature in every scoring-stage model |
| `off_rating_off_c`, `off_rating_def_c`, `def_rating_off_c`, `def_rating_def_c` | `cbb_sim.ratings.own_ratings`, already opponent-adjusted and league-centred | team strength, the same block every other sub-model carries |
| `period`, `seconds_remaining`, `score_diff`, `in_bonus` | game state | the same state block the rebound model uses, so the two agree on what "context" is |

Banned outright, and not built: the rebound outcome itself, anything about the
rebound that follows, the attempt's own post-outcome fields, and any column
computed from the test season's completed totals.

Bundles: `Ka` = miss type only; `Kb` = `Ka` + the three as-of block rates + site;
`Kc` = `Kb` + ratings + state. `K2`/`K3` run on `Kc`; `K1` is `Kb` as a cell rate.

### 1.6 Primary metric and gates

**Primary: log loss on fold 2.** Reported alongside: Brier, the pooled level
error `mean(p_hat) - actual blocked share` (this is the reading the rebound
`B3` feed actually consumes, and the reason the round exists), and per-shot-type
level error.

Gates, in the project's standing form:

1. **Calibration** -- worst decile gap <= 2.0 pp.
2. **Responsiveness / matchup-specificity** -- predicted block rate bucketed by
   the **defence's prior-season block rate quintile** must slope with the actual,
   monotone in at least 3 of 4 steps, with the slope ratio reported. A flat
   predictor that gets the league rate right is a FAIL: the whole point of the
   feed is that it differs by matchup. The same cut is reported for the
   offence's and the shooter's prior quintile.
3. **Level** -- |pooled level error| <= 0.25 pp on fold 2, and <= 0.50 pp within
   each of rim / jump2 / three.

### 1.7 Segments (every arm, both folds)

Shot type; defence prior-season block quintile (and its slope ratio); month of
the test season; home / away / neutral; conference vs non-conference; period and
five-minute game-minute bucket; early season (Nov-Dec) vs late. Minimum cell
n = 300, and every cell below it is printed `UNDERPOWERED` and excluded from
every pass/fail. Underpowered is never reported as signal or as absence of
signal.

### 1.8 Noise floor

A spec-identical retrain under a second seed for the leading tree arm and for the
reference, on fold 2. The floor is the observed log-loss spread; where this
model's own two-seed gap comes back smaller than a published multi-seed SD for a
comparable cell, the **larger** is the operative floor, the convention
`rebound/experiments.md` section 8.2 already fixed for this project.

### 1.9 Decision rule

1. An arm is eligible only if it beats `K0` on fold-2 log loss by more than the
   measured floor.
2. Among eligible arms the winner has the lowest fold-2 log loss; ties inside one
   floor go to the **simpler** arm, ordered `K0 < K1 < K1_oa < K2 < K3 <
   K3_conf < K3_prior`.
3. A winner must pass all three gates of 1.6.
4. A tree arm must beat the best eligible linear arm by more than the floor,
   otherwise the linear arm wins.
5. Fold 1 must not reverse the sign of the fold-2 margin.
6. **Nothing ships on offline evidence.** The winner's only use in this round is
   as the `B3` / `B3e` feed of `rebound/experiments.md` round 3; wiring it into
   the engine requires a DEFAULT-OFF flag, a bit-identical served path checked
   against `docs/ops/parity_reference_windows_v6.json`, and a paired closed-loop
   run with no gate regression.
7. If no arm clears rule 1, **no arm is adopted** and the round says so. The
   engine keeps feeding `blocked_f = 0.0` and G4's -0.740 pp channel stays open.

### 1.10 What this round may not do

No post-hoc multiplier, cap, clip, offset or calibration curve on the model's
output or on sim output. No arm may be fitted on, or read a level from, the test
season's completed totals. Season 2026 stays sealed. `fg_make` is not edited and
not refit: if this model's existence implies a change there, that is a finding to
report, not a change to make inside this round.

---

<!-- RESULTS APPENDED BELOW BY scripts/train_shot_block_v1.py -->
