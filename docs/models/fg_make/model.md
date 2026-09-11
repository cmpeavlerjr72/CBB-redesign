# L3 FIELD-GOAL MAKE: model

Bake-off run 2026-09-10. Full grid, every number, and the pre-registration it
executes: `docs/models/fg_make/experiments.md`. Feature provenance:
`docs/models/fg_make/features.md`. Code: `src/cbb_sim/models/fg_make.py`,
`scripts/train_fg_make_v1.py`, `scripts/train_fg_make_v2.py`,
`scripts/train_fg_make_v2b_s1.py`.

> ## STATUS 2026-09-10: ROUNDS 2 AND 2b SUPERSEDE THE ROUND-1 STATE BLOCK
>
> **Round 1's `score_diff` is a POST-OUTCOME feature and the round-1 winner
> must not be shipped.** It is read off the feed's running score on the
> attempt's own row, and that column is the score AFTER the play: a made three
> already carries its own three points. Evidence, proved three ways off the raw
> feed:
> `docs/tests/fg_make_state_confound_2026-09-10.md`. The leak manufactures
> 62.0 / 83.3 / 83.5% of the apparent margin effect (rim / jumper / three) and
> is the mechanism behind L23's margin SD 34.6 and home/away correlation -0.64.
>
> **Round 2 (experiments.md sections 13-14) re-parametrised the state block and
> adopted `S-C R2_C_safe_state` on all three classes**: the team block, the
> shooter block and `period`, `seconds_remaining`, `in_bonus`, `chance_number`,
> `chance_elapsed_s`, `is_transition_f` -- **no margin term of any kind**. It
> passes the `ARCHITECTURE_DECISIONS.md` Decision-10 closed-loop gate (margin SD
> 14.05 against the round-1 arm's 35.39 on the same paired 500-game x 5-seed
> run; home/away correlation +0.241 against -0.605) and improves calibration on
> all three classes. It costs 35.0 / 38.2 / 36.4 noise floors of log loss --
> that gap was the leak.
>
> **Round 2b (sections 15-16) adopted the standing S1 scheme** (L21, monthly
> in-season walk-forward refit) for the same arm: no gate regressed, log loss
> improved by 0.19-0.60 floors, and the closed-loop gate passed.
>
> **Round 3 (sections 17-18) re-keyed the shooter on `shot_shooter_id`** and
> found the `FGA_3` shooter-quintile span collapse from 35.9 pp to 2.8 pp. Its
> pre-registered rule read "not adopted"; L29 overrode that on data-integrity
> precedence (a model trained on a wrong label is ineligible to be served) and
> made the corrected arm the INTERIM served model.
>
> **Round 4 (sections 19-20) re-baked the shooter block from scratch on the
> correct label and its winner is `B1 R4_B1_shrunk`**: the team block, the S-C
> engine-safe state block and **one** shooter column,
> `shooter_shrunk_dev_c` -- the shooter's as-of rate shrunk to his own team's
> as-of rate with `m` = 75 / 200 / 300 attempts fitted on fold 1. It is the only
> arm that passes calibration and Decision 8 on all three classes, and the one
> column moves the Decision-8 shooter slope from 0.28/0.29/0.31 (no shooter
> block) to **1.04 / 0.84 / 0.92**, closing the `FGA_jump2` and `FGA_3` failures
> L29 left open. It passes the Decision-10 closed loop with the best margin SD
> (13.757 vs 12.127 actual) and home/away correlation (+0.242 vs +0.423) of any
> fg_make arm to date.
>
> **THE SERVED MODEL AND THE DEFAULT ARE `ENGINE_FG_MAKE=round3_shooter_S_C_s1`**
> (`data/processed/models/fg_make/round3_shooter/S_C_s1/`, through
> `cbb_sim.engine.manifest`), whose own missing Decision-10 closed-loop gate was
> run and PASSED this round (section 20.0). **The round-4 winner is not yet
> servable from the shared engine inputs**: `round4_B1` needs the slot columns
> `shooter_shrunk_dev_c__{class}`, which exist only in
> `data/processed/models/engine_fgm4/`; writing them into the shared
> `data/processed/models/engine/` means rewriting an array other workers are
> reading. Section 20.7 names the unblocking step. Sections 4-8 below describe
> ROUND 1 and are left exactly as they were written.
>
> **The engine's own shooter inputs carried a train/serve skew** and it was
> worth 0.97 pp of three-point make rate: `scripts/build_engine_inputs.py`
> builds the shooter slot block from the `participant_1_id`-keyed events, whose
> `shooter_make_c__three` correlates only 0.33 with the corrected one.
> `scripts/build_engine_inputs_shotshooter.py` closes it into a sibling
> directory (section 20.0).

**Headline (round 1, as written).** All three shot classes go to **LightGBM on the full bundle**
(`C_plus_state`), each with calibration inside the gate and with the shooter
driver tracked at slope 0.99-1.01. `FGA_rim` and `FGA_jump2` beat the best
passing non-tree arm by 33.7 and 32.4 noise floors. `FGA_3` is LightGBM under
the amended responsiveness gate of **`ARCHITECTURE_DECISIONS.md` Decision 8**,
which replaced the steps-only wording of this model's pre-registration after
that wording selected a model 92 noise floors worse on log loss whose shooter
slope ratio was 0.0086 (flat at the mean). Section 4.3 records both the
original verdict and the re-decision; `experiments.md` section 12 is the
mechanical re-application, and it flags one clause of Decision 8 that still
needs tightening.

---

## 1. Purpose

Given that a field-goal attempt happened, does it go in. This is one of the two
probabilities that turn the L3 possession-outcome model's chosen terminal event
into points (the other is the free throw, L18), and it is the sub-model that
gates G2 (points per possession) and G4 (eFG%) point at first when a scoring
number is wrong.

It is three models, not one. The pre-registration splits the target by shot
class — `FGA_rim`, `FGA_jump2`, `FGA_3` — because the base rate, the
shooter-skill signal and the defensive signal are different quantities in each,
and the data agrees emphatically: the shooter block is worth 46 noise floors on
threes and 3 floors on two-point jumpers (section 5.1). Nothing in
`cbb_sim.models.fg_make` ever fits across classes; `fit_by_class` slices the
frame and `tests/test_fg_make.py` proves that corrupting one class's rows
cannot move another class's predictions.

The engine calls it once per attempt the possession-outcome model produces, on
the class that model chose, with the shooter the L4 usage layer selected.

## 2. Target variable

- One row per field-goal attempt in the CBBD event stream
  (`cbb_sim.models.event_stream.build_stream` under the **v2** possessions
  build). Target: `made`.
- The class comes from `cbb_sim.pbp.events.classify_frame` WITH the L16
  rim-location override applied at the threshold the v2 build recorded
  (2.27 ft). This model is a large part of why that override exists: without
  it, ESPN's 2025 putback mistag moves a batch of near-rim attempts into
  `FGA_jump2` and changes two of the three class base rates in the selection
  fold's own test season.
- Population: D-I, non-truncated, **`pbp_complete`** games, seasons 2022-2025 —
  86.5% of the D-I non-truncated games, a stricter universe than the rebound
  and free-throw bake-offs used, so row counts are not comparable to theirs.
- 2,241,195 attempts; 132 (0.006%) carry no shooter id and are dropped rather
  than imputed. Class shares 36.5 / 25.6 / 38.0%, make rates 58.6 / 38.6 /
  33.8%.
- The two population rules are verified, not asserted: blocked shots are misses
  (123,222 blocked attempts; 52 of them, 0.04% of blocks, are logged MADE by
  the feed and are left exactly as the feed has them), and and-one attempts are
  makes (59,467 attempts, make rate 1.0 by construction).

## 3. Methodology at a glance

- Data window 2022-2025; season 2026 sealed (`cbb_sim.data.seal`).
- Temporal walk-forward, **no random split**. F1 trains {2022, 2023} tests
  2024; F2 trains {2022, 2023, 2024} tests 2025 and is the selection fold.
  `D_plus_lineup` has its own fold (train 2024, test 2025) because CBBD
  `onFloor` is empty at the source before 2024 (L13).
- Primary metric: attempt-level log loss on F2, per class — the engine samples
  from this probability rather than thresholding it, so the proper score that
  punishes a confident wrong probability is the right one.
- Gates: decile calibration (worst gated gap <= 2.0 pp) and responsiveness on
  BOTH drivers — the shooter's as-of class make rate and the defence's as-of
  allowed rate. Applying the matchup rule to the defence is stricter than any
  earlier L3 bake-off, and it is what decided `FGA_3`. The responsiveness rule
  is now **Decision 8**'s: slope ratio within [0.8, 1.2] AND monotone in at
  least 3 of 4 quintile steps, with the 4-of-4 requirement dropped when the
  driver's realised quintile span is below 2 pp. The pre-registration's
  steps-only wording is superseded; both readings are on record in
  `experiments.md` sections 9 and 12.
- Families: team-level logistic ridge (A only); empirical Bayes on the
  shooter's class rate combined with the defence's allowed rate on the logit
  scale, prior and both strengths fitted on the training fold; logistic ridge
  on the full bundle; LightGBM on the full bundle with its parameters searched
  on F1 only.
- Noise floor per class: a 200-replicate game-level block bootstrap SE for the
  non-tree arms and the SD over five seed-varied refits for the tree; the
  larger is used. The bootstrap dominates in all three classes (5.4e-4 to
  8.5e-4 against tree seed SDs of 0.9e-4 to 2.1e-4).

## 4. Winner

| class | winner | F2 log loss | worst decile gap | shooter steps / slope | defence steps / slope | margin over the best passing non-tree arm |
|---|---|---|---|---|---|---|
| `FGA_rim` | `lgbm` / `C_plus_state` | 0.641605 | 1.19 pp | 4/4, 1.008 | 4/4, 0.999 | 33.7 floors over `ridge` |
| `FGA_jump2` | `lgbm` / `C_plus_state` | 0.642003 | 1.58 pp | 4/4, 0.986 | 4/4, 0.828 | 32.4 floors over `ridge` |
| `FGA_3` | `lgbm` / `C_plus_state` (Decision 8; see 4.3) | 0.561085 | 1.01 pp | 4/4, 0.988 | 3/4, 0.474 on a 1.37 pp driver | it is the only passing arm; 32.1 floors over the best non-tree arm of any gate status (`ridge`) |

### 4.1 What beat what

On both two-point classes the ordering is the same and it is not close:
LightGBM < logistic ridge < empirical Bayes < team baseline, with the tree
ahead of the best linear arm by roughly 0.0185 log loss in each — about 32
noise floors, so the pre-registered "a tree must beat the best passing non-tree
arm by more than the floor" clause is satisfied by a wide margin and the
simplicity tie-break never fires. The tree's advantage is not a level
correction; every arm has the level roughly right (level shift 0.02-0.8 pp).
It is shape: the tree is the only arm whose predicted make rate tracks BOTH
driver quintile ladders at slope ~1 simultaneously.

### 4.2 L15 reproduces at the rim and on the three, and does NOT on the jumper

Adding the shooter block to the team bundle is worth 11.1 noise floors on
`FGA_rim`, **46.5** on `FGA_3`, and only 3.0 on `FGA_jump2`. The two-point
jumper is the shot where who takes it matters least — which is also the shot
whose share has been falling every season (27.0% of attempts in 2022, 23.6% in
2025). L15's "player identity dominates every player-game rate stat" is
therefore a per-shot-class statement here, not a blanket one, and the
free-throw result (L18, where the team arm fails by 6.8 pp of shape) is the
extreme end of the same axis.

### 4.3 `FGA_3`: the original verdict, and the Decision 8 re-decision

**As adopted now:** `lgbm`, F2 log loss 0.561085, calibration 1.01 pp, shooter
driver 4/4 at slope 0.988. It is the only arm that clears Decision 8's gate on
this class, so the "a tree must beat the best PASSING non-tree arm by more than
the floor" clause has no passing non-tree arm to bind against; the gap to the
best non-tree arm of any gate status (`ridge`, 0.588370 — which itself fails
calibration at 3.18 pp) is 0.027285 = 32.1 floors, so the stricter reading of
that clause selects the same arm.

**What the pre-registration's steps-only gate did, and why it was amended.** The
original rule selected `team_baseline` because it was the only arm passing a
4-of-4 monotone requirement on both drivers. The numbers behind that sentence:

| arm | F2 log loss | calibration | shooter driver | defence driver |
|---|---|---|---|---|
| `lgbm` | **0.561085** | PASS (1.01 pp) | 4/4 steps, slope 0.988 | **3/4 steps**, slope 0.474 |
| `ridge` | 0.588370 | FAIL (3.18 pp) | 4/4, slope 0.970 | 3/4, slope 0.710 |
| `eb_shrink` | 0.599327 | FAIL (2.04 pp) | 4/4, slope 1.066 | 4/4, slope 0.826 |
| `team_baseline` | 0.639094 | PASS (1.05 pp) | 4/4, **slope 0.009** | 4/4, slope 1.012 |

Two things are going on, and both are visible in the quintile ladders in
`experiments.md` section 11.1.

**The defence driver on threes carries almost no signal.** Across the five
quintiles of a defence's as-of three-point rate allowed, the ACTUAL make rate
moves from 32.95% to 34.32% — a span of 1.37 pp, against 4.98 pp for rim and
3.33 pp for the two-point jumper. That is the well-known result that a defence
controls opponent three-point percentage far less than it controls anything
else, arriving here on its own. A 4-of-4 monotone requirement on a 1.37 pp true
span is a requirement to reproduce noise: `lgbm` and `ridge` both slope the
right way overall and both dip at exactly one interior step. The arms that
survive it are the two that are (nearly) monotone FUNCTIONS of the driver by
construction — the linear team baseline and the EB arm, which puts the defence
rate straight into the logit.

**The gate as written cannot see flatness.** `team_baseline` passes the SHOOTER
responsiveness check with a predicted span of 0.31 pp against a realised span
of 35.90 pp — a slope ratio of 0.0086, or 116x too flat. It is precisely the
"flat at the mean" model `CLAUDE.md`'s matchup rule exists to reject, and the
monotone-steps reading of that rule waves it through because 0.31 pp of drift
happens to point the right way. `prob_metrics.quintile_responsiveness`
anticipates this in its own docstring — "an arm that slopes the right way but
flatter than reality still passes the SHAPE test and is caught by `slope_ratio`
instead" — but the pre-registration gates on the steps and reports the slope.

The difference between the two answers was 0.078 log loss, 92 noise floors, and
a model that cannot tell a 15% three-point shooter from a 51% one. The worker
did not amend the spec; the finding went to the PM (`experiments.md` section
11.1), and the PM amended the GATE generically in
`ARCHITECTURE_DECISIONS.md` **Decision 8**: slope ratio within [0.8, 1.2] AND
monotone in at least 3 of 4 quintile steps, with the 4-of-4 requirement dropped
when the driver's realised quintile span is below 2 pp, on every driver.

**Re-applying Decision 8 mechanically** (`experiments.md` section 12; the
decision step only, nothing retrained) changes one thing in this model and
sharpens two others:

- `FGA_3`: `team_baseline` -> `lgbm`. The flat arm now fails on its own shooter
  slope (0.0086, outside the band) instead of passing on direction alone.
- `FGA_rim`: unchanged winner, and `team_baseline` now fails responsiveness on
  the shooter slope (0.180) as well as calibration.
- `FGA_jump2`: unchanged winner, but the passing set narrows from four arms to
  two: `eb_shrink` now fails on a defence slope of 1.388 (over-steep, the other
  side of the band) and `team_baseline` on a shooter slope of 0.142.

**One clause of Decision 8 still needs tightening, and it is not rhetorical.**
Clause (b) exempts a sub-2 pp driver from the step count "the steps are then
noise"; clause (a) says the slope band "applies to every driver" and does not
say whether a sub-2 pp driver is exempt from IT. `FGA_3`'s defence driver spans
1.37 pp and `lgbm`'s slope on it is 0.474, so the two readings disagree exactly
once:

| reading | `FGA_rim` | `FGA_jump2` | `FGA_3` |
|---|---|---|---|
| `strict` — the band applies to every driver | `lgbm` | `lgbm` | **NONE passes** |
| `low_span_exempt` — a sub-2 pp driver is noise for BOTH clauses | `lgbm` | `lgbm` | `lgbm` |

Decision 8's own text states the corrected gate changes `FGA_3` to LightGBM, so
`low_span_exempt` is the reading adopted here and recorded as such. Both are
computed in `experiments.md` section 12 and in
`fg_make.decision8_verdict(..., reading=...)`, so nothing is hidden by the
choice. The PM should settle clause (a)'s scope in Decision 8 itself rather than
leaving it to a per-model reading.

## 5. Robustness checks

### 5.1 Feature blocks (F2, logistic ridge, nested bundles)

| class | A -> B (shooter block) | B -> C (state block) |
|---|---|---|
| `FGA_rim` | 0.005965 = 11.1 floors | 0.010265 = 19.0 floors |
| `FGA_jump2` | 0.001754 = 3.0 floors | 0.005728 = 9.9 floors |
| `FGA_3` | 0.039480 = 46.5 floors | 0.011244 = 13.2 floors |

Both blocks clear the floor in every class, so both are KEPT. The state block's
contribution is concentrated where you would expect: on `FGA_rim`, the team
baseline mis-states the CONTINUATION-chance (putback) make rate by 3.25 pp and
the shooter-only bundle by 3.00 pp, while the two arms that carry
`chance_number` / `chance_elapsed_s` land at 0.12 and 0.15 pp. A putback is an
easier shot and only the state block knows the attempt is one.

### 5.2 Transfers (the L15 natural experiment)

Every arm is slightly WORSE on players whose modal team changed since the prior
season than on continuing players, on all three classes, and worst of all on
players with no prior season — but the gaps are small and the EB arm's
predicted make rate tracks the realised rate to within 0.6 pp on transfers in
every class (e.g. `FGA_rim`: predicted 59.30% vs actual 59.27% on 73,708
transfer attempts). The fitted prior is the POSITION mean in all three classes
and on both folds, which is the same answer the free-throw model reached (L18)
and the same shape L15 predicted: a transferring player's prior should not lean
on his prior-season team context.

### 5.3 G4 — implied team eFG% (the gate this model is graded by)

Computed from the three class models on the test season's own shot mix: every
attempt a team actually took, weighted by its class model's predicted make
probability instead of by the outcome, so the mix is taken as given and this is
a check of the make models alone.

The adopted trio is now all-LightGBM (`experiments.md` section 12.4): implied
**50.932%** against an actual 50.874% (gap 0.058 pp), and it PASSES G4 on the
honest pregame grouping on both sides (worst tercile gap 0.71 pp on offence,
0.91 pp on defence, against a 1.0 pp tolerance). On the ORACLE grouping
(terciles of realised eFG%, reported because it is the grouping that exposes a
flat model) it passes on offence at 0.74 pp and misses defence at 1.03 pp.
Team-level implied-vs-actual correlation 0.77 (offence), 0.76 (defence).

The superseded mixed trio (`lgbm`, `lgbm`, `team_baseline`, `experiments.md`
section 10) had a tighter league LEVEL — implied 50.876% against 50.874%, gap
0.002 pp — and a worse SHAPE: it missed the oracle grouping on both sides
(1.21 pp offence, 1.09 pp defence), under-predicting the top tercile and
over-predicting the bottom. That is the too-narrow-spread signature, and it was
sourced in exactly the `FGA_3` decision Decision 8 reversed: swapping the flat
three-point model out closes 0.46 pp of the offence oracle gap. What remains
(defence oracle 1.03 pp) is not from that arm.

### 5.4 Is the defence lineup-level or team-level? — TEAM-LEVEL

On its own fold (train 2024, test 2025), on the 1,202,616 attempts (97.3%) that
carry all ten on-floor ids, with the defender-rate shrinkage FITTED on the
train season over [0, 100, 250, 500, 1000] pseudo-attempts (landing on 100):
adding the defensive five's as-of rim-protection and perimeter rates to the
full bundle is worth **0.30, -0.01 and 0.03 floors** on the ridge arm for rim,
jumper and three respectively, and is NEGATIVE on the tree arm in all three
classes (-0.16, -0.10, -0.82). Not a straddle like the rebound model's — a
clean nothing. The defence enters shot-making as a TEAM quantity; per-defender
rates belong to the player-attribution layer, not to this model.

### 5.5 Team form from all chances vs first chances only

The rebound model restricts its team rates to first-chance opportunities so a
continuation-chance labelling defect cannot reach a first-chance model's
predictors (ledger row B5). Here the target population is all attempts and the
L16 defect is repaired at the source by the v2 override, so the default is all
chances — and the cost of that choice is measured, not argued: rebuilding every
team rate from first chances only moves F2 ridge log loss by -0.00001 to
+0.00002, i.e. 0.02 floors, in all three classes, with the two sources
correlating 0.949-0.958. The restriction neither helps nor hurts here.

## 6. Decisions log

1. **Three models, not one with a class dummy.** Pre-registered, and the data
   backs it: the shooter block is worth 46.5 floors on threes and 3.0 on
   two-point jumpers, so a shared shooter coefficient would be wrong for at
   least one class.
2. **Shooter identity keyed on the CBBD player id**, ESPN id attached where the
   crosswalk resolves (0% in 2022-2023, 100% in 2024-2025). Same decision and
   the same measured reason as the free-throw model (ledger section B).
3. **`minutes-to-date` could not be built** and is replaced by
   `shooter_games_asof` + `shooter_fga_asof`, with the coverage that forced it
   reported per season. A column that is real in two seasons and structurally
   zero in two others is a season dummy wearing a minutes label.
4. **The assisted flag is not built at all**, and `blocked` / `and_one` are
   banned by name in `design_matrix`. All three are post-outcome in this feed.
5. **The chance-state block is at-release, not post-outcome**, and is
   deliberately NOT the `is_transition` the ledger bans at L5: that one is a
   function of when the chance ENDS, this one of when it STARTED plus the
   attempt's own clock. `features.md` section 4 carries the argument and the
   test that pins it.
6. **Team form from all chances** (5.5), measured against the first-chance-only
   alternative rather than assumed.
7. **The defence's EB shrinkage strength is fitted too**, not just the
   shooter's — a defence's as-of allowed rate rests on a handful of attempts in
   November and thousands in March, and the logit combination is unstable
   otherwise. Fitted values: m_def = 300 (rim, jumper), 1000 (three).
8. **LightGBM parameters searched on F1 only**, six rungs, and all three classes
   independently chose the same rung (31 leaves, min_child_samples 200). The
   winner is frozen before F2 is touched and is reused for the seed refits and
   the lineup fold.
9. **Centred rates are branched, not subtracted** (`where(rate exists,
   rate - league, 0.0)`). This is the one real bug this build hit: the
   subtraction form leaves a ~1e-8 float32 residue in a structurally constant
   column, a standardiser divides by it, and the linear arms diverge to a log
   loss of 10. Both ends are fixed and `RidgeArm` now uses a relative
   constant-column threshold.

## 7. Consumption from the sim

**Round 2b changed this.** The shipping model is a SCHEDULE, not three
objects: `data/processed/models/fg_make/round2b/S_C_s1/manifest_<class>.json`
plus one joblib per (class, refit date), loaded through
`cbb_sim.engine.manifest.ArtifactManifest` and selected per game by tipoff.
`cbb_sim.engine.adapters.FgMakeAdapter` does this when
`ENGINE_FG_MAKE=round2b_S_C_s1`; `ENGINE_FG_MAKE=round2_S_C` serves the static
round-2 winner, and the default `winner` serves the round-1 artifacts below.
The round-1 consumption pattern, unchanged and still valid for those artifacts:

```python
import joblib
from cbb_sim.models import fg_make as FG

# one artifact per shot class; the engine loads all three once at startup
models = {c: joblib.load(f"data/processed/models/fg_make/winner_{c}.joblib")
          for c in FG.SHOT_CLASSES}

fitted = models["FGA_rim"]          # FittedFgMake
p = FG.predict_arm(fitted.arm, fitted.models["FGA_rim"], rows)   # (n, 2)
p_make = p[:, FG.CLASS_INDEX["MAKE"]]
```

- `rows` is a frame with the columns of `FG.feature_set(fitted.feature_set)`,
  in any order (`design_matrix` selects and orders them) plus, for the
  `eb_shrink` arm, the raw columns its algebra reads. Feature provenance and
  fallbacks: `features.md`.
- Every missing as-of rate is **exactly 0.0** on the centred scale (the league
  mean), never a fabricated level. Do not impute anything else.
- Output is `(n, 2)` in the order `("MISS", "MAKE")`, `FG.CLASSES`.
- The class must be the one the possession-outcome model chose, under the SAME
  possessions version (v2). Scoring a v1-labelled attempt with a v2-trained rim
  model is a mismatch: `fitted.meta["possessions_version"]` records which.
- The sim must not call this per attempt in the hot loop; per `CLAUDE.md` it
  reads a lookup table built from these models at game setup.

## 8. Artifacts

| Path | What it is |
|---|---|
| `data/processed/models/fg_make/events_v2.parquet` | One row per field-goal attempt with its class, outcome, state, chance derivation and on-floor ids (cached; rebuilt with `--rebuild-events`) |
| `data/processed/models/fg_make/grid_results.csv` | Every (class, fold, arm) row of the bake-off |
| `data/processed/models/fg_make/run_report.json` | Everything the `experiments.md` section renders from, including the full EB grids, every decile calibration table and every responsiveness ladder |
| `data/processed/models/fg_make/lgbm_ladder_v2.json` | The F1-only parameter search and the frozen per-class parameters |
| `data/processed/models/fg_make/winner_{class}.joblib` | The adopted arm per class, refit on F2 train. All three are `lgbm` after the Decision 8 re-decision; `winner_FGA_3.joblib` carries `meta["gate"] = "ARCHITECTURE_DECISIONS.md Decision 8"` and `meta["supersedes_arm"] = "team_baseline"` |
| `data/processed/models/fg_make/reference_superseded_FGA_3.joblib` | The `team_baseline` arm the steps-only gate had selected for threes, kept for reference. NOT for use by the engine |
| `data/processed/models/fg_make/run_report_partial.json` | Checkpoint written before the lineup block |
| `data/processed/models/fg_make/state_confound.json` | ROUND 2 step 1: the leak proof and the confound decomposition (`scripts/diag_fg_make_state_confound.py`) |
| `data/processed/models/fg_make/round2/grid_results.csv`, `run_report.json`, `closed_loop.json` | ROUND 2: every (class, fold, arm) row, the full detail, and the Decision-10 closed-loop table |
| `data/processed/models/fg_make/round2/<arm>/fg_make_<class>_F2.joblib` | ROUND 2: one fitted LightGBM per (arm, class). `S_C`'s three carry `adopted=True` |
| `data/processed/models/fg_make/round2b/S_C_s1/<class>_<refit_date>.joblib` + `manifest_<class>.json` | ROUND 2b: the S1 schedule, 18 artifacts and three `cbb_sim.engine.manifest`-format manifests, all `adopted=True`. **This is the shipping model** |
| `results/engine_v0/fgmake_r2_<arm>/` | The six paired closed-loop engine runs |
| `data/processed/models/fg_make/round3_shooter/S_C_s1/` | ROUND 3: the corrected-label S-C-S1 schedule. **This is the SERVED model** (`ENGINE_FG_MAKE=round3_shooter_S_C_s1`, the engine default) |
| `data/processed/models/fg_make/round4/{m_fitted,leak_test,run_report,closed_loop,closed_loop_interim}.json` | ROUND 4: the fold-1 `m` grid, the change-form leak table, all six arms, and both closed-loop tables |
| `data/processed/models/fg_make/round4/<arm>/` | ROUND 4: one S1 schedule per arm (18 joblibs + three manifests). **`B1/` is the round-4 WINNER**, not yet the default (section 20.7) |
| `data/processed/models/fg_make/round4/run_report_v1_leaked_assisted.json`, `round4/B{3,4}_leaked_assisted/` | ROUND 4: the pre-repair reading of `sh_assisted_share`, kept on record and INELIGIBLE (section 20.2) |
| `data/processed/models/fg_make/shooter_skill_v1.json` | ROUND 4 step 2: the between-shooter skill evidence (`docs/tests/fg_make_shooter_skill_2026-09-10.md`) |
| `data/processed/models/engine_fgm4/` | The engine inputs with the fg_make shooter slot block keyed on `shot_shooter_id` plus the round-4 columns. Gitignored, rebuildable with `scripts/build_engine_inputs_shotshooter.py` |
| `results/engine_v0/fgm4_cl_*/` | The five paired round-3/round-4 closed-loop runs (engine code pinned to commit `6431772`) |

## 9. Known gaps / followups

0. **ROUND 2 / 2b SUPERSEDE ITEMS 1-7's CONTEXT.** Items 1-7 were written about
   the round-1 model. What changed: item 7 ("nothing here has been through a
   paired-seed sim run") is DONE -- six paired engine runs, `experiments.md`
   sections 14.5, 14.6 and 16.3 -- and it found the defect that item 0 of this
   list now names. Items 2-6 are unchanged and still open.

8. **The shooter is keyed on the WRONG participant column -- ROUND 3 RAN
   (2026-09-10); THE SERVED MODEL STILL CARRIES THE DEFECT.**
   `fg_make.build_fg_events`/`_season_events` gained a `shooter_key` parameter
   this round (defaulting to `participant_1_id`, byte-identical to every prior
   artifact) and round 3 refit the round-2b winner (S-C under S1) on
   `shot_shooter_id` (`experiments.md` section 17-18,
   `scripts/train_fg_make_v3_shooter.py`,
   `docs/tests/fg_make_shooter_key_2026-09-10.md`). **Result: NOT ADOPTED on
   any class**, per the pre-registered rule -- F2 log loss is worse than the
   round-2b reference by 51.6 / 46.5 / 8,182.6 noise floors (rim / jumper /
   three) and Decision 8 flips PASS -> FAIL on the jumper and the three. This
   is not a defect in the fix: the `shooter_make_c` Decision-8 quintile span
   collapses from 16.1/8.1/35.9 pp (reference) to 10.2/4.3/2.8 pp (corrected)
   -- a **96% collapse on `FGA_3`** -- because a 35.9 pp shooter-skill spread
   on three-point makes was never plausible on its own; it is the size the
   mislabelling produces when ~half of assisted makes are credited to the
   team's primary ball-handler, whose contaminated history then reads as a
   near-perfect proxy for team shot quality. **CLOSED BY ROUND 4
   (sections 19-20).** L29 overrode round 3's mechanical "not adopted" on
   data-integrity precedence and made the corrected arm the interim served
   model; round 4 added the `adapters.py` branch, ran the interim model's
   missing Decision-10 closed loop (PASS, section 20.0), and re-baked the
   shooter block on the correct label. Winner: **B1**, a single shrunk column.
   Nothing under `round2b/` was touched.

9. **G4's defence tercile regressed under the honest arms** (1.066 pp for S-C
   against a 1.0 pp tolerance, where the round-1 leaked trio read 0.914 pp).
   Every honest arm misses it, including the one with no state at all
   (S-B, 1.021 pp), so it is not the state parametrisation's doing; the leaked
   arm's extra (spurious) discriminating power on the test rows was flattering
   it. `experiments.md` section 14.7.

10. **The end-game effect this model declined to carry belongs upstream.** In
    the last two minutes a team trailing by 4-9 takes 52.0% of its attempts
    from three at 8.4 s on the ball, against 27.0% at 19.0 s for a team leading
    by 4-9. That is a SHOT-MIX and CLOCK effect, owned by `possession_outcome`
    and the clock model, and neither currently carries an end-game state term
    of that shape. `docs/tests/fg_make_state_confound_2026-09-10.md` section 3.

1. **The `FGA_3` gate question is RESOLVED** (Decision 8, 2026-09-10): the
   responsiveness gate gained the slope-ratio clause and all three classes take
   `lgbm`. What is still open is the SCOPE of clause (a) — whether a driver
   with a sub-2 pp realised span is exempt from the slope band as well as from
   the step count (4.3). The two readings disagree only on `FGA_3`, where the
   strict one adopts nothing at all; `low_span_exempt` is adopted here because
   Decision 8's own text states the outcome it expects. The PM should write that
   scope into Decision 8.
2. **`minutes-to-date`** is the obvious first addition once the L4 player layer
   supplies minutes in CBBD id space (6.3 above). It needs its own
   pre-registration.
3. **Season drift is carried by `season_idx` alone** and, as at L3 round 1 and
   L5, a linear season index is not the season-progress structure L11 asks for.
   The 2025 test season's three-point share jumps to 39.1% from 37.3%; the
   models' level is right on F2, but the mechanism is unpinned.
4. **The 52 attempts logged as blocked AND made** are a feed contradiction left
   as-is. They are 0.04% of blocks and nothing depends on them, but a future
   event-layer pass should decide whether the block row or the make flag is
   wrong.
5. **Shooter identity has no continuity model.** A transferring player's rate
   is shrunk toward his position, which is the fitted answer, but there is no
   explicit continuity weighting of the kind L15 item (4) describes. The
   transfer subset shows the current treatment is not obviously broken (5.2),
   not that it is optimal.
6. **`D_plus_lineup` was answered on two seasons** (L13). A clean nothing on
   both folds is a stronger result than the rebound model's straddle, but it is
   still two seasons of `onFloor` data.
7. **Nothing here has been through a paired-seed sim run.** Per `CLAUDE.md` an
   offline winner ships only after the gates are re-read with it wired in; G2
   and G4 are the ones to watch.
