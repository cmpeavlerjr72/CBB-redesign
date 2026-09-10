# L3 FIELD-GOAL MAKE: model

Bake-off run 2026-09-10. Full grid, every number, and the pre-registration it
executes: `docs/models/fg_make/experiments.md`. Feature provenance:
`docs/models/fg_make/features.md`. Code: `src/cbb_sim/models/fg_make.py`,
`scripts/train_fg_make_v1.py`.

**Headline.** Two of the three shot classes have a clean winner and one does
not, and the one that does not is a finding about the GATE rather than about
the models. `FGA_rim` and `FGA_jump2` go to LightGBM on the full bundle, each
beating the best passing non-tree arm by more than thirty noise floors with
calibration inside the gate and both responsiveness drivers monotone in 4 of 4
steps. `FGA_3` is adopted as the **team-level baseline** because it is the only
arm that passes both gates — while being 92 noise floors WORSE on log loss than
the arm that fails, and while carrying a shooter slope ratio of 0.0086, i.e.
essentially flat in the one dimension that matters most. Section 4.3 says
exactly what happened and what the PM has to decide.

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
- Gates: decile calibration (worst gated gap <= 2.0 pp) and BOTH responsiveness
  drivers monotone in 4 of 4 quintile steps — the shooter's as-of class make
  rate and the defence's as-of allowed rate. Applying the matchup rule to the
  defence is stricter than any earlier L3 bake-off, and it is what decided
  `FGA_3`.
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

| class | winner | F2 log loss | worst decile gap | shooter slope | defence slope | margin over the best passing non-tree arm |
|---|---|---|---|---|---|---|
| `FGA_rim` | `lgbm` / `C_plus_state` | 0.641605 | 1.19 pp | 1.008 | 0.999 | 33.7 floors over `ridge` |
| `FGA_jump2` | `lgbm` / `C_plus_state` | 0.642003 | 1.58 pp | 0.986 | 0.828 | 32.4 floors over `ridge` |
| `FGA_3` | `team_baseline` / `A_team` (by the rule; see 4.3) | 0.639094 | 1.05 pp | **0.009** | 1.012 | it is the only passing arm |

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

### 4.3 `FGA_3`: the rule's answer, and why it is perverse

The pre-registered rule selects `team_baseline` for threes because it is the
only arm that passes both gates. The numbers behind that sentence:

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

So the honest statement is: **by the pre-registered rule, `FGA_3` adopts
`team_baseline`.** Under any reading that also gates the slope ratio (say,
0.5-1.5 on both drivers), `FGA_3` adopts `lgbm` and all three classes agree.
The difference between the two answers is 0.078 log loss, 92 noise floors, and
a model that cannot tell a 15% three-point shooter from a 51% one. This is not
a decision a worker should take by amending the spec after seeing the numbers;
it is written up here and in `experiments.md` section 11.1 for the PM. Nothing was
tuned, dropped or re-run to produce either answer.

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

The adopted trio (`lgbm`, `lgbm`, `team_baseline`) lands the league level
essentially exactly — implied 50.876% against an actual 50.874% — and passes
G4 on the honest pregame grouping on both sides (worst tercile gap 0.98 pp on
offence, 0.87 pp on defence, against a 1.0 pp tolerance). On the ORACLE
grouping (terciles of realised eFG%, reported because it is the grouping that
exposes a flat model) it misses at 1.21 pp on offence and 1.09 pp on defence,
under-predicting the top tercile and over-predicting the bottom — the classic
too-narrow spread, and the `FGA_3` decision above is exactly where it comes
from. An all-`lgbm` trio passes the oracle grouping on offence too (0.74 pp)
and misses defence by 1.03 pp. Team-level correlation between implied and
actual eFG% is 0.82 for the adopted trio.

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
| `data/processed/models/fg_make/winner_{class}.joblib` | The adopted arm per class, refit on F2 train |
| `data/processed/models/fg_make/run_report_partial.json` | Checkpoint written before the lineup block |

## 9. Known gaps / followups

1. **The `FGA_3` decision is with the PM** (4.3). Either the responsiveness gate
   gains a slope-ratio clause — in which case all three classes take `lgbm` —
   or the 4-of-4 monotone rule stands on a driver whose true span is 1.37 pp and
   threes ship a team-level model. It should not be resolved by a worker after
   the fact.
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
