# L4 SHOT ALLOCATION (usage)

Pre-registration and the full grid: [`experiments.md`](experiments.md). Feature
provenance: [`features.md`](features.md). Code:
`src/cbb_sim/models/usage.py`. Trainers: `scripts/train_usage_v1.py` (round 1)
`scripts/train_usage_v2.py` (round 2, the shooter-label data fix) and
`scripts/train_usage_v2b.py` (round 2b, the S1 training-scheme confirmation).
Audit of the label defect: `scripts/diag_shooter_key_v1.py`.
Tests: `tests/test_usage.py`.

> **Status: ROUND 2 RUN 2026-09-10 (shooter-label data fix). WINNER UNCHANGED ON
> ALL FIVE CLASSES.** Round 1 keyed the field-goal shooter on
> `participant_1_id`, which is the ASSISTER on 48.98% of assisted makes;
> round 2 re-ran the identical pre-registration with the shooter keyed on
> `shot_shooter_id`. `lgbm` wins all five classes on the selection fold in both
> rounds (margins 16.9 -> 11.0, 2.1 -> 1.9, 1.2 -> 1.5, 3.8 -> 3.9 floors, and
> `FT_trip` the only eligible arm in both). What DID change is the eligibility
> verdict on `FGA_rim`: cleaning the label moves U1's top-3 usage-share gap from
> -2.17 pp to -1.79 pp and both simple arms become eligible -- credit handed back
> from the passer to the finisher, which is R3's named defect shrinking without
> anyone touching a share vector. `TOV` still reverses on the robustness fold and
> stays UNCONFIRMED. `experiments.md` section 8; audit
> `docs/tests/shooter_key_audit_2026-09-10.md`.
>
> **Round 1's finding stands otherwise:** CFB's "too narrow / too short" pair
> does NOT reproduce in CBB, and top-k usage share -- failing in the opposite
> direction, still negative in 49 of 50 F1 cells -- is what decides eligibility.
> Sections 4 and 5 describe round 1; section 8 of `experiments.md` is round 2.
>
> **Round 2b (S1 training scheme), run 2026-09-10: S1 ADOPTED on all five
> classes.** Monthly in-season walk-forward refit beats the static fit on log
> loss by 0.44-0.82 floors on every class and improves the worst calibration
> gap on four of five, with no gate lost anywhere. Unlike L21's possession-
> outcome result this is NOT a calibration rescue -- the static allocator was
> already at 0.24-0.54 pp against a 2.0 pp gate -- so S1 is adopted because it
> is the standing default and costs nothing here, not because this model needed
> it. 30 monthly artifacts in `data/processed/models/usage_s1/` with a manifest
> the engine selects by game month. `experiments.md` section 9.

---

## 1. Purpose

Given the five offensive players on the floor, the class of event that just
happened, and the game state, decide **which of the five is credited with it**.
This is the allocation half of the L4 player layer: the rotation model decides
who is on the floor, the L3 possession-outcome model decides what kind of event
a chance ends in, and this model turns that team-level event into a player-level
one. It feeds every player prop and it is the model G8 is measuring when it asks
for the share of team FGA held by the top-1 and top-3 players.

It is deliberately the smallest possible unit of that job. It does not decide
how many events there are (L3), it does not decide who is on the floor
(rotation), and it does not decide whether a shot goes in (fg_make / free
throw). The pre-registration's game-level checks re-allocate the **actual**
event sequence over the **actual** on-floor fives precisely so that a failure
here cannot be confused with a failure in either neighbour.

## 2. Target variable

One row per credited event, restricted to events whose OFFENSIVE five is fully
resolved. Five classes, each modelled as a separate choice among the five:

| class | credited player | source |
|---|---|---|
| `FGA_rim` | the shooter | `event_stream.cls == "FGA_rim"`, **`shot_shooter_id`** (round 2 data fix; round 1 used `participant_1_id`, which is the ASSISTER on 48.98% of assisted makes -- see below) |
| `FGA_jump2` | the shooter | same |
| `FGA_3` | the shooter | same |
| `TOV` | the player charged with the turnover | `participant_1_id` (no `shot_shooter_id` exists on a turnover row; unaffected by the round-2 fix) |
| `FT_trip` | the FOULED SHOOTER | first attempt of a foul-caused trip (`trip_pos == 1`, `trip_cause == "foul"`) |

**THE SHOOTER KEY (round-2 data fix, 2026-09-10).** CBBD's `participants` array
is not ordered shooter-first. On an assisted made field goal it holds the shooter
and the assister and the order is a coin flip: over every 2022-2025 row of this
universe, `participant_1_id == shot_shooter_id` on 51.02% of assisted makes, and
on the 48.98% that disagree `participant_1_id` is the assister on 100.000% of
rows. Inside this model's window that mislabels 11.9% / 5.0% / 14.1% of
`FGA_rim` / `FGA_jump2` / `FGA_3` rows in 2024 (12.0% / 4.9% / 14.1% in 2025),
at a rate that varies ~3x between classes and 5.3-22.3% between teams. Round 1's
`in_five` coverage filter could not catch it because the assister is a teammate
on the floor. The three FGA classes therefore key on `shot_shooter_id`
(`build_usage_events(shooter_key=...)`, default `"shot_shooter_id"`); rows with
no shooter id (0.04-0.27%) are dropped, never imputed and never fallen back to
`participant_1_id`. `TOV` and `FT_trip` are untouched and were never affected:
FTA rows agree with `shot_shooter_id` on 100.000% and TOV rows carry no
`shot_shooter_id` at all. Evidence:
[`docs/tests/shooter_key_audit_2026-09-10.md`](../../tests/shooter_key_audit_2026-09-10.md);
re-run: `experiments.md` section 8.

Technical free throws are excluded, for the same reason `free_throw` excludes
them: the shooter is chosen by the coach rather than by who was fouled, so the
two populations come from different shooter distributions and pooling them
would bias both.

Population filter: D-I, non-truncated, `pbp_complete` games, seasons 2024 and
2025 only. On-floor ids are empty at the source in 2022-2023 (L13), so there is
no earlier fold to build. Coverage per class and season -- how many events have a
resolved five, how many have a credited id, and how many are therefore modelled
-- is reported in `experiments.md` section 2.1 rather than filtered silently.

The five are sorted ascending by CBBD player id, so the alternative order is a
function of the lineup and never of the feed's column order, and `y` is the slot
index of the credited player. The baseline any arm has to beat is the uniform
over five: **log loss 1.609438**.

## 3. Methodology at a glance

- **Data window.** 2024 and 2025 (L13). 2026 is sealed; `fold_slices` and
  `walkforward_slices` both call `seal.assert_not_sealed`.
- **Split.** Temporal walk-forward, never random. F1 trains 2024 and tests 2025
  and is the selection fold. The robustness fold is a within-2025 walk-forward:
  train before 2025-01-15, test after.
- **Primary metric.** Per-class log loss of the credited player among the five.
  Eligibility is gated on share calibration by as-of-rate decile (<= 2 pp),
  responsiveness by as-of-rate quintile (monotone 4 of 4), and the game-level
  dispersion trio: the CFB "too narrow" SD ratio (0.9-1.1), the "too short"
  players-with-at-least-one-event count (+/- 0.5), and top-1/top-3 usage share
  (+/- 2 pp).
- **Families tested.** Five arms: a shrunk proportional allocator, a
  single-level Dirichlet, the CFB two-level hierarchical Dirichlet, a
  conditional logit, and a LightGBM scorer over the five. Nothing is chosen for
  familiarity: the shrinkage prior and strength, the Dirichlet concentrations,
  the ridge penalty and the tree parameters are all fitted on training data and
  the winner falls out of `train_usage_v1.decide`.

## 4. Winner

**LightGBM over the five, on all five event classes, on the selection fold.**
Two of the five replicate on the robustness fold, two straddle the noise floor,
and one (`TOV`) reverses outright and is recorded UNCONFIRMED. Full tables:
`experiments.md` sections 3, 4 and 6.

| class | winner | log loss | best other eligible arm | margin | floors | confidence |
|---|---|---:|---|---:|---:|---|
| FGA_rim | lgbm | 1.502348 | cond_logit 1.521954 | 0.019606 | 16.9 | replicated (8.4 floors on the robustness fold) |
| FGA_jump2 | lgbm | 1.491034 | cond_logit 1.494403 | 0.003369 | 2.1 | straddles (cond_logit leads by 0.001188 on the robustness fold, inside its floor) |
| FGA_3 | lgbm | 1.458927 | cond_logit 1.460492 | 0.001565 | 1.2 | straddles (the tree's 0.000615 lead is inside the robustness fold's floor) |
| TOV | lgbm | 1.577664 | cond_logit 1.580645 | 0.002981 | 3.8 | **UNCONFIRMED** -- 7.0 floors BEHIND `proportional` on the robustness fold |
| FT_trip | lgbm | 1.522266 | none -- the only eligible arm | -- | -- | replicated (4.2 floors on the robustness fold) |

The uniform-over-five baseline is 1.609438 on every class, so the winners run 2%
(`TOV`) to 9% (`FGA_3`) better than a coin flip among five, with top-1 accuracy
0.27-0.34 against a 0.20 baseline. That range is what a five-way choice with a
realistic share vector permits; L15's 60-89% residual on every player-game rate
stat predicted it.

**What actually decided it was eligibility, not log loss.** The proportional
allocator and the single-level Dirichlet are INELIGIBLE on `FGA_rim`,
`FGA_jump2` and `FT_trip` because they under-concentrate the top of the usage
distribution: the simulated top-3 share per team-game comes in 2.17, 2.20 and
2.95 pp below real against a +/- 2 pp gate. On `FT_trip` every arm but the tree
fails that gate. On the two classes where the top-share gate is slack (`FGA_3`,
`TOV`) all five arms are eligible and the margins collapse to 1.2-3.8 floors.

**Fitted parameters.** Shrinkage prior and strength, in pseudo on-floor events:
`FGA_rim` position/50, `FGA_jump2` league/50, `FGA_3` position/25, `TOV`
league/200, `FT_trip` position/200. The `prior_season` rung is unidentified on
F1 by construction (L13) and is reported as such; on the robustness fold, where
it IS available, it never wins -- the same transfer-attenuation shape `free_throw`
found independently. Tree parameters `num_leaves=15, learning_rate=0.08,
n_estimators=300, min_child_samples=200`, searched on 2024 only, the same rung on
every class; seed-varied refits move the log loss by 1.2e-05 to 1.5e-04.

**The two Dirichlet arms are not adopted anywhere, and the reason is the finding
below.** Their fitted concentrations came back at "no dispersion" on every class
(U2 on every knob, U3 on the between-role level), so U2 is identical to U1 to six
decimals and U3 costs log loss on four of five classes.

## 5. Robustness

### 5.1 The CFB "too narrow / too short" pair does not reproduce -- a third check does the work

This is the substantive result of the bake-off. CFB's fixed share table measured
per-player per-game count SD at 0.54-0.89x real and carried 5.2 rushers / 8.1
receivers with any share against 11.1 / 15.7 real; the hierarchical Dirichlet and
`extend_profile` exist to fix those two. Re-allocating the ACTUAL 2025 event
sequence over the ACTUAL on-floor fives, the plain proportional allocator lands
at:

| class | SD ratio (gate 0.9-1.1) | players with >=1 event, sim / real (gate +/- 0.5) | top-1 gap | top-3 gap (gate +/- 2 pp) |
|---|---:|---|---:|---:|
| FGA_rim | 1.0108 | 7.090 / 6.825 (+0.265) | -1.54 pp | **-2.17 pp** |
| FGA_jump2 | 1.0382 | 5.764 / 5.528 (+0.236) | -1.81 pp | **-2.20 pp** |
| FGA_3 | 1.0823 | 6.820 / 6.689 (+0.131) | -0.24 pp | -0.60 pp |
| TOV | 1.0473 | 5.856 / 5.825 (+0.031) | -0.33 pp | -0.39 pp |
| FT_trip | 1.0183 | 5.372 / 5.092 (+0.280) | -2.29 pp | **-2.95 pp** |

Every SD ratio passes and every one errs WIDE; every player count passes and
every one errs LONG. The mechanism is structural: CFB drew from one share vector
per team-game, so its only variance was multinomial noise around a fixed mean,
while here the allocation is conditioned on a five that changes between games and
within a game. Lineup variation already supplies the dispersion CFB had to
inject. The corollary is that if a future rotation model under-disperses the
lineups themselves, CFB's defect WOULD appear here -- which is why the
hierarchical arm stays implemented and tested rather than deleted.

The gate that does bite is the top-k share, and it fails in the OPPOSITE
direction from CFB's top-1 inflation: every arm under-concentrates, on every
class, on both statistics, in all 50 F1 cells (5 classes x 5 arms x 2 statistics). That is one named defect, left open
in the change ledger, not tuned away.

### 5.2 Fold agreement

`FGA_rim` and `FT_trip` replicate at 8.4 and 4.2 floors. `FGA_jump2` and `FGA_3`
straddle: the tree's lead is inside the robustness fold's floor and the rule
falls through to `cond_logit` and `proportional` respectively. `TOV` reverses --
3.8 floors ahead on F1, 7.0 floors behind `proportional` on the within-2025 fold,
where the tree is also the only arm whose calibration approaches the gate (1.54
pp) and the only one whose player count comes in LOW (-0.115). The two folds are
not independent (the second tests half of the season the first tests whole), so
the reversal is a warning rather than a refutation, and `TOV` is recorded
UNCONFIRMED.

### 5.3 Segment checks

- **Transfers.** The credited player is a transfer on 34,438-74,068 F1 events per
  class. Transfers score BETTER than continuing players on three classes and
  marginally worse on two -- which is a composition effect, not a transfer
  finding: no F1 arm uses the prior-season block at all, and transfers in this
  sample are higher-usage players whose within-lineup share is further from
  uniform. The real signal is that players with NO prior season are 0.03-0.07
  nats harder on every class, 20-50x the floor.
- **Roles.** The per-role SD ratio for the proportional arm runs 0.948
  (`FT_trip` handlers) to 1.175 (`FGA_3` bigs) -- inside the pooled band, with
  bigs and wings consistently the wide end and handlers the narrow end.
- **Calibration and responsiveness.** Every arm on every class passes the decile
  share-calibration gate except the mis-specified tree noted in section 6 item 6,
  and every arm is monotone 4 of 4 by as-of-rate quintile on every class and both
  folds. Neither gate discriminated between arms; the game-level checks did.

### 5.4 Coverage

95.0-98.4% of credited events are modelled. The loss is 0.9-4.4% to an unresolved
on-floor five (L13) plus, on turnovers only, 5.1-5.3% with no `participant_1_id`
-- CBBD leaves the charged player blank on team turnovers. Roster position
resolves on 99.98% of player-games and hoopR minutes join on 99.81%, so neither
the position prior nor `minutes_asof` rests on a large unmeasured fallback.

Round 2 adds one further, small loss: field-goal rows with no `shot_shooter_id`
are dropped (0.04-0.27% per class; 1,136 rows over the two seasons; per-team
median 0.00-0.18%, worst team 5.6%). Total modelled events 1,633,164 against
round 1's 1,634,792, a 0.10% reduction. Nothing is imputed and nothing falls back
to `participant_1_id`. `experiments.md` section 8.5.

## 6. Decisions log

1. **The event stream, not the chance table, is the source.** The chance table
   carries no `participant` id, and the possession table carries the on-floor
   five only once per possession (from its first event), so a substitution
   inside a possession would attribute the event to the wrong five. The event
   stream carries both on the event's own row. Same reason `rebound` and
   `free_throw` read the stream.
2. **Players are keyed on the CBBD id**, as in `free_throw`. The ESPN id is
   attached through the crosswalk where it resolves, for the prop-grading join;
   it is not the model key.
3. **Exposure is per on-floor event, not per game.** The pre-registration's
   "per-possession-on-floor rates, not per game" makes a bench player's rate
   comparable to a starter's, and it is the unit the fitted shrinkage strength
   is expressed in (pseudo on-floor events), so `m` reads directly as "how much
   history before a player's own rate outweighs the prior".
4. **U2 is U3 with one role, not U3 with the roles kept and the between level
   off.** A single flat `Dirichlet(a * w)` is the design the pre-registration
   names; keeping the role vector would pin a one-man role (a lineup's only
   centre) to zero variance, which is a per-role behaviour and not a
   single-level one. `ProfileSet.single_role` makes the two arms share one
   implementation and differ only in the role vector.
   `tests/test_usage.py::test_a_single_level_matches_the_analytic_dirichlet_sd`
   pins U2 to the closed-form Dirichlet SD.
5. **The Dirichlet is drawn once per (game, team, class) over the TEAM's
   profile, not per event over the lineup.** Usage persistence across
   substitutions is the whole point; a per-event draw would add noise without
   adding any game-to-game usage variance, which is the defect CFB measured as
   "too narrow".
6. **U5 is fitted with a grouped-softmax objective, not a binary one.** The
   pre-registration calls it a "ranker". A lambdarank score has an arbitrary
   scale, so softmaxing one would be miscalibrated by construction; and a plain
   binary objective softmaxed afterwards optimises a different likelihood and
   came out systematically over-sharpened -- measured on F1 `TOV`, top decile
   30.13% predicted against 27.89% real, worst gap 2.24 pp, which FAILED the
   calibration gate on an arm whose log loss looked fine. `_group_softmax_objective`
   optimises exactly the graded conditional likelihood (`grad = p - y`,
   `hess = p (1 - p)` within each group of five) and the same arm then reads
   0.27 pp. That is an objective fix, not a temperature fitted on the model's own
   output -- the latter is the shape `docs/SIM_GUARDRAILS.md` section 5 bans.
7. **U5 is offset by `log q`** (`lgbm_init_score`), so at zero trees it IS U1
   and what the tree fits is a multiplicative correction to the as-of share.
   Without the offset the comparison would be "can a tree relearn a
   normalisation", which is not the pre-registered question.
8. **Row bagging is off in U5.** A row is one alternative of a choice set, so
   `subsample` would split choice sets across the in-bag boundary. Column
   sampling is unaffected and is kept.
9. **State features enter U4 only as interactions.** A conditional logit
   differences out anything constant across the alternatives, so `score_diff`,
   `sec_remaining` and `chance_number` as plain columns are not weak but
   *unidentified*. They are kept, interacted with the alternative's own share
   and role. The raw columns go to U5 unchanged, where a tree can interact them
   itself.
10. **The prior-season block is dropped on F1 and recorded.** Not because it is
    constant -- `build_usage_design` falls it back to the position prior, so it
    varies -- but because its MEANING changes between a fold with no prior
    season and one with. Left in, the F1 conditional logit on `FGA_3` scored
    3.74 against U1's 1.46. `unidentified_features` catches it; the drop is in
    the results table; the within-2025 fold is where the block is actually
    measured. Same treatment `cbb_sim.models.clock` gives its degenerate
    chance-number column.
11. **The Dirichlet concentrations are fitted against dispersion, not
    likelihood.** A mean-preserving Dirichlet is centred on U1's shares, so the
    per-event likelihood is nearly flat in the concentration; the dispersion is
    what the layer exists to change, so the dispersion is what it is fitted
    against -- `CLAUDE.md`'s "dispersion comes from the model's own variance
    function and is validated against realised residual SD", applied as a fit.
    One moment per parameter: each role's own SD ratio for that role's
    concentration, then the pooled ratio for the between-role one.
12. **`chance_number` is a proxy and is labelled one.** The authoritative value
    lives on the chance table, which has no five to join on. See `features.md`
    section 1.3.

## 7. Consumption from the sim

```python
from cbb_sim.models import usage as U

# ---- once per (game, seed), at game setup -------------------------------
# `profiles` is class -> Profile over the TEAM's players with any as-of rate,
# built from the pregame as-of table (usage.build_profiles on the pregame
# design, or assembled directly from a shipped rate table).
state = U.new_game_state(
    game_id=game_id, seed=seed, profiles=profiles,
    alphas=None,                       # None = no dispersion (the U1 path)
)                                      # or U.hier_alphas(...) for the U3 path

# ---- once per credited event -------------------------------------------
shooter = U.draw_player(five_on_floor, "FGA_3", state)   # -> a CBBD player id
```

- Artifact: `data/processed/models/usage/usage_params_v1.json` carries, per
  class, the fitted shrinkage prior and strength, the fitted concentrations, the
  conditional-logit feature order and coefficients, and the tree parameters.
- `five_on_floor` is any iterable of exactly five CBBD player ids; the order does
  not matter (the model normalises over the set), and a length other than five
  raises.
- A lineup in which nobody carries any rate for the class falls back to the
  uniform over the five -- never a silent pick of the first id.
- One uniform of the `(seed, game_id, "usage")` stream is consumed per call, so
  a game's draws are reproducible and independent of every other game in the
  run. Paired bake-off arms line up game for game and seed for seed.
- For a tree or logit winner the sim needs the arm's own scorer rather than
  `draw_player`'s rate table; the per-event probabilities come from
  `U.LgbmChoiceArm.predict_proba` / `U.CondLogitArm.predict_proba` on the
  feature order recorded in the params file, and the categorical draw off the
  same stream. Section 9 records that the lookup-table form of that path is not
  built yet.

## 8. Artifacts

| Path | What it is |
|---|---|
| `data/processed/models/usage/events_{version}.parquet` | one row per credited event with its offensive five, its state, and the coverage flags |
| `data/processed/models/usage/asof_{version}.parquet` | one row per (season, player, game) with every pregame as-of input |
| `data/processed/models/usage/build_report_{version}.json` | universe, rim-override threshold, per-class coverage, crosswalk join rate, prior-season availability |
| `data/processed/models/usage/results_v1.json` | every arm's full metric block on both folds, including the calibration and responsiveness tables and the alpha grids |
| `data/processed/models/usage/usage_params_v1.json` | the fitted parameters the sim consumes |
| `data/processed/models/usage/report_v1.md` | the generated results markdown appended to `experiments.md` |
| `data/processed/models/usage/train_log_v1.txt` | the run log, including every grid rung |
| `data/processed/models/usage_v2/events_v2_shotshooter.parquet` | ROUND 2 (shooter keyed on `shot_shooter_id`): the same event table, relabelled |
| `data/processed/models/usage_v2/asof_v2_shotshooter.parquet` | round-2 as-of inputs |
| `data/processed/models/usage_v2/build_report_v2_shotshooter.json` | round-2 coverage, relabel counts and the per-season / per-team drop report |
| `data/processed/models/usage_v2/results_v2.json` | round-2 metric blocks, incl. the per-team segment |
| `data/processed/models/usage_v2/usage_params_v2.json` | round-2 fitted parameters (what the sim should consume once the PM adopts them) |
| `data/processed/models/usage_v2/report_v2.md` | the generated round-2 markdown appended to `experiments.md` section 8 |
| `data/processed/models/usage_v2/train_log_v2.txt` | the round-2 run log |
| `data/processed/models/usage_v2/shooter_key_audit*.json` | the shooter-key audit numbers behind `docs/tests/shooter_key_audit_2026-09-10.md` |
| `data/processed/models/usage_s1/{class}/{arm}_{YYYY-MM-DD}.joblib` | ROUND 2b: one S1 monthly refit; the engine takes the LATEST refit date at or before a game's date |
| `data/processed/models/usage_s1/{class}/{arm}_static.joblib` | the S0 control fit, used for games before the first refit date |
| `data/processed/models/usage_s1/s1_manifest.json` | every artifact row with its `refit_date`, `prior_kind`, `shrink_m`, `shooter_key` and `possessions_version` |
| `data/processed/models/usage_s1/results_v2b.json` | S0-vs-S1 metric blocks, the refit schedules and the per-scheme noise floors |

Round-1 artifacts are LEFT IN PLACE and unmodified: the engine reads
`usage/asof_v2.parquet` and `usage/usage_params_v1.json` (`scripts/build_engine_inputs.py`
lines 81-83) and must not be disturbed mid-run.

## 9. Known gaps / followups

1. **Every arm under-concentrates the top of the usage distribution.** Top-3
   share per team-game is 0.45-2.95 pp low in every one of the 25 F1 arm-class
   cells (and so is top-1 share, in all 25), and it is
   what makes the simple arms ineligible on three classes. The fix is a model
   that can represent matchup-specific concentration -- a "featured player" term,
   or an opponent-conditioned share -- not an exponent on the share vector.
   OPEN in the change ledger. **Round 2 shrank one contaminating term inside
   this defect but did not close it:** the shooter-label fix moved `FGA_rim`'s
   U1 top-3 gap from -2.17 to -1.79 pp (an assist credited as a shot is credit
   moved from the finisher to the passer, and the passer is usually not the
   lineup's highest-usage player), yet 49 of the 50 round-2 F1 cells are still
   negative. `experiments.md` R13.
1b. **`fg_make` reads the same wrong column and has NOT been re-run.**
   `models/fg_make.py:361` takes its `shooter_id` from `event_stream`'s
   `player_id`, i.e. `participant_1_id`, on every FGA row -- so its shooter
   as-of block (`shooter_make_c`, `shooter_att_c`, `shooter_games_asof`,
   `shooter_fga_asof`, `prior_season_make_c`) accumulates each attempt and its
   make/miss onto the wrong player on 4.9-14.1% of attempts. Feature set
   `B_plus_shooter` and everything above it is affected. The fix is one keyword
   (`ES.build_stream(..., shooter_key="shot_shooter_id")`); the re-run is a PM
   dispatch, not done here. `docs/tests/shooter_key_audit_2026-09-10.md` section 4.
2. **`TOV` is undecided.** The tree wins F1 by 3.8 floors and loses the
   robustness fold by 7.0. It needs a third fold, which means another season of
   on-floor ids (2026 is sealed).
3. **`FGA_jump2` and `FGA_3` are floor-thin.** 2.1 and 1.2 floors on the
   selection fold, both inside the floor on the robustness fold. If a cheaper arm
   matters at sim time, `cond_logit` on `FGA_jump2` and `proportional` on `FGA_3`
   are defensible on the evidence.
4. **The prior-season block has only ever been identified on one fold.** L13
   caps this model at two seasons of on-floor data, so `prior_rate` and
   `log_prior_share` are dropped on F1 and the within-2025 fold is the only place
   they are exercised -- where the shrinkage grid still never chooses them. A
   third season would make the transfer question answerable rather than merely
   reportable, and is the first thing to redo after the seal lifts.
5. **`chance_number` is a proxy.** The authoritative value lives on the chance
   table, which carries no on-floor five to join on. Making the possession
   builder emit a per-event chance index would remove the proxy; it is a small
   change to another worker's module and was not made here.
6. **No opponent term.** The allocator sees no defensive feature at all. That is
   the pre-registration's scope, and it is defensible (the rotation model already
   chose the five against this opponent, and the possession-outcome model already
   chose the event class), but it is also the most likely home for the
   concentration defect in item 1: "who gets the ball" against a switching
   defence is plausibly not "who gets the ball" against a zone.
7. **The sim path for a tree or logit winner is not a lookup table yet.**
   `draw_player` serves the rate-table arms directly; a tree winner needs its
   per-event scores materialised, and `CLAUDE.md` forbids live model calls in the
   sim loop. Building that table -- alternatives x fitted score, keyed on the
   pregame feature bins -- is the next engineering step and it is not done.
8. **Technical free throws have no allocator.** They are excluded from
   `FT_trip` because the shooter is coach-chosen. `free_throw/model.md` section 9
   flags the same gap for the make-probability model; both need one rule.
9. **`and_one` trips are double-counted across classes by construction.** The
   made field goal is one `FGA_*` event and the trip it draws is a separate
   `FT_trip` event, both credited to the same player. That is correct for
   per-class allocation but a consumer summing classes to get "possessions used"
   must know it.
10. **The game-level checks isolate allocation on purpose, and therefore say
    nothing about the composed player layer.** They re-allocate the actual event
    sequence over the actual fives. G8 on a full sim run -- rotation, event model
    and allocator composed -- is a different and harder test, and it has not been
    run.
