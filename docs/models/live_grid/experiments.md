# live_grid: pre-registered bake-off of the binning scheme for the CBB live-odds file (`live.json`)

Status: PRE-REGISTERED 2026-11-02 (builder worker, owner decision "build the live-odds layer, game level first"), BEFORE any exporter or
grader code and before any arm is scored. Append-only: results go in later sections; status changes go to `docs/models/change_ledger.md`
in the same commit. Design this tests: `docs/ops/live_trajectories_cbb_design_2026-11-02.md` section 2.2. Note: the design doc names
`docs/models/live_conditioning/experiments.md`; the owner brief moved it to this path. Pre-registration only; nothing here adopts a model.

Scope: the live line is a SLICE of the graded pregame ensemble (no live model, no hand adjustment). The question is only how the stored
trajectories are binned and pooled so a static JSON lookup (the client interpolates between bin midpoints) reproduces the conditional
distribution of the rest of the game. A kernel/per-seed conditioner is not entered: no kernel endpoint exists and the product is a static file.

## 1. Data (fixed now)

Research trajectories with the design-doc schema (`results/trajectories/<tag>/trajectory.parquet`, one row per possession):

* PRIMARY: `clk10_ctrl_o0_s25`, `clk10_ctrl_o1000_s25`, `clk10_ctrl_o2000_s25`, `clk10_ctrl_o3000_s25`, `clk10_ctrl_o4000_s25`: the same
  500 games, served-engine control, seeds 0-24, 1000-1024, ..., 4000-4024 = **125 distinct seeds per game, 500 games, 62,500 game-seeds**
  (8.6 M possession rows). `eg_ctrl_*` repeat the same row counts (duplicates of the control) and are not used.
* CONFIRMATORY (reported, not selecting): `replay394_s50` (394 games x 50 seeds, 19,700 game-seeds), same machinery with a 40/10 split.
* Not used: `daily_2026-11-02_s20_o0` (20 seeds), arm dirs `eg_EV1*`, `clk10_E1*` (non-served engines).

Seeds are the unit of holdout, not games: for every game the 125 seeds are randomly partitioned into TRAIN (100) and TEST (25); the
grid for a game is built from that game's own train seeds and scored on that game's own test seeds (conditional calibration WITHIN game).
No game contributes training information about another game, except the labelled league OT prior in section 4. Split = a seed-level random
draw per game, keyed (split_id, game_id). Four splits (ids 1-4). Production will have 2,500 seeds per game; a 100-seed training set is 4% of
that, so:

* minimum-n thresholds are SCALED by the same factor: production 200 (remaining curves) x 100/2500 = **MIN_N = 8**. Total-axis quantile bins:
  5 (about 20 per bin at 100 seeds).
* A training-size sensitivity (train 50 vs 100 seeds, same test) is reported for the winning structure to show the direction of the
  scaling. Transfer to 2,500 seeds is an extrapolation and is labelled as one; this study cannot test cells that only exist at 2,500 seeds.

## 2. Checkpoint convention (all arms, design doc 2.2 item 7)

* A checkpoint of a (game, seed) is a possession OPEN state: elapsed = (period-1)*1200 + (1200 - clock_start) for period <= 2; margin and
  total BEFORE the possession (previous row's after-score; 0-0 at poss_idx 0). Conditioning rows: period <= 2 only.
* Label = final score after the last row of the (game, seed), OT included. Win = final home score > final away score (no ties exist).
* Time bins: coarse 120 s (20 bins, midpoints 60..2340). Midpoint STATE = the last checkpoint at or before the midpoint (CFB rule). Remaining
  quantity at a bin = final - score at that midpoint state. Each seed counts once per cell.
* Kickoff row at elapsed 0 (state 0-0, all train seeds) and an exact end-of-regulation row at elapsed 2400 (section 4) are in EVERY arm
  except `A0` (literal CFB copy, fade-to-zero, no kickoff row).
* Lookup at elapsed t, current margin m, current total s: take the two adjacent row midpoints around t (in the arm's own midpoint sequence),
  look up the cell each row assigns to (m, s), and blend the two remaining-distributions linearly in t (client rule). Win probability
  = P(m + remaining margin > 0) from the blended remaining-margin distribution; projected total distribution = s + remaining total.
  Probabilities are floored/capped at the published per-mille resolution [0.001, 0.999] (curves are published as integers per mille).

## 3. Arms (candidates)

Structure axis (S) x pooling axis (P). Margin bins "coarse" = edges `[-11,-6,-2,3,7,12]` under `searchsorted(side="right")` (7 bins, mirror
symmetric for integers). Football copy edges `[-14,-7,0,1,8,15]`.

| arm | time rows | margin-keyed curves (margin AND win prob) | total-keyed curves | thin-cell rule |
|---|---|---|---|---|
| `R0` (diagnostic) | 20 coarse | none: all-seeds time row | none: time row | n/a |
| `A0` (diagnostic, literal CFB) | 20 coarse, NO exact end row (fade to zero over last 60 s unless tied) | football edges, 7 bins | margin-keyed (as CFB) | time row |
| `A_row` | 20 coarse + exact end | coarse edges, 7 bins | margin-keyed | time row |
| `A_pool` | same | same | same | neighbour pooling |
| `B_row` | same | coarse edges | 5 quantile bins of train total-so-far at that row (edges per row) | time row |
| `B_pool` | same | same | same | neighbour pooling |
| `C_row` | coarse rows to elapsed 2100 + late block: 8 rows of 30 s (midpoints 2175..2385) | coarse before 2160; late edges `[-16,-10,-6,-3,-1,2,4,7,11,17]` (11 bins) in the late block | total quantile bins before 2160; in the late block total curves keyed by the late margin bins (fouling regime) | that row's all-seeds row |
| `C_pool` | same | same | same | neighbour pooling |
| `C_pool2` | same | same | same | thin late cell -> the coarse 7-bin margin partition of the SAME 30 s row, then neighbour pooling |

Eligible for selection: the seven arms `A_row, A_pool, B_row, B_pool, C_row, C_pool, C_pool2`. `R0` and `A0` are diagnostics (reference
points for "does conditioning matter" and "does the CFB copy match"), never selected.
Neighbour pooling: a margin bin with n < MIN_N merges with the adjacent bin that has the larger n (ties: toward zero), repeating until
n >= MIN_N or all bins are merged; the cell histogram is the sum of its merged members; `widened` = the number of merged bins. Quantile total
bins are never thin by construction at n >= 5 x MIN_N.
Simplicity order for ties (simplest first): `A_row < A_pool < B_row < B_pool < C_row < C_pool < C_pool2`.

## 4. End-of-regulation row (exact, not faded)

At elapsed >= 2400 the lookup is exact: if the regulation margin is non-zero, remaining margin and total are 0; if tied, the remaining
distribution is the empirical OT remainder of train seeds that were tied at 2400 in that game; if the game has fewer than 3 such train seeds the
OT remainder pools the tied-at-2400 seeds of all TRAIN games of the same split (a league prior, labelled `pooled_ot`, count reported).
Between the last row midpoint and 2400 the client blends toward this row linearly.

## 5. Test set and metrics (identical for every arm; one script `scripts/grade_live_grid_v1.py`, arms scored blind by label)

Test checkpoints: for each test seed, 12 elapsed times t = `[240, 600, 960, 1200, 1500, 1800, 2040, 2160, 2220, 2280, 2340, 2395]`
(5 of 12 are in the last 4 minutes by design: that is where the structure differs). State at t = the last checkpoint at or before t.
Rows scored: 500 games x 25 seeds x 12 = 150,000 per split.

* M1 (primary): mean log loss of the home-win probability (per row, then mean over rows; per-mille floor/cap).
* M2: CRPS of remaining total (discrete, unit steps, mean over rows), from the blended remaining-total distribution.
* M3: CRPS of remaining margin, same.
* Also reported: Brier, and calibration (predicted-probability decile vs realised) with the responsiveness slope.

Segments (every metric): period (t < 1200 vs >= 1200), |margin| bucket at the checkpoint (0-5, 6-12, 13+), late block (t >= 2160) vs not,
OT games (final went to OT) vs not, pregame-favourite strength quintile (by the game's train-seed unconditional home-win probability), and
per-checkpoint-time. Per-game level: mean M1 per game per arm, paired against `A_row`; report the share of games where the arm is better and a
game-clustered bootstrap SE (1,000 draws) of the paired difference. Per-team is not separately reported (a team appears in few games; stated
underpowered). Segments with fewer than 30 rows are marked UNDERPOWERED, never read as signal.
Cell diagnostics: share of looked-up cells that were thin (pooled/fallback) per arm and per segment, mean `widened`.

## 6. Noise floor and decision rule

* Noise floor of an arm on a metric = max - min of the metric for that SAME arm across the four independent seed splits (same arm, same
  spec, different seed split). For a pairwise comparison the larger of the two arms' floors is used.
* Arm X BEATS arm Y on metric m if mean_over_splits(Y - X) > floor (larger of the two) AND X is better in all 4 splits (lower is better).
* Selection: take the arm with the best mean M1 among the seven eligible; the WINNER is the simplest arm (simplicity order above) that is
  within one floor of that best arm on M1, M2 AND M3 simultaneously and has no late-block veto. Ties go to the simpler arm.
* Late-block veto: an arm whose mean late-block (t >= 2160) M1 is worse than the best arm's late-block M1 by more than that floor is not
  eligible to win (the late block is the reason structure C exists).
* `R0` must lose to `A_row` on M1 by more than the floor, otherwise the result is reported as "conditioning on state does not matter at
  this seed count" and the exporter ships the simplest eligible arm with that caveat. If `A0` beats `A_row`, that is reported as a finding on
  edge scaling (not selecting).
* Total-axis gate (design B0): if no `B_*` arm beats its `A_*` sibling on M2 beyond the floor, the total axis is dropped and the file reverts
  to the CFB shape (this restates the rule above, it is not an extra degree of freedom).
* Pooling variants are compared inside their structure first (row vs pool) by the same BEATS rule.

## 7. Compliance with the project rules

Grading truth = the trajectory's own last-row score (same engine run): a SELF-CONSISTENCY test of a slice of the ensemble, not a test against
real games. It says nothing about live-market accuracy; the layer stays unvalidated until the shadow logger (design B8) has two weeks.
No post-hoc adjustment on engine output enters any arm; pooling uses only train seeds. Every grid is built from train seeds only; test seeds
are never read when building tables (enforced by construction, asserted by a unit test). Compute: local, no AWS.

## 8. ADDENDUM registered 2026-11-02 AFTER the primary 100-seed read (disclosed; the primary read had been scored when this was written)

The primary read (section 9, appended below by the run) showed the diagnostic `R0` (no state conditioning) ahead of every eligible arm at
100 training seeds, with the A_row - R0 M1 gap shrinking with seed count (25 -> 50 -> 100 training seeds). 100 seeds is 4% of production, so
the primary read cannot decide the production question. Registered now, BEFORE it runs:

* SCALE READ: 20 games (the first 20 ids of `results/trajectories/_inputs/replay_ids_394.parquet`), served-engine control, 2,500 seeds each
  (`run_engine_window_v2.py --seeds 2500 --trajectory-seeds 2500 --tag live_grid_scale20_s2500`), 4 splits of 2,000 TRAIN / 500 TEST seeds per
  game, MIN_N = round(200 x 2000/2500) = 160, same 12 checkpoints, same arms, same grader, same BEATS / floor / selection rules (floor =
  same arm across the 4 splits). 20 games is a small game sample (game-level results labelled accordingly); the seed count per game is the point.
* Interpretation rule: the section 6 `R0` clause is evaluated on the SCALE read as well. If `A_row` beats `R0` on M1 at 2,000 seeds, the
  production arm is the section 6 WINNER of the scale read (same selection rule). If `R0` still beats or ties `A_row`, the result is reported as
  "state conditioning by binning does not help at 2,000 seeds" and the exporter default stays the simplest eligible arm (`A_row`) with that
  caveat printed in the file's `rule` string and in the exporter doc; the PM decides whether a smoothed (non-binned) conditioner is a
  follow-up round (not entered here: it would need its own pre-registered arms).
* If the 100-seed section 6 winner and the scale-read winner differ, the scale-read winner is the exporter default (it is the production regime);
  the 100-seed winner stays selectable behind the flag.
* Implementation clarification (no change to rules): quantile total-so-far bins can be thin through ties; they take the same thin-cell rule as
  margin bins (time row for `*_row`, neighbour pooling for `*_pool`).

## 8b. AMENDMENT registered 2026-11-02 after the PM message "match the shipped client" (disclosed; before the re-run, after the pre-amendment read)

The site client (monte_site commit 6d2de56, `atStateBb`) fixes the lookup geometry. The grader's lookup is changed to be that client rule, so
the arm that wins is the arm the file can actually express:

1. No kickoff row (client clamps to the first coarse midpoint). All arms: `kickoff=False`.
2. Arm C keeps all 20 coarse rows to 2400 (client `coarseAt`); the late block (8 rows, midpoints 2175..2385) replaces the coarse lookup for
   elapsed >= 2160, interpolating linearly between late midpoints, clamping before 2175, and from 2385 blending toward the exact end row at 2400.
3. A thin late cell in `C_row` is OMITTED and the client falls back to the coarse block at the late row's midpoint (design wording "thin late
   cells fall back to the coarse block"); it replaces the earlier "late row all-seeds" rule. `C_pool` pools neighbours inside the late row (cells
   never omitted). `C_pool2` is DROPPED: its fallback (a coarse partition of the same 30 s row) cannot be expressed in the file.
4. The coarse total axis keeps the time-row total curve when a total cell is thin (client `time_total_sf`); the margin-keyed `total_sf_margin`
   fallback is not shipped.
Eligible arms are now `A_row, A_pool, B_row, B_pool, C_row, C_pool` (simplicity order unchanged). All other rules, data, splits, checkpoints,
metrics and the section 8 scale read stand. The pre-amendment 100-seed read (kickoff row, C_row late fallback = row, C_pool2) is kept in
section 9a as an additional record; the post-amendment read is the selection record.

## 8c. Reading of the section 6 selection sentence (disclosed; registered after the first post-amendment read)

"Within one floor of that best arm on M1, M2 AND M3" cannot be satisfied by one "best arm" when the metrics disagree (the late block lowers M2/M3
and the no-late arms are marginally better on M1). Fixed reading, applied to every read and shown beside the literal one-sided code result:
an arm is eligible to win when, on EACH of M1, M2, M3 and M1-late, it is within the larger of the two arms' floors of the best eligible arm ON THAT
METRIC. The simplest eligible arm wins. (The literal one-sided code, which lets a better arm always pass, is also printed.)

## 8d. ADDENDUM: two more arms to resolve the total-axis gate (disclosed; registered after the section 8 scale read, BEFORE the arms run)

The scale read and the 100-seed read both show no `B_*` arm beating its `A_*` sibling on M2 (the total-axis gate of section 6 says: drop the axis),
yet `C_*` (= B + late block) wins M2/M3 by a margin that is the late block / exact end blend, not the total axis. The arm set has no
"late block without the total axis", so the gate cannot be applied cleanly. Added, defined exactly as `C_*` with `total_axis=False`:

* `D_row` = `A_row` + late block (coarse totals keyed by margin as in A; late rows as in C; thin late cell omitted -> coarse fallback).
* `D_pool` = `A_pool` + late block (neighbour pooling everywhere).

New simplicity order (simplest first): `A_row < A_pool < D_row < D_pool < B_row < B_pool < C_row < C_pool` (the total axis, which has had no
support in either read, is ranked as the more complex feature). All reads (primary 100-seed, confirmatory, training-size sensitivity, scale) are
re-run with the 10-arm set; the total-axis gate is then the pair `C_x` vs `D_x` on M2 (and `B_x` vs `A_x`).

## 8e. AMENDMENT: push-as-half in the win probability (grader bug vs the client; disclosed; found after the first full read)

The client prices `pWinA = pCoverA(line 0)` with push counted as half (`pAbove`, `liveGrid.ts`). The grader's `win_prob` counted the exact
`m + remaining = 0` mass as a loss. That mass is an artefact of pairing a state with an independent remaining distribution (no real final ties),
about 2% of mass, and it showed up as a uniform ~1.1 point under-prediction of home wins even in-sample (detected by an in-sample consistency
check, R0 at t=2200: 0.528 vs 0.543). Fix: `P(win) = P(rem > -m) + 0.5 P(rem = -m)`, identical to the client. All reads are re-run; the
pre-fix reads (kept as `*_noD`/`*_pre8e` result dirs) selected the same arms.
