# L5 CLOCK CONSUMPTION

Status: **TWO BAKE-OFF ROUNDS RUN 2026-09-10 — NO ARM ADOPTED.**
Round 1: zero of twenty (arm × feature set) combinations passed the
pre-registered gates on fold 2. Round 2 changed the period-end state
representation and added a two-regime arm: zero of six passed. Nothing is
shipped and no post-hoc correction is applied.
Round-1 grid `experiments.md` section 2, diagnosis section 3, reading section 4;
round-2 pre-registration section 5, grid section 6, reading section 7.
Features: `features.md` (round 2 is section 8).

The pre-registration anticipated this outcome explicitly — *"If no arm passes
the emergent G1, report the failure and the diagnosis; adopt nothing"* — so
section 4 below is the report and section 5 is the diagnosis.

## 1. Purpose

How many seconds a possession takes. This is layer L5 of the cascade and it is
the reason the engine has a pace at all: `ARCHITECTURE_DECISIONS.md` Decision 7
removed the game-level possession sampler, so the possession count of a
simulated game is whatever falls out of running THIS model over two
1,200-second halves plus the overtime model.

Decision 7 exists because of L14: all 32 arms of the L2 game-level pace
bake-off failed PIT calibration for the same reason (overtime and fat tails),
and feature richness and model class were inside the noise floor. An emergent
count gets the overtime skew for free. The cost, written into Decision 7's own
"risk acknowledged", is that **G1 now depends on this model being right BY
STATE**, not merely on average. This run is the first measurement of that cost,
and it is larger than the tolerance.

Order of operations inside the engine, per the pre-registration: at each
possession start the engine draws the duration conditioned on state, THEN draws
the terminal event (L3) given the duration bucket and state. This model
therefore never sees the terminal event, which is why the pre-registered "mean
and SD of duration by terminal event class" table exists — it measures how much
of that unseen information the state recovers on its own.

## 2. Target variable

`duration_s` from `data/processed/possessions/possessions_{season}.parquet` —
the seconds between the previous possession's terminal event and this one's.
Integer, clipped at 0 by the possession builder
(`src/cbb_sim/pbp/possessions.py`), modelled on the support 0..90.

**Population (F1 + F2 windows, seasons 2022-2025):**

| step | games | possessions |
|---|---:|---:|
| D-I, hoopR feed not truncated | 21,969 | 3,029,695 |
| ...and CBBD-complete (possession points reconcile to the final score for both teams) | 18,902 | 2,607,355 |
| ...and `duration_s <= 90` | 18,902 | 2,607,192 |

The CBBD-completeness filter is the pre-registration's "CBBD-complete games"
and it is built here for the first time — `games_universe.pbp_truncated` is a
hoopR-side flag and does not catch it
(`docs/tests/possessions_build_2026-09-10.md` section 2). It removes 14.0% of
the universe, concentrated in 2022-2023 where the CBBD stream is short of the
final score in ~19% of games. Rows per season after every filter:
574,973 / 603,112 / 697,903 / 731,204.

163 rows (0.006%) exceed the 90-second cap. They are periods whose first logged
event is also their last, so the possession is stamped with the full period
length; they are EXCLUDED, not clipped, because clipping would invent a
duration the feed does not contain.

**Censoring.** 5,600 rows (0.215%) have `terminal_event == "end_period"`: the
possession ended at the horn, so the true duration is strictly larger and the
row is right-censored. Each arm's handling is stated in the module docstring of
`src/cbb_sim/models/clock.py` and re-stated in section 6 below.

**Distribution.** Unimodal but strongly asymmetric and with a real point mass
at the short end: 1.90% of possessions last 0 seconds, the mode sits at 17-18 s
(3.98% and 3.96%), and the mean is 17.48 s. That shape is the single most
important fact in this bake-off — see section 5.

**`unknown`-terminal possessions are KEPT.** 0.8% of rows carry
`terminal_event = "unknown"`, the mismatch-guard residue: an unlogged change of
possession, which L3 excludes because its terminal LABEL is a data gap. Here
the label is irrelevant — this model never sees it — and the row's DURATION is
real elapsed clock, so dropping it would punch holes in the tiling of the
period that the emergent test depends on. They are kept, and their consequence
is reported in section 4.5.

## 3. Methodology at a glance

- **Data window.** Seasons 2022-2025, 2,607,192 possessions. 2026 sealed;
  `assert_not_sealed` fires on every train and test slice.
- **Split.** Temporal walk-forward, no random split. F1 trains {2022, 2023} and
  tests 2024; F2 trains {2022, 2023, 2024} and tests 2025 and is the selection
  fold.
- **Primary metric.** CRPS of the predictive distribution, in its discrete
  (ranked-probability-score) form on the integer grid 0..90:
  `CRPS = sum_t (F(t) - 1{y <= t})^2`. CRPS rather than an error on the mean
  because the engine consumes the WHOLE distribution: a duration model with the
  right mean and the wrong spread produces the right possession count on
  average and the wrong count variance, which is the CFB INV-56 failure one
  level down.
- **One scoring object for five very different model classes.** Every arm
  exposes `pmf(df) -> (n, 91)`, a proper probability mass function on the
  integer seconds. That is what makes an empirical resampler, two parametric
  regressions, a discrete-time hazard and nine quantile regressions comparable
  on one number — and the sampler and the emergent simulator consume exactly
  the object that was scored, so there is no second implementation to drift.
- **Model families tested.** Empirical Kaplan-Meier state cells; heteroscedastic
  log-normal and Gamma with a censored MLE; a discrete-time per-second logistic
  hazard (full 38M person-period expansion, no subsampling); LightGBM quantile
  regression at 9 levels sampled by inverse CDF with linear interpolation.
- **Gates, applied before CRPS is consulted.** The emergent G1 (mean ± 1.0,
  SD ± 0.75, all powered months) and "no PIT K-S D > 0.05 in any powered state
  cell".
- **Noise floor.** Seed-varied refit of the tree arm (seed 20260910 vs
  20261910): worst |ΔCRPS| **0.000295**. Game-block bootstrap SE of the mean
  CRPS for every other arm: worst **0.006764**. The larger is used, which is
  the conservative choice.

## 4. Winner

**There is none.** `n_eligible = 0` of 20. The pre-registered rule eliminates
every arm before CRPS is consulted, and the standing rule ("no hand tuning on
engine output") forbids closing the gap with a correction.

Full F2 table in `experiments.md` section 2.2. The four things it says:

### 4.1 Nothing passes the emergent G1, and the failure is the MEAN, not the SD

Two arms pass the overall mean-and-SD reading — `gamma`/`C_plus_score`
(Δmean **+0.720**, ΔSD **−0.280**) and `gamma`/`B_plus_teams` (+0.887, −0.292)
— and then fail the "all powered months" clause: 4 of 5 and 3 of 5. January is
the month that breaks both (Δmean +1.119 and +1.250 against a ±1.0 tolerance).

Every other arm misses the overall mean outright: `empirical` +1.10 to +1.37,
`hazard` +1.81 to +2.51, `lgbm_quantile` +2.19 to +2.82, and `lognormal`
−1.35 to −2.01 in the other direction. Only `lognormal` and `gamma` get the SD
inside tolerance on every feature set; the rest are 0.26-1.86 possessions too
narrow.

The by-month structure is uniform across arms and is itself a finding: **every
arm's overshoot is smallest in November and largest in January-February**
(`gamma`/`C_plus_score`: +0.34 Nov, +0.57 Dec, +1.12 Jan, +0.82 Feb, +0.60
Mar). Real pace slows by 1.83 possessions per team from November to January
(69.49 → 67.65 on this test set); `gamma`'s chained sim slows by 1.06 over the
same months. L5 — the learning, not the layer — says the pace model needs a
season-progress term, and this is that learning showing up one level down, at
the possession. April is the only underpowered month (16 games) and is excluded
from the read, per the pre-registration.

### 4.2 Nothing is PIT-clean, and the parametric families are not close

| arm | PIT failures / 27 powered cells (F2) | worst K-S D |
|---|---:|---:|
| `empirical` | 5 | 0.081-0.083 |
| `lgbm_quantile` | 6-7 | 0.140-0.198 |
| `hazard` | 22-23 | 0.376-0.389 |
| `gamma` | 23 | 0.388-0.404 |
| `lognormal` | 26-27 | 0.330-0.346 |

The gap between the empirical arm and everything parametric is a
**distribution-family failure, not a fitting or a leak failure**, and it
reproduces in-sample: refitting `gamma`/`C_plus_score` on the training window
and scoring it on 120,000 of its OWN training rows gives a pooled PIT K-S D of
0.056 (deciles 0.90 / 0.78 / 0.85 / 0.93 / 1.03 / 1.13 / 1.22 / 1.28 / 1.23 /
0.67), while the empirical arm on the same rows gives 0.0033. A unimodal
continuous family cannot simultaneously carry the 2% point mass at zero
seconds, the shot-clock hump at 17-18 s, and the thin tail past 35 s. The
logistic discrete-time hazard fails for the same reason one step removed: its
baseline is flexible in elapsed time but the state enters through a single
linear index.

`empirical`'s five failures are the five smallest powered cells in the table:
four of the five `other` previous-end-type cells (`other | 300-599`, D 0.083,
n 830; `other | 120-299`, 0.075, n 548; `other | 0-34`, 0.061, n 422;
`other | 600+`, 0.058, n 1,365) plus `period_start | 300-599` (0.066, n 356).
`other` is the dead-ball / unlogged change of possession, 0.5% of possessions.
Every cell with n ≥ 2,000 is clean — the largest, `made_FG | 600+` at n =
137,113, has D = 0.005 — and on F1 the same arm has only 2 failures. The honest
reading: the empirical arm is within a whisker of PIT-clean on the states that
matter, and would still not be adopted, because it fails G1 by +1.10.

### 4.3 The tree arm wins CRPS by 18× the floor and is the worst engine component

`lgbm_quantile`/`D_plus_season` has the lowest F2 CRPS at **4.8082**, with
`C_plus_score` at 4.8102 — a 0.0021 gap, inside the 0.006764 floor, so those
two are tied and the simpler feature set would win the tie-break if the arm
were eligible. The best non-tree arm is `empirical`/`C_plus_score` at 4.9307,
so the tree's margin is **0.1225, about 18× the floor** — comfortably clearing
the pre-registered "a tree arm must beat the best non-tree arm by more than the
floor" clause it never got to use.

It is also the arm that would damage the engine most: emergent Δmean +2.19 to
+2.82 possessions per team per game, and the worst end-of-half profile in the
grid (section 4.4). The reason is in the pre-registered sampling scheme rather
than in the tree: nine quantiles inverted with linear interpolation cannot
cover the tail, so **7.0-7.6% of test rows receive ZERO predictive mass on the
duration that actually occurred** (the "undef %" column; every other arm is
under 0.72%). The truncated tail costs 0.19-0.26 seconds of mean duration,
which over ~137 possessions a game is most of the overshoot.

This is the clearest single result of the run: **the arm that scores best on
the primary metric is the arm that would fail the gate the model exists to
serve.** That is exactly what a gate-before-metric decision rule is for.

### 4.4 The end-of-half check fails everywhere, and it says where to look

The pre-registered check: the share of halves whose last possession starts with
fewer than 35 seconds remaining, and that possession's mean duration.

| arm (C_plus_score) | share < 35 s | mean duration |
|---|---:|---:|
| **actual** | **0.8832** | **12.06 s** |
| `lognormal` | 0.8934 | 11.79 |
| `gamma` | 0.9514 | 11.65 |
| `empirical` | 0.9829 | 9.50 |
| `hazard` | 0.9916 | 7.98 |
| `lgbm_quantile` | 0.9967 | 5.83 |

No arm reproduces the actual 88.3% share at better than 89.3% (`lognormal`,
which is wrong about everything else), and the two arms with the best CRPS
reproduce it at 99.7% and 98.3%. The sim instead squeezes in one more short
possession, which is the same defect as the +1 to +2.8 possession overshoot seen
from the other end.

> **CORRECTION, measured in round 2 (2026-09-10).** The first version of this
> section read the 11.7% of halves whose last possession starts with 35+ seconds
> left as clock management — "the offence takes the ball with 40 seconds on the
> clock and holds it". That is WRONG, and the round-2 run measured it: those
> halves' last logged possession has a mean END clock of 20.4 seconds and only
> 16.3% of them reach 0:00. They are halves where the CBBD event stream simply
> STOPS before the horn. Only 44.6% of 2025 halves have a last possession ending
> at exactly 0:00; the median unaccounted time is 1 s but the 90th percentile is
> 17 s. A sim that always runs its clock to zero cannot reproduce a half that
> stops at 0:20, so the 88.3% actual is not a number any correct model should
> match. On the CLOCK-COMPLETE halves (last possession ending within 2 s of the
> horn, 63.5% of 2025 halves) the actual share is **95.56%** and the actual mean
> duration **12.41 s** — which makes `gamma`'s 95.14% nearly exact rather than
> 6.8 points high. The round-1 CRPS, PIT and emergent-G1 numbers are unaffected;
> only this one interpretation was wrong. Full evidence and both readings:
> `experiments.md` section 6.3.

### 4.5 What DID pass

**Responsiveness passes emphatically on every arm** (`CLAUDE.md` standing rule,
`experiments.md` section 3.3). Bucketing F2 games by the pregame tempo-prior
quintile, the emergent possession count slopes with the actual in 4 of 4
quintile steps for all five arms, with sim-vs-actual span ratios of 0.79
(`empirical`), 0.92 (`lognormal`), 1.01 (`gamma`), 0.98 (`hazard`) and 1.06
(`lgbm_quantile`). The sim's per-game count correlates 0.36-0.52 with the
actual. So this is not a league-average model wearing a feature vector: the
matchup signal is there and it is correctly scaled. The failure is a
conditional-shape failure, not a responsiveness failure.

**The state recovers the ENDS of the terminal-event ordering and nothing in
between** (`duration_by_terminal_F2.csv`). The pre-registered diagnostic asks
whether a model blind to the terminal event still orders possessions the way the
terminal event does. On `empirical`/`C_plus_score` the actual means run
`FT_trip_bonus` 12.09 < `FT_trip_shooting` 15.37 < `TOV` 16.00 < `FGA_rim`
16.97 < `FGA_3` 18.91 < `FGA_jump2` 20.92 — an 8.8-second span. The
model-implied means run 15.51 / 17.64 / 17.73 / 17.37 / 17.66 / 17.97: the two
extremes are in the right places and the middle four are compressed into a
0.4-second band and mis-ordered. So the state carries the "an early foul ended
this possession before it developed" signal and essentially none of the
rim-vs-jumper signal — which is the L3 model's job to supply, not this one's,
but it does mean the engine cannot lean on duration to disambiguate shot type.

The same table carries a warning about the `unknown` terminal class: actual mean
6.48 s against a model-implied 19.31 s, a 12.8-second gap on 6,005 F2 rows
(0.8%). That is not a model failure — the shortness comes from the missing event
that created the mismatch-guard close, not from anything in the state — but it
biases the emergent count DOWNWARD by roughly 0.4 possessions per team, so the
true overshoot in section 4.1 is if anything understated.

**Censoring is handled non-degenerately by every arm.** Mean predicted
P(D > c) on the censored rows is 0.25 (`lgbm_quantile`), 0.43 (`empirical`),
0.50 (`hazard`), 0.61 (`lognormal`), 0.63 (`gamma`) — all strictly inside
(0, 1), so no arm has absorbed the horn as signal.

## 5. The diagnosis

Three measurements, all in `experiments.md` section 3.

### 5.1 D1 — the conditional mean of duration bends at the end of a period and only one arm bends with it

Actual mean duration by seconds remaining at the possession's start, against
each arm's own predicted mean on the same rows (F2, regulation):

| seconds left | n | actual | `empirical` | `lognormal` | `gamma` | `hazard` | `lgbm_quantile` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0-9 | 4,186 | **3.07** | 8.77 | 16.38 | 15.51 | 9.71 | **3.21** |
| 10-19 | 5,288 | **7.37** | 9.20 | 16.98 | 16.08 | 11.24 | **7.27** |
| 20-24 | 2,936 | **10.75** | 9.41 | 17.06 | 16.17 | 12.87 | **10.43** |
| 25-29 | 3,136 | **13.08** | 9.49 | 17.06 | 16.18 | 13.68 | **12.79** |
| 30-34 | 3,546 | **15.64** | 9.55 | 17.17 | 16.31 | 14.31 | **14.86** |
| 35-39 | 3,804 | 15.55 | 15.55 | 17.17 | 16.32 | 14.80 | 15.32 |
| 90+ | 667,164 | 17.88 | 17.88 | 18.39 | 17.66 | 17.49 | 17.69 |

The real conditional mean falls from 17.9 s to 3.1 s across the last 90 seconds
of a period. `empirical` is FLAT at ~9.5 s across the whole 0-34 range, because
its finest pre-registered seconds-remaining bucket is `0-34`. `lognormal` and
`gamma` are flat at 15-17 s across the entire range, because `seconds_remaining`
enters them linearly through a log link and the required curvature is not
representable. `hazard` bends about half as far as it should. Only
`lgbm_quantile` tracks the curve, and it is the arm whose tail truncation
breaks it elsewhere.

The emergent possession count is 2,400 divided by the mean duration realised
over the chain, so a conditional mean that is wrong only in the last 90 seconds
of each half still moves the count — and it moves the end-of-half check
directly.

### 5.2 D2 — the count SD is mostly reachable; the arms are 0.3-0.8 short of it

| quantity | value |
|---|---:|
| actual per-team possessions per game, SD | **5.088** |
| i.i.d. renewal reference (identical possessions, no game-to-game pace variation) | 3.264 |
| between-game spread the FEATURES carry (`feature_implied_count_sd`) | 3.12-3.59 |
| the two combined in quadrature | ~4.5-4.9 |
| sim SD achieved | 4.33 (`empirical`) to 5.27 (`lognormal`), `gamma` 4.81 |

So a state-conditional i.i.d. duration model can in principle reach about 4.7-4.9
of the 5.09, and `gamma` reaches 4.81. The count-SD half of G1 is therefore
*not* the blocking problem — it is nearly solved, and the residual 0.2-0.3 is
the part of real pace dispersion that is neither in the features nor in
possession-level noise (the same shape as L10 one layer down: independent draws
understate dispersion). The blocking problem is the MEAN, by month, and the
conditional shape near a period boundary.

### 5.3 D3 — responsiveness is fine

See section 4.5. The model is matchup-specific with a correctly scaled slope.
This matters for the diagnosis because it rules out the boring explanation: the
features are working, the conditional distribution is not.

## 6. Decisions log

- **Possession-level target, not chance-level.** The pre-registration's Target
  line says "duration_s of a possession". The consequence is that the
  pre-registered A_state feature "chance number within possession" is
  identically 1 in every row, because a possession begins on its first chance
  by the segmentation rule. It is KEPT in the feature list, reported as
  zero-variance, and dropped by every arm with the drop recorded — a
  pre-registered feature is never silently discarded. If a chance-level
  duration target was intended (the L3 outcome model is chance-level), that is
  a different target and needs its own pre-registration. Flagged in section 9.
- **`unknown`-terminal possessions are kept, unlike at L3.** L3 drops them
  because their terminal LABEL is a data gap and the label is L3's target. Here
  the target is elapsed clock, which is observed regardless of the label, and
  dropping them would leave gaps in the tiling of the period that the emergent
  test walks through. The cost is measured and reported (section 4.5) rather
  than avoided.
- **`is_transition` is banned outright rather than flagged.** It is defined as
  `duration_s <= 8 AND start_reason in {DREB, TOV}` — a function of the target.
  The change ledger already carries it as a CONFIRMED-DEFECT at L3 for being
  contemporaneous with the outcome; here it would be a perfect leak.
  `tests/test_clock.py` proves the identity on the real table and proves the
  column never reaches a feature set or a fitted arm.
- **One CRPS definition for every arm, via a shared `pmf()` interface.** The
  alternative — a continuous CRPS for the parametric arms, a sample-based CRPS
  for the resampler, a pinball loss for the quantile arm — would have compared
  scoring conventions rather than models.
- **Every arm is conditioned on D ≤ 90**, the same conditioning the target's own
  exclusion rule applies, so the scored object and the observed object live on
  the same support. Piling the excess onto the last grid cell instead would put
  visible mass on a 90-second possession, which is exactly what the cap exists
  to keep out.
- **Censored rows are excluded from CRPS / log score / PIT** and reported
  separately as a predicted-survival diagnostic. Scoring a censored row against
  its observed value would reward an arm for predicting the horn.
- **Censoring in the fit, per arm.** Kaplan-Meier inside the cell
  (`empirical`); a `log S(c)` likelihood contribution (`lognormal`, `gamma`);
  native person-period truncation (`hazard`); and EXCLUSION with the count and
  the direction of bias stated (`lgbm_quantile` — LightGBM's quantile objective
  has no censored form, and imputing a censored row as complete is a
  known-wrong answer rather than an unknown one).
- **The hazard arm is fit on the full 38M person-period expansion, not a
  subsample.** The static block is constant within a possession, so its gradient
  is `X' rsum` over per-possession residual sums and the baseline's is a
  `bincount` over elapsed second; neither needs the expanded design matrix to be
  materialised. Subsampling would have made its noise floor incomparable to the
  others'.
- **`seconds_remaining_now` enters the hazard only through indicators.** Its
  linear part is an exact linear combination of the static `seconds_remaining`
  feature and the elapsed-time baseline, so adding it would be collinear by
  construction.
- **The tree arm's crossing quantiles are sorted per row.** A monotonicity
  repair, not a tuning knob, and stated so that the 7.5% zero-mass share is
  read as a property of nine-quantile inversion rather than of the repair.
- **The emergent test overrides only the clock.** Draw *j* of a half takes its
  whole feature row from the *j*-th REAL possession of that half — previous end
  type, score difference, bonus, which team has the ball, both teams' priors —
  and only `seconds_remaining`, `period` and their interaction come from the
  simulated clock. No event model and no score model is involved, so the count
  that comes out is a property of this model alone. When a half's real sequence
  runs out the wrap SKIPS index 0, because the first possession of a half is the
  only one whose previous-end type is `period_start` and injecting that state
  mid-half would be a state the game never reaches. 38% (`lognormal`, the arm
  that runs slow) to 64% (`lgbm_quantile`, the arm that runs fast) of the 10,638
  F2 half-games wrap by at least one possession; the count is reported per arm
  in `grid_results.csv`.
- **CBBD-completeness is defined by points reconciliation**, the definition
  `docs/tests/possessions_build_2026-09-10.md` section 2 already uses. Clock
  accounting was considered as a second filter and REJECTED: the signed
  deviation of a half's summed possession durations from 1,200 s is −0.75 s
  (0.06% of a half), while filtering at ±2 s would remove two thirds of the
  universe. It is reported as a diagnostic instead.
- **Nothing is adopted and no level correction is applied.** The obvious "fix" —
  scaling every drawn duration by 1.016 so the count lands on 68.13 — is
  precisely the post-hoc multiplier `CLAUDE.md` and `docs/SIM_GUARDRAILS.md`
  ban, and it would be fitted on the answer.

## 7. Consumption from the sim

**Nothing from this run should be wired into the engine.** The two arms the
next iteration will want to diff against are persisted with `adopted: False`:

| file | why it is kept |
|---|---|
| `reference_not_adopted_lgbm_quantile.pkl` | lowest F2 CRPS |
| `reference_not_adopted_empirical.pkl` | fewest PIT failures |

When an arm does pass the gates, the call shape is:

```python
import pickle
from cbb_sim.models import clock as ck

with open("data/processed/models/clock/winner.pkl", "rb") as fh:
    arm = pickle.load(fh)

# design_rows must come from ck.build_design (or from the sim's own lookup
# builder using the identical definitions in features.md) -- never by hand,
# because the ratings are centred on an as-of league mean the caller does not
# otherwise have.
pmf = arm.pmf(design_rows)                       # (n, 91) over integer seconds
draw = ck.sample_durations(arm, design_rows, seed=SEED, index=j)
```

- **RNG:** `sample_durations` and `chain_halves` both go through
  `cbb_sim.control.rng` keyed on `(seed, game_id, "clock")`, the engine-wide
  contract. A game's draws depend on nothing but that triple, so paired arms and
  paired seeds difference game by game and dropping a game from a run never
  moves another game's draws (`tests/test_clock.py`).
- **NaN handling:** `build_design` fills a missing rating with 0.0 (the league
  mean on a centred scale), a missing tempo ratio with 1.0 (the league mean on a
  ratio scale), and a missing tempo prior with the as-of league tempo mean. The
  sim must use the same fallbacks and never a training-set median.
- **Output shape:** `pmf` rows are non-negative and sum to 1 over the integer
  seconds 0..90.
- Per `CLAUDE.md`, the sim loop uses lookup tables, not live model calls: the
  deployment step for an adopted arm is to tabulate `pmf` over the discretised
  state grid, not to call it inside the possession loop. Every arm here is
  already a function of a small discrete-ish state, so that tabulation is cheap.

## 8. Artifacts

| Path | What it is |
|---|---|
| `data/processed/models/clock/design.parquet` | the possession-level design table every arm shares (2,607,192 rows) |
| `data/processed/models/clock/build_diagnostics.json` | every universe/exclusion count |
| `data/processed/models/clock/grid_results.csv` | one row per (fold, arm, feature set) |
| `data/processed/models/clock/pit_cells_F2.csv` | K-S of the PIT per state cell, every F2 arm, with the powered flag |
| `data/processed/models/clock/emergent_F2.csv` | the emergent-G1 summary per arm |
| `data/processed/models/clock/emergent_F2_by_month.csv` | the by-month G1 read per arm |
| `data/processed/models/clock/duration_by_terminal_F2.csv` | duration mean/SD by terminal event, actual vs model-implied |
| `data/processed/models/clock/noise_floor.json` | tree seed-varied refits and the game-block bootstrap |
| `data/processed/models/clock/verdict.json` | the decision rule applied, and what failed which gate |
| `data/processed/models/clock/diagnosis_by_clock_band_F2.csv` | D1 |
| `data/processed/models/clock/diagnosis_count_variance_F2.json` | D2 |
| `data/processed/models/clock/diagnosis_responsiveness_F2.csv` | D3 |
| `data/processed/models/clock/reference_not_adopted_{arm}.pkl` | the two reference arms, `adopted: False` |

Code: `src/cbb_sim/models/clock.py`, `scripts/train_clock_v1.py`,
`scripts/diag_clock_v1.py`, `tests/test_clock.py`.

## 9. Known gaps and followups

Ordered by how much of the failure each would close. All of them are
pre-registration candidates, not changes to make now.

1. **End-of-period state resolution is the blocking defect** (D1). Candidates:
   seconds-remaining buckets that are fine below 45 s (0-4 / 5-9 / ... / 40-44)
   for the empirical arm; a spline or a piecewise-linear basis in
   `seconds_remaining` for the parametric and hazard arms; or an explicit
   two-regime model (normal clock vs end-of-period). The tree arm already
   demonstrates that the information IS in the state — it reproduces the curve
   to within 0.8 s in every band — so this is a representation problem, not a
   data problem.
2. **Nine quantiles cannot carry the tail** (section 4.3). 7.0-7.6% of rows get
   zero predictive mass and the truncation costs 0.19-0.26 s of mean duration.
   Candidates: more quantile levels (19 or 39), an explicit parametric tail
   above q(0.9), or replacing quantile regression with a distributional tree
   objective. Any of these changes the arm's definition and needs its own
   pre-registration.
3. **A unimodal continuous family cannot represent the duration law**
   (section 4.2, confirmed in-sample). If a parametric arm is wanted at all, it
   has to be a mixture (a short-possession component plus a shot-clock
   component) or a discrete distribution fitted on the grid directly. The
   Gamma/log-normal pair is REFUTED as a predictive-distribution family for this
   target.
4. **The within-season pace slowdown is under-tracked** (section 4.1). Every
   arm's overshoot is smallest in November and largest in January. The
   `D_plus_season` block (`season_idx`, `days_since_start`) is in the grid and
   does not fix it — on the emergent test D is WORSE than C on every arm. L5
   (the learning) asks for a season-progress term or a recency structure; a
   linear day counter is evidently not it. Same open question as the L3
   season-drift row in the change ledger, and the two should be pre-registered
   together.
5. **The count SD's residual 0.2-0.3 possessions** (D2). A game-level pace
   random effect — one shared multiplicative draw per game applied to every
   possession's duration, the possession-level analogue of the Control's shared
   pace draw — is the obvious candidate and is exactly the kind of shared-shock
   structure L3/L10 says independent draws need. It must be pre-registered and
   baked off, not assumed, and it interacts with item 1 (some of the missing
   SD may be end-of-period behaviour rather than pace).
6. **The possession-vs-chance target ambiguity** (section 6). L3 is a
   chance-level model and L5 is a possession-level one, so the engine currently
   has no way to spend clock inside a possession that contains an offensive
   rebound (13.2% of possessions, 1.15 chances each). Whether the engine draws
   one duration per possession or one per chance is an architecture question
   that should be settled before the next L5 pre-registration.
7. **No overtime.** Every number here is regulation-only. The overtime model is
   a separate open item (`change_ledger.md`, Control OT stub) and the emergent
   G1 will need re-reading once it exists, because OT games are 5.6% of games
   and carry +8.2 possessions (L14).
8. **The `other` previous-end type is 0.5% of possessions and the empirical
   arm's only PIT failures.** It is the dead-ball / unlogged change of
   possession — the mismatch-guard residue. A cleaner segmentation of that
   state (or its removal by fixing the underlying feed gap) would likely make
   the empirical arm PIT-clean.
9. **CBBD feed incompleteness removes 14% of the 2022-2025 universe** and is
   concentrated in the two earliest training seasons. The filter is built here
   for the first time; the same flag should be pushed back into
   `games_universe` so every model uses one definition
   (`change_ledger.md` section D has the open row).

## 10. Round 2 (2026-09-10)

Pre-registration: `experiments.md` section 5. Trainer: `scripts/train_clock_v2.py`
(round 1's script and artifacts are untouched; round-2 artifacts are `v2_*`).
What changed: `seconds_remaining` entered as ten FINE buckets crossed with the
period type {first half, second half/OT} and the offence's score state
{trailing, tied, leading}, plus a last-shot (<= 30 s) and a two-for-one (30-45 s)
window; a two-regime arm was added; the end-of-half check was promoted from a
diagnostic to a GATE. Target, universe, folds, metrics, noise floor and decision
rules unchanged.

### 10.1 Verdict

**NO ARM ADOPTED.** 0 of 6 pass the emergent G1, 0 are PIT-clean, 0 pass the
end-of-half gate. Floor 0.006753. F1-only choices: tree parameters
`{num_leaves: 63, min_child_samples: 500}` (the complexity ladder spanned only
0.0024 of F1 CRPS, inside the floor, so round 1's default survived) and
**T = 45** for the two-regime arm (flat across {30, 45, 60, 90}).

### 10.2 The finer state representation works on the metric and backfires on the gate

F2 CRPS falls for every non-tree arm by 9-45x the floor — `empirical` 4.9307 ->
4.8698, `gamma` 5.0394 -> 4.9035, `hazard` 5.0294 -> 4.9505, `lognormal` 5.2002
-> 5.0590 — and the tree arm does not move (4.8082 -> 4.8091, inside the floor).
The EMERGENT COUNT gets WORSE on four of five round-1 arms: `empirical` +1.124
-> +1.669, `gamma` +0.720 -> +1.531, `hazard` +1.810 -> +2.598, `lgbm_quantile`
+2.190 -> +2.525; only `lognormal` improves, to -1.334.

**Why, and this is the round-3 finding.** The conditional law of duration given
"7 seconds left" is short because it is TRUNCATED BY THE HORN in the training
data — 50.5% of possessions with under 10 s left consume all of it. A model that
learns that law draws short there; the sim then has 4 s left, draws shorter
again, and subdivides the tail of the half into possessions the game never
played. Round 1's coarse 0-34 s bucket hid this by being wrong in the other
direction. Round 2 made the conditional mean right and the count worse, which is
exactly the outcome a metric-only read would have missed.

### 10.3 The end-of-half gate, and the feed-truncation finding

The pre-registered actual value (0.8832 of halves starting their last possession
under 35 s) is contaminated: the CBBD stream stops before the horn in a third of
halves, those halves' last LOGGED possession has a mean END clock of 20.4 s, and
only 44.6% of 2025 halves end at exactly 0:00. On the clock-complete halves
(63.5% of 2025) the actual share is **0.9556** and `gamma` reaches 0.9465 (gap
**-0.009**) and `hazard` 0.9588 (gap **+0.003**) — inside the 0.00634 floor. So
round 2 DID fix the end-of-half SHARE. The gate still reads FAIL because a gate
is not re-based after the fact, and both readings are reported
(`experiments.md` section 6.3).

The DURATION half of the check survives the correction and is the real defect:
-2.53 s (`lognormal`) to -6.29 s (`lgbm_quantile`) against a 0.194 s floor, on
clock-complete halves. Same mechanism as 10.2.

### 10.4 What still passes

Responsiveness, on every arm: 4 of 4 monotone quintile steps against the
pregame tempo prior, slope ratios 0.78 / 0.95 / 1.04 / 1.00 / 1.05 / 1.03
(empirical / lognormal / gamma / hazard / lgbm / two_regime). And the count SD
is now INSIDE the +/- 0.75 tolerance on three arms (`gamma` -0.194, `hazard`
-0.248, `lognormal` +0.259). Neither dispersion nor matchup signal is the
blocker.

### 10.5 What round 3 must change

1. **Treat every possession that ends at the horn as right-censored**, not just
   the 0.2% whose terminal event is literally `end_period`. About 50% of
   possessions with under 10 s left consume all of it; those are observations of
   "at least this long", not "exactly this long". Fitting the INTENDED duration
   and letting the engine truncate it at the horn is the structural fix for
   10.2, and every arm here already has a censored likelihood for it.
2. **The two-regime idea is refuted as specified** (10.1): its behaviour is
   dominated by the upper regime and T is flat. A regime split helps only if the
   lower regime models intended duration rather than observed.
3. **A unimodal continuous family is still refuted** (PIT: `gamma` 19/27,
   `hazard` 22/27, `lognormal` 27/27 leak-sized cells). The state
   representation was never the reason.
4. **The nine-quantile tail is still the tree arm's ceiling** (7.4-7.5% of rows
   with zero predictive mass).
5. **Fix the grading truth before re-reading the end-of-half gate**: a
   CBBD-side clock-completeness flag belongs in `games_universe` so the actual
   statistic is computed on halves that reach 0:00.
