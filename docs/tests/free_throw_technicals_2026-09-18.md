# Free-throw technicals round 1: offline results (2026-09-18)

Lane: `docs/models/free_throw/`. Runs `experiments.md` section 9 (PROPOSED,
committed 2026-09-18 by the G4 diagnostic lane) as amended by section 9.9
(PM conditions, committed **before this script ran**, commit `04302c7`).
**OFFLINE ONLY.** No engine flag was wired, no served default changed, no
paired closed-loop simulation was run this round -- see section 6 for why,
stated in advance as permitted ("if time does not allow, stop at the offline
table and say so").

Script: `scripts/exp_free_throw_technicals_v1.py`. Output:
`results/free_throw_technicals/round1_offline.json` (gitignored, per
CLAUDE.md bulk-data rule). Wall clock: four runs while the script was being
built and extended (52 s, then two iterations adding `X2` and the per-team
responsiveness check, final run 73 s), all on `.venv/Scripts/python.exe`,
`PYTHONIOENCODING=utf-8`, thread env vars pinned to 1, single process (well
under the 2-process cap; no other process was started or signalled).

---

## 1. Measurement confirmation (job step 1a)

Full detail is in `experiments.md` section 9.9.1 (committed before any
modelling). Summary:

- **Identification.** Both feeds use one flat category, `"Technical Foul"`
  (CBBD `playType`, hoopR `type_text`), with no flagrant/intentional/
  administrative/unsporting subtype in either feed (confirmed directly on
  hoopR's 2025 `type_text` values here, extending the existing CBBD-side
  finding in `ft_trip_reconciliation_2026-09-10.md`). hoopR's free text
  distinguishes team/bench technicals (`"Technical Foul on <Team>."`) from
  individual ones (`"Technical Foul on <Player>."`); this is reported, not
  used by any arm.
- **Box includes technicals; the box-vs-pbp gap is already quantified and is
  re-cited, not re-derived**: `ft_trip_reconciliation_2026-09-10.md` sections
  1-2, netting technical FTA out of the CBBD event layer closes 94.3% of all
  disagreeing team-games EXACTLY and 96-109% of the season aggregate gap,
  every season 2022-2025.
- **New cross-source check.** hoopR's own pbp technical-foul events, netted
  of offsetting simultaneous (one-per-team, no-FT) pairs, imply 1,211-2,115
  "should-produce-a-trip" moments a season against CBBD's own derived
  879-1,355 technical trips -- CBBD's own trip table (the training target for
  every arm below) sits at **64-73%** of hoopR's implied count, every season,
  rising but not closing (2022 0.680 -> 2025 0.726). **Checked for a
  clock-precision artefact and ruled out**: collapsing hoopR's technical
  events within a 0/3/5/10-second same-team-or-not tolerance window moves the
  2025 single-team-moment count only 1211 -> 1195 -> 1193 -> 1185, 2% at
  most, nowhere near enough to explain a 27-36% gap. The mechanism (CBBD raw
  feed under-logging technicals vendor-to-vendor vs. some hoopR-side
  overcount not yet identified) is **not adjudicated by this round** and is
  flagged for a future one, not corrected here (9.8 bans re-scaling the rate
  to hit a target).
- **No era boundary** in technical trip length 2022-2026 (2-attempt is modal,
  54-66% every season); 2023's outlier 1-attempt share (522 vs 73-202 other
  seasons) is flagged as a feed-completeness wrinkle, not a rule change.
  `FT.TRIP_RULES["technical"]` is unchanged.
- **Possession is retained**, confirmed by reading `possessions.py` (not
  assumed): a technical trip never opens, closes or advances a `_Chance`.
  Every arm below is expressed as an ADDITIVE rate that adds free throws
  without consuming a simulated chance or changing who is next to inbound.
- **Who shoots has data support**: `trips_v1_era.parquet` carries
  `shooter_id` on technical rows; section 4 below uses it directly.

---

## 2. The blind table (all five arms, both folds)

One script scores every arm identically (`score_predicted_vs_actual` in
`exp_free_throw_technicals_v1.py`); nothing here differs by arm except the
predicted rate function. Primary offline quantity: predicted vs actual
technical-trip count on the test fold's real chance-level exposure, and the
pooled Poisson deviance of that one aggregate comparison (**not** a
per-attempt log loss -- section 9.4 already establishes that likelihood
comparison is degenerate for `X0`).

**F2 (train 2022-2024, test 2025) -- SELECTION**

| arm | predicted trips | actual trips | predicted rate/chance | actual rate/chance | pooled deviance |
|---|---:|---:|---:|---:|---:|
| X0 (reference) | 0.0 | 879 | 0 | 0.0004956 | 58734.4 (degenerate) |
| X1 (league-constant, per-chance) | 1098.2 | 879 | 0.0006192 | 0.0004956 | 46.98 |
| X2 (pre-game: conf-game x site) | 1094.9 | 879 | 0.0006174 | 0.0004956 | 45.70 |
| X4 (in-game: phase x margin x site) | 1094.3 | 879 | 0.0006170 | 0.0004956 | 45.44 |
| X5 (team as-of, EB-shrunk) | 1102.1 | 879 | 0.0006214 | 0.0004956 | 48.57 |

**F1 (train 2022-2023, test 2024)**

| arm | predicted trips | actual trips | predicted rate/chance | actual rate/chance | pooled deviance |
|---|---:|---:|---:|---:|---:|
| X0 | 0.0 | 954 | 0 | 0.0005380 | 63902.1 |
| X1 | 1172.8 | 954 | 0.0006614 | 0.0005380 | 43.63 |
| X2 | 1170.2 | 954 | 0.0006599 | 0.0005380 | 42.64 |
| X4 | 1167.4 | 954 | 0.0006584 | 0.0005380 | 41.63 |
| X5 | 1176.2 | 954 | 0.0006633 | 0.0005380 | 44.89 |

**Reading.** `X0` is degenerate on both folds, as pre-declared. `X1/X2/X4/X5`
sit within 1-3 deviance points of each other on numbers of scale 42-49, on
BOTH folds -- not distinguishable by this metric. All four **overshoot** the
test season's own realised rate by 22-25%, in both folds, because the pooled
training window (F2: 2022-2024) is inflated by 2023's anomalous spike
(1,355 trips against 879-1,045 every other season). `X2`'s internal-validation
shrinkage search (grid 0 to 1e7 pseudo-exposures) picked the GRID MAXIMUM on
both folds -- its two conditioning cells (conference-game x site) carry no
measurable signal at all and it collapses fully to `X1`. `X4`'s search
plateaus around k=51,200 (a shrinkage strength far larger than any single
cell's own exposure) with only a ~6% relative deviance gain over k=0 on both
folds -- a small, non-conclusive edge. `X5`'s search plateaus around
k=12,800-51,200. **No conditioning axis beats `X1` by more than the
noise floor on this aggregate metric, on either fold.**

**Noise floor** (game-level block bootstrap SE on the pooled per-chance rate,
200 replicates, two seeds -- this project's standing offline convention,
since no closed-loop paired-seed run exists this round):

| fold | seed 0 | seed 1 | floor used |
|---|---:|---:|---:|
| F1 | 1.575e-05 | 1.621e-05 | 1.621e-05 |
| F2 | 1.132e-05 | 1.232e-05 | 1.232e-05 |

In pp-of-FTA/FGA terms this floor is about **0.008 pp** -- two orders of
magnitude below the ~0.32-0.38 pp closure any positive-rate arm produces
(section 5), so rule 9.7.1 eligibility is clear for `X1/X2/X4/X5` on both
folds; only `X0` fails it (by construction).

---

## 3. Multi-level evidence (F2, test season 2025; X1 shown, X2/X4/X5 track it
within noise everywhere below unless stated)

| cut | actual rate | X1 predicted rate | note |
|---|---:|---:|---|
| **game_phase** H1_early (min 0-9) | 0.000246 | 0.000619 | powered (106 trips) |
| H1_late (10-19) | 0.000513 | 0.000619 | powered (222) |
| H2_early (20-29) | 0.000547 | 0.000619 | powered (241) |
| H2_late_OT (30-40+OT) | 0.000661 | 0.000619 | powered (310) |
| **margin_bucket** trailing (<=-5) | 0.000684 | 0.000619 | powered (360) |
| close (-4..4) | 0.000362 | 0.000619 | powered (261) |
| leading (>=5) | 0.000490 | 0.000619 | powered (258) |
| **site** home | 0.000449 | 0.000619 | powered (347) |
| away | 0.000551 | 0.000619 | powered (426) |
| neutral | 0.000469 | 0.000619 | powered (106) |
| **conference** non-conf | 0.000516 | 0.000619 | powered (336) |
| conf | 0.000484 | 0.000619 | powered (543) |
| **month** Nov | 0.000503 | 0.000619 | powered (189) |
| Dec | 0.000522 | 0.000619 | powered (150) |
| Jan | 0.000425 | 0.000619 | powered (188) |
| Feb | 0.000529 | 0.000619 | powered (223) |
| Mar | 0.000504 | 0.000619 | powered (121) |
| Apr | 0.001466 | 0.000619 | **UNDERPOWERED** (n=8) |

Real, directionally sensible variation exists (trailing teams commit more
technicals than leading or close teams; away teams draw more than home,
consistent with the possession-outcome G4 finding that the home whistle is
under-reproduced elsewhere in this cascade; the second half runs hotter than
the first) but **none of it survived `X4`'s own internal-validation search**
(section 2) -- the per-cell counts here (100-540) look powered in isolation,
but the search is choosing among ~36 cells at once, most of which are far
thinner than these one-way marginals, and the search's own held-out deviance
says the safest bet is still the pooled constant. This is reported as
evidence, not smoothed into "no signal" -- see the quintile section below,
where the SAME kind of aggregation (pooling many thin cells into 5 buckets)
finds something `X4`'s finer 36-cell grid could not.

### 3.1 Team-level quintile responsiveness -- the power calculation, not just a label

Teams bucketed by 2024 (prior-season) technical rate, >=500 chances that
season, graded on 2025:

| quintile | teams | prior rate | actual 2025 rate | Poisson SE (rate) | X1 predicted | X5 predicted |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 73 | 0.000050 | 0.000380 | 0.000038 | 0.000619 | 0.000482 |
| 2 | 72 | 0.000252 | 0.000473 | 0.000038 | 0.000619 | 0.000516 |
| 3 | 72 | 0.000447 | 0.000419 | 0.000038 | 0.000619 | 0.000588 |
| 4 | 72 | 0.000698 | 0.000548 | 0.000037 | 0.000619 | 0.000677 |
| 5 | 73 | 0.001210 | 0.000644 | 0.000037 | 0.000619 | 0.000835 |

Expected trips per quintile-season at the league rate are 172-180, each with
a Poisson SE of ~13 trips (~7.4% relative) -- **this cut is POWERED**, not the
underpowered case the pre-registration warned about (that case is the
per-cell 36-way `X4` grid and the raw per-team `X5` cells, both of which
individually have single-digit expected counts; the quintile AGGREGATION is
what buys the power). Q1-to-Q5 actual rates rise 0.000380 -> 0.000644, a real,
gap of ~2.7x the combined SE of the two end quintiles (~5.5 sigma) -- **real,
powered, team-level persistence in technical-foul incidence exists in the
data.**

**`X1` is flat by construction (slope ratio 0.0) and fails this responsiveness
check on its face, the same way any pure league-constant arm does everywhere
else in this project.** `X5` (heavily shrunk in section 2's aggregate
deviance search) nonetheless **reproduces the shape**: monotone in 4 of 4
steps on BOTH folds (F2: 0.000482/0.000516/0.000588/0.000677/0.000835, slope
ratio **1.336**; F1: 0.000484/0.000558/0.000634/0.000707/0.000924, slope
ratio **0.893**) -- a genuine, two-fold-replicated matchup-specific signal
that survives despite `X5` scoring marginally worst on the AGGREGATE deviance
in section 2. `X2` and `X4` are not team-keyed and this specific check is not
natural for them (9.9.4/9.9.3 restricts it to `X0`/`X1`/`X5`; not attempted
for `X2`/`X4`, stated rather than silently skipped).

**This is a tension the decision rule as written does not resolve cleanly**
(section 6).

---

## 4. Who shoots (X1 vs X3), n = 4,090 technical trips 2022-2025 with a
resolvable shooter and team context

| quantity | value |
|---|---:|
| Actual technical shooter's own as-of FT rate (mean) | 0.7982 |
| Team's attempt-share-weighted average shooter that game (`X1`'s assumption) | 0.7084 |
| Team's BEST as-of shooter that game (`X3`'s assumption) | 0.8616 |
| Observed technical make rate | 0.7956 |
| Observed non-technical make rate | 0.7180 |

The actual chosen shooter sits **59% of the way from the team average to the
team's best shooter**, closer to `X3`'s assumption than to `X1`'s (|0.7982 -
0.8616| = 0.0634 vs |0.7982 - 0.7084| = 0.0898), and the as-of-rate machinery
already fitted for FT-2 explains essentially all of the elevated technical
make rate (0.798 predicted vs 0.796 observed) purely through who is
selected to shoot. **`X3` (best-shooter rule) is the better-supported
assumption, though neither arm is exactly right** -- the true selection is
between "average" and "best", not at either pole.

---

## 5. Offline analytic FTA/FGA closure (the pre-registered primary, section
9.4, approximated without a closed-loop run -- section 6 explains why)

Because a technical trip is additive and never touches an existing chance
(section 1), its FTA/FGA contribution can be computed in closed form from
already-measured quantities without re-simulating a game: predicted trips
per team-game x average attempts per technical trip (2.229, 2025) gives
delta FTA/team-game; FGA is untouched by construction.

**Real/box-consistent** (test-fold's own realised chance exposure, 158.55
chances/team-game 2025, and box FGA/team-game 58.006):

| arm | delta FTA/team-game | delta pp FTA/FGA | % of the -0.282 pp technical channel |
|---|---:|---:|---:|
| X1 | 0.2188 | +0.377 | 134% |
| X2 | 0.2182 | +0.376 | 133% |
| X4 | 0.2180 | +0.376 | 133% |
| X5 | 0.2196 | +0.379 | 134% |

**Sim-consistent** (layered onto the SERVED v5b F2/2025 75-seed run's own
FGA/FTA/possession averages, `results/engine_v0/F2_2025_s200_v5b_A`, using
2x its `possessions` column as a chance-exposure proxy -- an approximation,
since the served run stores true end-to-end possessions, not the finer
`_Chance` count this rate was fit against, so this number likely
UNDERSTATES what a correctly chance-keyed wiring would produce):

| arm | delta FTA/team-game | delta pp FTA/FGA | % of -0.282 pp channel | % of -1.229 pp total gap |
|---|---:|---:|---:|---:|
| X1 | 0.1928 | +0.325 | 115% | 26.5% |
| X2 | 0.1923 | +0.324 | 115% | 26.4% |
| X4 | 0.1921 | +0.324 | 115% | 26.4% |
| X5 | 0.1935 | +0.326 | 116% | 26.6% |

All four arms give essentially the same answer (section 2's finding
restated): **the offline projection closes 115-134% of the technical channel
and roughly 26-27% of the FTA/FGA total gap**, i.e. slightly MORE than the
channel's own 23.0% pre-registered size. **This is not read as evidence the
arm is well-calibrated.** It is the net of two unrelated, imprecisely-known
biases pointing opposite ways: the pooled training rate overshoots 2025's
own realised rate by ~25% (section 2, the 2023-anomaly effect), while CBBD's
own trip table -- the training target -- sits at 64-73% of what hoopR's pbp
implies the true incidence is (section 1). These two numbers are similar
enough in magnitude to net out near 100% by coincidence; neither is
corrected here (9.8), and a different training window or a resolved
hoopR/CBBD reconciliation could easily move this net closure well off 100%
in either direction. A secondary points effect follows the same arithmetic:
+0.15-0.17 points/team-game from technical makes (using the observed 0.7956
make rate), i.e. roughly +0.30-0.35 points a game combined -- a small, one-
directional addition, not evaluated against G9 this round because that
requires an actual sim run.

---

## 6. Parity / closed loop: NOT RUN this round

Per the job's own permission ("if time does not allow, stop at the offline
table and say so"), this round stops here. Reasons, stated rather than
assumed: (1) wiring a technical-FT rule into `src/cbb_sim/engine/loop.py`
and `GameState` needs a new event/chance-retention path (section 1's
possession-retention finding is a real constraint on that wiring, not a
detail) and a genuinely new lookup-table export -- a non-trivial, first-time
change to shared engine files; (2) the concurrency notice for this task
names possession-outcome, rebound, late-game and rotation lanes as actively
running closed loops on the SAME working tree RIGHT NOW, with a hard 2-worker
compute cap already in effect; stacking a new engine-file change and a
500-game x 25-seed run on top of that is the kind of load the worker
discipline rules exist to prevent. No engine file was read for editing and
none was touched. `docs/ops/parity_reference_windows_v6.json` was not
exercised. This is a deviation from job step 2, made explicitly rather than
silently, and the PM can schedule the wiring + closed-loop step separately
once the shared tree is less loaded.

---

## 7. Recommended verdict (offline only; the PM decides)

Applying the pre-registered decision rule (9.7) and its extended tie-break
(9.9.3, `X0 < X1 < X3 < X2 < X4 < X5`) MECHANICALLY: `X0` fails eligibility
(rule 1); `X1/X2/X4/X5` are statistically tied on the primary (section 2,
differences inside a few deviance points on numbers that size, both folds);
ties go to the simplest, so the rule selects **`X1`** (paired with `X3`'s
shooter rule per section 4's evidence, since `X1` and `X3` share one rate by
construction and only differ on who shoots).

**Caveat the PM should weigh before ratifying that mechanically:** `X1` is a
pure league constant and is therefore PROVABLY, PERMANENTLY flat on the
team-level responsiveness check (section 3.1) -- the same category of defect
`ARCHITECTURE_DECISIONS.md` Decision 8 was written to stop an aggregate gate
from quietly passing elsewhere in this cascade. That check is POWERED
(section 3.1, ~5.5 sigma end-to-end) and `X5` reproduces its shape on BOTH
folds despite being statistically indistinguishable from `X1` on the
aggregate closure metric. Rule 9.7's tie-break was written before this
tension was visible and does not instruct which axis wins when they
disagree. This round does not resolve it -- it hands the PM both readings
side by side, as instructed.

Either way, no default changes from this round: **NOTHING IS ADOPTED.**
