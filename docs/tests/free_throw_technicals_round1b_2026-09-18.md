# Free-throw technicals round 1b: target adjudication + offline results (2026-09-18)

Lane: `docs/models/free_throw/`. Pre-registration (target adjudication + round
1b arms) committed and pushed BEFORE this script ran:
`docs/models/free_throw/experiments.md` section 10 (commit `b8d20c1`). Full
adjudication detail lives there and is not repeated in full here; this doc
covers the run itself. **OFFLINE ONLY**, same concurrency conditions as round
1 (other lanes running closed loops on this tree under the 2-worker cap): no
engine wiring, no closed-loop run this round.

Scripts: `scripts/build_ft_technical_target_v1.py` (job step 1, target
adjudication), `scripts/build_ft_technical_target_v2_verified.py` (the
verified target artifact), `scripts/grade_ft_technical_round1b_v1.py` (job
step 3, one blind grading script). Outputs: `data/processed/models/
free_throw/technical_target_{hoopr,cbbd_raw,reconciliation,verified,
verified_trips}_v1.parquet` (tracked, all well under 20MB), `results/
free_throw_technicals/{target_adjudication,round1b_offline}.json`
(gitignored). Wall clock: target build ~65s (v1) + ~27s (v2, both re-runs
included) + 54s (grading, both folds) = **under 3 minutes of actual model/
scan compute**, plus the exploratory/investigative work (hand-reading pbp
transcripts, iterating on the scan algorithm) that is not separately timed.
All runs on `.venv/Scripts/python.exe`, `PYTHONIOENCODING=utf-8`, thread env
vars pinned to 1, one process at a time (well under the 2-process cap; no
process started or signalled that this lane did not start itself).

---

## 1. Target adjudication (job step 1) -- summary

Full narrative, the hand-read transcripts, and the exact mechanism are in
`experiments.md` section 10.1 (committed before this run). Headline numbers,
reproduced here for a self-contained results doc:

**The three-way comparison, at the level it actually resolves:**

| source | what it is | 2022 | 2023 | 2024 | 2025 |
|---|---|---:|---:|---:|---:|
| hoopR raw-pbp same-clock scan (2nd source) | independent vendor, attempts | 2,411 | 3,285 | 2,666 | 2,420 |
| CBBD raw-pbp same-clock scan (NEW, not `trips_v1_era`) | same vendor as the training target, raw feed | 2,411 | 3,285 | 2,666 | 2,435 |
| `trips_v1_era` (round 1's target) | CBBD, through the `possessions.py` pipeline | 1,804 | 2,439 | 2,052 | 1,959 |
| box-implied (`box_fta - ev_fta`, the job's suggested 3rd estimate) | CBBD box vs CBBD event layer | 1,654 | 2,487 | 2,059 | 2,044 |

**Finding.** The two VENDORS (hoopR, CBBD) agree on raw technical-attempt
incidence to within 0-0.6% every season -- round 1's 64-73% "vendor gap" was
never a vendor gap. It is `possessions.py::_handle_technical`'s one-row
free-throw lookahead being defeated by an administrative row CBBD's feed
routinely inserts between a `Technical Foul` event and its free throws (an
automated census of the row immediately following the event in all 1,228
affected team-games: 89% `Lost Ball Turnover`, 10% `PersonalFoul`; confirmed
by hand-reading 15 full pbp transcripts, each showing an exact `(game,
period, clock, team_id, opp_id)` match between the raw scan and a
`trips_v1_era` row mis-tagged `foul_class in {"foul","none"}`). Box-implied
is confounded by the SAME bug in the opposite direction (the mis-tagged
attempts inflate `ev_fta`'s non-technical bucket) and is rejected as an
adjudication source -- it is a residual of two entangled biases, not a clean
third estimate, despite being the job's own suggested construction.

**Verified target**: a moment-level union of the CBBD raw scan and
`trips_v1_era`'s own trips (each resolves a small, disjoint edge case the
other misses -- section 10.1), cross-validated against hoopR's independent
scan to a quantified, fully-explained residual of **8-10% every season**
(the double/multi-technical-same-clock scanning limitation, shared
symmetrically by both vendors' own same-clock scans, plus one hand-confirmed
hoopR sequence-ordering anomaly in 11 of 11,179 2025 team-games). This
residual is reported, not patched (`CLAUDE.md` 9.8): it is materially
smaller and, unlike round 1's target, its cause is named.

| season | verified trips | verified FTA | 2nd-source (hoopR) FTA | residual |
|---|---:|---:|---:|---:|
| 2022 | 1,281 | 2,606 | 2,411 | +8.1% |
| 2023 | 2,082 | 3,610 | 3,285 | +9.9% |
| 2024 | 1,342 | 2,907 | 2,666 | +9.0% |
| 2025 | 1,204 | 2,644 | 2,420 | +9.3% |

**A newly-discovered downstream contamination, flagged not fixed**: the same
`possessions.py::_handle_technical` bug LEAKS technical attempts into
`attempts_v1_era.parquet` (FT-2's training universe) tagged as ordinary
trips, diluting FT-2's training population with coach-selected-shooter
attempts (make rate ~0.80) mislabelled as ordinary shooting/bonus attempts
(make rate ~0.68-0.72). Not corrected here -- this lane may not touch
`src/cbb_sim/` or retrain FT-2; flagged for the engine-core owner alongside
the `_handle_technical` fix itself.

---

## 2. Round 1b blind table (both folds, verified target)

One script (`grade_ft_technical_round1b_v1.py`) scores every arm identically,
importing round 1's own helpers (`exp_free_throw_technicals_v1.py`) rather
than re-deriving them, so the ONLY thing that changed from round 1 is the
target and the arm list (section 10.2).

**F2 (train 2022-2024, test 2025) -- SELECTION**

| arm | predicted trips | actual trips | predicted rate | actual rate | pooled deviance | level-calib rel. gap | eligibility 2 (|gap|<=10%) | team-keyed | eligibility 1 (responsiveness) |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| X0 (reference) | 0.0 | 1,204 | 0 | 0.0006789 | 81,208 (degenerate) | -100.0% | FAIL | no | FAIL (flat by construction) |
| X1 (league constant) | 1,609.1 | 1,204 | 0.0009073 | 0.0006789 | 111.8 | +33.6% | FAIL | no | FAIL (flat by construction) |
| X5 (team as-of, shrunk to X1) | 1,582.5 | 1,204 | 0.0008923 | 0.0006789 | 98.8 | +31.4% | FAIL | **yes** | **PASS** (slope 1.187, 4/4) |
| X6 (recency-weighted window) | 1,609.1 | 1,204 | 0.0009073 | 0.0006789 | 111.8 | +33.6% | FAIL | no | not attempted (structurally ineligible) |
| X5r (team as-of, shrunk to X6) | 1,582.5 | 1,204 | 0.0008923 | 0.0006789 | 98.8 | +31.4% | FAIL | **yes** | **PASS** (slope 1.187, 4/4) |
| X7 (league in-season as-of x site) | 1,221.9 | 1,204 | 0.0006890 | 0.0006789 | **0.26** | **+1.5%** | **PASS** | no | not attempted (structurally ineligible) |

**F1 (train 2022-2023, test 2024)**

| arm | predicted trips | actual trips | predicted rate | actual rate | pooled deviance | level-calib rel. gap | eligibility 2 | team-keyed | eligibility 1 |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| X0 | 0.0 | 1,342 | 0 | 0.0007568 | 90,808 | -100.0% | FAIL | no | FAIL |
| X1 | 1,747.5 | 1,342 | 0.0009855 | 0.0007568 | 102.4 | +30.2% | FAIL | no | FAIL |
| X5 | 1,723.2 | 1,342 | 0.0009718 | 0.0007568 | 91.3 | +28.4% | FAIL | **yes** | **PASS** (slope 0.929, 4/4) |
| X6 | 1,780.0 | 1,342 | 0.0010038 | 0.0007568 | 117.9 | +32.6% | FAIL | no | not attempted |
| X5r | 1,742.0 | 1,342 | 0.0009824 | 0.0007568 | 99.8 | +29.8% | FAIL | **yes** | **PASS** (slope 0.929, 4/4) |
| X7 | 1,488.3 | 1,342 | 0.0008393 | 0.0007568 | 14.9 | +10.9% | **FAIL** (just over 10%) | no | not attempted |

**Noise floor** (200-replicate game-block bootstrap on the pooled rate,
2 seeds): F1 = 1.932e-05, F2 = 1.478e-05. Two further orders of magnitude
below the 10% level-calibration threshold and below every deviance gap
between arms above -- not the binding constraint on this round's decision.

**Reading.** `X6` (recency-weighted training window) essentially reproduces
`X1` on both folds (predicted rate identical to 4 decimal places on F2): the
grid search's own internal validation (F2: predicting 2024 from
2022-weighted-vs-2023-weighted) found UNIFORM pooling beats every
recency-weighting candidate, because the anomalous season (2023, still the
highest-rate season on the verified target too: 2,082 trips vs 1,281 / 1,342
/ 1,204 elsewhere) sits in the MIDDLE of the training window, so weighting it
MORE heavily (what recency-weighting does) makes the fit worse, not better --
the reverse of the mechanism round 1 hypothesized. `X7` (league in-season
as-of rate) is the only arm that materially closes the level gap (33.6% ->
1.5% on F2), but this is largely definitional: an in-season cumulative
average necessarily converges toward that season's own realised rate as the
season progresses, and the pooled deviance metric is dominated by the
better-informed, later-season rows -- reported plainly rather than oversold
as "skill" `X5`/`X1` lack. `X5`/`X5r` reproduce the team-level responsiveness
`X5` already showed in round 1 (POWERED: quintile power **6.93 sigma** on F2,
**9.76 sigma** on F1, both well above round 1's cited ~5.5 sigma, since the
verified target's larger counts buy more power) -- but neither closes the
level gap at all (X5r vs X5: 31.4% vs 31.4% on F2, 29.8% vs 28.4% on F1 --
shrinking toward X6 instead of X1 changes essentially nothing, because X6
barely differs from X1, per the paragraph above).

**No arm clears BOTH pre-registered eligibility lines on either fold.**
`X0`/`X1`/`X6` fail responsiveness by construction (flat); `X7` is not
team-keyed and is pre-declared structurally ineligible for adoption
regardless of its (fold-1-inconsistent: PASS on F2, FAIL on F1) level score;
`X5`/`X5r` pass responsiveness but fail level calibration by ~28-34% on both
folds -- essentially unchanged from round 1's target despite using the fully
adjudicated, verified count. **NOTHING IS ELIGIBLE. NOTHING IS ADOPTED.**
This is the pre-registered legitimate outcome (section 10.2, decision rule),
not a failure to reach one.

**What this round's negative result actually shows**: fixing the TARGET
(section 1) did not, by itself, fix round 1's tension, and neither did
adding a recency-weighted prior (which turned out not to help, for a
specific, understood reason) or an in-season-adaptive league rate (which
helps the level but cannot be team-keyed by construction). The one
combination that could plausibly clear both lines at once --
`X5i`, a team-as-of rate shrunk toward `X7`'s in-season-adaptive level
instead of a fixed pre-season prior -- was explicitly named and explicitly
NOT attempted this round (section 10.2, a stated scope limit), and is the
natural next candidate.

---

## 3. Multi-level evidence (F2, `X1` predicted rate shown; game/margin/site
cuts are on the OFFENDER's own state, matching round 1's convention)

| cut | actual rate | X1 predicted | note |
|---|---:|---:|---|
| game_phase H1_early | 0.000308 | 0.000907 | powered (133) |
| H1_late | 0.000698 | 0.000907 | powered (302) |
| H2_early | 0.000788 | 0.000907 | powered (347) |
| H2_late_OT | 0.000900 | 0.000907 | powered (422) |
| margin close | 0.000507 | 0.000907 | powered (365) |
| leading | 0.000693 | 0.000907 | powered (365) |
| trailing | 0.000900 | 0.000907 | powered (474) |
| site home | 0.000608 | 0.000907 | powered (470) |
| away | 0.000767 | 0.000907 | powered (593) |
| neutral | 0.000623 | 0.000907 | powered (141) |

Same directional shape as round 1 on the un-verified target (trailing >
leading > close; away > home; second half hotter than first) -- the
verified target does not change WHICH real effects exist, only the level.
No cell is underpowered this round (counts 133-593, all above the
`n>=20`/`n>=30` conventions used elsewhere in this lane).

### 3.1 Team-level quintile responsiveness (the power calculation)

F2: quintiles built on 2024 prior-rate, graded on 2025 (verified target),
`n=72-73` teams/quintile, `>=500` prior-season chances required:

| quintile | prior rate | actual 2025 rate | Poisson SE | X1 | X5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.000115 | 0.000489 | 0.0000443 | 0.000907 | 0.000687 |
| 2 | 0.000386 | 0.000586 | 0.0000442 | 0.000907 | 0.000755 |
| 3 | 0.000613 | 0.000563 | 0.0000443 | 0.000907 | 0.000869 |
| 4 | 0.000918 | 0.000714 | 0.0000433 | 0.000907 | 0.000936 |
| 5 | 0.001579 | 0.000919 | 0.0000433 | 0.000907 | 0.001197 |

Q1-to-Q5 actual gap 0.000430, combined end-quintile SE 0.0000620 -> **6.93
sigma**, powered. `X1` flat by construction (slope 0 by definition, not by a
noisy monotone count this round -- section 2 above). `X5`/`X5r` monotone 4/4,
slope ratio 1.187 (F2) / 0.929 (F1) -- inside or near the responsiveness
band both folds, the same real, two-fold-replicated persistence round 1
found, now on a verified target.

---

## 4. X3 shooter-rule variants (train/test-safe -- round 1's own version
pooled all four seasons including the test season, which is leakage for a
per-fold decision; fixed here)

| fold | actual test shooter rate | X3-avg gap (pp) | X3-best gap (pp) | X3-blend gap (pp), fitted f |
|---|---:|---:|---:|---:|
| F1 | 0.7931 | -8.40 | +6.87 | **-0.29**, f=0.531 (fit on 2022-2023 only) |
| F2 | 0.8101 | -9.41 | +5.83 | **-1.58**, f=0.514 (fit on 2022-2024 only) |

`X3-blend` (a linear interpolation between the team's average and best as-of
shooter, with the blend fraction FITTED out-of-sample per fold, never on the
test season) is the clear best-supported shooter rule on both folds by a
wide margin -- consistent with round 1's finding (chosen shooter sits ~59%
of the way from average to best) but now leakage-free and fold-specific
(f=0.51-0.53, close to round 1's pooled 0.59). This axis is orthogonal to
the rate arms above (the technical is additive, section 9.9.1) and is
reported as informative evidence, not gated by the round's eligibility
lines (no arm above was adopted for the rate to pair it with).

---

## 5. Decision

Applying the pre-registered decision rule (section 10.2) mechanically: no
arm is both team-keyed and passing both eligibility lines, on either fold.
**NO ARM IS ADOPTED.** No served default changes; no engine file was read
for editing or touched; 2025-26 stays sealed throughout (`assert_not_sealed`
called on every load in both scripts).

Ledger: see `docs/models/change_ledger.md` for whether this constitutes a
status change worth a new row (re-read immediately before appending, per
job discipline).
