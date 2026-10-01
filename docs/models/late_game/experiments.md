# L7 LATE-GAME REGIME -- experiments

Append-only. Status changes go to `docs/models/change_ledger.md` in the same
commit.

---

## 1. PROPOSED -- Round 1 pre-registration: does the engine need an end-game regime, and where does it belong? (written 2026-09-11 by the late-game lane, BEFORE any modelling; NOT RUN, NOT ADOPTED)

**Nothing below has been fitted, graded or served. No engine default is
changed by this section. It is committed in this state so that the spec
predates the experiment, as CLAUDE.md's bake-off rule requires. It is a
PROPOSAL to the PM, not a decision.**

Motivating evidence: `docs/tests/late_game_regime_2026-09-11.md`. Three numbers
from it, and nothing else is restated here:

- The end-game kernel carries **70.3%** of the tie deficit and the upstream
  margin-at-2:00 distribution **26.9%** (interaction 2.8%).
- A regime-free kernel measured on real first-half basketball, mixed against
  the CORRECT margin-at-2:00 distribution, yields a tie rate of **0.0289**; the
  engine yields **0.0300** on 285,500 rows. The engine's end-game is
  arithmetically indistinguishable from no end-game.
- The trailing-minus-leading split inside the window is **-0.158** on
  three-point share and **+0.265** on the bonus-free-throw rate in the data;
  the engine's are **+0.007** (wrong sign) and **+0.024**.

### 1.0 Preconditions, both blocking

1. **Re-read the duration lines against the served default.** The evidence was
   measured with run A's `ENGINE_CLOCK=v3c_srfloor_P3_s1`; the served default
   became `v5b_glat_pmean` at `e3ccce5`. Re-run `scripts/diag_late_game_tap_v1.py`
   with the default flag set before any clock-side arm is fitted. If the
   default clock already carries a score-conditioned duration split, arms B and
   C change shape on the clock half and this spec is amended in a numbered
   section before the round runs.
2. **A 200-seed re-read of the sim kernel.** The per-cell kernel behind the
   (a)/(b) split sits on 30-57 simulations per cell. The attribution's DIRECTION
   is corroborated by two constructions that use no sim kernel, but the 26.9 /
   70.3 split itself is not a precise number until the re-read exists.

### 1.1 The regime gate (frozen before any fit)

`period == 2` and `seconds_remaining <= 120` and `|score_diff| <= 6`, read on
the live state block at the START of the possession. 2.6% of regulation
possessions. Two sensitivity members, declared now and NOT used for selection:
`(150 s, 8 pts)` and `(90 s, 5 pts)`. Selection is on the `(120 s, 6 pts)` cut
only; the other two are reported to show the result is not a cut-point artefact.

### 1.2 Grid configuration

| Dimension | Values |
|---|---|
| Arms | **A** reference (served cascade, unchanged) / **B** state enrichment in the EXISTING sub-models (`possession_outcome` and clock refit on ALL possessions with the new columns, one pooled fit) / **C** regime-conditioned REFIT (same model classes, same bundles, fitted on window rows only, served behind the gate) / **D** dedicated end-game model (its own class over the full `L3_gates` / `L4_team` state) |
| Feature sets | `L0_reference`, `L1_role`, `L2_role_poss`, `L3_gates`, `L4_team` (`features.md` section 2) |
| Model classes | LightGBM multiclass for the event half; the served clock's own family for the duration half. No new family is introduced; the question is the CONDITIONING, not the learner |
| Targets | the six `possession_outcome.CLASSES`; possession duration |
| Folds | fold 1: train through 2022-23, test 2023-24. fold 2: train through 2023-24, test 2024-25. **Fold 2 selects.** 2025-26 sealed |
| Refit cadence | S1 monthly, the served scheme; a static arm is carried as the Decision 9 alignment cell |
| Mandatory Decision 9 arms | opponent adjustment of the as-of window rates, `is_conf_game`, and the refit-cadence cell, wherever team or player rates enter |
| Seeds | offline: 2 seeds per cell (the arm and its own floor). Closed loop: 25 paired seeds for the screen, 200 for the gate |

Arms A and B are fitted on all regulation possessions; arms C and D on window
rows only. **All four are GRADED on window rows only**, by one script, blind.

### 1.3 Primary metric, and why it is not the tie rate

- **Offline primary:** multiclass log loss on fold-2 window possessions.
- **Closed-loop primary:** the regulation-margin density near zero --
  `P(|margin| = 0)`, `P(|margin| = 1)` and the ratio `P(0) / P(1)`. Target
  from 2024-25: 0.0566 / 0.0366 / **1.546**; the engine today is 0.0300 /
  0.0551 / **0.545**.
- **Named secondaries:** the regulation tie rate (= G7 OT rate) and the
  regulation-margin excess kurtosis (engine +0.294, actual +0.740).

**The tie rate is deliberately NOT the primary.** An arm can raise it by being
wrong in a compensating direction; it cannot hit the `P(0)/P(1)` ratio AND the
kurtosis AND the role splits without representing the behaviour. Any arm that
moves the tie rate while leaving `P(0)/P(1)` below 1.0 is recorded as a FAILURE
regardless of G7, and a submission that reaches the tie rate by scaling,
clipping or re-weighting anything is rejected unread under the no-hand-tuning
rule.

### 1.4 Pre-registered closed-loop target (so that success is not moved afterwards)

A regime-only fix does not own the upstream margin distribution. On the
measured 70.3% share, an arm that fixes the kernel completely and touches
nothing upstream predicts a tie rate of
`0.0289 + 0.703 x (0.0510 - 0.0289) ~= 0.044` on the pbp universe, i.e. about
**0.044-0.048 on the verified universe**, not 0.0557. **The closed-loop band is
therefore 0.042 <= tie rate <= 0.050.** An arm landing ABOVE 0.050 is as much a
failure as one landing below 0.042: it has bought G7 from somewhere it does not
own, and the diagnosis is then wrong and must be re-opened.

### 1.5 Noise floor

- **Offline:** a spec-identical retrain of each arm under a second seed. A
  winner must beat that floor on fold 2, not merely beat arm A.
- **Closed loop:** a seed-offset paired run at matched seed count. The floor
  already on record at 20 seeds (`docs/tests/gates_pair_seedfloor_20_2026-09-11.md`)
  puts G7 at **0.0009** and the margin SD ratio at **0.0060**; a 200-seed floor
  is run as part of this round rather than borrowed across seed counts, because
  several G5 and G9 lines are seed-count dependent by construction.

### 1.6 Segment gates (all mandatory, all with a pre-declared direction)

| # | Gate | Pass condition |
|---|---|---|
| R1 | **role sign, three-point** | trailing-minus-leading three-point share is NEGATIVE and at least half the actual -0.158 |
| R2 | **role sign, fouling** | leading-minus-trailing bonus-FT rate is POSITIVE and at least half the actual +0.265 |
| R3 | **team responsiveness** | the fouling channel SLOPES across team PPG quintiles (CLAUDE.md matchup-specific rule); flat is a failure |
| R4 | **clock profile** | duration by seconds-remaining bucket tracks the actual's fall to 6.56 s and 2.89 s; the current ~10.4 s flatline inside 60 seconds is the named failure mode |
| R5 | **possession count** | window possessions per game slopes with `|margin at 2:00|` (actual 9.24 -> 11.34) |
| R6 | **no upstream damage** | the reference window's rates move by less than the seed floor. An arm that changes the first 38 minutes to fix the last 2 is rejected outright, per the bottom-up rule |
| R7 | **cut-point robustness** | the sign of R1 and R2 survives both sensitivity gates in 1.1 |

Cells below 200 possessions are labelled UNDERPOWERED and are never read as
signal or as absence of signal.

### 1.7 Decision rule

1. An arm must beat arm A on fold-2 window log loss **beyond its own noise
   floor**. Fold 1 is reported and is not the selector.
2. It must pass R1-R7. A gate failure is disqualifying, not a caveat.
3. **Ties go to the simpler arm, in the order A > B > C > D.** If B (adding
   `role` to the models that already exist) matches C or D within the floor,
   B ships and this folder records that the "regime layer" hypothesis was
   REJECTED -- which is a legitimate and welcome outcome, and is why arm B is in
   the grid at all.
4. An offline winner ships only after a paired-seed closed loop in which **no
   gate regressed**, per Decision 10, with the 1.4 band as the acceptance
   condition on G7.
5. If no arm clears its floor: **NO ARM ADOPTED**, the served default is
   unchanged, and the round is recorded as such.

### 1.8 Ranked hypothesis under test (stated so it can be refuted)

From the measured deficits, the lane's ranking of the three behaviours is
**fouling / free-throw supply first, duration and the hold second, three-point
share third** -- with the three-point channel explicitly flagged as the one
most at risk of being over-fixed, since the engine's aggregate three-point
share and its clock ramp are already right and only the role split is missing.
**This ranking is a hypothesis, not a finding: no single-channel counterfactual
was simulated.** Arm D's ablation (fit D, then serve it with each of the three
channels reverted to the reference in turn, on aligned seed streams) is the
pre-registered measurement that decides it, and a result that contradicts the
ranking is recorded as a correction to `docs/tests/late_game_regime_2026-09-11.md`
section 3.3 rather than quietly dropped.

### 1.9 Cost and scope

Offline: 4 arms x 5 bundles x 2 folds x 2 seeds on ~21k window rows per season
is small -- the window is 2.6% of possessions, so the whole grid is minutes,
not hours. The expensive half is the closed loop (25-seed screen, 200-seed
gate), which is scheduled only for an arm that has already cleared 1.7 steps 1
and 2. Out of scope for this round: overtime periods (the OT module is not the
defect), the margin-at-2:00 distribution (the other 27%), and player
attribution inside the window.

---

## 2. AMENDMENT -- the two blocking preconditions fired, and they change three of the motivating numbers (written 2026-09-18 by the late-game lane, BEFORE any arm was fitted; append-only, section 1 is unchanged)

Section 1.0 made two things blocking preconditions on this round and said, in
terms, that if the default clock already carried a score-conditioned duration
split then "arms B and C change shape on the clock half and this spec is
amended in a numbered section before the round runs." Both preconditions have
now been run and both fired. This section is that amendment. **No arm has been
fitted at the time of writing. Nothing below adopts anything or changes any
served default.**

Evidence: `scripts/diag_late_game_tap_v2.py` (12,000 simulations against the
original tap's 1,000, four sharded processes, SERVED defaults resolved by
`Adapters.load` rather than pinned), `scripts/diag_late_game_compare_v2.py`,
output `results/late_game/compare_v5b_2025.txt`; the margin LEVELS come from the
full 75-seed served-stack run `results/engine_v0/F2_2025_s200_v5b_A` (428,250
simulations over all 5,710 games), because a 250-game tap cannot carry a level.
The actual side is unchanged throughout: the comparison moves only the sim side.

### 2.1 Precondition 1, part one: the LEVELS are unmoved by the clock adoption

| | n sims | P(0) | P(1) | P(0)/P(1) | margin SD | excess kurtosis |
|---|---:|---:|---:|---:|---:|---:|
| **v5b served** (`v5b_glat_pmean`, 75 seeds) | 428,250 | 0.02996 | 0.05483 | **0.546** | 15.512 | +0.320 |
| v3c run A (the 2026-09-11 evidence) | 1,142,000 | 0.03034 | 0.05487 | 0.553 | 15.501 | +0.280 |
| **ACTUAL 2024-25** (verified finals) | 5,710 | 0.05660 | 0.03660 | **1.546** | 13.8 | +0.740 |

**The defect this folder exists for is entirely intact under the served stack.**
The clock adoption moved `P(0)/P(1)` by 0.007 and the kurtosis by +0.04. Section
1's primary metric, its target and its statement of the problem all stand
unchanged, and nothing in 1.3 is amended.

### 2.2 Precondition 1, part two: TWO DEFECTS in the 2026-09-11 sim-side measurement

Both are in the SIM side only. The actual side of
`docs/tests/late_game_regime_2026-09-11.md` is correct as published and is
re-used here unaltered.

**(i) The anchored role was not signed per possession on the sim side.** The
actual-side script signs the 2:00 anchor for whichever team is on offence
(`diag_late_game_actual_v1.py` line 62, `off_m120 = m120 * sgn`).
`diag_late_game_compare_v1.py` could not: the v1 tap recorded no offence-side
flag, so it took the offence-perspective `score_diff` of the FIRST window
possession and applied that one signed number to **every** possession of the
simulation, including the opponent's, which carry the opposite sign. The sim's
"trailing" and "leading" cells were therefore a near 50/50 mixture of the two
roles and any real split cancelled inside them. v2 records the offence's side on
both taps -- the event adapter is handed `off` directly, the clock adapter is
handed the team-static row -- and asserts the clock/event row alignment on four
columns before using it.

With the anchor signed per possession, on the SERVED stack:

| split, final 2:00, `abs m@2:00 <= 6` | SIM v5b | ACTUAL | reproduced |
|---|---:|---:|---:|
| trailing-minus-leading **three-point share of FGA** | **+0.1523** | +0.1575 | **96.7%** |
| leading-minus-trailing **bonus-FT rate** | **+0.2256** | +0.2652 | **85.1%** |
| (as published 2026-09-11, unsigned anchor) | +0.007 / +0.024 | -- | 0% / 9% |

| role cell | SIM 3PA share | ACT | SIM bonus-FT | ACT |
|---|---:|---:|---:|---:|
| trailing offence | 0.4811 | 0.4788 | 0.1383 | 0.1564 |
| leading offence | 0.3288 | 0.3213 | 0.3639 | 0.4216 |

**Section 1's headline "the engine reproduces 0% of the three-point role
asymmetry and 9% of the fouling role asymmetry" is WITHDRAWN. It was a
measurement artefact.** The served `possession_outcome` round-2 arm reproduces
97% of the three-point role split and 85% of the fouling role split. The same
holds on the LIVE margin, computed identically on both sides (sim +0.1829 /
+0.3097 against actual +0.2004 / +0.3903), and it holds bucket by bucket down
the clock. Section 5 point 2 of the motivating document -- "it still delivers 9%
of the fouling asymmetry and 0% of the three-point asymmetry" -- falls with it,
and with it the strongest single argument that the event half needs a regime
layer at all. Arm A on the event half is now the arm to beat, not the arm to
replace, and arms C and D on that half are testing a much smaller residual than
section 1 assumed. **This does not cancel the round; it is exactly what a
pre-registered precondition is for.**

**(ii) `CELL_DIMS` is not the served clock's cell grid.** Section 1.2 of the
motivating document, and the change-ledger row that quotes it, state that the
served clock "carries no `score_diff` in its cell grid", citing
`clock_adapter_v3.CELL_DIMS`. That constant is the `ENGINE_CLOCK_DIAG=1`
accumulator's grid. The served arm is `empirical_km3_srfloor | P3`, whose cells
are `clock_v3.EMPIRICAL_DIMS["P3_dummy"] = (prev_end, r2_bucket,
r2_period_type, eg_regime, tempo_tercile)` with `eg_regime = {0 outside, 1
trailing by >= 4, 2 leading by >= 4, 3 close}` inside the last 120 s of H2/OT --
a role-conditioned, clock-gated end-game dimension. It is doing work: the
served clock's window durations are 11.15 s when trailing by >= 4, 15.20 s when
close and 12.36 s when leading by >= 4.

**The real structural limit is different and it is exact.**
`clock_v3.SR_FLOOR_BUCKET = 5` and the served arm sets
`sr_floor_bucket = 5`, so `bucket = max(bucket, 5)`: **every possession with
fewer than 45 seconds left is served the 45-59 s law**, by construction, and
"the whole end-of-period effect is left to the engine's truncation"
(`EmpiricalArmV3` docstring). That is the mechanism behind the named failure
mode in gate R4, and the re-read prices it:

| final 2:00, duration (s) | SIM trail | ACT trail | SIM lead | ACT lead |
|---|---:|---:|---:|---:|
| (60,120] | 15.96 | 13.52 | 19.32 | 18.38 |
| (30,60] | 10.63 | 10.38 | 10.92 | 12.14 |
| **(10,30]** | **10.34** | **7.77** | **10.63** | **4.88** |
| **(0,10]** | **10.53** | **3.53** | **10.72** | **2.15** |

Above 45 seconds the served clock is close to right on both roles. Below it the
law is frozen and both roles sit at 10.3-10.7 s against an actual 7.77/3.53 and
4.88/2.15. Window duration overall 13.46 s against 10.56; `P(dur <= 8)` 0.390
against 0.522; window possessions per game **8.720** against 9.24-11.34.

A related correction to the same section: the motivating document says the sim
"has no slope in `abs m@2:00` to produce because its clock never sees the
margin". It has most of one. Window possessions per game by `abs m@2:00` run
7.90 / 8.00 / 8.31 / 8.54 / 9.06 / 9.41 / **9.64** against an actual 9.24 to
11.34 -- a span of +1.74 against +2.10, i.e. **83% of the slope and a level
1.3-1.7 possessions short everywhere**. The defect is a level defect, not an
absent response.

**(iii) A sign transcription in R1.** Section 1.6's gate R1 asks for a NEGATIVE
trailing-minus-leading three-point share "at least half the actual -0.158". The
evidence table it quotes has the trailing offence at 0.4788 and the leading
offence at 0.3213, i.e. trailing MINUS leading = **+0.1575**. The minus sign is
a transcription error and the substantive gate -- the trailing team shoots MORE
threes -- is unchanged. **R1 is restated as: trailing-minus-leading three-point
share is POSITIVE and at least +0.0788.** R2 is unchanged.

### 2.3 Precondition 2: the kernel re-read, and the (a)/(b) split MOVES

12,000 simulations against 1,000. Every kernel cell shown in
`results/late_game/compare_v5b_2025.txt` now sits on 337-690 simulations rather
than 30-57, and the cells below 200 are labelled.

| | 2026-09-11 (v3c, 1,000 sims) | **2026-09-18 (v5b, 12,000 sims)** |
|---|---:|---:|
| (a) upstream margin-at-2:00 distribution | +26.9% | **+12.5%** |
| (b) the end-game kernel | +70.3% | **+80.7%** |
| interaction | +2.8% | +6.8% |
| `P(abs m@2:00 <= 6)` sim/actual | 0.900 | 0.9228 |

**The kernel's share went UP, not down**, and the upstream distribution's share
roughly halved. The direction of section 3.2's conclusion is unchanged and
strengthened; the numbers 70.3 / 26.9 are superseded by **80.7 / 12.5**.

**Consequence for the pre-registered closed-loop band (section 1.4).** The band
was derived from the 70.3% share as `0.0289 + 0.703 x (0.0510 - 0.0289) = 0.044`
on the pbp universe. On the corrected share it is
`0.02892 + 0.807 x (0.05096 - 0.02892) = 0.0467` pbp, which is **0.0511 on the
verified universe** (the pbp/verified ratio 0.0557/0.05096 = 1.093).
**The closed-loop band is therefore restated as `0.046 <= tie rate <= 0.055` on
the verified universe.** It remains TWO-SIDED for the reason section 1.4 gives:
an arm landing above 0.055 has bought G7 from somewhere it does not own.

### 2.4 The ranked hypothesis of section 1.8 is RE-ORDERED before the round runs

Section 1.8 ranked the three behaviours **fouling first, duration second,
three-point third**, and said in terms that "this ranking is a hypothesis, not a
finding" and that a contradicting result is recorded as a correction. The
corrected measurement contradicts it. The re-ordered ranking, pre-registered
here before any arm is fitted:

1. **Duration and the hold.** The only channel with a large, structural,
   mechanically identified defect: the 45-second cell floor, 10.3-10.7 s against
   3.53/2.15 inside 10 seconds, and 8.72 window possessions per game against
   9.24-11.34.
2. **Free-throw SUPPLY LEVEL inside 30 seconds.** The role SPLIT is 85%
   reproduced, but the leading offence's bonus-FT rate inside 30 s is 0.6641 /
   0.6973 against an actual 0.8407 / 0.8464 -- a residual of about 18%, not the
   91% section 1.8 assumed.
3. **Three-point share.** Essentially closed (96.7% of the split, and the
   clock-conditioned ramp tracks). The one open cell is the LEADING offence
   inside 10 seconds: sim 0.2708 against an actual 0.0833, i.e. the engine still
   lets a leading team shoot a three when the real one is at the line -- which
   is the same fouling defect seen from the shot side, not an independent one.

Arm D's ablation (section 1.8) remains the pre-registered instrument that
decides the ranking inside the closed loop; this is a re-ranking on measured
deficits, and it is stated now so that the round cannot be read as having
confirmed the ranking it was designed with.

### 2.5 What section 1 left open, fixed here BEFORE any fit

Each of these is required by the bake-off rule and was not pinned by section 1.
None is searched, and none may be changed after this commit.

1. **Refit cadence.** Every arm's PRIMARY grid cell is **S0 (static)**, so the
   grid varies the CONDITIONING with the cadence held fixed and no arm can win
   the primary by being refit more often than its reference. The Decision-9
   cadence cell (S1 monthly, the served scheme) is run for arm A and for the
   fold-2 winner only. This is a COST decision -- an S1 cell is six LightGBM
   fits on up to 2.7M rows at about two minutes each, and the full grid under S1
   is roughly four machine-hours on a 4-process cap shared with other lanes --
   and it is declared rather than discovered.
2. **Hyperparameters.** Two fixed sets, neither searched:
   `possession_outcome.LgbmArm.PARAMS` for every fit on the full design, and
   `train_late_game_v1.WINDOW_PARAMS` (`n_estimators=300, learning_rate=0.05,
   num_leaves=15, min_child_samples=50`) for every fit on window rows only. The
   full-data value `min_child_samples=400` on a 50k-row window fit would return
   a near-constant model, which would sandbag arms C and D rather than test
   them. Each set is identical across every bundle, fold and seed in its half.
3. **The noise floor for DETERMINISTIC arms.** Section 1.5 specifies a
   spec-identical retrain under a second seed. That moves the LightGBM arms and
   does nothing at all to the cascade's logits or to the Kaplan-Meier cell law.
   Following `possession_outcome`'s own precedent, the floor for every arm is
   **`max(seed floor, game-level block-bootstrap SE)`**, both reported
   separately, with the game as the resampling unit.
4. **Arm D's reading.** "A dedicated end-game model (its own class over the full
   `L3_gates` / `L4_team` state)" is realised as: populations POOLED (with
   `chance_number` and `is_cont` as features) and the conditional law estimated
   **separately per role** (trailing / tied / leading) on window rows. That is
   the direct test of section 5's claim that the regime is "a conditional law
   that a single pooled fit averages away", and it is what distinguishes D from
   C, which is one pooled window-only fit.
5. **The DURATION half's arms and metric**, which section 1.2 named only as "the
   served clock's own family". Six cells, all the same Kaplan-Meier cell family
   so the grid tests conditioning only: `A_clk` the served spec (P3 dims, floor
   5, all rows); `B1_clk` (P3 dims, **floor removed**, all rows); `B2_clk`
   (floor removed and `eg_regime` refined from 4 levels to 6 -- trailing>=4 /
   trailing 1-3 / tied / leading 1-3 / leading>=4 -- all rows); `C_clk` and
   `C2_clk`, those two specs fitted on WINDOW ROWS ONLY; `D_clk`, its own grid
   (role3 x fine clock bucket x bonus x prev_end) on window rows.
   **Duration primary: CRPS of the truncated law on uncensored fold-2 window
   rows**, with the censored log-likelihood beside it -- the clock lane's own
   blind metric (`clock_v3.score_arm_v3`), unchanged.
6. **The two behavioural-gate cut points** of `features.md`: `trail_must_foul`
   uses `k = 60 s` (the midpoint of the 120 s window) and `lead_can_hold` uses
   `35 s` (`clock_adapter_v3.EOH_WINDOW_S`, a constant the clock lane already
   uses). Both are fixed a priori and neither is tuned on any outcome.
7. **The prediction universe.** Every event-half cell is scored on the WIDEST
   sensitivity gate (150 s, 8 pts), which is a strict superset of the selection
   gate and of the narrow sensitivity gate, so **R7 is computable without a
   second pass**; plus a fixed seeded sample of 60,000 reference-window rows so
   **R6 is measurable** for the arms that refit on all possessions. Window-only
   arms are GATED OFF outside the window in the engine, so they are not scored
   there and R6 is recorded for them as satisfied by construction. **Selection
   remains on the (120 s, 6 pt) subset only.**
8. **`L4_team` carries FOUR as-of columns, not two**: the offence's own and the
   defence's conceded late-window bonus-FT rate and three-point share, each an
   expanding mean over that team's strictly prior games and centred on the
   league's own as-of mean. `features.md` names the conceded one; a rate is
   produced by the pair, so both sides are carried and this is a strict
   superset. **Leak test (mandatory, CLAUDE.md): change-form correlation with
   own-week margin = -0.0151 / +0.0109 / +0.0041 / -0.0051 on 22,275 week
   changes, all inside the 0.15 bar.** They sit BELOW the 0.04-0.08 honest
   baseline, which is a statement that they carry little signal, not that they
   are clean by luck; that is reported, not hidden.
9. **Pre-outcome discipline.** `start_score_diff` is re-asserted pre-outcome by
   this lane's own L27 own-row delta test (98.45% on scoring chances against
   0.90% for the post-outcome alternative). `duration_s` is POST-OUTCOME and is
   a target only. `is_transition` in the chances table is `possession duration
   <= 8 AND start_reason in {DREB, TOV}` -- post-outcome but PRE-EVENT, because
   the engine draws the duration before the event model runs (`loop.py` step
   (a)) -- so it stays in the event half's bundles, exactly as the served arm
   has it, and is FORBIDDEN in the duration half, where it would be the target.

### 2.6 What this amendment does NOT change

The regime gate (1.1), the offline primary metric (1.3), the sensitivity
members, the fold structure and the seal, the arm set A/B/C/D, gates R2-R6, the
underpowered rule, the decision rule (1.7) and the tie-break order A > B > C > D
are all unchanged. Section 1 is not edited; this section supersedes only the
four numbered items it names (the 0%/9% role-split headline, the `CELL_DIMS`
mechanism, the 70.3/26.9 attribution with the band that follows from it, and
R1's sign), and it fixes the nine items section 1 left open.

---

## 3. RESULTS -- round 1, RUN 2026-09-18 (status PROPOSED -> RUN; NO ARM ADOPTED, no served default changed)

Full evidence, every table and every caveat:
`docs/tests/late_game_round1_2026-09-18.md`. Raw output:
`results/late_game/round1/{event,clock}_results.csv` and `*_detail.json`.
The grid was fitted by `scripts/train_late_game_v1.py` and
`scripts/train_late_game_clock_v1.py`, which compute no metric, and scored by
`scripts/grade_late_game_v1.py`, the single blind grader, which has no branch on
which arm produced a cell.

### 3.1 Event half, fold 2 (selection), first chances, n = 17,596

Primary = multiclass log loss on the selection gate. Binding floor =
`max(seed floor, game-level block-bootstrap SE)`.

| arm | bundle | log loss | Δ vs A | binding floor | floors beaten | R1 | R2 | R3 | R3b | R7 |
|---|---|---:|---:|---:|---:|:--:|:--:|:--:|:--:|:--:|
| A | L0_reference (served) | 1.32443 | 0 | 0.00645 | -- | PASS | PASS | PASS | PASS | PASS |
| B | L1_role | 1.32466 | +0.00023 | 0.00655 | -0.04 | PASS | PASS | PASS | PASS | PASS |
| B | L2_role_poss | 1.31378 | -0.01065 | 0.00654 | 1.63 | PASS | PASS | PASS | PASS | PASS |
| **B** | **L3_gates** | **1.31263** | **-0.01180** | 0.00659 | **1.79** | PASS | PASS | PASS | PASS | PASS |
| B | L4_team | 1.31356 | -0.01087 | 0.00657 | 1.66 | PASS | PASS | PASS | PASS | PASS |
| C | L0_reference | 1.32718 | +0.00275 | 0.00671 | -0.41 | PASS | PASS | PASS | FAIL | PASS |
| C | L1_role | 1.32605 | +0.00162 | 0.00677 | -0.24 | PASS | PASS | PASS | FAIL | PASS |
| C | L2_role_poss | 1.31611 | -0.00832 | 0.00683 | 1.22 | PASS | PASS | PASS | FAIL | PASS |
| C | L3_gates | 1.31635 | -0.00808 | 0.00697 | 1.16 | PASS | PASS | PASS | FAIL | PASS |
| C | L4_team | 1.31666 | -0.00777 | 0.00698 | 1.11 | PASS | PASS | PASS | FAIL | PASS |
| D | L3_gates | 1.33599 | +0.01156 | 0.00746 | -1.55 | PASS | PASS | PASS | FAIL | PASS |
| D | L4_team | 1.33429 | +0.00986 | 0.00741 | -1.33 | PASS | PASS | PASS | FAIL | PASS |

`B | L3_gates` is the only family that clears its floor with every gate passing.
Arm C clears by 1.1-1.2 floors and FAILS the as-of-prior responsiveness cut
(R3b, slope ratio 0.884-0.941 against arm A's 0.974). Arm D is WORSE than the
reference by 1.3-1.6 floors: splitting 44k window rows three ways by role costs
more than the sign flip is worth. **On the pre-registered tie-break
A > B > C > D, B wins, which is decision rule 1.7.3's named outcome: the
regime-LAYER hypothesis is REJECTED on the event half in favour of columns in
the sub-model that already exists.**

The size of the win has to be stated with it. Arm A's own role splits on these
rows are +0.197 / +0.397 against actuals of +0.198 / +0.393, its per-role
bonus-FT rates are 0.1411/0.1496 (trailing) and 0.5381/0.5430 (leading), and the
best arm buys 0.0118 nats on 2.4% of possessions. Per-team bonus-FT MAE over 361
teams: A 4.50 pp / corr 0.791, B_L3 4.24 pp / 0.812. R6: no full-scope arm moves
the reference window by more than 0.017 pp relative to A.

### 3.2 Duration half, fold 2 (selection), n = 17,710 (1,249 censored)

Primary = CRPS of the truncated law on uncensored window rows. The cell law is
deterministic, so the floor is the block-bootstrap SE.

| arm | scope | cells | sr floor | CRPS | Δ vs A | floors beaten | cens. loglik | R4 max bucket gap (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A_clk | all rows | P3 | 5 | 4.14955 | 0 | -- | -3.1851 | 2.577 |
| B1_clk | all rows | P3 | 0 | 4.01063 | -0.13892 | 4.62 | -3.1252 | 0.802 |
| B2_clk | all rows | P3R | 0 | 4.01012 | -0.13943 | 4.23 | -3.1036 | 1.004 |
| C_clk | window | P3 | 0 | 3.98171 | -0.16785 | 5.51 | -3.1864 | **0.139** |
| **C2_clk** | window | P3R | 0 | **3.96353** | **-0.18602** | **5.69** | -3.1554 | 0.943 |
| D_clk | window | role3 x bucket x bonus x prev_end | 0 | 4.02353 | -0.12603 | 3.88 | **-3.0945** | 0.478 |

**Every arm beats the served reference beyond the floor, by 3.9 to 5.7 floors.
This is the round's finding.** `C2_clk` wins the pre-registered primary;
`D_clk` wins the named secondary and is the only arm that reproduces the
leading team being fouled off the ball inside 10 seconds (1.40 s predicted
against 1.39 actual, where the served arm says 1.90). The two metrics disagree
on C2 versus D and that is reported rather than resolved by choosing one.

### 3.3 Status and what is NOT concluded

**Status: RUN. NO ARM ADOPTED.** Decision rule 1.7.4 is unmet for both halves:
no closed loop was run, the round having hit a hard machine deadline. Three
things are PARTIAL and each is listed with its resume command in
`docs/tests/late_game_round1_2026-09-18.md` section 7: no closed loop; five
fold-2 seed-1 floor cells and four fold-1 ones not reached (every measured seed
floor is 2.6-11x smaller than the block SE, so the binding floor is unaffected
unless a missing one is 3.8x the largest observed); and the event half graded on
FIRST chances only (86.6% of window rows -- the first-chance predictions are
bit-identical with or without the continuation fit, so the graded POPULATION is
restricted and no arm is perturbed).

The prior evidence that names round 2's arm: clock round 4 already closed-loop
rejected a GLOBAL floor removal (`v4_nofloor_P3_s1`: best offline CRPS, worst
G1 at +1.561 possessions/game against the served +1.156, floor 0.180) because it
manufactures possessions at the FIRST half's horn (`clock/experiments.md`
15.3-15.5). `C_clk`, `C2_clk` and `D_clk` are fitted and served on window rows
only, so a gate at `period == 2 and seconds_remaining <= 120 and
|score_diff| <= 6` removes the floor exactly where round 4 shows it is wrong and
nowhere near where round 4 shows it is load-bearing. **That is the recommended
round-2 candidate, and it needs a DEFAULT-OFF composite clock adapter that this
round did not write.**

---

## 4. PROPOSED -- Round 2 pre-registration: window-gated arms, wired DEFAULT-OFF, paired closed loop (written 2026-09-30 by lane D, BEFORE any arm was wired or run; NOT RUN, NOT ADOPTED)

**Nothing below has been simulated. No served default is changed by this round;
every arm is reachable only through a new DEFAULT-OFF flag. The PM decides.**
Evidence carried in: section 3 and `docs/tests/late_game_round1_2026-09-18.md`.
The only closed-loop numbers seen before this section was written are the two
EXISTING reference runs' own `P(0)/P(1)` (`po4b_R_s25` 0.551, `po4b_R_s25_floor`
0.489, read from their `games.parquet`); no arm output of any kind has been seen.

### 4.1 The window (frozen; the bounds are section 1.1's, not chosen here)

`period == 2` AND `seconds_remaining <= 120` AND `|score_diff| <= 6`, read on the
engine's live state block with `score_diff` in the OFFENCE's perspective, at the
START of the possession for the clock draw and at the start of the CHANCE for
the event draw. **Overtime is OUTSIDE the window**: periods >= 3 keep the served
law in every arm, because round 1 fitted and graded every window arm on
`period == 2` rows only and section 1.9 put overtime out of scope. The first
half is outside the window by construction, so the first-half-horn possessions
that sank clock round 4's global floor removal are untouched. No bound is varied
in this round; an OT-inclusive or wider window would be its own pre-registered
arm and is NOT in this grid.

### 4.2 Arms (flag `ENGINE_LATE_GAME`; unset or `off` = served, bit-identical)

| arm | flag value | what changes inside the window | outside the window |
|---|---|---|---|
| **R** | unset | nothing (served reference) | served |
| R_floor | unset, seeds 1000-1024 | nothing -- the spec-identical reseed that defines the floor | served |
| **W_C2** | `clk_C2` | clock: intended duration drawn from round 1's `C2_clk` law (P3R cells, `sr_floor_bucket = 0`, fitted on window rows) | served, bit-identical |
| **W_D** | `clk_D` | clock: round 1's `D_clk` law (role3 x fine bucket x bonus x prev_end, window rows) | served, bit-identical |
| **E_BL3** | `ev_BL3` | event, FIRST chances: round 1's `B / L3_gates` LightGBM (served `C_plus_state` plus the nine late-game columns) | served, bit-identical |
| E_L0S0 (control) | `ev_L0S0` | event, FIRST chances: round 1's arm-A `L0_reference` LightGBM, static S0 | served, bit-identical |
| **W_best+E_BL3** | `clk_<best>+ev_BL3` | both | served, bit-identical |

Fixed details, none tuned on any output:

1. **Artifacts** are the round-1 SELECTED cells refit spec-identically: fold 2,
   seed 0, S0 (trained on the 2022-2024 seasons, so `max_train_date` precedes
   every 2024-25 tipoff). Each refit is verified against round 1's saved
   test-row predictions (`round1_clock/pmf/{C2,D}_clk_F2.npy`,
   `round1/pred/c027.npy` and `c024.npy`) before it is served: Kaplan-Meier arms
   must match exactly; LightGBM arms to max |diff| <= 1e-4 (thread-count
   summation order). A cell that does not reproduce is not served and the round
   stops on it.
2. **Clock latent.** The served clock is `v5b_glat_pmean`: the cell law scaled
   by ONE per-game latent `A`. The window law is scaled by the SAME `A` (same
   uniform, same location), so the game keeps one pace realisation, and the
   window draw uses the served draw's own duration uniform. Outside the window
   the served draw is returned untouched.
3. **Event populations.** Only first chances inside the window are re-served.
   Continuations keep the served cascade: round 1 graded first chances only
   (its section 7 item 3), so no continuation fit has evidence behind it.
4. **Cadence control.** Inside the window the served event model is an S1
   monthly schedule while `B / L3_gates` is static S0. `E_L0S0` serves the
   reference bundle under the same S0 cadence, so `E_BL3 - E_L0S0` is the
   enrichment and `E_L0S0 - R` is the cadence. It is a control, not a candidate.
5. **`in_double_bonus`** is an `L3_gates` column the engine state block lacks.
   It is APPENDED to `STATE_COLS` and filled from `GameState.in_double_bonus()`
   -- the engine's own rule-era counter -- so no existing plan's column index
   moves and the default path stays bit-identical.
6. **The "best" duration arm** for the combined arm = whichever of W_C2 / W_D
   lands `P(0)/P(1)` closer to the 1.546 target without a veto failure; if both
   fail a veto, W_C2 (round 1's primary winner). Named in the report.

### 4.3 Sample, seeds, floor

The standing 500-game subset (F2 2025 slate sorted by `game_id`, every 11th row,
first 500 -- identical to `po4b_R_s25`), seeds 0-24 for every arm, paired by the
`(seed, game_id, family)` streams. Served stack pinned and asserted exactly as
`scripts/run_po4b_closed_loop.py` does; no players file (G8 is not a line here).
**Floor per line = |R_floor - R|** on that line (the project's seed-offset
convention, seeds 1000-1024 as in `po4b_R_s25_floor`). A game-level paired
bootstrap SE of each delta vs R is reported beside it, secondary, not the
decision floor. R and R_floor are RE-RUN under this round's tap at the current
HEAD rather than borrowed, and R is also checked bit-identical against
`po4b_R_s25/games.parquet`.

### 4.4 Primary, and power

**Primary:** the regulation-margin density near zero. `P(0)` = share of
simulations reaching overtime; `P(1)` = share ending regulation at
`|margin| = 1`; and the ratio `P(0)/P(1)`. Target 2024-25 verified: 1.546, OT
rate 0.0557; the restated two-sided band `0.046 <= OT rate <= 0.055` (section
2.3) stands. Reported per arm in floors vs R. Section 1.3's rule is carried
verbatim: an arm that moves the tie rate while leaving `P(0)/P(1)` below 1.0 is
recorded as NOT fixing the defect, regardless of G7.

**Power.** The stride sample is declared adequate for the primary if R's floor
on `P(0)/P(1)` is <= 0.10 (one tenth of the 1.0 gap to target) AND R has >= 300
regulation ties. If not, the arms are re-run on a CLOSE-GAME-ENRICHED sample:
the 500 F2-2025 games with the smallest `|mean simulated margin|` in the served
75-seed run `results/engine_v0/F2_2025_s200_v5b_A` (a sim-side, pregame-
determined quantity; no actual outcome enters the selection), same seeds, same
floor construction. The enriched sample's actual `P(0)/P(1)` is descriptive
only.

### 4.5 Vetoes (no regression)

A line REGRESSES if the arm's absolute error against the actual exceeds R's by
more than one floor on that line.

| veto | line(s) |
|---|---|
| G1 | possessions per game: pooled mean and pooled SD |
| G5 | margin SD ratio, total SD ratio (`|ratio - 1|`) |
| G9 | margin bias, total bias (`|bias|`) |
| first half | per-simulation first-half points (both teams) and first-half possession count must be BIT-IDENTICAL to R in every simulation; one mismatch fails the veto |
| half share | pooled first-half share of regulation points vs the G7 target |

### 4.6 Reported lines (not decision lines)

Regulation-margin distribution for `|m| = 0..5`; window possessions per
simulation; window duration by role x clock bucket (the closed-loop R4);
bonus-FT rate and three-point share of FGA on window FIRST chances by LIVE
role, signed by the offence side (the tap reads the offence's own `score_diff`
on every row, so no row carries the opponent's sign); **final-2:00 FTA per
game** (both teams, from the first H2 possession starting at `<= 120 s` to the
end of regulation); per-game paired deltas; per-team cells where n >= 200, else
UNDERPOWERED. Per-player: not applicable.

### 4.7 Order under the wall clock

R, R_floor, W_C2, W_D, E_BL3, E_L0S0, W_best+E_BL3. Anything not reached is
listed with its exact resume command; nothing is read from a partial run.

---

## 5. RESULTS -- round 2, RUN 2026-09-30 (status PROPOSED -> RUN; NO ARM ADOPTED, no served default changed)

Full evidence: `docs/tests/late_game_round2_2026-09-30.md`. Default path
bit-identical (v6 digest PASS; R vs `po4b_R_s25` 28/28 columns); first half
bit-identical to R in 12,500/12,500 simulations in every arm. Power adequate
(floor on `P(0)/P(1)` 0.0623, 390 ties); enriched sample not triggered.

| arm | P(0)/P(1) | floors vs R | OT | vetoes |
|---|---:|---:|---:|---|
| R | 0.551 | -- | 0.0312 | -- |
| W_C2 | 0.543 | -0.13 | 0.0306 | FAIL G1 mean, half share |
| W_D | 0.854 | +4.88 | 0.0437 | FAIL G1 mean, G1 SD, half share |
| E_BL3 | 0.579 | +0.46 | 0.0318 | PASS |
| E_L0S0 (control) | 0.576 | +0.41 | 0.0326 | PASS |
| W_C2+E_BL3 (4.2.6 combo) | 0.607 | +0.90 | 0.0314 | FAIL G1 mean, half share |
| W_D+E_BL3 (exploratory) | 0.851 | +4.82 | 0.0420 | FAIL G1 mean, G1 SD, half share |

Only `D_clk` moves the primary, and under 1.3 it does NOT fix the defect
(ratio < 1.0; OT below the 2.3 band). The mechanism is the TIED offence holding
for the last shot, which D represents (role first in its fallback hierarchy)
and C2 does not (tied rows fall back below `eg_role6`). Every duration arm
over-produces window possessions against the like-for-like pbp count, which is
what fails G1 and the half share. The event half is not the lever (enrichment
~+0.05 floors net of its cadence control). **Status: RUN. NO ARM ADOPTED.**

## 6. PROPOSED -- Round 3 pre-registration: the hold without the extra possessions, on the corrected foul state (written 2026-09-30 by lane C, BEFORE any arm was wired or run; NOT RUN, NOT ADOPTED)

**Nothing below has been simulated or wired. No served default changes. The PM decides whether and when
it runs.** Evidence carried in: section 5 and `docs/tests/late_game_round2_2026-09-30.md`; foul rounds 7-9
(`possession_outcome/experiments.md` sections 20-26, `docs/tests/foul_round9_first_half_2026-09-30.md`).

### 6.1 Why a round 3, and what changed underneath it

1. Round 2's `W_D` (`clk_D`) is the only arm that moves `P(0)/P(1)` (+4.88 floors, 0.551 -> 0.854) because
   its tied offence HOLDS for the last shot (tied (0,10] intended 20.4 s vs 12.2). It fails the G1-mean,
   G1-SD and half-share vetoes through REAL window over-production (+0.3-0.9 window possessions per game
   against the like-for-like pbp count), and its LEADING rows draw very short durations (2.6 / 3.6 s
   intended vs 10.3 / 10.8 served).
2. Round 2 attributed the leading-offence bonus-FT gap (0.41 closed loop vs 0.54 actual, while the event
   model predicts 0.538 on actual states) to the bonus STATE the engine arrives in, i.e. foul accrual. Round
   2 ran on the served foul stack (constant accrual, labelled-state PO), whose final-2:00 bonus occupancy is
   0.787 vs 0.920 actual on the verified sample (round-9 taps, minute 38-40, 500 x 25). Under
   `ENGINE_FOUL_JOINT=R8b` it is 0.885 and under `R9ao1` 0.890 (round 9 changes only the and-one rate).
3. So two things are confounded in round 2's `W_D`: the hold (wanted) and short leading possessions that
   are, in the data, mostly the trailing team FOULING to stop the clock. With the served foul state the
   foul does not produce the bonus trip it produces in the data, so the leading team's short possession
   becomes a cheap extra possession rather than a free-throw trip.

### 6.2 Base stacks (fixed before any run)

- `B8` = served + `ENGINE_FOUL_JOINT=R8b` (round 8's VALIDATED-PENDING-SHIP-ACTION arm).
- `B9` = served + `ENGINE_FOUL_JOINT=R9ao1` (or `R9ao3`, whichever the PM names; round 9 results: possession_outcome experiments.md section 28) only if round 9 is VALIDATED by the PM before this
  round runs; otherwise `B9` is NOT RUN and that is stated.
- `B0` = served (round 2's base), carried for the attribution of the foul-state effect.
Inputs: engine inputs v3 (S0 tag), verified 500-game sample `stride500_verified_minswap_v1`, verified truth.

### 6.3 Arms (new values of `ENGINE_LATE_GAME`; unset / `off` = served, bit-identical; window = section 4.1)

| arm | base | inside the window |
|---|---|---|
| `R8` | B8 | nothing |
| `R8_floor` | B8, seeds +1000 (four offsets 1000-4000, Decision 12) | nothing |
| `D8` | B8 | `clk_D` (round 2's `W_D`, unchanged artifact) |
| `Dt8` | B8 | `clk_D` law for TIED offence rows only (`score_diff == 0`); leading and trailing rows keep the served law -- the hold without the leading-role short draws |
| `Dtt8` | B8 | `clk_D` for tied AND trailing offence rows; leading rows served |
| `D0` | B0 | `clk_D` (= round 2's `W_D`, re-run on v3 inputs as the foul-state attribution) |
| `Dt0` | B0 | as `Dt8` on the served foul stack |
| `Dt9` | B9 | as `Dt8`, only if B9 exists |

Role is the offence's LIVE `score_diff` at possession start, as in round 2 (`role3`). The tied-only and
tied+trailing gates are pre-registered here; no other split is tried. No artifact is refit: `clk_D` is
served exactly as in round 2 (Kaplan-Meier cells, same verification), with the same per-game clock latent.

### 6.4 Primary, vetoes, reported lines

- **Primary:** `P(0)/P(1)` (target 1.546; OT rate band 0.046-0.055; section 1.3's rule: below 1.0 = NOT
  fixing the defect), in floors vs the arm's own base reference (`R8` for B8 arms, served R for B0 arms).
- **Floor (Decision 12):** max(SD over the base reference + four seed-offset draws, 2 x paired game-bootstrap
  SE of arm - base). 500 x 25 is the screening size; any arm that clears the primary is re-read at 500 x 200
  on the box before any statement beyond "screen".
- **Vetoes:** section 4.5's (G1 mean and SD, G5 margin and total SD ratios, G9 biases, first half
  bit-identical to the base in every simulation, half share) PLUS a new window line: **window possessions per
  game by `|m@2:00| = k` (k = 0..6) must not exceed the like-for-like pbp actual (round 2 section 5 count,
  `possessions_v4` when available) by more than one floor in any k**, which is the "without the extra
  possessions" condition stated as a line.
- **Reported (not decision lines):** leading-offence and trailing-offence bonus-FT rate on window first chances
  (actual 0.543 leading), final-2:00 FTA per game (pbp 5.10 / 6.00 close), window bonus occupancy by role,
  window duration by role x bucket, the tie kernel P(tie | k), per game, per team (n >= 200 else UNDERPOWERED).

### 6.5 Decision rule

An arm is a candidate only if it moves the primary toward target by > 1 floor, lands `P(0)/P(1)` >= 1.0, and
passes every veto. Among candidates, the simplest in the order `Dt` < `Dtt` < `D`. The B0 arms are
attribution only and cannot be candidates. If `Dt8` passes the window-possession veto while `D8` fails it, the
extra possessions are attributed to the leading-role draws; if both fail, to the hold itself (then the next
object is the clock's censoring at the horn, not the role gate).

### 6.6 What this round needs before it can run (not done here)

1. Two new `ENGINE_LATE_GAME` values (`clk_Dt`, `clk_Dtt`) in `src/cbb_sim/engine/late_game_adapter.py`,
   default off, with parity: unset and `clk_D` bit-identical to round 2's runs at the same HEAD.
2. The round-2 tap (`scripts/run_late_game_r2_closed_loop.py`) taking `--sample-file`, `--input-dir` and an
   `ENGINE_FOUL_JOINT` env (v3 inputs need the S0 event-block overlay: the box docker mount, or locally
   `R9_ENGINE_DIR` as in `scripts/run_foul_joint_tap_v2.py`).
3. The window-possession actual on `possessions_v4` (lane B's event layer v4) for the new veto.

---

## 7. AMENDMENT -- round 3 restated on served stack v2 (written 2026-10-01 06:20-06:45 EDT by lane L, BEFORE any round-3 arm was wired or run; append-only, section 6 is unchanged)

**Nothing has been wired or simulated for round 3 at the time of writing. No served default changes.**
Section 6 was written on 2026-09-30, before served stack v2 was adopted (02:06 EDT 2026-10-01,
`docs/tests/adoption_served_v2_2026-10-01.md`). Its base stacks no longer describe what is served, so the
round cannot run as written. This section restates only what the stack change forces, and adds the
pre-arm diagnostic and an offline line the PM's brief asks for. Everything section 6 does not name here
stands.

### 7.1 What changed underneath section 6

1. **The corrected foul state is now SERVED.** `ENGINE_FOUL_JOINT=R9ao3` (R8b's trip repair + the AO3
   and-one model) is the default. Section 6.2 conditioned `B9` on round 9 being VALIDATED by the PM;
   adoption is stronger than that, and the PM named `R9ao3`.
2. **The rest of the stack moved too:** clock `v5b_r6L2_glat_pmean` (same P3 / `sr_floor_bucket = 5` cell
   family as `v5b_glat_pmean`, refit on event layer v4, its own latent parameters), `K2_Ocell`, `G3`, `KD`,
   event team block v3. The reference for every paired read is a plain default run (parity reference v9,
   `docs/ops/parity_reference_windows_v9.json`, inputs `data/processed/models/engine_v3`). The
   in-process event-block overlay of 6.6.2 is no longer needed: the v3 block is the default.
3. **`R8b` is no longer a meaningful base.** It is `R9ao3` minus the and-one model, a stack nobody serves.

### 7.2 Base stacks, restated (replaces 6.2 for this round)

- `B9` = served stack v2, NO `ENGINE_*` variable set. **This is the candidate base.**
- `B8` (served v1 + R8b): **NOT RUN, superseded** by 7.1.1/7.1.3.
- `B0` = served stack v2 with `ENGINE_FOUL_JOINT=reference` only (the constant-accrual foul state round 2 ran
  on). Attribution only, exactly as in 6.2/6.5; it isolates the foul state and nothing else, which the full
  `SERVED_V1` would not.

Inputs: engine inputs v3 (`data/processed/models/engine_v3`, tag `F2_2025`), the verified 500-game sample
named in 6.2 (`data/processed/truth/stride500_verified_minswap_v1_F2_2025.parquet`), `CBB_TRUTH=verified_v1`.

### 7.3 Arms, restated (replaces the table in 6.3; flag values unchanged in meaning)

| arm | base | `ENGINE_LATE_GAME` | inside the window | role |
|---|---|---|---|---|
| `R9` | B9 | unset | nothing | reference |
| `R9_f1..f4` | B9 | unset, seed offsets 1000/2000/3000/4000 | nothing | floor draws (Decision 12) |
| `D9` | B9 | `clk_D` | round 2's `clk_D` law, every window row | candidate |
| `Dt9` | B9 | `clk_Dt` | `clk_D` law on TIED offence rows only (`score_diff == 0`) | candidate |
| `Dtt9` | B9 | `clk_Dtt` | `clk_D` law on tied AND trailing offence rows | candidate |
| `R0` | B0 | unset | nothing | attribution reference |
| `D0`, `Dt0` | B0 | `clk_D`, `clk_Dt` | as above | attribution only |

`clk_D` is served exactly as in round 2: the same pickle (`data/processed/models/late_game/round2/clk_D.pkl`,
verified against round 1's saved PMFs by `train_late_game_r2_v1.py`), the same duration uniform, scaled by
the SERVED clock's own per-game latent (now L2's). No artifact is refit. The two new flag values gate the
SAME law by the offence's live `score_diff` sign at possession start; rows outside the gate return the
served draw untouched. Off path (`ENGINE_LATE_GAME` unset) must reproduce parity v9 bit-for-bit, and
`clk_D` must reproduce itself before and after the edit.

### 7.4 Primary, floor, vetoes, decision rule

Unchanged from 6.4/6.5 with the bases of 7.2: primary `P(0)/P(1)` in floors vs the arm's own base
reference (`R9` for B9 arms, `R0` for B0 arms); floor = max(SD of the line over `R9` + `R9_f1..f4`, 2 x
paired game-bootstrap SE of arm - base); vetoes as 6.4. Two clarifications fixed before any output exists:

1. **Window-possession veto, actual.** From `data/processed/possessions_v4/possessions_2025.parquet` on
   the same 500 games: per game, the anchor `k = |home margin|` at the start of the first period-2
   possession starting at `<= 120` s, and the count of period-2 possessions starting at `<= 120` s with
   `|start_score_diff| <= 6`. The sim count is the tap's identical definition. Floor per k = max(SD over the
   five B9 draws, 2 x paired game-bootstrap SE). **If `R9` itself exceeds the actual by more than one floor
   in some k, that k is read against `R9` instead** (the arm may not add more than one floor over the
   served count), and the report says so. The veto exists to stop an arm ADDING possessions; it is not
   meant to fail every arm on a served-stack excess.
2. **The OT band** (0.046-0.055, verified universe) and section 1.3's `< 1.0` rule are carried unchanged.
   The sample's own actual (OT 0.068) is descriptive only.

The B9 decision rule is 6.5's with `R8`/`D8`/`Dt8`/`Dtt8` read as `R9`/`D9`/`Dt9`/`Dtt9`.

### 7.5 Offline line (new; both folds; a GUARD, not the selector)

Round 3 fits nothing, so its offline evidence is the COMPOSED window law graded on round 1's held-out
window rows (`data/processed/models/late_game/round1_clock/window_test_F{1,2}.parquet` and the saved
PMFs `pmf/{A,D}_clk_F{1,2}.npy`): `Dt` = D's PMF on tied rows and A's elsewhere; `Dtt` = D's on tied and
trailing rows. Metric: round 1's duration primary (CRPS of the horn-truncated law on uncensored rows) and the
censored log-likelihood, overall and per role. Graded by one script for all four (A, D, Dt, Dtt). Floor:
game-level block-bootstrap SE of the paired delta vs A (the Kaplan-Meier laws are deterministic, so the
reseed floor is zero; section 2.5.3). **Guard: a candidate that is worse than A on fold 2 by more than one
floor on the rows it replaces is not a candidate.** Fold 1 is reported. Caveat stated now: A_clk is
round 1's P3 / floor-5 law fitted on the round-1 training table, the same family as the served L2 clock
but not its refit; the guard reads structure, not the served artifact.

### 7.6 Pre-arm diagnostic: where the regulation ties are lost (reported, no decision line)

On `R9` (and `R0`) vs the actual (possessions_v4 2025 for the states, verified finals for OT), all with one
script and identical definitions on both sides:

1. Margin `m(T)` = home margin at the start of the first period-2 possession starting at `<= T` seconds
   (end-of-regulation margin if none), T = 120, 60, 30, 10. Distribution of `|m(T)|` for k = 0..8+.
2. Kernel `P(regulation tie | |m(T)| = k)` at every T, and a shift-share of the tie-rate gap at each T into
   arrival (the distribution of `|m(T)|`) and conversion (the kernel), both substitution orders averaged.
   Where the arrival share jumps between two T values is where ties are lost.
3. Behaviour cells inside the final 2:00, by offence role (trailing 1-3, trailing 4-6, tied, leading 1-3,
   leading 4-6) and clock bucket: possession duration used; share of possessions that are free-throw-only
   (FTA > 0, no FGA, no TOV: the foul-to-stop-the-clock outcome when the defence trails); three-point
   share of FGA; points per possession; FT make rate; OREB continuation.
4. Tied-game final possessions: possessions starting tied with `<= 35` s left: duration used, share that
   runs to the horn, points distribution, and the share of those games that end regulation tied.
Season actual (all verified 2024-25 games) and sample actual are both shown; sim cells under 200
possessions and actual cells under 200 are labelled UNDERPOWERED.

### 7.7 What cannot run, stated now

- **Fold 1 closed loop: NOT RUN.** No 2023-24 engine inputs v3 or fold-1 serving artifacts exist (HANDOFF
  open item 10). Fold 1 enters only through 7.5.
- Screening size is 500 x 25 as in 6.4. A candidate that clears it is re-read at 500 x 200 and 5,710 x 200
  on the box (request written, PENDING); a local closed-loop tap sized from a measured slice is direction
  only.

---

## 8. RESULTS -- round 3, RUN 2026-10-01 (lane L; status PROPOSED -> RUN; NO ARM ADOPTED, no served default changed)

Full evidence: `docs/tests/late_game_round3_2026-10-01.md`. Base = served stack v2 (section 7). Off path parity v9
PASS bit-identical; first half bit-identical in every arm; all 11 runs re-verified bit-identical on 20-game
slices after dirty-tree starts. 500 verified games x 25 paired seeds; floor = max(draw SD over R9 + 4 offsets,
2 x paired game-bootstrap SE).

| arm | P(0)/P(1) | floors vs base | OT | vetoes | candidate |
|---|---:|---:|---:|---|---|
| R9 (served v2) | 0.552 | -- | 0.0290 | -- | -- |
| Dt9 | 0.672 | +2.13 | 0.0341 | PASS | no (< 1.0) |
| Dtt9 | 0.709 | +1.76 | 0.0319 | PASS | no (< 1.0) |
| D9 | 0.841 | +2.83 | 0.0402 | FAIL G1 mean, G1 SD, half share, window possessions | no |
| R0 (v2, foul reference) vs R9 | 0.560 | +0.08 | 0.0297 | -- | attribution |
| Dt0 / D0 vs R0 | 0.676 / 0.807 | +1.94 / +2.67 | 0.0342 / 0.0425 | -- | attribution |

Offline guard (7.5) passes for every arm on both folds (tied rows: D law -1.11 CRPS, +9.3 floors F2, +7.9 F1).
Decision rule 6.5: no candidate. Attribution: the extra window possessions are D's LEADING-role draws (Dt and Dtt
pass the veto); the corrected foul state does not move the tie rate. Where the ties are lost (7.6): arrival at
2:00 is 16% of the gap; the loss accrues between 1:00 and 0:10 (arrival share 73% at 0:10), through the tied offence
not holding (fixed by Dt), end-of-clock possessions scoring ~1.0 PPP vs ~0.6 actual (NOT fixed; tied final
possessions score 0 in 50% vs 64%), and the trailing team fouling late (leading offence 10 s vs 4 s inside 30 s).
Box 500 x 200 re-read requested (`docs/ops/box_queue/d1001_L_1.md`), PENDING at writing.

---

## 9. PROPOSED -- Round 4 pre-registration: end-of-period possession value and the leading team's clock, composed on `clk_Dt` (written 2026-10-01 09:20-09:45 EDT by lane L, BEFORE any round-4 arm was fitted, wired or run; NOT RUN, NOT ADOPTED)

**Nothing below has been fitted or simulated. No served default changes; every arm is a new default-off flag.**
PM brief 2026-10-01 after round 3 (section 8): (a) last-second possession value, (b) the trailing team's late
fouling / leading team's time use inside 30 s without extra possessions, (c) price the leading-team late FT
make defect, do not fix it.

### 9.1 Motivating evidence (diagnostic only; no arm involved)

`scripts/diag_late_game_r4_owner_v1.py` on `lg4_R9_s25` (served v2, round-4 tap: both halves, made shots,
a per-shot log; its games are bit-identical to round 3's `lg3_R9_s25`) vs `possessions_v4` 2025. Output
`results/late_game/round4/owner_R9.json`. Possessions STARTING at <= 35 s:

- **No-shot at the horn.** The data end 18-29% of first-half possessions starting inside 10 s with no
  terminal event (`end_period`), and 21-29% of tied / 20-25% of trailing period-2 ones inside 6 s. The engine
  produces 0%: `possession_outcome` drops `end_period` chances as "the clock model's job" and the engine
  never draws one -- every possession ends in a TOV, shot or trip however little time is left.
- **Make rate of the shot at the horn.** First-chance shots with <= 1 s left at the shot: sim (mean fg_make
  probability) three 0.267 / jumper 0.364 / rim 0.558 in H1 (0.223 / 0.370 / 0.638 in H2) against actual
  0.163 / 0.213 / 0.369 (0.133 / 0.169 / 0.441). fg_make carries `seconds_remaining` and `chance_elapsed_s`
  but does not reproduce the buzzer effect in the closed loop. Above 1 s the sim is not too high.
- **PPP shift-share** (sim - season actual): H1 (-1,3] +0.18, (3,6] +0.22, (6,10] +0.21; tied P2 +0.28 to
  +0.39; trailing P2 (-1,3] +0.34. The volume term (no-shot + TOV + mix) and the make term carry it.
- Leading offence inside 35 s (period 2): TOV 0.21-0.29 per possession vs 0.07-0.18, FTA 0.97-1.09 vs
  1.30-1.71: the trailing team does not foul it, which is (b).

### 9.2 Arms, offline (one grader per family, both folds, fold 2 selects)

Folds: F1 trains 2022-2023 and tests 2023-24; F2 trains 2022-2024 and tests 2024-25. 2025-26 sealed. All
laws are deterministic cell laws: the reseed floor is zero (section 2.5.3) and the floor is the game-block
bootstrap SE of the PAIRED per-row delta. Hierarchical cells shrink to their parent with m = 50 pseudo-rows.
Periods 1-2 only; overtime keeps the served behaviour (OT-period foul-state rows are not trustworthy, PM note).

**(a1) No-shot law `BZ`** (`scripts/train_late_game_r4_buzzer_v1.py`): P(possession ends `end_period` |
start state), possessions starting at <= 35 s. Start buckets (0,1] (1,3] (3,6] (6,10] (10,15] (15,20] (20,35].
`BZ0` period; `BZ1` period x bucket; `BZ2` x role3; `BZ3` x start type (made / DREB / TOV / other).
Metric: log loss on test-fold rows. Selection: the simplest arm within one floor of the best on F2; it must
beat `BZ0` by > 1 floor on F2 with the same sign on F1 (the time term is real), else (a1) is NOT RUN.

**(a2) Buzzer make law `MK`** (`scripts/train_late_game_r4_make_v1.py`): first-chance FGA whose time left AT
THE SHOT is <= 1 s (data: chance end clock; engine: possession-start clock minus the fed
`chance_elapsed_s`). `MK0` = make rate by class x period over ALL first-chance shots (no buzzer term);
`MK1` = class x period on the <= 1 s rows; `MK2` = class x period x {0 s, 1 s}. Metric: log loss on test-fold
<= 1 s rows. Same selection rule against `MK0`. Stated limit: the served fg_make's own offline prediction on
these rows is not produced (its inputs are per-game engine arrays); the closed loop is the comparison with
the served model, and the 9.1 table is the motivating one. Responsiveness: the realised <= 1 s make rate by
offence prior make-quintile is reported; if it slopes in the data and the cell law is flat, that is recorded
as a known limitation, not hidden.

**(b) Leading-team clock law `LGL`** (`scripts/train_late_game_r4_clock_v1.py`): `clk_D`'s Kaplan-Meier cell
family with role3 refined to round 1's five bands (`eg_role6`: trail >= 4, trail 1-3, tied, lead 1-3, lead
>= 4) x fine bucket x bonus x prev_end, window rows, no floor. The trainer first reproduces round 1's D PMF
exactly (else stop). Served on LEADING offence rows only, with `clk_D` on tied rows (`clk_Dt`). Offline:
section 7.5's grader extended with the composite `Dt+LGL` (tied: D, leading: LGL, trailing: A), CRPS and
censored log-likelihood per role and per role x bucket. Guard: on leading rows, `LGL` must not be worse than
A by > 1 floor on F2.

### 9.3 Closed-loop arms (500 verified games x 25 paired seeds, fold 2, served v2 base)

Flags (default off; unset = served, parity v9 bit-identical): `ENGINE_LATE_GAME=clk_Dt` (round 3),
`ENGINE_LATE_GAME=clk_DtL` (tied: D law, leading: LGL law; window as section 4.1), `ENGINE_LG_BUZZER=<BZ arm>`
(no-shot draw on its own stream (seed, game_id, "lg_buzzer"); a no-shot possession runs to the horn, scores
nothing, records no box event), `ENGINE_LG_MAKE=<MK arm>` (replaces fg_make's probability on first-chance
shots with <= 1 s left at the shot, periods 1-2).

| arm | flags | role |
|---|---|---|
| R9, R9_f1..f4 | none | reference + floor draws: round 3's runs, reused (their games are bit-identical under the round-4 tap; parity v9 is re-checked at the round-4 HEAD before any arm runs) |
| Dt | `clk_Dt` | round 3's `lg3_Dt9_s25`, reused |
| Dt+a1 | `clk_Dt` + BZ | |
| Dt+a | `clk_Dt` + BZ + MK | the (a) set |
| Dt+b | `clk_DtL` | the (b) set |
| Dt+a+b | `clk_DtL` + BZ + MK | the full set |

If (a1) or (a2) is NOT RUN under 9.2, the arms containing it are dropped and the rest run.

### 9.4 Primary, floors, vetoes, decision rule

- **Primary and qualification: section 1.3 unchanged.** P(0)/P(1) in floors vs R9; a candidate needs > 1
  floor toward 1.546, P(0)/P(1) >= 1.0, and every veto passing. The OT band 0.046-0.055 is reported.
- **Floor:** Decision 12, as section 7.4 (draw SD over R9 + four offsets, 2 x paired game-bootstrap SE).
- **Vetoes:** section 7.4's (G1 mean and SD, G5 margin and total SD ratios, G9 margin and total bias, half
  share, window possessions by k with 7.4.1's reading) **except the first-half bit-identity veto**, which
  (a) arms cannot satisfy by design (they act at the first-half horn). It is replaced by the **first-half
  buzzer test**: H1 points per possession for possessions starting at <= 10 s must move TOWARD the actual
  (season 2024-25) and must not overshoot it by more than one floor (floor = max(draw SD over the five R9
  runs, 2 x paired bootstrap SE)). The first half outside the final 35 s is reported.
- **Decision.** Among candidates, the simplest in the order Dt+a1 < Dt+a < Dt+b < Dt+a+b. If none qualifies,
  the **clear best** is the arm that passes every veto and whose primary exceeds every other arm's by > 1
  floor; the PM's brief then applies (default-off flags, parity v9 on a clean `src/`, box request
  `d1001_L_2.md` for the full-size paired read). If no arm is a candidate or a clear best: NO ARM ADOPTED.
- **Multi-level reported lines:** by time bucket, score state (role bands), half, site (home / away / neutral
  offence), per game (paired), per team (UNDERPOWERED below 200 sims), the 7.6 tie-loss diagnostic and the
  9.1 owner table on every arm.

### 9.5 Pricing the leading-team FT make defect (c): reported, not an arm

The FT model is not touched. The expected tie-rate cost of fixing it is estimated on the R9 possession log by a
first-order re-scoring, stated as such: for each period-2 possession in the final 2:00 whose offence leads,
by role band x clock bucket where the sim's FT make is below the actual (round 3 section 2.2), each missed
FT is converted to a make with probability (act - sim) / (1 - sim) on a fixed seeded stream; the margin
change is carried to the end of regulation with no behavioural response, and the change in P(0) and
P(0)/P(1) is reported with a game-bootstrap SE. It is an estimate for Decision 11 bookkeeping, never applied
to any sim output.

### 9.6 Lane H's clock change

The window arms wrap whatever the served clock adapter is, through its `_latent` and `inner._frame`
(section 4.2.2). They survive a lane-H refit that keeps the `LatentClockAdapter` interface (the window law is
then scaled by the new latent). A clock that conditions duration on the possession's OUTCOME would invert the
engine's draw order (duration before event); the window laws, which are outcome-free, would then need
re-deriving as conditional laws. This is stated now and checked against lane H's committed code at report time.

---

## 10. RESULTS -- round 4, RUN 2026-10-01 (lane L; status PROPOSED -> RUN; NO ARM ADOPTED, no served default changed)

Full evidence: `docs/tests/late_game_round4_2026-10-01.md`. Offline guards pass on both folds:
- `BZ3`: +18.2 floors vs `BZ0` on F2, +16.3 on F1.
- `MK2`: +12.7 / +15.8 vs `MK0`.
- `LGL` on leading rows: +13.1 / +13.2 vs A.

Closed loop, 500 x 25 vs R9 (Decision 12 floors; the floor draws were re-run under the round-4 tap and are bit-identical to round 3's):

| arm | P(0)/P(1) | floors | OT | vetoes failing |
|---|---:|---:|---:|---|
| Dt | 0.672 | +2.13 | 0.0341 | none |
| Dt+a1 (BZ3) | 0.796 | +2.93 | 0.0383 | G9 total, half share, G1 SD |
| Dt+a (BZ3+MK2) | 0.832 | +3.06 | 0.0382 | G9 total, half share, G1 SD, H1 buzzer (overshoot 0.642 vs 0.730) |
| Dt+b (clk_DtL) | 0.701 | +1.87 | 0.0410 | G1 mean, G1 SD, half share, window possessions |
| Dt+a+b | 0.848 | +3.02 | 0.0454 | G1 mean, G1 SD, G9 total, half share, window possessions, H1 buzzer |

No candidate. No clear best: the set arms are within one floor of each other. No full-size request was filed.

Owners found:
- **No-shot at the horn.** The engine produces none against 18-29% in the data. BZ3 fixes the H1 buzzer PPP: 0.764 vs 0.730, base 0.943.
- **Buzzer make.** MK2 overshoots, because fg_make under-rates quick shots at 1-10 s left.
- **The first-half horn possession count.** The sim has about 35-80% too many; this is a clock defect behind the half-share veto.
- **Trailing-offence time use.** It is behind the leading law's extra possessions.

(c) pricing: fixing the leading late FT make costs -0.0014 OT rate (SE 0.0005) and adds +0.037 to P(0)/P(1) (SE 0.015).

Round-3 500 x 200 box re-read: `clk_Dt` +6.84 floors, ratio 0.679, and it also fails G1 SD (+2.3); verdict unchanged.

---

## 11. PROPOSED -- Round 5 pre-registration: BZ3 + a five-band window law on trailing AND leading rows, `clk_D` on tied rows (written 2026-10-01 ~10:30 EDT by lane L, BEFORE any round-5 arm was wired or run; NOT RUN, NOT ADOPTED)

**No served default changes.** PM ruling after round 4: BZ3 is VALIDATED-PENDING-SHIP-ACTION (its total-bias
veto is a Decision 11 exposure). The make law (MK2) is held out. The trailing-role use of the five-band law
(`LGL`, round-4 artifact, unchanged; offline +10 floors on trailing rows in round 4's grade) is REGISTERED
here, before any closed loop uses it.

### 11.1 Arms

New flag value `ENGINE_LATE_GAME=clk_DtLL`: window rows (section 4.1) with a TIED offence get `clk_D`, rows with
a trailing or leading offence get `LGL`; no refit, same per-game latent and uniform as rounds 2-4.

| arm | flags | role |
|---|---|---|
| R9, R9_f1..f4 | none | reference + floors: round 4's `lg4_R9_*` (round-4 tap), reused |
| Dt | `clk_Dt` | round 3's run, reused (comparison) |
| Dt+a1 | `clk_Dt` + `ENGINE_LG_BUZZER=BZ3` | round 4's run, reused (comparison) |
| Dt+LL | `clk_DtLL` | does the trailing law remove the leading law's extra possessions? |
| **Dt+LL+a1 (the set)** | `clk_DtLL` + `ENGINE_LG_BUZZER=BZ3` | candidate |

### 11.2 Offline line (both folds, fold 2 selects; guard)

Round 4's clock grader extended with the composite `DtLL` (tied: D, trailing and leading: LGL) on round 1's
window test rows. Guard: on trailing rows and on leading rows separately, `LGL` must not be worse than A by > 1
floor on F2 (block-bootstrap SE of the paired delta). BZ3's offline evidence is round 4's (unchanged artifact).

### 11.3 Primary, vetoes, one registered change of framing

- **Primary and qualification: section 1.3 unchanged** (P(0)/P(1) > 1 floor toward 1.546 and >= 1.0; OT band
  reported). Floors as 9.4.
- **HARD vetoes:** G1 possession mean and SD, half share, window possessions by k (7.4.1 reading), first-half
  buzzer test (9.4), G5 margin and total SD ratios, G9 margin bias.
- **Changed framing (registered now, before any run):** G9 TOTAL bias is NOT a disqualifier for the set. It
  is reported as a priced exposure: the set's change in points per game vs R9, split by half (H1 / H2), with
  paired SE. Rationale: PM ruling that BZ3 removes points the stack was already short of (Decision 11).
- **Reported separately (the remaining gap):** the share of simulations reaching a tie at 1:00 / 0:30 / 0:10
  (arrival) and the one-point-finish rate P(1), each vs R9 and the actual, with the tie-loss shift-share.
- **Decision.** The set is a candidate if it qualifies under 1.3 and passes every hard veto. If not, it is the
  **clear best** if it passes every hard veto and its primary exceeds that of every other arm in 11.1 that
  also passes every hard veto by more than one floor (an arm failing a hard veto is not a competitor). If
  the set is a candidate or the clear best: flags stay default-off, parity v9 on a clean `src/`, and a box
  request `d1001_L_2.md` for the full-size paired read (5,710 x 200, four floor draws vs the plain default on
  lane D's S2 floors `d1001D_S2f{1..4}`). Otherwise NO ARM ADOPTED.
