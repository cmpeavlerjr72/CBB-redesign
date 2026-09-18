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
