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
