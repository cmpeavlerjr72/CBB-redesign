# Late-game regime, ROUND 1: the blind bake-off (2026-09-18)

Lane: late-game regime. **Offline only. NOTHING IS ADOPTED, no served default is
changed, no closed loop was run** (the round hit a hard wall-clock deadline;
section 7 says exactly what remains and how to resume it). Pre-registration:
`docs/models/late_game/experiments.md` section 1 (2026-09-11) and its section-2
AMENDMENT (2026-09-18), both committed before any arm was fitted -- the
amendment at `e68395e`, before the first trainer started.

Inputs and code:

    scripts/diag_late_game_tap_v2.py         the served-stack tap (12,000 sims, 4 shards)
    scripts/diag_late_game_compare_v2.py     the re-read; results/late_game/compare_v5b_2025.txt
    scripts/build_late_game_design_v1.py     the event-half design + the leak test
    scripts/train_late_game_v1.py            the event half (arms A/B/C/D)
    scripts/train_late_game_clock_v1.py      the duration half (A_clk..D_clk)
    scripts/grade_late_game_v1.py            the ONE blind grader for both halves

    results/late_game/round1/event_results.csv   clock_results.csv   *_detail.json
    results/engine_v0/F2_2025_s200_v5b_A         the 75-seed served run (levels)

---

## 1. The v5b re-read, and why it rewrote the diagnosis

Full table in `experiments.md` section 2. Three results, in order of how much
they change the lane:

**(a) The defect is intact under the served stack.** `P(0)/P(1)` is 0.546 under
`v5b_glat_pmean` (428,250 sims) against 0.553 under the v3c run A and an actual
1.546; excess kurtosis +0.320 against +0.740. The clock adoption did not touch
the tie/OT shortfall, so the round's premise stands.

**(b) The 2026-09-11 headline was a measurement artefact, and it is withdrawn.**
The v1 tap recorded no offence-side flag, so the sim's "role at the 2:00 mark"
applied ONE offence's signed anchor to both teams' possessions; each role cell
was a ~50/50 mixture and any real split cancelled inside it. The actual side
never had the defect. Signed per possession:

| split | SIM v5b | ACTUAL | reproduced | published 2026-09-11 |
|---|---:|---:|---:|---|
| trailing-minus-leading three-point share | +0.1523 | +0.1575 | **96.7%** | +0.007 ("0%") |
| leading-minus-trailing bonus-FT rate | +0.2256 | +0.2652 | **85.1%** | +0.024 ("9%") |

**(c) `CELL_DIMS` is not the served clock's cell grid.** The served arm is
`empirical_km3_srfloor | P3`, whose cells include `eg_regime` (trailing by >= 4
/ leading by >= 4 / close, inside the last 120 s) -- a role-conditioned,
clock-gated dimension that is doing work. The real structural limit is
`sr_floor_bucket = 5`: every possession with under 45 seconds left is served the
45-59 s law. The `(a)/(b)` attribution moves to **(b) 80.7% / (a) 12.5%** on 12x
the simulations, and the closed-loop band to 0.046-0.055.

---

## 2. Event half -- fold 2 (SELECTION), the blind table

Graded on FIRST chances of the selection gate (`period == 2`,
`seconds_remaining <= 120`, `|score_diff| <= 6`), **n = 17,596**, by
`grade_late_game_v1.py`, one code path per cell. Primary = multiclass log loss.
Floors: `max(seed floor, game-level block-bootstrap SE)`; the seed floor is a
spec-identical retrain under seed 1.

| arm | bundle | log loss | Δ vs A | seed floor | block SE | binding floor | floors beaten | R1 | R2 | R3 | R3b | R6 (pp) |
|---|---|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|:--:|---:|
| **A** | L0_reference (served) | 1.32443 | 0 | 0.00061 | 0.00645 | 0.00645 | -- | PASS | PASS | PASS | PASS | 1.628 |
| B | L1_role | 1.32466 | +0.00023 | 0.00172 | 0.00655 | 0.00655 | -0.04 | PASS | PASS | PASS | PASS | 1.616 |
| B | L2_role_poss | 1.31378 | -0.01065 | *n/r* | 0.00654 | 0.00654 | **1.63** | PASS | PASS | PASS | PASS | 1.620 |
| **B** | **L3_gates** | **1.31263** | **-0.01180** | *n/r* | 0.00659 | 0.00659 | **1.79** | PASS | PASS | PASS | PASS | 1.611 |
| B | L4_team | 1.31356 | -0.01087 | *n/r* | 0.00657 | 0.00657 | **1.66** | PASS | PASS | PASS | PASS | 1.614 |
| C | L0_reference | 1.32718 | +0.00275 | 0.00010 | 0.00671 | 0.00671 | -0.41 | PASS | PASS | PASS | FAIL | gated |
| C | L1_role | 1.32605 | +0.00162 | 0.00145 | 0.00677 | 0.00677 | -0.24 | PASS | PASS | PASS | FAIL | gated |
| C | L2_role_poss | 1.31611 | -0.00832 | 0.00100 | 0.00683 | 0.00683 | 1.22 | PASS | PASS | PASS | FAIL | gated |
| C | L3_gates | 1.31635 | -0.00808 | 0.00063 | 0.00697 | 0.00697 | 1.16 | PASS | PASS | PASS | FAIL | gated |
| C | L4_team | 1.31666 | -0.00777 | 0.00123 | 0.00698 | 0.00698 | 1.11 | PASS | PASS | PASS | FAIL | gated |
| D | L3_gates | 1.33599 | +0.01156 | *n/r* | 0.00746 | 0.00746 | -1.55 | PASS | PASS | PASS | FAIL | gated |
| D | L4_team | 1.33429 | +0.00986 | *n/r* | 0.00741 | 0.00741 | -1.33 | PASS | PASS | PASS | FAIL | gated |

*n/r* = the seed-1 retrain was not reached before the deadline (section 7).
**Every seed floor that WAS measured is 0.00010-0.00172, i.e. 2.6-11x SMALLER
than the block-bootstrap SE, so the block SE is the binding floor on every row
and a missing seed retrain cannot change a verdict unless it exceeded the
largest observed seed floor by 3.8x.** That is stated as a bound, not as a
claim that the cells were run.

**Reading.** `B | L3_gates` -- adding `role`, `gt_margin`, `poss_deficit`,
`in_double_bonus`, the two interactions and the two behavioural gates to the
EXISTING pooled `possession_outcome` fit -- is the only family that clears its
floor with every gate passing, at **1.79 floors**. Arm C (the regime-conditioned
refit) clears by 1.1-1.2 floors and **fails R3b**. Arm D (the dedicated,
role-stratified end-game model) is WORSE than the reference by 1.3-1.6 floors:
splitting 44k window rows three ways costs more than the sign flip is worth.
Per the pre-registered tie-break A > B > C > D, **B beats C and D**, and
`experiments.md` 1.7.3 names this outcome explicitly: the "regime layer"
hypothesis is REJECTED on the event half in favour of columns in the model that
already exists.

**But the gain is small and the reference is already good.** Arm A's own
predictions inside the window, against the actuals on the same rows:

| cell (fold 2, first chances) | n | pred bonus-FT | actual | pred 3PA share | actual |
|---|---:|---:|---:|---:|---:|
| trailing offence | 8,222 | 0.1411 | 0.1496 | 0.4871 | 0.4870 |
| tied | 1,169 | 0.1299 | 0.1292 | 0.3466 | 0.3230 |
| leading offence | 8,205 | 0.5381 | 0.5430 | 0.2900 | 0.2889 |
| sec (60,120] | 6,657 | 0.1869 | 0.1881 | 0.3484 | 0.3586 |
| sec (30,60] | 4,106 | 0.3372 | 0.3324 | 0.3890 | 0.3636 |
| sec (10,30] | 4,305 | 0.4429 | 0.4606 | 0.4909 | 0.4795 |
| sec (0,10] | 2,528 | 0.4714 | 0.4889 | 0.6468 | 0.6870 |

Arm A's role splits are +0.197 / +0.397 against actuals of +0.198 / +0.393. No
cell is underpowered. **The event half is not the defect**, and 0.0118 nats on
2.4% of possessions is what the best arm buys.

**Per-team (the multi-level rule).** Bonus-FT rate per offence team, 361 teams
with >= 20 window chances: arm A MAE 4.50 pp, corr 0.791; `B | L3_gates` 4.24
pp, 0.812. **Responsiveness:** R3 (season-PPG quintile, the gate as written)
slope ratio A 0.918, B_L3 1.014; R3b (the as-of prior rating quintile, added by
the amendment because CLAUDE.md asks for a PRIOR) A 0.974, B_L3 0.973, and every
C and D arm 0.849-0.941, outside a [0.95, 1.05] read -- the window-only arms
lose team responsiveness, which is what 44k rows buys you. **R7:** the sign of
R1 and R2 survives both sensitivity gates for every arm. **R6:** no full-scope
arm moves the reference window by more than 0.017 pp relative to A.

---

## 3. Duration half -- fold 2 (SELECTION), the blind table

Window rows of the clock design, **n = 17,710 (1,249 horn-censored)**. Primary =
CRPS of the truncated law on uncensored rows; censored log-likelihood beside it.
The cell law is deterministic, so the floor is the block-bootstrap SE.

| arm | scope | cells | floor | CRPS | Δ vs A | block SE | floors beaten | cens. loglik | R4 max bucket gap (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **A_clk** | all rows | P3 | 5 | 4.14955 | 0 | 0.03298 | -- | -3.1851 | 2.577 |
| B1_clk | all rows | P3 | 0 | 4.01063 | -0.13892 | 0.03005 | 4.62 | -3.1252 | 0.802 |
| B2_clk | all rows | P3R (6-level) | 0 | 4.01012 | -0.13943 | 0.03300 | 4.23 | -3.1036 | 1.004 |
| C_clk | window | P3 | 0 | 3.98171 | -0.16785 | 0.03047 | 5.51 | -3.1864 | **0.139** |
| **C2_clk** | window | P3R (6-level) | 0 | **3.96353** | **-0.18602** | 0.03272 | **5.69** | -3.1554 | 0.943 |
| D_clk | window | role3 x bucket x bonus x prev_end | 0 | 4.02353 | -0.12603 | 0.03250 | 3.88 | **-3.0945** | 0.478 |

**Every arm beats the served reference beyond the floor, by 3.9 to 5.7 floors.
`C2_clk` wins the pre-registered primary.** `D_clk` wins the named secondary
(censored log-likelihood, 5.9 floors) and the role profile; `C_clk` wins the
aggregate clock profile. The two metrics disagree on the ordering of C2 and D
and that is reported, not resolved by picking one.

**R4, the clock profile** (truncated predictive mean vs the actual on uncensored
rows -- the same population on both sides; every cell 598-6,584 rows, none
underpowered):

| arm | (60,120] | (30,60] | (10,30] | (0,10] | (10,30] trail/lead | (0,10] trail/lead |
|---|---:|---:|---:|---:|---|---|
| A_clk (served) | 18.28 | **10.45** | 6.84 | 2.27 | 7.67 / **5.95** | 2.98 / **1.90** |
| B1_clk | 18.28 | 13.82 | 6.75 | 2.12 | 7.80 / 5.58 | 2.85 / 1.74 |
| B2_clk | 18.01 | 12.62 | 6.42 | 2.12 | 8.27 / 4.45 | 3.05 / 1.62 |
| C_clk | 18.88 | 13.03 | 6.54 | 2.13 | 7.45 / 5.51 | 2.71 / 1.82 |
| C2_clk | 18.58 | 12.08 | 6.13 | 2.13 | 7.75 / 4.40 | 2.83 / 1.76 |
| D_clk | 18.89 | 12.54 | 6.33 | 2.12 | **8.55 / 3.60** | **3.43 / 1.40** |
| **ACTUAL** | 19.01 | 13.02 | 6.50 | 2.22 | 8.90 / 3.40 | 3.57 / 1.39 |

Two things this table settles. First, the floor's damage is **concentrated in
(30,60]**, where truncation cannot hide it: the served arm predicts 10.45 s
against 13.02, and every no-floor arm lands at 12.1-13.8. Second, the served arm
cannot represent the **leading team being fouled off the ball**: 5.95 s and
1.90 s where the actual is 3.40 and 1.39. `D_clk` reproduces both cells almost
exactly (3.60 / 1.40) and is the only arm that does.

**R5** (offline proxy: mean duration by the live `|score_diff|`; the gate's own
possession-count form needs the engine). Actual span from `|m| = 0` to
`|m| = 6`: 19.38 -> 10.08 s, i.e. 9.30 s. Predicted spans: A 4.82, B1 6.00,
C 5.82, D 6.15, C2 3.66, B2 3.06. **Every arm, including the winner, is short**,
and the missing behaviour is specific: at a TIE inside the final 2:00 the real
offence holds for the last shot (19.38 s) and no arm predicts above 14.6.

---

## 4. The verdict, against the pre-registered decision rule

**NO ARM IS ADOPTED BY THIS DOCUMENT. The PM decides.** What the round
establishes, against `experiments.md` 1.7:

1. **Event half.** `B | L3_gates` clears arm A beyond its floor (1.79) with
   every gate passing, and the tie-break puts it ahead of C and D. Per 1.7.3
   this REJECTS the regime-layer hypothesis on the event half. The gain is
   0.0118 nats on 2.4% of possessions and the reference's role splits are
   already right to three decimals, so the honest description is "a small,
   real, safe improvement to an already-adequate sub-model", not a fix for G7.
2. **Duration half.** The served clock is wrong inside the window by 3.9-5.7
   floors on the pre-registered primary, and the mechanism is identified
   (`sr_floor_bucket = 5`). `C2_clk` wins the primary; `D_clk` wins the
   secondary and the role profile. **This is the round's finding.**
3. **Neither half has been through a closed loop**, so 1.7.4 is unmet and
   nothing may ship. Section 7.

**The one piece of prior evidence that decides what to run next.** Clock round 4
already closed-loop-tested a GLOBAL floor removal (`v4_nofloor_P3_s1`, arm A4)
and rejected it: offline it beat the served arm by 1.75 CRPS floors, and in the
25-seed paired loop it was the WORST arm on G1 (+1.561 possessions/game against
the served +1.156, floor 0.180), because removing the floor everywhere
manufactures extra possessions **at the horn of the FIRST half**
(`clock/experiments.md` 15.3-15.5). Arms `C_clk`, `C2_clk` and `D_clk` are
fitted and served on window rows only, so a gate at `period == 2 and
seconds_remaining <= 120 and |score_diff| <= 6` removes the floor **exactly
where round 4 shows it is wrong and nowhere near where round 4 shows it is
load-bearing**. That is a genuinely new arm, not a re-run, and it is the
recommended round-2 candidate.

---

## 5. Evidence levels, as required

Overall (sections 2, 3); per-game (the block-bootstrap floor resamples GAMES,
not chances); per-team (361 teams, section 2); per-possession-type (role, first
/ continuation, bonus / no bonus, four clock buckets, section 2; prev_end and
bonus enter the duration cells); per-player: **not applicable and not claimed**
-- player attribution inside the window is out of this round's scope
(`experiments.md` 1.9). Every cell carries its n and the grader labels anything
below 200 rows UNDERPOWERED; no cell in any table above is underpowered.

---

## 6. What was checked and is clean

- **Pre-outcome.** `start_score_diff` re-asserted by this lane's own L27 own-row
  delta test: 98.45% of scoring chances move the NEXT chance's start margin by
  exactly this chance's own points, against 0.90% for the post-outcome
  alternative. `duration_s` is the duration half's TARGET and appears in no
  feature or cell. `is_transition` is post-outcome but PRE-EVENT (the engine
  draws the duration first), so it stays in the event half and is banned from
  the duration half.
- **Home/away/neutral** is a three-level feature in every event bundle
  (`site_home`, `site_away`, `site_neutral`) and in the duration cells' team
  block.
- **Rule-era flags** stay in GameState; no era constant is baked into any fitted
  object here.
- **Leak test** on the four as-of window team rates: change-form correlation with
  own-week margin -0.0151 / +0.0109 / +0.0041 / -0.0051 on 22,275 week changes,
  all inside the 0.15 bar -- and all BELOW the 0.04-0.08 honest baseline, which
  says they carry little signal. `L4_team` is duly no better than `L3_gates`.
- **Seal.** `PO.fold_slices` and `ck.fold_slices` guard every slice; 2025-26
  never enters a fold.
- **No hand tuning.** No multiplier, cap, clip, offset or blend anywhere.

---

## 7. What is PARTIAL, and the commands to finish it

The round ran into a hard machine deadline (21:30 ET, 2026-09-18). Three things
were started and not finished, and nothing else was cut.

1. **No closed loop was run.** `experiments.md` 1.7.4 is therefore unmet and no
   arm can ship on this evidence. The cheapest informative next step needs NO
   new code, because a no-floor P3 S1 schedule is already fitted and already
   wired: a paired screen of `ENGINE_CLOCK=v4_nofloor_P3_s1` against
   `v3c_srfloor_P3_s1` prices the floor on THIS lane's primary (`P(0)`, `P(1)`,
   the tie rate), which round 4 never read:

        ENGINE_CLOCK=v3c_srfloor_P3_s1 .venv/Scripts/python.exe scripts/run_engine.py \
          --max-games 500 --seeds 25 --no-players --tag lg1_floor_R
        ENGINE_CLOCK=v4_nofloor_P3_s1 .venv/Scripts/python.exe scripts/run_engine.py \
          --max-games 500 --seeds 25 --no-players --tag lg1_floor_N

   The GATED arm (`C2_clk` / `D_clk`) needs a composite adapter -- served law
   outside the window, window law inside -- behind a DEFAULT-OFF flag. That is
   new engine code and was not written.

2. **Five event-half seed-1 floor cells were not reached** on fold 2
   (`c038` B_L2, `c039` B_L3, `c040` B_L4, `c046` D_L3, `c047` D_L4) and four on
   fold 1 (`c013`-`c016`, the full-scope B cells; fold 1 is not the selector).
   Every seed floor that WAS measured is 2.6-11x smaller than the block SE, so
   the binding floor is unaffected unless a missing one is 3.8x the largest
   observed. Resume:

        .venv/Scripts/python.exe scripts/train_late_game_v1.py \
          --only c038,c039,c040,c046,c047,c013,c014,c015,c016 --pops first --meta-suffix _r
        .venv/Scripts/python.exe scripts/grade_late_game_v1.py

3. **The event half is graded on FIRST chances only** (17,596 of 20,312 window
   rows; continuations are 13.4%). The continuation cascade fit is the wall-clock
   cost and was dropped under the deadline. The first-chance predictions are
   BIT-IDENTICAL either way -- each population is fitted on its own rows only --
   so this restricts the graded population, it does not perturb any arm. Fold 1
   seed 0 has both populations if a cross-check is wanted. Resume by dropping
   `--pops first`.

Also not run, and named in the pre-registration rather than cut here: the S1
cadence cell (the amendment fixed S0 as every arm's primary cell), arm D's
three-channel ablation (a closed-loop instrument), and the full 200-seed kernel
re-read (this round used 12,000 simulations against the original 1,000).

---

## 8. What this document does NOT establish

1. **No arm has been through a closed loop.** Offline CRPS and log loss rank
   information; only a paired-stream engine run prices feedback (L26). The
   duration half's whole point is that it changes the NUMBER of possessions, and
   that is a feedback quantity by construction.
2. **The duration comparison is a bound on both sides.** The actual `duration_s`
   is post-outcome (L5) and censoring is informative; the predicted side is read
   on the truncated law so the two are the same conditional event, but neither
   is the intended duration.
3. **R5 is an offline proxy.** The gate asks for window possessions per game to
   slope with `|margin at 2:00|`; only the engine produces that. What is reported
   is mean duration by live `|score_diff|`.
4. **The event-half winner is small.** 1.79 floors on 2.4% of possessions, on a
   reference whose role splits are already right to three decimals. It is not a
   candidate to fix G7 and is not offered as one.
5. **The corrected attribution has its own noise.** 80.7 / 12.5 sits on 12,000
   simulations, not on 200 seeds; the DIRECTION is corroborated by the
   regime-free-kernel construction that uses no sim kernel at all, and only that
   direction is claimed.
6. **Nothing here explains the remaining `P(0)/P(1)` gap.** The engine is at
   0.546 against 1.546 and no arm in this round has been shown to move it.
