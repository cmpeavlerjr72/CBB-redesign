# Late game: the margin-magnitude clock in lopsided end-of-half states, the bonus state, and the final-possession mix (DRAFT, NOT COMMITTED, NOT RUN)

Status: DRAFT pre-registration for PM review, written 2026-10-09 by a Sonnet worker, covering BOTH D3 (period-end possession surplus, garbage-time scoring) and D4 (overtime rate) because they share an owner family (end-of-half behaviour inside clock / late-game / foul accrual). Becomes the next section of `docs/models/late_game/experiments.md` (append-only) when the PM commits it; NOTHING runs before that commit. Post-freeze round (freeze ON). Adopts nothing; no served default changes; live `experiments.md` files were not edited. Parity v10 stays the off-path reference.

## 1. Why this round (diagnostic evidence; no arm involved)

`docs/tests/period_end_possessions_2026-10-09.md` and `docs/tests/ot_rate_endgame_2026-10-09.md` (fold 2 / 2024-25; 5,500 games with a matched pbp path x 50 seeds; served stack v3):

- **Second-half possession surplus is real** (segmentation ruled out by two independent raw-pbp counts): minutes 36-40 the sim plays 1.23 more possessions than the module's real count (raw Oliver 1.32, same-team runs 0.93; CIs exclude 0). It lives in lopsided states: 13+ at 2:00 (39% of games) the sim plays 9.06 possessions in the last two minutes vs 6.52 real and scores 9.68 vs 7.10 points. The SIM LEADER is too fast: seconds per possession leading by 13+ is 17.17 real vs 12.76 sim at 2:00 (17.92 vs 14.94 at 4:00), by 8-12 15.86 vs 13.87 and by 4-7 15.98 vs 14.31 at 4:00; trailers are faster in reality at 4-12 points (13.51 vs 14.41; 12.33 vs 14.01 at 4:00). Leader TOV rate in 13+ is 22.1% vs 19.0% at 4:00 (25.4% vs 21.1% at 2:00). The first-half end (0.15-0.56) is method-dependent and out of this round's selection.
- **OT deficit is conversion, not arrival and not variance**: regulation ties 3.54% sim vs 5.62% real (-2.08 pp), one-point margins 5.72% vs 3.64% (mirror); cumulative |m| <= 1 and <= 3 equal (9.26 vs 9.25%, 19.39 vs 19.80%). P(tie | |margin| at 2:00) is 0.55-0.75 of real in every state. The leader's free-throw trips per possession are 1.040 vs 0.924 (lead 1-3) and 1.269 vs 1.170 (lead 4-6); the team with the ball is in the bonus 88.6-89.6% vs 94.1-95.8% (leading by 1-6); trailing teams foul 0.65 / 0.75 / 0.46 times per leader possession (pbp, broader definition) vs 0.48 / 0.61 / 0.40 in the engine. Down 1-3 with under 30 s: 3PA share 36.1% vs 31.1%, 2PA 26.0% vs 29.7%, points per possession 0.893 vs 1.104.

What is already closed or registered (do NOT re-propose): late-game rounds 2-6 (`clk_D`, `clk_Dt`, `clk_Dtt`, `clk_DtL`, `clk_DtLL`, the five-band role law `LGL`, `BZ3` no-shot-at-the-horn, `MK2` buzzer make, `BL3` intentional-foul cells) all declared on `period == 2`, `seconds_remaining <= 120`, `|score_diff| <= 6`; clock rounds 1-8 (cell, tempo, outcome-conditioned K2, calendar). Not covered by any of them, hence new here: (i) margin-MAGNITUDE bands above 6 and the 4:00-2:00 span for the DURATION law; (ii) the event mix of the leading team in those states. Possession_outcome `experiments.md` section 29 (late foul accrual: window count law `LF0/LF1/LF2` for the trailing defence's non-trip fouls in the final 2:00, flag `ENGINE_LATE_FOUL`) is pre-registered and NEVER RUN; this draft does not rewrite it, it promotes it as block F below with the extra gates D4 supplies.

## 2. Blocks and arms (flags default-off; fold 2 selects, fold 1 confirms; 2025-26 sealed)

**Window (frozen before any fit):** regulation, periods 1-2, possession START `seconds_remaining <= 240`, score_diff in the offence's perspective read at the start of the possession; overtime and `> 240 s` keep the served law. Role = leader / trailer / tied; band of `|score_diff|` = 0-3, 4-7, 8-12, 13+; time bucket (180, 240], (120, 180], (60, 120], (30, 60], (0, 30]. SELECTION is on period 2; period 1 is fitted by the same arm and REPORTED only (its horn defect is owned by `BZ3`/clock round 4).

| block | arm | law inside the window | rank |
|---|---|---|---|
| C (clock) | `GT0` | served K2 (control) | 0 |
| C | `GT0s1` | spec-identical refit under seed 1 (noise-floor arm; never selectable) | floor |
| C | `GT2` | smooth margin response: AFT log-scale `b_L * max(m,0)/12 + b_T * max(-m,0)/12`, `m` = score_diff, times a fixed time taper `w(t) = min(1, (240 - t)/120)` (no fitted constant); 2 fitted parameters per half | 1 |
| C | `GT1` | cell law: first-chance duration pmf by role x band x time bucket (KM, hierarchical shrink to the time bucket then to GT0 with m = 50 pseudo-rows, as section 29) | 2 |
| E (event) | `EV0` | served `possession_outcome` class probabilities | 0 |
| E | `EV1` | class probabilities (the six `possession_outcome.CLASSES`) refit on window rows by role x band x time bucket, shrunk to EV0 with m = 50; the leading team's TOV/FT/shot mix in 8-12 and 13+ is the target | 1 |
| F (foul state) | `LF0`, `LF1`, `LF2` | exactly as `possession_outcome/experiments.md` section 29.2 (window and cell law unchanged); NOT re-specified here | 0 / 1 / 2 |

Composition arms only after a block wins alone: `GTw+EVw`, `GTw+LFw`, `GTw+EVw+LFw` (w = the block winner).

Home/away/neutral stays first-class in the base laws (audit every arm's feature list; the cells above are site-pooled, with a site main effect carried from the base law). No multiplier, cap, clip, offset, curve or blend on sim output: all arms enter a sub-model's law or state.

## 3. Primary metrics

**Offline (one blind grader per block, both folds, window rows of the test season, fold 2 selects):** C: held-out duration deviance (as clock sections 32/37) on window rows PLUS the cell calibration |model - actual| mean seconds per role x band x time cell; E: multiclass log loss of the six classes; F: section 29's min(K, 3) multinomial log loss. **Mandatory responsiveness gates (quintile slopes, not optional):** (a) model band means on actual band means of seconds per possession, separately for leaders and trailers, target 1 (the control's leader slope in 13+ is about 0.3); (b) the same by offence tempo quintile within the window (existing clock line; must not fall below the control's by more than its floor); (c) for E, predicted leader TOV rate by band slopes with actual; (d) for F, expected K by defence foul count must fall from 4-5 to 6+ as in the data.

**Closed loop (500 verified fold-2 games x 25 paired seeds, served v3 base; four control floor draws at disjoint seed offsets):**
- Block C primary: the H2 minutes 36-40 possession gap, both teams (M-end count; real 15.52, served sim 16.75), toward 0 by more than its floor; co-primary: last-2-minute points per game in the 13+ state (real 7.10, served sim 9.68).
- Block F and compositions: regulation tie rate and `P(0)/P(1)` (section 1.3's qualification line retained; real tie rate 5.62% in these games; section 1.3 target unchanged), the leader's bonus occupancy by role (real 94-96% leading 1-6; served 89%), trailing-team fouls per leader possession (real 0.65 / 0.75 / 0.46; served 0.48 / 0.61 / 0.40).
- Block E: leader TOV rate in 8-12 and 13+ (real 19.0 / 21.1% at 4:00 / 2:00 refs; served 22.1 / 25.4%).
- Reported, not selecting: down-1-3 final-possession mix and points per possession (D4 c), one-point finishes, lead-change and time-of-decision slopes, per-state tables of D3 (b).

## 4. Noise floor, vetoes, decision rule

**Floor.** Offline fits are deterministic: reseed floor 0, proved by refitting the selected arm under seed 1 (pmfs max abs diff 0); binding floor = 2 x paired game-block bootstrap SE (200 draws, seed 12345) of (arm - control). Closed loop: max(2 x SD of the four control draws, 2 x paired game-bootstrap SE), per line.

**Hard vetoes (any regression beyond floor on any, vs the served stack, fails the arm):** G1 possession mean and SD; first-half and second-half share; window possessions by k (round-4 reading); G5 margin and total SD ratios; G9 margin bias and slope; first-half buzzer test (round 4); the D1 lines (home margin non-neutral within its seed SE of +0.09; eFG home-minus-away gap), the D2 lines (game-possession slopes on `env` and `gap`, total slope on `env`, lead-change slope; none may fall), and the OT/tie rate must not move away from the actual. G9 total bias is reported as priced exposure by half (Decision 11 framing, as rounds 5-6), not a disqualifier by itself.

**Decision.** Offline an arm WINS a block if on fold 2 it beats its control by more than 1 floor on the block primary, all mandatory slopes hold, and fold 1 keeps the sign; the simplest winner (lowest rank) whose primary is within one floor of the best wins. In the closed loop a block winner or composition is a candidate if its closed-loop primary moves toward the actual by more than one floor and every hard veto passes; otherwise the clear best is the arm that passes every veto and beats every other vetoes-passing arm by more than one floor; otherwise NO ARM ADOPTED. A candidate gets parity v10 on a clean `src/` and a full-size 5,710 x 200 paired read request (box; any paid compute needs the user's approval), then the PM decides.

## 5. Order and cost

Offline for C, E, F is cheap (deterministic fits, two folds). Closed loop: C alone, F alone, then compositions only for block winners. D3's per-state tables and D4's conversion table are rerun on each arm's taps with `scripts/diag_c4_d34_endgame_v1.py` (the trajectory recorder, `--trajectory-seeds`) so that the multi-level evidence (overall, per game, per team quintile, per possession type, per state) is on the same footing as the diagnostic. Open questions for the PM: Q1 whether section 29 (block F) should be committed and run first, since the bonus gap (about 5 pp) is the larger OT lever and the clock block mostly removes points in decided games; Q2 whether block E is wanted at all before C and F are read (its evidence is the leader TOV overshoot only); Q3 H1 window arms are reported only; a separate first-half round is needed if the PM wants the 0.15-0.56 H1 surplus owned.
