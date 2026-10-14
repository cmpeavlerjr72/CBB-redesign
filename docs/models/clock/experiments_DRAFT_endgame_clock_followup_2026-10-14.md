# Clock + late game: cell term on the served window law, final-possession conversion, bounded env arms (DRAFT, NOT COMMITTED, NOT RUN)

Status: DRAFT pre-registration for PM review, written 2026-10-14 by a Sonnet worker. Becomes a section of `docs/models/clock/experiments.md` (append-only; cross-referenced from `docs/models/late_game/experiments.md`) when the PM promotes it; NOTHING fits or runs before that commit. Post-freeze round (freeze ON). Adopts nothing, changes no served default. Served stack v3 (K2 clock) = control. Inputs: PROJECT_STATUS 10-13 findings (1)-(3), 10-14 findings (4)-(5); clock sections 41-45; late_game sections 15-16; `docs/tests/endgame_round_2026-10-13.md`; `docs/tests/clock_env_round10_full_2026-10-14.md`.

## 1. Why this round (evidence only; no arm involved)

| # | finding | source | owner | block |
|---|---|---|---|---|
| 1 | Served K2 window law miscalibrated by cell, worst (30,60]: 13+ leaders 9.0 vs 17.1 s, tied 11.4 vs 20.9 s. Pooled cell law (GT1) fixes cells (2.29 -> 0.34 s) but tempo slope 0.365 -> -0.009 (gate (b) fail). So: cell term ON the served law, not a replacement | endgame s9.1, late_game s16 | clock | A |
| 2 | Deviance primary is zero-mass smoothing: served K2 gives zero mass to 0.19% (F2) / 0.28% (F1) of window rows carrying 0.106 / 0.156 of its deviance; round 9 E1's full-set win was +0.0006 on positive-mass rows (floor 0.0017) | late_game s16, clock s42 | grader | all |
| 3 | Tie-rate gap is conversion in the 3-5 / 6-9 states, not bonus state: P(tie / 3-5) real 0.147 vs sim 0.076; down 1-3 with < 30 s: PPP real 0.893 vs sim 1.104, 3PA share 36.1 vs 31.1%, 2PA 26.0 vs 29.7% (5,500 games); leader in bonus 88.6-89.6% vs 94.1-95.8% (LF2 moved it ~ +1 pp, tie rate did not follow) | ot_rate_endgame s(c), endgame s6, s9.3 | possession_outcome (mix), fg_make (make), foul accrual A2 / LF2 (bonus) | B |
| 4 | E1 env term over-responds: possessions-on-env 0.207 vs actual 0.156 (control 0.015); env Q5 possession gap -0.27 -> -2.44; offence-quintile possession slope -0.20 -> +0.89 vs real +0.35; overshoot fails G1 SD, pace SD Q2, G9 slope, H1 share, D1 eFG gap at 200 vs 200 | clock s45 | clock | C |
| 5 | Served control G1 possession SD already 5.629 vs actual 5.474 (floor 0.014; +0.155 = 11 floors); owner = clock latent / variance function | clock s45 | clock | D |

Closed, do not re-propose: tempo terms A1-A3, M1/M2/M2D, G2D, K1/K2/K2M, calendar, home-site term, late-game rounds 2-6 arms, GT1/GT2 as replacements (GT1 stays a diagnostic reference only), EV1 (wins offline, moves MD3/MD5 the wrong way), LF2 as a standalone (selected, not adopted), Decision 9 arms, E1 itself (refuted s45).

## 2. Arms (all default-off flags over the one shared core; fold 2 selects, fold 1 confirms; 2025-26 sealed)

Common: no multiplier, cap, clip, offset, curve or blend on sim output; every term enters a sub-model's law or state; home/away/neutral stays in the base law (audit each arm's feature list); features league-centred at the as-of snapshot; rule-era flag from GameState; every training and grading row satisfies `created_at < tipoff`, asserted in code. Control `E0` = served K2 (`r8_K2/F{1,2}`); each block has a reseeded noise-floor arm (spec-identical refit under seed 1; never selectable).

### 2.A Block A: cell term ON the served K2 window law

Window: regulation, period 2 selects (period 1 fitted, reported only), possession start <= 240 s. Cell = margin bucket x time bucket: margin = {tied, leader x {1-3, 4-7, 8-12, 13+}, trailer x same} (9 buckets, offence perspective at possession start) x time (180,240], (120,180], (60,120], (30,60], (0,30] = 45 cells. Both arms keep the served row-level law (K2 cells, tempo tercile, chance structure, game latent) as the base so the tempo response is inherited, not refit.

| arm | form | params | note |
|---|---|---|---|
| `CA0` | served K2 (control) | 0 | |
| `CA0s1` | K2 refit seed 1 | floor | never selectable |
| `CA1` | RATIO form: `p(d \| row, cell) = p0(d \| row) * r_cell(bin(d)) / Z`, `r_cell` per cell over 8 duration bins, shrunk to 1 with m = 50 pseudo-rows then to the time bucket's pooled ratio (shrink as late_game s15.A GT1). Keeps served row heterogeneity | 45 x 7, shrunk | cannot move mass onto zero-mass rows (reported by the dual deviance, not hidden) |
| `CA2` | CELL-AFT form: served pmf scaled by `k_cell = exp(beta_cell)`, `p_k(t) = F0((t+0.5)/k) - F0((t-0.5)/k)` via `clock_r7.scale_pmf`; M2 Poisson pseudo-likelihood with the K2 conditional-cell fixed effects (clock s41.B choice 3); `beta_cell` hierarchically shrunk (m = 50) cell -> time bucket -> 0 | 45, shrunk | moves support, so zero-mass rows can gain mass: dual deviance is the arbiter |
| `CA3` | `CA2` with `beta` smooth in margin per time bucket: `b_L * max(m,0)/12 + b_T * max(-m,0)/12` (5 x 2 = 10 params) | 10 | simplest rung; wins ties |

Rank (ties to simpler): `CA3` < `CA2` < `CA1`.

### 2.B Block B: final-possession conversion

State `FPS`: regulation period 2 (period 1 reported), possession start <= 30 s, offence trailing by 1-3 (primary cell); trailing 4-6 and tied are reported cells. Offline rows are few (about 1,600 real rows per 5,500 games): UNDERPOWERED per cell, labelled.

| arm | law changed | owner | rank |
|---|---|---|---|
| `FP0` / `FP0s1` | served `possession_outcome` + `fg_make` / seed-1 refit | | control / floor |
| `FP1` | class probabilities (six `CLASSES`) refit on `FPS` rows (cell: trailing 1-3, shrunk to FP0 with m = 50): target 3PA 36.1 vs 31.1% and 2PA 26.0 vs 29.7% | possession_outcome | 1 |
| `FP2` | `FP1` + an `FPS` state term (cell intercept, shrunk m = 50 to 0) on the `fg_make` logit of shots taken in `FPS`: target PPP 0.893 vs 1.104 | fg_make | 2 |
| `FP3` | `FP2` read on top of `LF2` (late-foul accrual, possession_outcome s29, selected 10-13, not adopted) as the base | foul-accrual boundary | composition |

**Owner boundary.** Leader-in-bonus occupancy (88.6-89.6% vs 94.1-95.8%) and trailing-defence fouls per leader possession belong to foul accrual (A2; possession_outcome s29 `LF0-LF2`, already run). This round proposes NO clock-side arm for the bonus channel: the clock owns when the foul stops the clock (durations), not how many fouls accrue; a clock-side bonus arm would reopen `BL3` intentional-foul cells (closed) and risk compensating an upstream bias. Bonus occupancy is a REPORTED line and the base of `FP3` only. No post-hoc multiplier on points per possession; the make term is a feature of the fg_make law.

### 2.C Block C: bounded env arms (round-8 M2 estimator, walk-forward, `clock_env` serving path)

Finding (4): the response must be bounded by feature construction, never by shrinking a coefficient. Each arm states its bound; the bound is VERIFIED offline (C-pre) before any closed loop.

| arm | log k adds | why the response is bounded by construction (hypothesis, tested by C-pre) |
|---|---|---|
| `ED` | `c_o * q(off_c(offence)) + c_d * q(def_c(defence))`, `q` = within-snapshot quantile score in [-0.5, 0.5] | off / def split lets the fit allocate slope per side (E1's offence-quintile slope overshot to +0.89 vs +0.35 where the defence side may not); rank scores cap each tail contribution at \|c\|/2, so tails (env Q4/Q5 gap) cannot extrapolate |
| `EG` | `c * z_env_game`, identical for both offences in a game (mean of the two offence/defence pair values) | one game-level scale feeds both sides: cannot create a within-game possession imbalance (G1 SD, pace SD Q2 vetoes); fitted and applied at the same (game) level as the target slope; honours "one pace realisation per game" |
| `EM` | `d * (z_gap - mu_gap)`, no env term | mismatch-only, centred on the training mean: cannot move the env mean slope E1 overshot; the target gap slope is small (real +0.050); offline gap line is unresolvable (floor 1.7-11.8), so selected on the closed loop only |
| `EH` | `c_1 * z_off` on H1 rows, `c_2 * z_off` on H2 rows (separate fits; E1's `z_off`) | each coefficient is fitted where its own half slope is measured (actual H1 +0.080 vs E1 +0.101); no pooled term to over-shoot H1; H2 end-game state stays with the served window law |

Rank: `EG` < `ED` < `EM` < `EH` (fewest degrees of freedom first; PM may reorder). Reseed arms: `<arm>s1`.

**C-pre (offline, hard gate before any closed loop):** (i) possessions-on-env ratio (arm / actual) in [0.7, 1.15] (E1 was 1.299 offline and 1.33 in the sim); (ii) offence-z_off quintile possession slope <= real (+0.35) + floor; (iii) env-quintile possession gap not further from 0 than control in any quintile beyond floor (E1: Q4, Q5 failed); (iv) first-half env ratio in [0.7, 1.15]; (v) the round-9 positive-mass-row deviance gate.

### 2.D Block D: G1 possession SD as a registered line

Served control: **5.629 vs actual 5.474 (floor 0.014; 11.1 floors too wide)**; E1 5.650. Printed in every table header and registered for EVERY arm:

| block | role |
|---|---|
| A, B | hard no-regression veto (must not move away beyond floor); toward-actual reported |
| C | CO-PRIMARY with the two env primaries: toward 5.474 by more than its floor, or held within floor while both env primaries clear; moving away beyond floor is a veto (this refuted E1) |
| D arm (optional, Q4) | `VD1`: game-latent sigma and chance-count dispersion refit with the s41.B estimator; the SD is scored, never matched; a variance-function arm, not a scale |

## 3. Folds, grader, noise floor

Fold 1 trains through 2022-23, tests 2023-24; fold 2 trains through 2023-24, tests 2024-25; S1 monthly refits within the test season; fold 2 selects, fold 1 confirms sign (served-equivalent control, labelled as late_game s15.0 fold-1 limitation). Design `r6_L2/design_v2.parquet` joined to as-of ratings. 2025-26 sealed.

**ONE blind grader, new versioned file `scripts/grade_endgame_clock_v2.py`** (coded arm labels; modes offline | loop | full; reuses line definitions of `grade_clock_env_full_v2.py` and `grade_endgame_v1.py` unchanged; committed before any fit):

| line group | content | role |
|---|---|---|
| Dual deviance (finding 2) | per-possession duration deviance on (i) ALL rows and (ii) rows where the control assigns positive mass, with the excluded share (0.12-0.3%); per cell too. A win on (i) that does not survive on (ii) beyond its floor does NOT win | gate on (ii); (i) reported |
| Cell calibration | \|model - actual\| mean first-chance seconds over 45 cells (period 2), plus E[min(T, R)] | block A primary |
| Responsiveness | offence-tempo quintile slope not below control beyond floor (control 0.405 offline; absolute range [0.85, 1.15] NOT used, control-infeasible s41.B); z_off, defence-z, tempo elasticity not below control; every slope also by TEAM prior quintile | gates |
| Round-10 s45 line list | EVERY s45 line kept as a hard veto: G1 mean, G1 SD, H1/H2 share, G5 margin/total SD, G9 bias/slope, tempo elasticity, team pace slope, OREB/FGM/TOV/FTA within-game slopes, OT/tie rate, D1 margin and eFG gap, margin SD ratio overall and by \|spread\| Q1-Q5 (0.009 band), pace SD ratio overall [0.95, 1.05] and Q1-Q5, regulation total SD 0.994 [0.974, 1.014], H1 env magnitude (R2), env-quintile possession gap (R4), overall margin SD rise | hard vetoes |
| MD1-MD6 | late_game s15.1 as written (MD1 cov -12.04 -> -20.92; MD2 SD ratio by \|spread\| quintile; MD3 OT 3.54% -> 5.6%, never below 3.5%; MD4 H2 36-40 possessions 16.75 -> 15.52; MD5 13+ last-2-min points 9.68 -> 7.10; MD6 margin SD ratio 1.049 no rise) | primary for A (MD4, MD5) and B (MD3); no-regression for C |
| Multi-level | overall; per game (\|possessions sim - real\|, within-game SD); per team (prior quintile, share of teams with right sign); per possession type (start type; `FPS` class mix); per cell (A: 45 cells); month; site; H1 vs H2. Cells < 3,000 rows (slopes), < 150 games (elasticities), < 200 rows (cells): UNDERPOWERED | reported |

**Noise floor.** Spec-identical retrain under another seed (`CA0s1`, `FP0s1`, `<C arm>s1`); offline binding floor = max(reseed difference, 2 x paired game-block bootstrap SE, 200 draws, seed 12345). Closed loop and full size: max(2 x SD of four 200-seed control draws, 2 x paired bootstrap SE), per line. Ties to the simpler model.

## 4. Decision rule

1. **Offline (fold 2 selects; fold 1 same sign).** An arm WINS its block if it beats its control by more than 1 floor on the block primary (A: positive-mass-row deviance and cell calibration; B: `FPS` log loss and make-logit deviance; C: C-pre only, the env primaries are closed-loop lines), every responsiveness gate holds, and the dual-deviance gate holds. Simplest winner within one floor of the best wins. No winner: block REFUTED.
2. **Closed-loop sample = SCREEN only** (PM process ruling 2026-10-14): 500 verified fold-2 games x 25 paired seeds, local, four control draws, seed-count check (raise to 50 and relabel if a deciding floor is unresolved). It can eliminate an arm; it NEVER declares a candidate.
3. **CANDIDATE only on the full-size box read**: 5,710 games x 200 paired seeds vs served v3, four 200-seed control draws, every line read 200 vs 200 (50-seed-draw read reported beside it). Candidate = block primaries clear their floors (A: MD4 and MD5; B: MD3 and `FPS` PPP; C: both env primaries plus G1 SD) AND every hard veto, MD1-MD6 and R2-R4 pass. Otherwise NO ARM ADOPTED.
4. **Adoption** needs a paired-seed sim with no gate regressed on both folds, parity v10 bit-identical flags off on a clean `src/`, AND an explicit PM freeze lift per adoption. G9 total bias reported as priced exposure by channel (Decision 11).
5. Compositions (A+B, A+C, B+LF2) only for blocks that passed step 3 alone, as new paired reads.

## 5. Compute plan (AWS spot c7a.48xlarge, 200 GB disk, parity v10 check on the box first; user approval 2026-10-14: spend at PM discretion)

Reference: round-10 full read = 5 runs + post-processing, 101 min, about $4.9 (~12 min / ~$0.6 per 200-seed run). Offline fits and screens are local (no cost).

| block | offline | screen (local 500 x 25) | full size per candidate | est. cost |
|---|---|---|---|---|
| A (CA1-CA3) | local, minutes | winner(s), <= 2 | 1 arm run + controls | $1.5 reuse controls / $5 fresh |
| B (FP1-FP3) | local, minutes | winner(s) | 1 arm run + controls | same |
| C (ED, EG, EM, EH) | local, minutes; C-pre may kill arms free | up to 4, <= 2 survive | 1-2 arm runs + controls | $2 - $6 |
| D (VD1, only if approved) | local | 1 | 1 | $1.5 - $5 |
| Compositions | | | 1 run each | ~$1 each |

Planning cap about $15 for the round if each block yields one candidate and the four control draws (offsets 1000-4000) are reused from `results/clock_env10` on HF (valid only if engine SHA, inputs and parity v10 are unchanged; verify and state); fresh draws add ~$2.5 per full read. One box session per block group; reap at end; confirm instances terminated.

## 6. OPEN QUESTIONS FOR THE PM

1. Reuse the four 200-seed served-v3 control draws from round 10 (offsets 1000-4000) for every full read if engine SHA and inputs are unchanged? Recommend: yes, after a sha/parity check on the box; saves ~$2.5 per read.
2. One combined round with one grader, or A+B first and C later? Recommend: one spec and grader; run A and B first, C second (C-pre can reject C arms for free).
3. Allow `FP3` (final-possession arm on top of the selected-but-unadopted LF2)? Recommend: yes, composition-only; otherwise B reads on served v3 and the bonus channel stays report-only.
4. Add the optional variance-function arm `VD1` for finding (5), or keep G1 SD a registered line only? Recommend: line only this round (separate mechanism); it is a veto for A and B and a co-primary for C.
5. Accept `EG` (game-symmetric env) beyond the three forms named in finding (4)? Recommend: yes, it most directly honours "one pace realisation per game"; drop it if only the three named forms are wanted.
6. Tighten the C-pre env ratio window to [0.7, 1.15] and require offence-quintile slope <= real + floor (E1 passed [0.7, 1.3] offline and overshot at full size)? Recommend: yes.
7. Cell design (45 cells, m = 50 shrink, 8 duration bins for CA1) or collapse margin buckets? Recommend: keep 45 cells with hierarchical shrink; the (30,60] finding is cell-specific and underpowered cells shrink rather than drop.
8. Period 1 fitted, reported only (as endgame Q3)? Recommend: yes.
9. `FPS` scope: trailing 1-3 as the only primary cell, 4-6 and tied reported? Recommend: yes; the evidence (1,589 real rows) supports only the 1-3 cell.
10. Compute: AWS spot for full reads only, screens and fits local? Recommend: yes.

## 7. PM RULINGS on section 6 (Fable 5.1, 2026-10-14; binding at promotion)

1. Control draws: REUSE the four 200-seed served-v3 draws from round 10 (offsets 1000-4000) for every full read, conditional on an engine sha + parity v10 check on the box and unchanged fold-2 inputs; any mismatch means fresh draws, logged.
2. One spec, one grader. Run order: blocks A and B first, C second. Each block's offline read and screen may land on its own Opus day; the section is appended per block, append-only.
3. FP3 on LF2: ALLOWED as a composition arm, read alongside FP2 on served v3. A composite can only be adopted jointly with LF2, and only if the composite passes every gate LF2 alone failed (MD1-MD5). No partial adoption.
4. VD1: NOT this round. G1 possession SD is a registered line only (veto for A and B, co-primary for C). The variance function gets its own spec if finding (5) survives this round.
5. EG: ACCEPTED as a fourth C arm.
6. C-pre: ACCEPTED as tightened (ratio in [0.7, 1.15], offence-quintile slope <= real + floor, no env-quintile gap worse than control). It is an offline gate on responsiveness, not a tuning target; no arm is refit to pass it.
7. Cells: keep 45 with hierarchical shrinkage. The shrink strength is NOT fixed at m = 50 by convenience: register a grid {20, 50, 100} chosen on fold 1 only, then frozen for fold 2. Underpowered cells shrink, never drop, and are labelled underpowered in every table.
8. Period 1: fitted, reported only.
9. FP scope: trailing 1-3 is the only primary cell; 4-6 and tied reported only.
10. Compute: full reads on the box. Offline fits and screens run locally only when under about 2 h wall clock; anything longer also goes to the box (user authorisation 2026-10-14). The "bounded by construction" claims in block C are hypotheses; C-pre is their test and the section must say which arms it rejected.

Status: DRAFT with rulings. Promote verbatim (plus rulings) into experiments.md in the commit before the first fit runs. Nothing in this file has run.
