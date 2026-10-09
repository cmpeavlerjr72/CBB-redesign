# Clock: scoring-environment and mismatch response of possession duration (DRAFT, NOT COMMITTED, NOT RUN)

Status: DRAFT pre-registration for PM review, written 2026-10-09 by a Sonnet worker. Becomes the next section of `docs/models/clock/experiments.md` (append-only) when the PM commits it; NOTHING runs before that commit. Post-freeze round (the model freeze is ON until the PM lifts it). Adopts nothing, changes no served default. The live `experiments.md` was not edited.

## 1. Why this round (diagnostic evidence; no arm involved)

`docs/tests/strength_responsiveness_decomp_2026-10-09.md` (fold 2 / 2024-25, 5,710 games x 50 seeds, served stack v3 = K2 clock; verdict section) found the sim's totals flatter than reality across team strength, and located the compression in the NUMBER OF POSSESSIONS, not in the rating or the per-possession efficiency model:

- Game possessions (both teams, regulation) on the scoring environment `env` = both offences' `off_c` + both defences' `def_c` (own as-of ratings, day before; SD 8.3): **real +0.152 per unit, sim +0.011**. On the mismatch `gap` = |home net - away net| (SD 8.0): **real +0.050, sim -0.008**. Jointly the same (+0.152 / +0.050 vs +0.011 / -0.008). Script `scripts/diag_c4_d2_supp_v1.py`.
- Per team-game, slope on the offence's scoring strength: possessions sim/real 0.06, seconds per possession 0.28 (CIs of the difference exclude 0); first-half only, so not an end-of-game artefact: seconds per possession fall 18.09 -> 17.80 across home-rating quintiles in reality, 17.84 -> 17.81 in the sim.
- The per-possession efficiency response is within about 7% (PPP slope sim/real 0.97, margin on own rating 0.992), so about 0.2 of the 0.25 points-per-unit game-total slope gap (total on env: real 0.600, sim 0.408, ratio 0.67, diff [0.146, 0.251], n = 5,424) is possessions.
- Real-minus-sim total gap rises from -0.76 (home-rating Q1) to +1.18 (Q5); about 3/4 of the rise is possessions.

What is already closed in this model (do NOT re-propose): tempo-only terms `A1`-`A3` (section 32), mean-scale Poisson tempo `M1`/`M2`/`M2D` (and calendar term), `G2D` (section 36-37), outcome-conditioned first-chance law `K1`/`K2`/`K2M` (sections 38-40; K2 is served), team random effects `A3`, dispersion and per-game latent rounds (sections 16-27), home-site duration term (sections 30-31). The own efficiency ratings (`off_c`, `def_c`) entered the clock only as columns of the early tree/dummy redesigns (round 1-2 `B_plus_teams`, features.md section 3), never as a continuous scale term on the served law, and those rounds were closed for other reasons. The mismatch `gap` has never been a clock input.

## 2. Arms (fold 2 selects; fold 1 confirms; 2025-26 sealed)

All arms keep the served K2 first-chance law (cells, `srfloor`, horn censoring, S1 monthly refits, game latent) and add a continuous accelerated-failure-time scale on top, served through the existing round-7/8 continuous-CDF path (`clock_r7`/`clock_r8`: `p_k(t) = F0((t+0.5)/k) - F0((t-0.5)/k)`), fitted by the round-8 Poisson pseudo-likelihood with baseline-cell fixed effects on every uncensored regulation row (`M2` estimator, the one section 37 found mean-consistent). Features are league-centred at the as-of snapshot (modeling rule: no raw levels) and strictly pre-tipoff (`created_at < tipoff` asserted in code; day-before ratings).

| arm | log k adds | params | rank |
|---|---|---|---|
| `E0` | nothing: the served K2 (control) | 0 | 0 |
| `E0s1` | spec-identical K2 retrain and latent refit under seed 1 (noise-floor arm; never selectable) | 0 | floor |
| `E1` | `c * z_off`, `z_off = (off_c(offence) + def_c(defence)) / SD`, the possession-level view of `env` | 1 | 1 |
| `E2` | `E1 + d * z_gap`, `z_gap = |net(offence) - net(defence)| / SD` (mismatch) | 2 | 2 |
| `E3` | `E2` with `c`, `d` interacted with the 6 start types | 12 | 3 |

Optional reference only (not selectable, already run in round 8): `K2M` (tempo AFT) to show the arm is not recovering the tempo term. Home/away/neutral stays a first-class feature of the base law; nothing in an arm removes it (audit each arm's feature list).

## 3. Folds and training

Fold 1 trains through 2022-23, tests 2023-24; fold 2 trains through 2023-24, tests 2024-25; S1 monthly refits within the test season. Fold 2 is the selection fold. Design `data/processed/models/clock/r6_L2/design_v2.parquet` joined to the as-of ratings; deterministic fits. Seed 0 except `E0s1`.

## 4. Primary metric and gates

**Offline primary (fold 2):** held-out duration deviance per possession (as sections 32/37). **Responsiveness line (the reason for the round, mandatory gate):** implied game possessions (1200 / mean implied duration, team-game level) regressed on `env` and on `gap`, the ratio (arm slope / actual slope) on the SAME held-out games, target 1.0; the control's ratios are about 0.07 and -0.16. An arm that is flat at the mean fails.

**Mandatory quintile responsiveness slopes (fold 2 and fold 1):** model quintile means of implied duration on actual means, for quintiles of (a) offence tempo (existing line; the arm must stay in [0.85, 1.15] and not fall below `E0`'s by more than its floor), (b) `z_off` (target 1; `E0` is about 0.3), (c) `z_gap` (target 1), (d) defence `z_off`. Also game-prior (tempo) quintile and the game-level tempo elasticity ratio of round 8 (must stay within [0.9, 1.1]; the served K2 value is the reference).

**Segments for every table:** overall; month (Nov-Mar, April underpowered); site (offence home / away / neutral); start type; conference tier; first half vs second half (the effect must show in the first half, where there is no end-game state); per team (offence-team quintile, plus the share of teams whose implied minus actual mean duration has the right sign); per possession type (start type). Cells below 3,000 rows (slopes) or 150 games (elasticities) are labelled UNDERPOWERED and never read as signal.

**Noise floor.** The fits are deterministic (Poisson + KM), so a spec-identical retrain is the identical object (proved by refitting the selected arm's F2 schedule under `--seed 1` and comparing pmfs, max abs diff 0; the latent sigma refit under seed 1 is `E0s1`). The binding offline floor per line is 2 x the paired game-block bootstrap SE (200 draws over games, seed 12345) of each (arm - `E0`) difference. Closed loop: four control draws at disjoint seed offsets (as `S0f1..f4`); floor = max(2 x their SD, 2 x paired game-bootstrap SE).

## 5. Decision rule

Offline, an arm WINS if on fold 2 (a) deviance beats `E0` by more than its floor, (b) the `env` and `gap` possession-slope ratios each move toward 1 by more than their floors and land in [0.7, 1.3], (c) the quintile slopes (a)-(d) above hold, (d) fold 1 does not reverse the sign of (a) or (b). Vetoes (any refuses the arm): implied possessions per team-game |model - actual| worse than `E0`'s by more than 0.10; any start type's mean gap worse by more than 0.20 s; any of Nov-Mar worse by more than 0.25 s. Among winners the simplest arm (`E1` < `E2` < `E3`) whose deviance and slope ratios are each within one floor of the best winner's. If none wins the round is REFUTED and reported as such.

## 6. Closed loop (offline winner only) and paired-seed gates that must not regress

Default-off `ENGINE_CLOCK=v5b_r8K2_env<arm>_glat_pmean` (new mode key; the adapter reads the rating columns it already holds; flag-off path bit-identical to `docs/ops/parity_reference_windows_v10.json`; the B1 latent sigma refitted on the arm's law with the unedited `exp_clk5b_mean_consistent.py --fit-only`). Screening: 500 verified fold-2 games x 25 paired seeds vs the served stack (only `ENGINE_CLOCK` differs, seeds shared); a candidate is re-read full size (5,710 x 200, box request, user approval for any paid compute) against four control floor draws.

**Primary (closed loop):** the sim's game-possession slopes on `env` and `gap` (D2 lines; real +0.152 / +0.050; control +0.011 / -0.008) and the game-total slope on `env` (real 0.600; control 0.408), each moving toward the actual by more than its floor.

**Hard vetoes (any regression beyond floor fails; same set as rounds 5-8):** G1 possession mean and SD; first-half and second-half share; G5 margin and total SD ratios; G9 margin bias and slope; G9 total bias (reported as priced exposure per Decision 11, a regression is explained by channel, not hidden); tempo elasticity ratio and team pace slope (round-8 lines); OREB/FGM/TOV/FTA possession slopes of round 8; OT rate and tie rate (must not move away from the actual); home margin non-neutral gap `|+0.09|` of D1 (must stay within its seed SE) and the eFG home-minus-away gap; period-end possession lines of D3 (reported, not selectors). Reported, not selecting: lead-change and time-of-decision slopes against |spread| (D2 section c; real -0.118 / -37.9 s per point, control -0.093 / -28.8), per-quintile total gap (Q1 -0.76 ... Q5 +1.18 for the control).

**Ship path.** Winner on fold 2 offline, fold 1 sign-consistent, closed loop passes every hard veto and the primary clears its floor, then parity v10 on a clean `src/` and the full-size read; the PM adopts. No post-hoc multiplier, cap, clip or blend on sim output at any stage: the scale enters the law's input only.

## 7. Cost and open questions for the PM

Offline about the cost of round 8 (deterministic fits, two folds, five arms); screening 500 x 25 on this machine. Q1: confirm the rating columns (`off_c`, `def_c`) are served identically in the live path (day-1 2027 ratings snapshot) before wiring. Q2: if `E1` wins offline but the game-total slope is still below 0.9 of real in the loop, the remainder is the efficiency adapters (FTA-rate slope ratio 0.80, eFG 0.93 in D2) and belongs to a possession_outcome / fg_make responsiveness round, not here.
