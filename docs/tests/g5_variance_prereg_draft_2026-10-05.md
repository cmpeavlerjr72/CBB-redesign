# DRAFT pre-registration: G5 total variance and score correlation, fixes in the owning sub-models (2026-10-05)

**DRAFT, NOT REGISTERED.** It is not in any `experiments.md`, and nothing in it governs a run until the PM moves it, possibly amended, into the owning `experiments.md` and commits it before the run. Evidence: `docs/tests/g5_variance_decomp_2026-10-05.md` (DX below). No training happened today, and 2025-26 stays sealed.

The rules from CLAUDE.md apply in full:
- folds F1 (test 2023-24) and F2 (test 2024-25, selection);
- one blind grader;
- the noise floor is a spec-identical reseed;
- ties go to the simpler arm;
- Decision 12 floors;
- Decision 11: a fix that exposes a compensation ships as a set and is judged on the combined stack.

No candidate may add a post-hoc multiplier on sim output. Each candidate changes a sub-model's own law or variance function.

**Common closed-loop lines for every candidate** (verified truth, 200 seeds full size on the box; 25-seed local reads are provisional only):
- G5 total SD ratio, home/away corr and margin SD ratio;
- the DX component table (a)-(e) and the channel blocks, reported side by side with realised;
- G9 total bias, total MAE, calibration slope and pred-total terciles;
- G1 mean, SD and by-month;
- G7 OT rate.

**Report-only line:** regulation-only Var(P) net of the persistent team pace error (DX 3.3), to expose the pace-latent compensation.

## Ranking

| rank | candidate | owner | DX gap it targets (F2 / F1, pts^2) | build cost |
|---|---|---|---|---|
| 1 | C1: clock K2 outcome-conditioned possession time, adoption read | clock (round 8) | (d) +24.2 / +27.7; K2 F2 measured: (d) -45.6 -> -23.4 (real -22.1) | none for F2 (artifacts and full-size read exist); fold-1 run needed |
| 2 | C2: late-game FT shooter / FT pressure law | free_throw + late_game (FT trip shooter selection) | FT-make coupling +12.9 / +16.4 | tap + offline fit |
| 3 | C3: bivariate game-level whistle in the joint foul model | foul_joint (R9ao3 successor) | whistle blocks +15.4 / +9.8 | offline fit |

Two items are out of scope as candidates but named owners. **G7 OT rate** (about +13.5 pts^2 of Var(total) at the real OT rate) is already queued with the late-game rounds. **The pace latent's sigma** may be over-dispersed in regulation. It is entered as a veto/report line, not a candidate; if C1 exposes it, it becomes clock round 9.

---

## C1. Clock K2 (`ENGINE_CLOCK=v5b_r8K2_glat_pmean`): adoption read under Decision 12

**Motivating evidence (DX 3.1).**
- The pace x efficiency term is the largest owner: 57% / 59% of the gap, all of it in the FG-value channel (sim -37.4 vs real +4.2, F1 -38.7 vs +9.4).
- K2 already exists, and its full-size F2 read (`d1001_H_3`, 200 seeds, seeds paired with served v2) puts every pace channel at reality.
- Gate effects on that read: G5 total ratio 0.925 -> 0.973, corr 0.126 -> 0.170, margin SD ratio 1.042 -> 1.041, G9 calibration slope 0.948 -> 0.949, total bias -0.34 -> -0.13, OT 0.0305 -> 0.0351.
- The ledger status is still TESTING (section 39); the full-size read was never ruled on.

**Arms.** `S` = served v2. `K2` = served v2 + `ENGINE_CLOCK=v5b_r8K2_glat_pmean`. Nothing is refitted.

**Folds.**
- F2: the existing reads (`v3full_COMB9GCTKD_s200_o0` vs `laneH_v3full_K2_s200_o0`) plus floors from the served draws (`v3full_COMB9GCTKD`, `d1001D_S2f1..4`) and 2x the paired game-bootstrap SE.
- F1: a new 200-seed pair on the fold-1 overlay (`overrides_V2.json` + `r8_K2/F1` artifacts), seeds 0-199, with V1 floors (offsets 1000-4000) reused.

**Primary.** G5 total SD ratio, |ratio - 1| (F2). Secondary: home/away corr gap.

**Decision rule.** K2 is selected if:
1. on F2, |ratio - 1| improves by more than 2 floors;
2. F1 moves in the same direction;
3. no veto line regresses by more than 2 floors.

The veto lines are G9 total MAE, calibration slope, pred-total tercile cells, G1 by month, and G1 SD (K2 5.629 vs 5.474).

**Decision 11 note.** K2 raises regulation Var(P) further: 89.2 -> 91.8, against about 72-74 net of persistent team error (DX 3.3). Read the report-only possession line. If it overshoots past the G1 SD tolerance, K2 ships with a clock round 9 latent refit (below) as a set.

**Clock round 9, conditional.** Refit `sigma` walk-forward with a team-season random effect in the method-of-moments residual, so persistent team pace error is not absorbed into the per-game latent. This is a model change in the clock's own variance function, not a sim-output adjustment. Arm: `K2 + sigma_net`. Same rule.

## C2. Late-game FT shooter selection / FT pressure law

**Motivating evidence (DX 3.2).** The engine carries a negative cross-team covariance between a team's FT% and the opponent's FG value in every stack, S0 included:

| line | F2 sim | F2 real | F1 sim | F1 real |
|---|---|---|---|---|
| val x ftp block (pts^2) | -13.6 | +1.6 | -12.8 | +4.9 |
| corr(FT%, own margin) | 0.26 | 0.16 | 0.25 | 0.15 |
| FT% deviation when trailing at the horn | -0.027 | -0.010 | -0.026 | -0.001 |

The FT block is worth +12.9 / +16.4 pts^2 of total variance, and it also inflates margin variance, which compensates G5 margin SD. **Expect the margin SD ratio to move when this is fixed; it is a Decision 11 set line.**

**Step 1 (diagnostic, before any fit; no training).** A per-FT-trip tap of served v2 on the 500-game verified sample x 10 seeds, recording shooter id, score_diff, seconds left, intentional-window flag and shooter as-of FT%. Compare against pbp 2023-24 / 2024-25 trips. Measure two things:
- (i) the fouled shooter's as-of FT% minus the team's on-court mean, in the intentional window, by leading or trailing;
- (ii) the served FT model's make-probability response to `score_diff` at engine states vs pbp states.

The step decides which arm family runs. If (i) is out by more than 2 SE, run C2a; if (ii), run C2b; if both, run both.

**Arms (offline, F1 / F2 walk-forward).**
- `C2-0`: the served shooter law and FT model (reference).
- `C2a`: a late-game foul-target law. The fouled player is drawn from the on-court lineup with weights fitted on training-season pbp intentional-window trips, as a multinomial on the as-of FT% rank and minutes share.
- `C2b`: the FT make model with `score_diff` / pressure terms removed (the simpler arm), against the served model.
- `C2c`: `C2a` + `C2b`.

**Offline primary.**
- `C2a`: log loss of fouled-player identity in the window (F2).
- `C2b`: FT make log loss on F2, within the window and overall.

The usual noise floor applies (reseed / bootstrap).

**Closed-loop primary.** The val x ftp block and corr(FT%, own margin), both against realised, plus the common lines.

**Responsiveness check.** By team FT% quintile, the fouled-shooter FT% gap must slope with actuals.

## C3. Bivariate game-level whistle in the joint foul model

**Motivating evidence (DX 3.2, 0).**
- The residual corr of FTA per possession across the two teams is 0.241 real vs 0.136 sim (F1 0.219 vs 0.124).
- The whistle blocks are short by +15.4 / +9.8 pts^2.
- The marginal team FTA-channel variance is already right: 46.9 real vs 48.2 sim. So the fix must re-partition, not add, variance: a shared game component with a smaller idiosyncratic one.
- Round 8's univariate whistle latent added only +0.03 / +0.04 of corr and was refuted as the MAIN channel. This candidate is the bivariate version on the R9ao3 foul state, sized against the 15 pts^2 it targets, not against the whole G5 gap.

**Arms (offline, F1 / F2).**
- `W0`: R9ao3 as served.
- `W1`: R9ao3 + a per-game shared log-rate whistle effect `u_g ~ N(0, s_g^2)`, with the per-team-game dispersion refitted jointly (`s_g`, `s_t` fitted walk-forward by marginal likelihood on the two-team FTA counts per game, with the as-of mean held fixed).
- `W2`: `W1` with `s_g` a function of site (neutral vs home/away). DX 4 shows neutral shared PPP 88 vs 65 on F2, about 2.8 SE, the same direction on F1, underpowered.

**Offline primary.** Held-out two-team FTA joint log-likelihood per game (F2), with a reseed floor.

**Offline secondaries.**
- the realised fta x fta block and the FTA/poss residual corr implied by the fit vs realised;
- marginal team FTA variance unchanged within the floor (guard against adding variance).

**Responsiveness check.** By team FT-rate prior quintile, predicted FTA slopes with actuals.

**Closed loop.** It runs only if `W1` or `W2` wins offline. The primaries are the fta x fta block and the G5 corr.

**Decision 11.** C3 raises FT trips' shared variance and moves G9 total only through variance, not level, so a level shift above the floor is a veto.

---

## Known interactions to read before ruling

1. **C1 and the pace latent:** C1 alone is the cleanest single step, but it deepens the regulation possession over-dispersion.
2. **C2 and the margin SD ratio:** C2 removes a negative FT block that currently props margin SD toward 1.04.
3. **The honest G5 target is about 0.97, not 1.00.** 4-6% of realised residual variance is persistent team mean-model error (DX 0). Nothing here proposes changing the gate; the PM should know that a ratio of 0.97-0.98 is consistent with a correct within-game variance function.
