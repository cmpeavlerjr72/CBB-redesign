# Shared (game-level) shooting latent: experiments (append-only)

Owner lane: B (overnight 2026-09-30). Sub-model: a per-game random effect on the
`fg_make` logit, shared by both teams, drawn once per simulated game. It is a
variance component of `fg_make`, not a new mean model. NOTHING in this file adopts
anything or changes a served default; the PM decides from the results section.

Sources read before writing (not re-derived):
`docs/tests/g1_g5_possessions_corr_diagnostic_2026-09-30.md` (section 2: G5 gap
0.136 = whistle 37%, shared shooting 24%, 0-0 finals 18%, pace x efficiency 14%),
`docs/tests/foul_round8_whistle_2026-09-30.md` (the whistle latent, refuted as the
main G5 channel), `docs/tests/engine_gates_F2_2025_s200_v3_S0_2026-09-30.md` (the
baseline), HANDOFF.md "SUMMARY FOR USER, 2026-09-30" cross-cutting finding 2,
`src/cbb_sim/engine/clock_adapter_v3.py` (the v5b pace latent pattern) and
`src/cbb_sim/engine/foul_joint.py` (the round-8 whistle wiring pattern).

---

## 1. Pre-registration: round 1 (written 2026-09-30 ~20:50 EDT, COMMITTED BEFORE THE BAKE-OFF STAGE RAN)

### 1.0 What was seen before this was written (disclosure)

The brief orders a measurement first. `scripts/exp_shared_shooting_v1.py measure`
ran at 20:47 EDT on all three residual seasons (2023, 2024, 2025), i.e. it saw the
TEST seasons' between-team moments. Its numbers are in the results doc and are
not repeated here. The candidate list, metric, floor and decision rule below
were not changed by them except in one place, stated openly: the per-type
measurement showed the three-point shared variance near zero while rim and
jumper are clearly positive, which is the reason the per-type arm `G3` is in the
list next to the pooled `G1` (both were in the brief's examples anyway). The
bake-off stage (`... bakeoff`) had NOT run when this was committed.

### 1.1 The object

Per game `g`, side `i` (home/away), shot type `k` (rim, jump2, three), with the
held-out make probabilities `p` of the served fg_make spec (round 4 `B1`, S1
monthly refits; `scripts/exp_shared_shooting_preds_v1.py`: 2025 = the served
artifacts themselves, 2024 = the same spec refitted on F1, 2023 = the same spec
refitted on 2022 only):

    R_ik = sum over the side's type-k FGA of (y - p)       residual makes
    W_ik = sum of p (1 - p)                                 binomial variance

A logit shift `u_k` shared by both teams gives, to first order,
`Cov(R_hk, R_al) = E[W_hk W_al] Sigma_kl`. Sigma is fitted by method of moments on
BETWEEN-team cross products only (a team's own hot night cannot enter them),
after removing the (season, ISO week, type) league residual (drift is the season
anchor's object, not this one). Verified finals only (`verified_finals=True`).

### 1.2 Candidates (four arms; simplicity order N < G1 < GP < G3)

| arm | form | fitted parameters |
|---|---|---|
| `N` | null: no shared latent (the served engine) | -- |
| `G1` | one logit-scale game effect `u ~ N(0, s^2)` on every shot type, both teams | `s^2` = sum_g (sum_k R_hk)(sum_k R_ak) / sum_g W_h W_a |
| `GP` | `G1` coupled to the clock's own per-game pace latent: `u = s (rho z_pace + sqrt(1 - rho^2) z)`, `z_pace` the v5b latent's normal (same key, same ordinal as `clock_adapter_v3.LATENT_ORDINAL`) | `s^2` as G1; `c = Cov(sum R, q) / sum W` with `q` = ln(box possessions) minus an as-of pace line fitted on train; `rho = clip(-c / (s * sigma_clk), -1, 1)`, `sigma_clk` = served B1 sigma 0.04703 |
| `G3` | per-type effects `u ~ N(0, Sigma)`, Sigma 3x3 with fitted cross-type covariances (PSD-projected) | 6 |

Control (closed loop only, if budget allows): `U1` = per-TEAM independent effects
with G1's `s^2` (separates "more variance" from "shared variance", as the
whistle round's `R8bU`).

FT make is MEASURED, not a candidate (its as-of expectation here is an
approximation, not the served free_throw model; its points scale is 1 per make).

### 1.3 Folds (CLAUDE.md)

- Fold 1: fit on 2023 residuals (preds from a 2022-trained model), test 2024.
- Fold 2 (selection): fit on 2023 + 2024 residuals, test 2025 (served predictions).
- 2025-26 sealed. Nothing is fitted on a test season or on sim output.

### 1.4 Primary metric

Held-out per-game Gaussian log density of the 7-vector
`x_g = [R_h,rim, R_h,jump2, R_h,three, R_a,rim, R_a,jump2, R_a,three, q_g]` with

    V_g = blockdiag_i( W_i (Sigma + T) W_i + diag(W_i) )         within team (binomial + latent)
        + between-team block  W_h Sigma W_a
        + Cov(R_ik, q) = W_ik c   (GP only; 0 otherwise),  Var(q) = v_q (train)

`T` = the within-team second moment over binomial minus Sigma (unshared team-game
variance), fitted per arm on train, so every arm carries the same within-team
total and the arms differ only in what they call shared. Reported as the mean
test log density minus arm `N`'s.

### 1.5 Noise floor

Floor = max(a, b): (a) SD of the paired test-mean difference under a 200-replicate
Poisson game bootstrap of the test season; (b) the absolute change in the arm's
test mean log density when the arm is refitted on a seed-1 Poisson bootstrap of
its TRAINING games (the spec-identical retrain under another seed: MoM has no
optimiser seed).

### 1.6 Decision rule (fixed now)

1. Eligible: fold-2 gain over `N` > 2 floors AND fold-1 gain over `N` > 0.
2. Winner: the eligible arm with the largest fold-2 gain; any eligible arm within
   1 floor of it ties, and ties go to the simplest (N < G1 < GP < G3).
3. No eligible arm: `N` wins and the latent is REFUTED offline.

Secondary (reported, not deciding): held-out between-team covariance observed /
predicted, by type pair and total; per-team-game residual-variance calibration by
type and in points (the "per-game eFG dispersion" line); segments home/away vs
neutral, month, as-of tempo tercile; per team (UNDERPOWERED, about 30 games each).

### 1.7 Engine wiring and closed loop (fixed now)

- Module `src/cbb_sim/engine/shared_shooting.py`, env `ENGINE_SHARED_SHOOTING`
  (unset / `reference` = OFF: no draw, no stream touched, bit-identical; parity
  proved against `docs/ops/parity_reference_windows_v6.json` and against the box's
  S0 rows). Arms `G1`, `GP`, `G3`, `U1`. Parameters read from
  `data/processed/models/shared_shooting/params_v1.json` (the fold-2 TRAIN fit
  written by the bake-off; never tuned on sim output).
- One draw per simulated game from its own family `shared_shooting` keyed on
  (seed, game_id), the round-8 whistle pattern; `GP` reads the clock latent's
  normal from the clock stream key at `LATENT_ORDINAL` without advancing it.
  Applied in `loop.py` as `logit(p_make) + u[type]` (both teams). Not
  probability-mean-corrected: the logit-normal Jensen shift is about
  0.5 s^2 p(1-p)(1-2p), under 0.1 pp at the fitted sizes; the closed loop reports
  the realised eFG% line.
- Closed loop: winner vs S0 on v3 inputs (`engine_v3`, verified truth,
  `CBB_TRUTH=verified_v1`), full slate 5,710 games, paired seeds. Local: seeds
  0-24 (core cap 4) against the box's `v3full_S0_s200_o0_off0_n25`; floors per
  Decision 12 = max(SD across the five 25-seed reference draws S0, S0f1..S0f4,
  paired game bootstrap). Box (request through `docs/ops/box_queue/`): seeds
  0-199 against `v3full_S0_s200_o0` with the four 200-seed floor draws, and `U1`
  if time allows.
- Closed-loop reading (Decision 11): every G1-G9 line reported, G5 by component
  (Var home, Var away, 2 Cov; margin and total). Status VALIDATED-PENDING-SHIP-ACTION
  if the G5 home/away correlation and the total SD ratio both move toward target
  beyond the floor at the 200-seed read and no other line regresses beyond floor
  except through a named compensation; REFUTED if they do not move beyond floor;
  UNDERPOWERED if only the 25-seed read exists (Decision 12: it cannot decide a
  G5 line). Mechanism line: sim within-game between-team covariance of made FG by
  type against the fitted `W_h Sigma W_a`.


---

## 2. Results, round 1 (appended 2026-09-30 23:09 EDT; full evidence in `docs/tests/shared_shooting_latent_2026-09-30.md`)

**Offline (rule 1.6).** Bake-off at 20:50 EDT, after commit `86e3d0f`.

| arm | fold-1 gain (floors) | fold-2 gain (floors) | eligible |
|---|---|---|---|
| G1 | +0.70 | +1.55 | no |
| GP | +0.78 | +1.84 | no |
| **G3** | **+3.55** | **+3.65** | **yes: WINNER** |

- Served candidate: `params_v1.json` (fold-2 train Sigma).
- Secondary: held-out between-team covariance obs/pred is 0.73 (fold 2) and 0.66 (fold 1). The training seasons carry more shared variance than the test seasons. Reported, not corrected.

**Closed loop (rule 1.7).** G3 vs S0, v3 inputs, verified truth, 5,705 graded games.
- **200 seeds (box):**
  - home/away corr 0.1174 -> 0.1350, +20 floors;
  - total SD ratio 0.9306 -> 0.9495, +24 floors;
  - margin SD ratio 1.0500 -> 1.0494, -0.3 floors;
  - no other line beyond 2 floors except possession SD (+2.7, PASS -> PASS).
- **G5 by component:** Var(home) +2.47, Var(away) +2.51, Cov +2.65.
- **Control U1 (25 seeds, local):** corr -0.003, Cov -0.09, margin SD ratio +0.022 (+11.8 floors, away from 1).

**STATUS: VALIDATED-PENDING-SHIP-ACTION (Decision 11)**, arm `G3`. Nothing is adopted and no default changes; the PM decides inside the ship set.

## ADOPTED 2026-10-01 -- arm `G3` is served (adoption executor)

The Decision 11 set (clock L2 + shot_block K2_Ocell + foul R9ao3 + shared_shooting G3 + chance_time KD) was ADOPTED as ONE served stack by the PM under the user's delegation of 2026-09-30. Evidence: `docs/tests/engine_gates_F2_2025_s200_v3_COMB9GKD_full_2026-10-01.md` (5,710 x 200, verified truth, Decision 12 floors, paired vs S0: G1 possessions mean FAIL -> PASS, OREB% FAIL -> PASS, total and margin bias stay PASS, slope 0.917 -> 0.948, home/away corr 0.117 -> 0.126, G2 cells 3/9 -> 6/9, no verdict regresses). Engine default changed in the adoption commit; the pre-adoption stack stays reachable with the SERVED_V1 env (`docs/tests/adoption_served_v2_2026-10-01.md`), which reproduces parity references v6 and v7 bit-identically.

- `shared_shooting.DEFAULT = "G3"`; `ENGINE_SHARED_SHOOTING=reference` = served-v1 (no draw).
- Artifact: `data/processed/models/shared_shooting/params_v1.json` (tracked).

---

## 3. Round 2 pre-registration: walk-forward refit of G3's Sigma (lane B, 2026-10-01 ~06:40 EDT; COMMITTED BEFORE THE `wf` STAGE RUNS)

### 3.0 What was seen before this was written (disclosure)

- Round 1 (sections 1-2): held-out between-team covariance obs/pred 0.73 (fold 2) and 0.66 (fold 1); the per-season Sigma table (2023 / 2024 / 2025) in `docs/tests/shared_shooting_latent_2026-09-30.md` 1.2. Those numbers saw the test seasons and are the reason this round exists.
- Today's delivery diagnostic (`scripts/diag_g3_delivery_v1.py`, existing 200-seed runs only): at the RATE level (made FG minus attempts x the game's own make rate) the engine delivers 94-95% of `E[W_h Sigma W_a]`. The 58% at the count level is the attempts' response (fewer misses -> fewer OREB -> fewer attempts; the clock). The latent is applied on the scale it was fitted. A Sigma "corrected" for that dilution would be a fit to sim output and is NOT an arm.
- The `wf` stage below has NOT run.

### 3.1 Candidates (simplicity order G3P < G3L < G3A)

| arm | Sigma for a test-season game | note |
|---|---|---|
| `G3P` | round-1 G3: `psd(sigma_between)` pooled over every train season | the served spec (reference) |
| `G3L` | the same fit on the LAST train season only | fold 1 trains on 2023 only, so G3L == G3P by construction there (fold-1 gain exactly 0) |
| `G3A` | as-of in-season update: `psd((n0 * S_P + num_cur) / (n0 + den_cur))` elementwise on the MoM numerator and denominator; `S_P` = G3P; `n0` = the train seasons' mean per-season between-team mass `sum_g W_hk W_al` (one season of prior weight, fixed, not tuned); `num_cur` / `den_cur` = the test season's centred between-team cross products from ISO weeks strictly before the game's week | strictly as-of: completed earlier weeks of the same season only |

Within-team unshared `T` is fitted per arm on train exactly as in round 1 (`psd(within_cross - S)`), with `S` the arm's train-time Sigma (G3A uses G3P's T).

### 3.2 Data, folds, metric, floor (identical to round 1 unless stated)

- Residuals: `results/shared_shooting/preds_v1.parquet` (unchanged), verified finals, (season, ISO week, type) centring.
- Fold 1: train 2023, test 2024. Fold 2 (selection): train 2023-24, test 2025. 2025-26 sealed.
- Primary: held-out per-game Gaussian log density of the 6-vector `[R_h(3), R_a(3)]` (round 1's pace element is dropped: no arm here couples to pace), mean over test games, reported as the gain over `G3P`.
- Floor: max(SD of the paired test-mean difference under a 200-replicate Poisson game bootstrap; |change| of the arm's test mean log density when refitted on a seed-1 Poisson bootstrap of its training games).

### 3.3 Decision rule (fixed now)

1. Eligible: fold-2 gain over `G3P` > 2 floors AND fold-1 gain over `G3P` > 0. `G3L` therefore cannot be eligible (its fold 1 is identical by construction); it is reported as a fold-2-only read.
2. Winner: the eligible arm with the largest fold-2 gain; arms within 1 floor tie; ties go to the simplest.
3. No eligible arm: `G3P` stands (the served Sigma) and the walk-forward refit is REFUTED offline for this round.

Secondary (reported, not deciding): held-out between-team covariance obs/pred by arm; per-month gain; the G3A Sigma path through the season.

### 3.4 Closed loop (only if an arm other than G3P wins)

- New arm values `G3L` / `G3A` of `ENGINE_SHARED_SHOOTING` in `src/cbb_sim/engine/shared_shooting.py` (the default stays `G3`). G3A reads a per-game Sigma table built offline for the fold-2 slate from completed earlier weeks only.
- Read: paired vs served v2, full slate, 200 seeds on the box with four floor draws (Decision 12). A local tap is direction only. Status per section 1.7's wording.
