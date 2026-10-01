# Shared (game-level) possession latents: offensive rebounds and turnovers (append-only)

Owner lane: B (day session 2026-10-01). Sub-model: a per-game random effect on the
rebound model's OREB-vs-DREB logit and / or the possession-outcome TOV logit, shared by
both teams, drawn once per simulated game. It is a variance component of those
sub-models, not a new mean model. Lane I owns the rebound level today and lane H the
clock: this round edits neither model; the effect sits behind its own default-off flag.
NOTHING here adopts anything or changes a served default.

Sources read first: `docs/tests/shared_shooting_latent_2026-09-30.md`,
`docs/models/shared_shooting/experiments.md` (the G3 form this round copies),
`docs/tests/g1_g5_possessions_corr_diagnostic_2026-09-30.md` 2.3, and today's ownership
table (`scripts/diag_g5_channels_v1.py`, `results/g5_channels/channels_v1.json`).

---

## 1. Round 1 pre-registration (2026-10-01 ~06:40 EDT, COMMITTED BEFORE THE `measure` OR `bakeoff` STAGE RUNS)

### 1.0 What was seen before this was written (disclosure)

The step-1 ownership table on served v2 (2025, 5,443 games, sim 200 seeds) was read.
On the points scale (channel `v0 (OREB - o0 N)`, residuals against the sim's per-game
mean) the between-team covariance, actual vs sim, is: OREB 0.66 vs -0.04 pts^2 (gap
0.0051 corr units, 3.4 SE); TOV 0.76 vs 0.84 (the engine already matches or exceeds
it). The per-miss and per-chance between-team moments used below have NOT been computed.

### 1.1 The object

Per game `g`, side `i`, channel `c`:
- `oreb`: opportunities `n` = live rebounds of side i's misses = side i's chances that start `OREB` plus the opponent's possessions that start `DREB` (pbp chances table `data/processed/possessions_v2/chances_<season>.parquet`); `y` = side i's `OREB` starts.
- `tov`: `n` = side i's chances; `y` = chances whose terminal event is `TOV`.

`R_ic = y - n p_ic`, `W_ic = n p_ic (1 - p_ic)`, with an as-of expectation
`p_ic = expit(logit(L_c) + d_off,i + d_def,j + s_c * site_i)`:
- `L_c`: the league season-to-date rate (strictly earlier dates);
- `d_off,i` = `logit((y_cum + K L_c) / (n_cum + K)) - logit(L_c)` over side i's earlier games this season; `d_def,j` the same for what the opponent allowed; `K` = 200 opportunities (fixed);
- `s_c`: the train-season home/away log-odds half-difference (+ home, - away, 0 neutral).

Residuals are centred by (season, ISO week, channel). Verified finals only.

A logit shift shared by both teams gives `Cov(R_hc, R_al) = E[W_hc W_al] Sigma_cl`. Sigma
is fitted by method of moments on BETWEEN-team cross products only.

### 1.2 Candidates (simplicity N < O1 = T1 < OT)

| arm | form |
|---|---|
| `N` | no shared latent (served) |
| `O1` | shared OREB logit effect `u ~ N(0, s_o^2)` |
| `T1` | shared TOV logit effect `u ~ N(0, s_t^2)` |
| `OT` | joint 2x2 Sigma (OREB, TOV) with a cross term, PSD-projected |

### 1.3 Folds, metric, floor, rule (the round-1 shared-shooting design)

- Fold 1: fit 2023, test 2024. Fold 2 (selection): fit 2023-24, test 2025. 2025-26 sealed.
- Primary: held-out per-game Gaussian log density of `[R_h,oreb, R_h,tov, R_a,oreb, R_a,tov]` with `V = blockdiag_i(W_i (Sigma + T) W_i + diag W_i)` plus the between block `W_h Sigma W_a`; `T = psd(within second moment over binomial - Sigma)` fitted per arm on train. Gain = mean test log density minus N's.
- Floor: max(200-replicate Poisson game bootstrap SD of the paired mean difference; |change| under a seed-1 training-game bootstrap refit).
- Rule: eligible = fold-2 gain > 2 floors AND fold-1 gain > 0. Winner = the largest fold-2 gain; arms within 1 floor tie, ties go to the simpler; an O1 / T1 tie goes to the arm that passes the pre-condition in 1.4. No eligible arm: N wins and the latent is REFUTED offline.
- Secondary: held-out between-team covariance obs/pred; per-team-game dispersion obs/pred; segments site, month, conference; per-season Sigma 2023 / 2024 / 2025 (the 2025-artefact check).

### 1.4 Closed-loop pre-condition (fixed now)

An arm is wired only for a channel the served engine UNDER-produces: the served-v2 sim
between-team covariance of that channel must sit below the actual by more than 2 actual
SE (step-1 table). TOV fails (sim 0.84 >= actual 0.76), so `T1` is never wired, and an
`OT` winner is wired as its OREB component only if `O1` is itself eligible; otherwise
nothing is wired.

### 1.5 Engine wiring and closed loop (fixed now)

- Module `src/cbb_sim/engine/shared_possession.py`, env `ENGINE_SHARED_POSS` (unset / `reference` = OFF: no draw, no stream touched; parity proved against `docs/ops/parity_reference_windows_v9.json`). Arm `O1`. Params `data/processed/models/shared_possession/params_v1.json` = the fold-2 TRAIN fit, never tuned on sim output.
- One draw per simulated game from its own stream family `shared_poss` keyed on (seed, game_id), added to the logit of the binary OREB / (OREB + DREB) probability of BOTH teams' live misses before the fixed dead-ball share is composed. Not mean-corrected (the Jensen shift is reported through the realised OREB% line).
- Closed loop: paired vs the served default (v2), v3 inputs, verified truth. Local tap for direction and parity only (Decision 12: underpowered for G5). Full size on the box: 5,710 x 200 with four served-default floor draws.
- Status: VALIDATED-PENDING-SHIP-ACTION if the G5 home/away corr and the total SD ratio both move toward target beyond floor at 200 seeds and no other line regresses beyond floor except through a named compensation (the G4 OREB% level is included: a Jensen level move beyond floor counts as a regression); REFUTED if neither moves beyond floor; UNDERPOWERED if only the tap exists. Mechanism line: sim between-team OREB covariance vs `E[W_h s_o^2 W_a]`.

---

## 2. Results, round 1 (appended 2026-10-01 ~06:45 EDT; evidence `docs/tests/g5_variance_channels_2026-10-01.md` section 3)

`scripts/exp_shared_poss_v1.py measure` and `bakeoff` ran at 06:31 EDT, after commit `5a62b8e`. Outputs: `results/shared_possession/{measure,bakeoff}_v1.json`.

**Measurement (2023-24 pooled, logit^2):**
- OREB shared s^2 is 0.0013 (SE 0.0016): not detectable.
- TOV shared s^2 is 0.0074 (0.0011).
- The OREB x TOV term is -0.0046 (0.0009).
- 2025 is not an outlier.

| arm | fold 1 gain (floors) | fold 2 gain (floors) | eligible |
|---|---|---|---|
| O1 | -0.00002 (-0.13) | +0.00010 (+0.74) | no |
| T1 | +0.00294 (+3.76) | +0.00184 (+2.23) | yes |
| OT | +0.00386 (+3.48) | +0.00294 (+2.68) | yes; winner by the letter (beats T1 by 0.001101 against a 0.001097 floor) |

**STATUS:**
- `O1` is REFUTED offline.
- `T1` and `OT` are eligible offline but NOT WIRED under pre-condition 1.4. The served engine already reproduces between-team TOV covariance in points (0.84 vs 0.76 pts^2).
- No engine module was written, `params_v1.json` was not written, and there is no closed loop.
