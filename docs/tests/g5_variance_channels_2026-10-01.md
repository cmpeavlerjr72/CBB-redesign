# G5 on served stack v2: channel ownership of the correlation and total-variance gaps, the G3 delivery question, walk-forward Sigma, and shared OREB / TOV latents (lane B, 2026-10-01)

Lane B, day session 2026-10-01, wall clock 06:15 -> 06:40 EDT (system clock). **NOTHING IS ADOPTED, NO DEFAULT CHANGES, NOTHING NEW IS WIRED INTO THE ENGINE.** No closed loop ran; no box request was filed (no arm reached the closed-loop stage).

- Pre-registrations: `docs/models/shared_shooting/experiments.md` section 3 and `docs/models/shared_possession/experiments.md` section 1, commit `5a62b8e` (06:2x EDT), before either bake-off stage ran.
- Ledger: two rows (`docs/models/change_ledger.md`), status updated in the results commit.
- Run: served v2 at full size, `results/engine_v0/v3full_COMB9GCTKD_s200_o0` (5,710 x 200, the adoption read), verified truth (`CBB_TRUTH=verified_v1`). References: `v3full_COMB9CTKD_s200_o0` (v2 without G3), `v3full_S0_s200_o0` and its four floor draws `S0f1..f4`, `laneB_v3full_G3_s200_o0`.

## 0. Verdicts

| item | result | status |
|---|---|---|
| Step 1: ownership of the G5 gaps on served v2 | Closes exactly (corr gap 0.1056 on the 5,443-game pbp subset; total-variance gap 42.0 pts^2). **The largest owner is the clock's outcome-blind possession time: pace x makes owns +0.078 (74%) of the corr gap and +42 pts^2 (100%) of the total-variance gap, partly offset by pace x OREB -0.040 (-38%).** Next: cross-type make residuals +0.039 (37%), mean x residual +0.013 (12%). Shared make, FT, whistle, OREB and TOV channels own 5% or less each. Every channel is stable across 2023 / 2024 / 2025 on real data: none is a 2025-only artefact | measured (diagnostic) |
| Step 2: why G3 delivers 58% | **Not scale and not season mismatch.** At the rate level (makes minus attempts x the game's own make rate) the engine delivers 94-95% of `E[W_h Sigma W_a]` (scale is right). The rest is the attempts' response to makes: -0.44 of +1.03 makes^2. Routes: fewer misses -> fewer OREB (83% of the volume route) and fewer possessions (59%), offset by TOV / FTA (-41%) | measured (diagnostic) |
| Step 2: walk-forward refit of G3's Sigma (`G3L`, `G3A` vs served `G3P`) | `G3L` fold 2 +0.000003 (0.001 floors; fold 1 identical by construction); `G3A` fold 2 -0.0001 (-0.22 floors), fold 1 -0.00055 (-0.61). No arm eligible | **REFUTED offline; G3P stands** |
| Step 3: shared OREB latent `O1` | per-miss shared logit variance 0.0013 (SE 0.0016) on 2023-24; fold 2 +0.74 floors, fold 1 -0.13 | **REFUTED offline** |
| Step 3: shared TOV latent `T1`, joint `OT` | real (TOV 0.0064-0.0086, 5-6 SE per season; OREB x TOV -0.004, 3-4 SE). `T1` fold 2 +2.23 / fold 1 +3.76 floors; `OT` +2.68 / +3.48; `OT` wins by the letter (beats `T1` by 1.004 of its floor, at the tie boundary) | **eligible offline, NOT WIRED** under pre-condition 1.4: the engine already reproduces between-team TOV covariance in points (0.84 vs 0.76 pts^2; corr gap -0.0008, SE 0.0011) |
| Step 4: closed loop / box | nothing to wire | NOT RUN (no winner reached it) |

---

## 1. Ownership table (step 1)

### 1.1 Method

`scripts/diag_g5_channels_v1.py`. It uses an exact per-team-game points identity, with every term linear:

    pts_i = e0 N                              pace (game possessions, box estimator, mean of the two sides)
          + e0 (N_i - N)                      possession imbalance
          + v0 (OREB_i - o0 N_i)              OREB
          - v0 (TOV_i - t0 N_i)               TOV
          + (f0 - 0.44 v0)(FTA_i - a0 N_i)    FTA volume (whistle)
          + 3PA-share and rim/jumper-split mix terms at expected make rates
          + 2(FGM_rim - FGA_rim p0) + 2(FGM_jump - ...) + 3(FGM_3 - ...) + (FTM - FTA f0)   make residuals
          + data remainder (actual only; mean |rem| 0.002 pts)

The base constants are the sim's own per-game, per-side 200-seed ratio-of-means (as-of).
- **Sim:** within-game deviations.
- **Actual:** residuals against the sim's per-game mean.
- The G5 correlation chain is the 09-30 diagnostic's: between / denominator + mean x residual + residual covariance, with the residual covariance split by channel pair.
- Total variance: `Var(T) = sum_kl Cov(C_k, C_l)`.
- Both close to rounding error.
- Rim / jumper split of the actual box comes from the pbp fg_make design rows. That restricts the read to 5,443 of 5,705 graded games, where actual corr is 0.2309 (gate 0.2283) and sim corr is 0.1253.
- Actual SE: 200 Poisson game bootstraps.
- Sim floor (SD across the five S0 200-seed draws): 0.0005 corr units or less on every row, negligible against the actual SE.

### 1.2 The table (served v2, 2025, corr gap 0.1056, total-variance gap 42.0 pts^2)

| channel | corr units actual / sim | gap (SE) | share of corr gap | total-var pts^2 actual / sim | gap (SE) | share of var gap |
|---|---|---|---:|---|---|---:|
| between-game / denominator | | +0.0021 | 2% | | | |
| mean x residual (per-game mean miscalibration) | | +0.0128 | 12% | | | |
| pace x pace | 0.1951 / 0.1889 | +0.0061 (0.0059) | 6% | 108.9 / 102.1 | +6.8 (3.3) | 16% |
| **pace x makes (rim, jump, three, FT)** | **+0.0040 / -0.0744** | **+0.0784 (0.0078)** | **74%** | **2.0 / -40.0** | **+42.0 (4.3)** | **100%** |
| **pace x OREB** | **-0.0481 / -0.0079** | **-0.0403 (0.0034)** | **-38%** | -27.0 / -4.3 | -22.7 (1.9) | -54% |
| pace x TOV | -0.0181 / -0.0152 | -0.0029 (0.0027) | -3% | -10.0 / -8.3 | -1.6 (1.5) | -4% |
| pace x FTA | +0.0170 / +0.0108 | +0.0061 (0.0017) | 6% | 9.4 / 5.8 | +3.5 (0.9) | 8% |
| pace x shot mix | +0.0101 / +0.0030 | +0.0071 (0.0016) | 7% | 5.7 / 1.6 | +4.1 (0.9) | 10% |
| two-point block (rim, jumper, rim/jumper split, all pairs; G3's channel) | +0.0074 / +0.0121 | -0.0046 (0.0038) | -4% | 78.2 / 77.7 | +0.5 (1.5) | 1% |
| three make | -0.0011 / +0.0028 | -0.0039 (0.0044) | -4% | 93.1 / 93.4 | -0.3 (1.8) | -1% |
| FT make | +0.0012 / -0.0019 | +0.0030 (0.0004) | 3% | 9.6 / 8.1 | +1.4 (0.2) | 3% |
| FTA volume (whistle) | +0.0056 / +0.0031 | +0.0025 (0.0003) | 2% | 8.1 / 7.6 | +0.4 (0.2) | 1% |
| OREB | +0.0048 / -0.0003 | +0.0051 (0.0015) | 5% | 32.1 / 25.0 | +7.1 (0.6) | 17% |
| TOV | +0.0055 / +0.0063 | -0.0008 (0.0011) | -1% | 24.4 / 20.6 | +3.7 (0.5) | 9% |
| 3PA share / imbalance (own) | 0 / -0.0066 vs -0.0037 | -0.0029 | -3% | ~0 | ~0 | 0% |
| **makes cross-type (three and FT with each other and with twos)** | **+0.0150 / -0.0235** | **+0.0386 (0.0070)** | **37%** | 5.4 / 1.8 | +3.7 (2.5) | 9% |
| OREB x makes / shot mix | +0.0033 / +0.0287 | -0.0254 (0.0060) | -24% | -59.8 / -51.1 | -8.8 (2.3) | -21% |
| TOV x non-pace | +0.0584 / +0.0640 | -0.0056 (0.0044) | -5% | 11.9 / 13.8 | -1.9 (1.9) | -5% |
| FTA x non-pace | +0.0104 / +0.0000 | +0.0104 (0.0024) | 10% | 0.8 / -3.4 | +4.2 (1.1) | 10% |
| possession imbalance x non-pace (box-estimator accounting) | +0.0164 / -0.0027 | +0.0192 (0.0012) | 18% | 0.1 / 0.0 | +0.1 (0.1) | 0% |
| remaining pairs | | +0.0007 (0.0004) | 1% | | -0.3 | -1% |
| **total** | 0.2309 / 0.1253 | **0.1056** | 100% | 292.5 / 250.5 | **+42.0** | 100% (ratio 0.925) |

**Reading.**
1. **Pace x makes is the dominant owner of both lines.**
   - Real games show no covariance between possessions and make residuals: +0.004 corr units, and -0.2 to -0.6 pts^2 in every season (1.4).
   - In the engine, a game with more makes has fewer possessions: -0.074 corr units, -40 pts^2 of total variance.
   - The adopted stack made this worse. Sim pace x makes was -0.055 on S0, -0.066 on v2 without G3, and is -0.074 on v2. G3 itself adds -0.009 to -0.010, because it raises shared makes, which feed this coupling.
2. **Pace x OREB runs the other way.** Real games with more OREB have fewer possessions (-0.048); the engine has almost none (-0.008).
3. **The 09-30 "14% pace x efficiency" figure was the net of these two, on v5b.** On served v2 the net is +0.048, 46% of the corr gap and 60% of the total-variance gap.
4. **Makes cross-type +0.039 (37%).** The engine's between-team residuals across shot types are NEGATIVE at baseline (-0.033 on S0, before any latent). The FT x opponent-FG pairs dominate: three x FT -1.93 pts^2 sim vs -0.23 actual; rim x FT -0.80 vs +0.35. Reality is +0.015. **Unattributed.** The candidate is score-state feedback through who shoots the free throws (late game, usage). It is NOT tested here.
5. **Possession imbalance x non-pace +0.019 (18%).** This is box-estimator accounting: one side's own-estimator possessions minus the game's. It is partly definitional (team rebounds, the 0.44 FTA coefficient). Its owner is grading-truth definition, not a sub-model. Flagged, not pursued.
6. **The shooting, FT, whistle, OREB and TOV shared channels are each 5% or less of the corr gap.**
   - The two-point block, where G3 acts, is already slightly past reality (-0.0046, 1.2 SE).
   - The OREB and TOV channels' total-variance shares (17%, 9%) are mostly within-team variance (sim Var(OREB channel) 13.4 / 11.6 vs actual 16.9 / 13.9), not covariance.

### 1.3 By game type (served v2, 2025; gap in corr units, SE in brackets)

| segment | games | corr actual / sim | total SD ratio | mean x resid | pace x makes | pace x OREB | makes cross-type | OREB | FTA x non-pace | imbalance x non-pace |
|---|---:|---|---:|---:|---|---|---|---|---|---|
| conference | 3,433 | 0.332 / 0.191 | 0.927 | +0.034 | +0.079 (0.009) | -0.038 (0.004) | +0.042 (0.009) | +0.006 (0.002) | +0.013 (0.003) | +0.021 (0.002) |
| non-conference | 2,010 | 0.119 / 0.054 | 0.928 | -0.004 | +0.079 (0.011) | -0.046 (0.005) | +0.034 (0.010) | +0.004 (0.002) | +0.006 (0.004) | +0.016 (0.002) |
| home / away site | 4,736 | 0.213 / 0.117 | 0.925 | +0.009 | +0.078 (0.008) | -0.040 (0.003) | +0.038 (0.007) | +0.005 (0.002) | +0.009 (0.002) | +0.018 (0.001) |
| neutral | 707 | 0.378 / 0.208 | 0.928 | +0.029 | +0.084 (0.021) | -0.045 (0.009) | +0.046 (0.020) | +0.004 (0.004) | +0.022 (0.008) | +0.025 (0.004) |
| predicted margin, close tercile | 1,815 | 0.462 / 0.340 | 0.902 | +0.005 | +0.079 (0.014) | -0.043 (0.007) | +0.042 (0.011) | +0.004 (0.003) | +0.012 (0.004) | +0.024 (0.002) |
| predicted margin, mid tercile | 1,814 | 0.344 / 0.253 | 0.948 | +0.014 | +0.091 (0.014) | -0.044 (0.005) | +0.035 (0.013) | +0.007 (0.003) | +0.017 (0.004) | +0.022 (0.002) |
| predicted margin, wide tercile | 1,814 | 0.096 / 0.012 | 0.930 | +0.006 | +0.079 (0.011) | -0.040 (0.005) | +0.044 (0.012) | +0.005 (0.002) | +0.006 (0.004) | +0.016 (0.002) |

All values, with every other channel per segment: `results/g5_channels/channels_v1.json` (`segments`).

- **The two big owners are flat across every cut.** Pace x makes sits at +0.078 to +0.091 and pace x OREB at -0.038 to -0.046 in every segment: a structural defect, not a tier, site or conference effect.
- **Mean x residual is the one owner that moves with game type:** conference +0.034 vs non-conference -0.004. That is lane F's / G9's per-game mean object.
- **Power.** Neutral (707 games) and each margin tercile (about 1,800) are powered for the big rows (SE 0.007-0.021) but UNDERPOWERED for the small ones (OREB, FTA, whistle: SE about equal to the value). Per-team cells were not computed: about 30 games per team cannot resolve rows this size (UNDERPOWERED by construction).

### 1.4 Per season (real data only; between-team covariance in pts^2, team-season FE residuals, SE in brackets)

| season | games | resid corr (pts) | pace x pace | pace x makes | pace x OREB | two-point block | three | FT make | FTA | OREB | TOV | other cross |
|---|---:|---:|---|---|---|---|---|---|---|---|---|---|
| 2023 | 4,519 | 0.362 | 18.3 (0.4) | -0.2 (0.8) | -5.2 (0.4) | 1.3 (0.5) | 0.6 (0.6) | 0.07 (0.05) | 0.40 (0.04) | 0.3 (0.2) | 1.0 (0.1) | 14.3 (1.1) |
| 2024 | 5,176 | 0.375 | 21.0 (0.7) | -0.2 (0.9) | -5.7 (0.4) | 2.1 (0.5) | 0.7 (0.5) | 0.09 (0.05) | 0.56 (0.04) | 0.2 (0.2) | 0.8 (0.1) | 13.3 (1.0) |
| 2025 | 5,443 | 0.367 | 21.2 (0.7) | -0.6 (0.9) | -6.1 (0.4) | 1.0 (0.4) | 0.0 (0.6) | 0.13 (0.05) | 0.71 (0.04) | 0.5 (0.2) | 0.8 (0.1) | 14.4 (0.9) |

- **No channel is a 2025-only artefact.** Every row sits within about 2 SE across seasons.
- The exception is the whistle (FTA 0.40 -> 0.56 -> 0.71), which RISES into 2025.
- The two-point block's 2024 high (2.1) is the season-level variation that round 2 of shared_shooting tested (section 2.2): it does not help prediction.
- The engine exists only for 2025 (fold 2): there is no served-v2 sim of 2023 or 2024 (HANDOFF open item 10), so the per-season table is real-data only.

### 1.5 What decides a G5 line, and what is underpowered

- The table reads the existing full-size 200-seed run (Decision 12 size). It attributes the gap; it decides no arm.
- No arm in this doc reached a closed loop, so no G5 line is decided here.
- Segment and per-season rows are labelled with their SE. Per-team rows are UNDERPOWERED and were not produced.

---

## 2. Why the engine delivers 58% of G3's fitted make covariance (step 2)

### 2.1 Scale vs dilution: exact split of the made-FG covariance

`scripts/diag_g3_delivery_v1.py`. Pairs: G3 on vs off, same seeds. `FGM = A p0 + R`, where `p0` is the arm's own per-game rate; Delta = arm minus no-G3. Units: makes^2.

| pair | Delta count cov | volume x volume | volume x rate | rate x rate | predicted `E[W Sigma W]` | rate / predicted |
|---|---:|---:|---:|---:|---:|---:|
| served v2 vs v2 without G3 (all 9 type pairs) | +0.590 | +0.056 | **-0.437** | **+0.971** | 1.034 | **0.94** |
| S0 + G3 vs S0 | +0.615 | +0.066 | -0.476 | +1.024 | 1.074 | 0.95 |

By type (v2), rate / predicted: rim 0.94, jumper 0.96, three 0.93.

- **The latent is applied on the scale it was fitted.** The 5-6% shortfall at the rate level is within the logit-normal Jensen term and W's per-game approximation.
- **The 58% at the count level is volume:** a team's make residual drags attempts down. Between-team route of `Cov(FGA_i, R_j)` (S0 + G3; v2 in brackets):

| route (FGA = N + OREB - TOV - 0.44 FTA) | Delta cov |
|---|---:|
| possessions N | -0.58 (-0.55) |
| OREB | -0.81 (-0.77) |
| TOV | +0.22 (+0.23) |
| FTA | +0.17 (+0.16) |
| total FGA | -1.00 (-0.94) |

- **The OREB route is physical.** Fewer misses mean fewer offensive rebounds, in real games too. The per-shot residual the fit uses is net of it, so it is not a defect.
- **The possession route is the clock defect of section 1 seen from the other side.** In the engine more makes buy fewer possessions; in real games they do not (pace x makes ~0, 1.4). So part of G3's covariance is cancelled by the same coupling that owns 74% of the corr gap.
- **Season mismatch is not why.** It runs the other way: the train Sigma OVER-states 2025 (obs/pred 0.73).

### 2.2 Walk-forward refit of G3's Sigma (pre-registered, experiments.md section 3)

`scripts/exp_shared_shooting_v2.py wf`, `results/shared_shooting/wf_v2.json`, run 06:29 EDT after commit `5a62b8e`.

| arm | fold 1 (test 2024): gain vs G3P / floor / floors | fold 2 (test 2025) | between-team obs/pred F1 / F2 | eligible |
|---|---|---|---|---|
| `G3P` (served, pooled train) | 0 | 0 | 0.66 / 0.73 | reference |
| `G3L` (last train season) | 0 (identical by construction) | +0.000003 / 0.00313 / +0.001 | 0.66 / 0.90 | no |
| `G3A` (as-of in-season update, prior = one season) | -0.00055 / 0.00090 / -0.61 | -0.00010 / 0.00045 / -0.22 | 0.77 / 0.85 | no |

- **REFUTED offline: `G3P` stands.**
- The refits do fix the level of the between-team covariance (obs/pred 0.73 -> 0.85-0.90). But the shared term is small against the binomial noise, so the per-game held-out density cannot tell the arms apart.
- Over-stating 2025 by about 37% (obs/pred 0.73) costs nothing measurable.
- G3A's fold-2 path: rim 0.020 -> 0.017, jumper 0.024 -> 0.028 -> 0.025, three 0.004 -> 0.001-0.002, from November to March.
- Months (fold 2, G3A): Nov +0.0001, Dec +0.0011, Jan -0.0008, Feb -0.0002, Mar -0.0005. No month beyond its own noise.

---

## 3. Shared per-game OREB / TOV latents (step 3; pre-registered, `shared_possession/experiments.md` section 1)

`scripts/exp_shared_poss_v1.py`, run 06:31 EDT after commit `5a62b8e`.

### 3.1 Measurement

16,684 verified games, pbp chances tables. The as-of expectation is league season-to-date plus shrunk (K = 200) team offence / opponent defence log-odds plus a train site term. Centred by (season, ISO week).

| season | OREB shared s^2 (SE) | TOV shared s^2 (SE) | OREB x TOV (SE) | within-team excess OREB / TOV | rates OREB / TOV |
|---|---|---|---|---|---|
| 2023 | 0.0023 (0.0022) | 0.0064 (0.0014) | -0.0050 (0.0012) | 0.029 / 0.012 | 0.292 / 0.158 |
| 2024 | 0.0005 (0.0019) | 0.0086 (0.0015) | -0.0042 (0.0013) | 0.026 / 0.014 | 0.297 / 0.148 |
| 2025 | 0.0020 (0.0022) | 0.0064 (0.0015) | -0.0041 (0.0013) | 0.027 / 0.014 | 0.305 / 0.149 |
| 2023-24 | 0.0013 (0.0016) | 0.0074 (0.0011) | -0.0046 (0.0009) | 0.027 / 0.013 | |

- **Per-miss OREB carries no detectable shared component.** The points-scale OREB gap of section 1 (+0.005) is therefore NOT a shared rebounding rate. It comes through misses and possessions, which are the shooting and clock channels.
- **The TOV rate is shared (5-6 SE per season), and so is a negative OREB x TOV term.** Games in which both teams turn it over more are games in which both rebound their own misses less.
- 2025 is not an outlier on any of the three.
- Segments, 2025 TOV s^2: home/away 0.0052 (0.0017), neutral 0.0147 (0.0041); conference 0.0071, non-conference 0.0053; positive in every month. The neutral and month cells are powered only for TOV.

### 3.2 Bake-off

| arm | fold 1 gain / floor / floors | fold 2 gain / floor / floors | eligible |
|---|---|---|---|
| `O1` | -0.00002 / 0.00017 / -0.13 | +0.00010 / 0.00013 / +0.74 | no |
| `T1` | +0.00294 / 0.00078 / +3.76 | +0.00184 / 0.00082 / +2.23 | yes |
| `OT` | +0.00386 / 0.00111 / +3.48 | +0.00294 / 0.00110 / +2.68 | yes: **winner by the letter** (beats T1 by 0.001101 against a 0.001097 floor) |

- Secondary: held-out between-team obs/pred for TOV is 1.35 (F1) and 0.87 (F2). Within-team dispersion obs/pred is 0.98-1.02 in every arm.
- **Wiring (pre-condition 1.4, fixed before the run): NOTHING IS WIRED.**
  - `T1` and `OT` model a channel the engine already reproduces in points. Between-team TOV covariance is sim 0.84 vs actual 0.76 pts^2 (gap -0.0008 corr units, SE 0.0011). Wiring would roughly double it, past reality.
  - `O1`, the only arm the pre-condition allows, is not eligible.
- Why the engine already shares TOV in points while its event model has no shared TOV term is NOT investigated. The likely route is the pace latent through the per-possession accounting.

---

## 4. Owners and the recommended next step

| owner | corr gap | total-var gap | evidence |
|---|---:|---:|---|
| **clock: possession time drawn once, before and marginal over the chance cascade** (pace x makes + pace x OREB, net) | +0.038 (36%) net; the parts are +0.078 / -0.040 | +19.3 pts^2 net | 1.2; mechanism 4.1 |
| unattributed: makes cross-type (FT x opponent FG) | +0.039 (37%) | +3.7 | 1.2 point 4 |
| per-game mean (lane F / G9) | +0.013 (12%) | | 1.2 |
| box-estimator possession accounting (imbalance) | +0.016 (15%) | ~0 | 1.2 point 5 |
| pace x FTA / shot mix / FTA x non-pace | +0.024 (23%) | +11.8 | 1.2 |
| shared make, FT, whistle, OREB, TOV channels; between-game; remainder | the rest, each 5% or less | | 1.2 |

### 4.1 The clock mechanism (measured; owner lane H / the clock)

`scripts/diag_pace_eff_mech_v1.py` (served v2 vs actual 2025; durations from the 2023-25 pbp possessions):

| regression of game N on the game's totals | sim (within-game) | actual (residual, SE) |
|---|---:|---:|
| OREB | **+0.099** | **-0.035 (0.009)** |
| FGM | +0.362 | +0.388 (0.008) |
| TOV | +0.462 | +0.418 (0.008) |
| FTA | +0.173 | +0.176 (0.004) |

Real possession durations (seconds):
- **Each extra chance (OREB) adds 8.1 s, for every start type.** DREB start: 13.2 / 21.4 / 29.9 s for 1 / 2 / 3 chances. Made-FG start: 20.4 / 28.4 / 36.8 s.
- **A one-chance possession ending in a make is 2.6-5.5 s SHORTER than one ending in a missed shot.** DREB start 12.4 vs 16.4 s; made-FG start 20.1 vs 22.8 s.

What the engine does:
- `loop.py` draws one duration per possession BEFORE the cascade (`ad.clock.draw`, section (a)), from a pmf fitted marginal over chains and outcomes.
- So an OREB costs no time: wrong sign on N.
- A make does not shorten its own possession. It only lengthens the next one (the made-FG start type), so makes slow the game about twice as much as in reality: pace x makes -0.074 vs +0.004.

**Proposed for the clock owner (NOT registered or run here; lane H owns the clock today):** a duration conditioned on the possession's realised chance count and terminal class (or per-chance durations), drawn after the cascade or jointly with it. Its interaction with the KD chance-time feed and the transition flag must be designed. Pre-register three lines:
1. N-on-counts slopes (OREB -0.035, FGM +0.388);
2. pace x makes and pace x OREB from this table;
3. G1 count and SD.

Arithmetic expectation: the two parts move in opposite directions on correlation (+0.078 and -0.040), so the net is about +0.04 corr and about +19 pts^2 of total variance. That is enough to move the total SD ratio from 0.925 to about 0.96 if nothing else moves. This is ARITHMETIC, not a sim result.

---

## 5. Not run

- No closed loop, local tap or box request: no arm reached that stage.
  - G3L / G3A are refuted offline.
  - O1 is refuted offline.
  - T1 / OT are blocked by pre-condition 1.4.
- The clock-mechanism arm (4.1): proposed only (lane H's model).
- The makes cross-type owner: not investigated beyond the pair table.
- Served-v2 floor draws: not needed. Every sim-side row here is a 200-seed read whose S0 draw SD is 0.0005 corr units or less.
- Fold-1 (2024) sim reads of served v2: no inputs exist (HANDOFF open item 10).

## 6. Reproduce

    set CBB_TRUTH=verified_v1
    .venv/Scripts/python.exe scripts/diag_g5_channels_v1.py                     # ~20 s; results/g5_channels/channels_v1.json
    .venv/Scripts/python.exe scripts/diag_g5_channels_v1.py --no-segments --boot 0 --seasons-desc --runs <refs...> --out results/g5_channels/channels_refs_v1.json
    .venv/Scripts/python.exe scripts/diag_g3_delivery_v1.py                     # ~25 s
    .venv/Scripts/python.exe scripts/diag_pace_eff_mech_v1.py                   # ~30 s
    .venv/Scripts/python.exe scripts/exp_shared_shooting_v2.py wf               # ~3 s
    .venv/Scripts/python.exe scripts/exp_shared_poss_v1.py measure              # ~5 s
    .venv/Scripts/python.exe scripts/exp_shared_poss_v1.py bakeoff

## 7. Incidents

- None.
  - No process of another lane was touched.
  - At most one process was running at a time, at 2 threads.
  - No engine file was edited.
  - The ledger was edited as bytes with its CRLF endings preserved; `git diff --stat` shows only this lane's rows.
- The shared tree had other lanes' uncommitted edits during this work (`clock_adapter_v3.py` and several scripts). Nothing here ran the engine, so none of it entered a result.


---

## 8. Second task (PM, 07:00 EDT): who owns the cross-type make residual (+0.039, 37%)? (appended 2026-10-01 ~08:40 EDT)

**Answer: the free_throw model's `score_diff` feature, through who shoots and when.**
- Cross-sectionally, `score_diff` carries a real association: FT% is higher when a team leads. That association is a proxy for the team's same-game form.
- Inside the engine the term becomes a CAUSAL response: an opponent make lowers this team's margin, and the model lowers this team's FT make probability.
- It owns about 62% of the group (about +0.024 corr units, 23% of the whole correlation gap).
- It has no fix yet. Rounds 14 and 15 (two feature swaps) are both REFUTED offline. A structural round is proposed (8.5).
- Nothing is adopted. Flag `ENGINE_FT_SCORE` is default-off and parity v9 PASSES with it off.

### 8.1 Possession-level tap

`scripts/diag_ftfg_tap_v1.py`:
- It wraps `loop._shoot_trip` and `ad.ft.predict` in its own process only.
- Coverage: 5,710 games x 6 seeds, 718k FT trips. It is bit-identical to the served 200-seed rows (34,260 rows, 0 mismatched cells). `src/` was clean.
- The FT make channel `R_ft = FTM - FTA f0` splits exactly per trip into:
  - **luck** = sum(made - p);
  - **composition** = sum(p) - f0 FTA (who shoots, in which state, at which p).
- Contexts: half (H1 / H2 before the last 4:00 / last 4:00 + OT), lead state at the trip (trail 4+ / close / lead 4+), trip kind (and-one / one-and-one / 2-3 shot).
- Grader: `scripts/diag_ftfg_report_v1.py` (`results/g5_channels/ftfg_report_v1.json`).

Between-team `Cov(FG make residual of one team, FT residual of the other)`, both directions, summed over rim / jumper / three, pts^2:

| cut | sim luck | sim composition | actual 2025 (SE) |
|---|---:|---:|---:|
| **all** | -0.34 | **-3.19** | **+0.28 (0.36)** |
| H1 | -0.03 | -1.09 | -0.19 (0.23) |
| H2 before 4:00 | -0.18 | -1.18 | +0.52 (0.22) |
| last 4:00 + OT | -0.13 | -0.91 | -0.06 (0.16) |
| trailing 4+ | -0.19 | -1.40 | +0.46 (0.23) |
| close | -0.07 | -0.25 | -0.14 (0.19) |
| leading 4+ | -0.08 | -1.54 | -0.04 (0.21) |
| 2-3 shot trips | -0.27 | -2.31 | +0.12 (0.30) |
| one-and-one | -0.02 | -0.65 | +0.19 (0.14) |
| and-one | -0.05 | -0.23 | -0.04 (0.10) |

- **Predicted-margin tiers (2025).** Sim composition is -3.46 / -3.28 / -2.81 (close / mid / wide). Actual is +0.84 (0.66) / -1.24 (0.72) / +1.20 (0.70).
- **Per season (real, as-of expectations).** 2023 +0.45 (0.45), 2024 +0.49 (0.38), 2025 -0.12 (0.38). Real data shows no coupling in any season: not a 2025 artefact.
- **Per game:** UNDERPOWERED, not produced; about 0.5 pts^2 against per-game binomial noise.

**What the table says.**
- The engine's coupling is composition, not luck.
- It is spread in proportion to FT volume over every half, lead state, trip kind and margin tier.
- That rules out the four brief candidates as owners:
  - the and-one draw (-0.23, its volume share);
  - bonus / one-and-one state (-0.65, its volume share);
  - late-game fouling (the last 4:00 carries 29%, about its share);
  - box accounting of FT possessions. That one is a volume effect, the separate FTA rows in 8.4.

### 8.2 The served FT model's margin term

- Served seg_27 on 2025 attempts, partial dependence on `score_diff`: 0.668 at -20, 0.712 at 0, 0.768 at +20. Already in the first half: 0.673 at -10 to 0.749 at +10.
- In the sim, a trip's first-attempt p rises 0.0019 per point of own margin within (game, side).

### 8.3 Attribution by ablation, and the two refuted feature swaps

**Ablation.** `ENGINE_FT_SCORE=FTn` serves `FT_FEATURES` minus `score_diff`, on the served S1_conf_aligned calendar. Run: 2,000 games x 6 seeds, paired seeds with the served tap. `scripts/diag_ftfg_ablation_v1.py`, `results/g5_channels/ftfg_ablation_v1.json`.

| line | served | FTn |
|---|---:|---:|
| FT composition x opponent FG (pts^2) | -3.35 | **-0.05** |
| per half, H1 / early H2 / late | -1.11 / -1.25 / -0.99 | -0.02 / +0.03 / -0.06 |
| FT x own FG, within team (actual about +0.9) | +3.12 | +0.10 |
| home/away points covariance | 24.10 | 27.79 (+3.7, direction only: Decision 12, a tap does not decide G5) |
| FT% (actual 0.7213) | 0.7192 | 0.7239 |

Removing the term removes the coupling in every cut. **Owner: free_throw `score_diff`.**

**Rounds 14 and 15 (pre-registered, `docs/models/free_throw/experiments.md` sections 14-15; commits `1ef9a3a`, `15d51d7`).**
- Round 14 drops the term (`FTn`, `FTnE`).
- Round 15 replaces it with the pregame team block (`FTp`, `FTpE`).
- **Both are REFUTED offline (FT0 stands).**

Fold-2 guards against FT0:

| arm | log loss vs FT0 | margin-bin calibration |
|---|---:|---:|
| FTn | +0.0045 | 3.9 pp |
| FTnE | +0.0041 | 4.0 pp |
| FTp | +0.0046 (19 floors) | 3.5 pp |
| FTpE | +0.0042 | 3.4 pp |

Fold 1 is the same.

**Why the swaps fail.**
- The pregame team block does not recover the loss, so the margin term is NOT cross-sectional team strength.
- It is same-game information. The real within-(game, team) association of FT% with own margin is +0.0040 per point. Real FT x opponent-FG covariance is about 0, while own FG x FT is positive (about +0.45 per side). So the margin is a proxy for the team's FORM that night, not a cause.
- The engine has no form latent, so the proxy becomes a causal path.
- Round 14's registered primary (within-game slope) was mis-specified for this reason. That is stated in section 15.1; no guard was replaced.

### 8.4 Closing the +0.039 group (2025, pts^2 of between-team covariance, gap = actual minus sim)

| piece | gap | share of group (~5.2 pts^2) | owner |
|---|---:|---:|---|
| FT composition x opponent FG | +3.2 | ~62% (+0.024 corr) | **free_throw `score_diff`** (ablation 8.3) |
| FT luck x opponent FG, and the actual's sampling | +0.6 | ~12% | none (sim luck -0.34 is noise-level; actual +0.28 +/- 0.36) |
| FG x FG cross-type (rim x three +1.05, jumper x three +0.27) | +1.3 | ~25% | unattributed: fg_make has no margin term; G3's fitted rim x three term is 0.0018 |
| remainder (shot-mix pairs) | +0.1 | ~2% | |

**Split of the "+0.024 pace x FTA / shot mix / FTA x non-pace" row: different owners, not this one.**
- **pace x FTA (+0.006) and pace x shot mix (+0.007).** Pace terms of the clock structure in section 4 (lane H).
- **FTA volume x opponent FG (+0.010).** Sim -2.84 vs actual +1.98 (1.23) count-cov. The last 4:00 MATCHES (sim -3.67, actual -3.68). The gap sits in H1 (-0.63 vs +1.02) and early H2 (+1.46 vs +4.64).
  - So it is foul generation outside late game (possession_outcome trip classes / R9ao3), not late-game fouling and not the FT model. FTn leaves it unchanged (-2.91 -> -4.02).
  - Real as-of seasons read -3.5 / -4.5 / -3.4. That residual definition carries team-strength errors, so the sim-mean version above is the comparable one.

### 8.5 Not run, and the proposed next round

- **Not run:** a full-size read of FTn. It did not win offline, so per the brief no box request was filed.
  - If the PM wants the ablation's G5 effect at full size as an ATTRIBUTION read: the flag is ready; the FTn artifacts (78 MB, gitignored, `free_throw/s1_scorediff/FTn/`) need an HF push first.
- **Not run:** the FG x FG cross-type piece (`scripts/diag_fgctx_tap_v1.py` is written and committed).
- **Proposed (not registered):** a per-team-game FORM latent in the shared_shooting form.
  - One draw per team per game, shared between that team's FG and FT make logits, fitted from within-team FT x FG residual covariance (real about +0.45 pts^2 per side);
  - together with an FT model without `score_diff`;
  - judged in the engine (Decision 10), because offline log loss favours the proxy by construction.
- **Incidents:**
  - Two offline launches failed on my own bugs (JSON keys; a NameError) and were rerun; no other lane was affected.
  - A speculative FTp artifact build was stopped (own PID) and its partial directory removed.
  - `.gitignore` gained one line for `free_throw/s1_scorediff/`.
  - Parity ran with another lane's uncommitted `shot_block.py` in the tree; the digest was still bit-identical.
