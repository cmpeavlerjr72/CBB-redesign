# Shared (game-level) shooting latent: measurement, bake-off and closed loop (lane B, 2026-09-30)

Lane B, overnight 2026-09-30, wall clock 20:40 -> 23:10 EDT. **NOTHING IS ADOPTED AND NO DEFAULT IS CHANGED.** The engine hook is default-off and its off path is proved bit-identical.

- Pre-registration: `docs/models/shared_shooting/experiments.md` section 1, commit `86e3d0f` (20:49 EDT). The bake-off ran at 20:50, after that commit.
- Ledger row: `docs/models/change_ledger.md` section A.

## 0. Verdict

| item | result | status |
|---|---|---|
| Step 1: is there a shared shooting component? | **Yes, on two-point shots.** Between-team shared logit variance: rim 0.014-0.021, jumper 0.022-0.026 (4-7 SE per season); three 0.004 or less (inside 2 SE); rim x jumper NEGATIVE. FT make: shared, but about 1% of either gap. If fitted perfectly (arithmetic, 2025): **20% of the correlation gap** (17% of the legacy 0.136) and **36% of the total-variance gap**. Pace x shooting coupling: small and of the WRONG sign to own the pace x efficiency channel | measured |
| Offline bake-off (N / G1 / GP / G3) | **G3 (per-type Sigma) WINS**: +3.65 floors on fold 2, +3.55 on fold 1. G1 (+1.55) and GP (+1.84) are not eligible | pre-registered rule applied |
| Off-path parity | v6 smoke digest bit-identical. Local v3 off path = box S0 rows bit-for-bit (78,000 cells). Box vs local G3 rows bit-for-bit (3.7M cells) | PASS |
| **Closed loop, 200 seeds, full slate, verified truth (box)** | home/away corr **0.1174 -> 0.1350 (+20 floors)**; total SD ratio **0.9306 -> 0.9495 (+24 floors)**; margin SD ratio 1.0500 -> 1.0494 (-0.3 floors); every other G1-G9 line inside 2 floors except possession SD (+2.7, PASS -> PASS). By component: Var(home) +2.47, Var(away) +2.51, Cov +2.65: a pure shared effect | **VALIDATED-PENDING-SHIP-ACTION** |
| Control U1 (unshared, 25 seeds, local) | corr -0.003, Cov -0.09, margin SD ratio **+0.022 AWAY (+11.8 floors)**: unshared variance widens the margin and adds no correlation. The SHARING is what does the work | control as designed |

**Two points for the PM.**
1. **The margin did not fall.** The brief expected a mean-preserving shared latent to lower margin SD. It does not, beyond noise, and arithmetically cannot (1.4, 5.2). Margin SD sits at the 1.05 edge because the sim's margin variance is ABOVE actual (152 vs 137 pts^2), not because missing covariance offsets missing per-team variance. That over-spread belongs to the aggregation object under G9.
2. **G5 still FAILS.** G3 closes 16% of the verified correlation gap and leaves the total SD ratio 0.0005 below the band. It belongs in the Decision-11 ship set with L2 / K2_Ocell / R8b, which remove total-variance props (7.2).

---

## 1. Measurement (step 1)

`scripts/exp_shared_shooting_v1.py measure`, run at 20:47 EDT. Output: `results/shared_shooting/measure_v1.json`.

### 1.1 Predictions and data

Every FGA of 2023, 2024 and 2025 gets a make probability from the served fg_make spec (round 4 `B1`, S1 monthly refits). Each probability comes from a model fitted only on data before the shot's month (`scripts/exp_shared_shooting_preds_v1.py`, 106 s on 4 threads).

| season | role | source of p | FGA | mean y | mean p | design FGA / box FGA |
|---|---|---|---:|---:|---:|---:|
| 2023 | train (both folds) | B1 spec refitted on 2022 (plus earlier 2023 months) | 516,475 | 0.4406 | 0.4392 | 0.9986 |
| 2024 | fold-1 test, fold-2 train | B1 spec refitted on F1 (plus earlier 2024 months) | 603,849 | 0.4412 | 0.4414 | 0.9989 |
| 2025 | fold-2 test | the served `round4/B1` artifacts, read as they are | 630,904 | 0.4422 | 0.4421 | 0.9990 |

- Truth: verified finals only (`verified_finals=True`, `CBB_TRUTH=verified_v1`).
- Games: those with a box row and design rows on both sides, 4,519 / 5,176 / 5,443.

**Notation.** Per game, side and shot type: `R` = sum of (y - p), the residual makes; `W` = sum of p(1-p), the binomial variance.

**The estimator.** A logit shift `u_k` that both teams share gives `Cov(R_hk, R_al) = E[W_hk W_al] Sigma_kl`.
- Sigma is estimated from BETWEEN-team cross products only, so a team's own hot night cannot enter it.
- Residuals are centred by (season, ISO week, type). That removes league drift, which is the season anchor's object; the raw values are reported alongside.
- Units are logit^2. Standard errors are from a 200-replicate Poisson game bootstrap.

### 1.2 The between-team (shared) shooting covariance, by shot type

| season | games | rim | jump2 | three | rim x jump2 | rim x three | jump2 x three | one pooled effect (G1) | raw G1 (not centred) |
|---|---:|---|---|---|---|---|---|---|---|
| 2023 | 4,519 | 0.0184 (0.0037) | 0.0256 (0.0044) | 0.0042 (0.0034) | -0.0028 (0.0028) | 0.0031 (0.0024) | 0.0090 (0.0027) | 0.0071 (0.0013) | 0.0077 |
| 2024 | 5,176 | 0.0211 (0.0032) | 0.0219 (0.0045) | 0.0043 (0.0029) | -0.0043 (0.0028) | 0.0006 (0.0022) | 0.0005 (0.0025) | 0.0047 (0.0011) | 0.0053 |
| 2025 | 5,443 | 0.0140 (0.0033) | 0.0260 (0.0044) | -0.0002 (0.0023) | -0.0059 (0.0029) | 0.0025 (0.0022) | 0.0048 (0.0025) | 0.0041 (0.0012) | 0.0046 |
| 2023-24 (fold-2 train) | 9,695 | 0.0199 (0.0024) | 0.0237 (0.0032) | 0.0043 (0.0020) | -0.0036 (0.0019) | 0.0018 (0.0015) | 0.0045 (0.0018) | 0.0058 (0.0009) | 0.0064 |

**The shared shooting component is real, and it is a two-point component.**
- Rim and mid-range jumper make rates share a game-level effect in every season: SD about 0.14-0.16 on the logit scale, 4-7 SE each.
- The three-point share is near zero (0.004 or less, inside 2 SE in every season).
- The rim x jumper cross term is NEGATIVE in all three seasons (-0.003 to -0.006, 1-2 SE each). A game in which both teams finish better at the rim tends to be one in which both shoot worse from mid-range. That pattern fits a game-level shot-LABELLING component (where the scorer or tracking draws the rim / jumper line) at least as well as a venue or officiating component. It is NOT established; see 1.5.
- The shared variance is most of the within-team excess over binomial. Within-team excess (shared plus unshared) is rim 0.031-0.044, jump2 0.028-0.030, three -0.001 to 0.006. So a team's "hot night" on two-point shots is about half to most a both-teams night.
- Week-centring removes 8-11% of the pooled variance (drift). The size declines across seasons (G1 0.0071 / 0.0047 / 0.0041). 2023 vs 2025 is 1.7 SE apart, so a trend is NOT established.
- Points scale: the correlation of the two teams' points-weighted residual makes is +0.073 / +0.046 / +0.041. The engine's equivalent eFG residual correlation is -0.030 (g1/g5 diagnostic 2.3).

### 1.3 Free-throw make (measured, not a candidate)

Shared FT% variance is 0.0049 (0.0047) / 0.0067 (0.0040) / 0.0075 (0.0038), and 0.0065 (0.0025) pooled.
- The expectation here is an as-of team FT% shrunk to the league season-to-date, an APPROXIMATION and not the served free_throw model. Between-team covariance is insensitive to that choice, because the two teams' prediction errors are independent.
- At about 1 point per make and about 20 attempts per team, it is negligible on the points scale (1.4).

### 1.4 Shares of the G5 gaps if each shared component were fitted perfectly (ARITHMETIC, 2025)

Base: the S0 v3 rows (`v3full_S0_s200_o0_off{0,25}_n25`, 50 seeds, 5,443 games) and the 2025 measured Sigma, applied with each game's actual `W` and points per make. A pure shared effect adds `a_h' Sigma a_h` to Var(home), the same form to Var(away), and `a_h' Sigma a_a` to Cov, where `a = W x points`.

| component | added Cov (pts^2) | added Var home / away | corr 0.117 -> | share of the corr gap (0.114 verified; 0.136 legacy) | total SD ratio 0.920 -> | share of the total-variance gap (42.6 pts^2) | margin SD ratio 1.0386 -> |
|---|---:|---|---:|---|---:|---|---:|
| shooting, per-type Sigma | 3.73 | 3.99 / 3.98 | 0.140 | 20% (17%) | 0.948 | 36% | 1.0403 |
| FT make | 0.12 | 0.14 / 0.11 | 0.118 | 0.6% | 0.921 | 1.2% | 1.0387 |
| both | 3.85 | 4.13 / 4.09 | 0.141 | 21% | 0.949 | 37% | 1.0404 |

- The correlation-gap share (20%) agrees with the diagnostic's 24% for the FG x FG channel. The diagnostic's channel also contains the volume x volume term.
- A pure shared effect cannot lower the margin SD arithmetically. Var(margin) gains `(a_h - a_a)' Sigma (a_h - a_a)`, which is not negative. The brief's expectation that margin SD falls could only come through engine feedback; the closed loop shows none beyond noise (section 5.2).

### 1.5 Game-level vs explained by features the model already has, and what is not identified

- **Features the model already has:** the residual is taken against the served model's own probabilities. Game state (period, clock, bonus, transition, chance elapsed), site, team form and shooter skill are inside `p` already, so none of them can be the source.
- **Pace** (as-of tempo terciles, 2025): 0.0028 (0.0020) / 0.0079 (0.0019) / 0.0014 (0.0019). In 2023-24 the terciles read 0.0071 / 0.0032 / 0.0063. There is no stable gradient with tempo.
- **Site:** home/away 0.0041 (0.0011) vs neutral 0.0025 (0.0032) in 2025. Neutral is UNDERPOWERED (707 games) and not distinguishable from home/away.
- **Month:** 2025 Nov 0.0003 (0.0028), Dec 0.0064, Jan 0.0056, Feb 0.0038, Mar 0.0053 (SE 0.002-0.003); 2023-24 has 0.004-0.007 in every month. Nov 2025 is low, but it is one month inside 2 SE.
- **Per team** (2025, 364 teams, median 30 games): median 0.0037, 62% positive, IQR -0.005 to +0.014. UNDERPOWERED; it is a distribution only.
- **Per game:** 52.0% of 2025 games have a positive between-team product. The effect is small per game and visible only in aggregate.
- **Venue vs officials vs labelling: NOT identified.** There are no official assignments on disk. Venue is the home team, which is confounded with the home defence's unmodelled form. Nothing here separates the three.

### 1.6 Pace x efficiency (the 14% channel), measured on the shooting residual

The real-data covariance of residual makes (both teams, per unit W) with the as-of pace residual `q = ln(box poss) - (a + b * as-of tempo)`:

| fold part | Cov(u, q) | SE | corr(residual makes, q) |
|---|---:|---:|---:|
| F1 train (2023) | -0.00038 | 0.00022 | -0.025 |
| F1 test (2024) | -0.00028 | 0.00025 | -0.019 |
| F2 train (2023-24) | -0.00032 | 0.00015 | -0.022 |
| F2 test (2025) | -0.00040 | 0.00019 | -0.027 |

- Faster-than-expected games shoot slightly WORSE than the model says (about 2 SE).
- In the engine's own terms (rho = -Cov / (s x sigma_clk)) that is a pace-latent correlation of only +0.09.
- Its sign WIDENS the measured pace x efficiency gap, because the engine is already more negative than reality (-0.060 vs -0.041 corr units, diagnostic 2.2).
- **So the pace x efficiency channel is NOT owned by a missing shooting-pace coupling.** The likelier owner is the clock's structure: the possession's duration is drawn before the chance cascade, marginal over OREB chains (diagnostic 1.4). That structure is not tested here.

---

## 2. Pre-registration (summary; the full text is in experiments.md section 1)

| arm | form |
|---|---|
| N | null (served) |
| G1 | one shared logit effect on every shot type |
| GP | G1 coupled to the clock's own v5b pace latent |
| G3 | per-type shared effects with a fitted 3x3 Sigma |
| U1 | control, closed loop only: the effect is unshared, one per team, with G1's variance |

- **Primary:** held-out per-game Gaussian log density of the two-team residual vector (plus the pace residual).
- **Floor:** max(paired game bootstrap, seed-1 training-bootstrap refit).
- **Rule:** fold-2 gain above 2 floors and fold-1 gain above 0. Arms within 1 floor tie; ties go to the simpler arm.
- **Disclosure:** the step-1 measurement (which saw the test seasons) ran before the registration, as the brief ordered. The per-type result is the stated reason G3 sits next to G1. The bake-off itself ran after the commit.

## 3. Offline bake-off

`results/shared_shooting/bakeoff_v1.json`, run 20:50 EDT.

| arm | fold 1 (fit 2023, test 2024): gain / floor / floors | fold 2 (fit 2023-24, test 2025): gain / floor / floors | eligible |
|---|---|---|---|
| N | 0 | 0 | ref |
| G1 (s^2 = 0.00708 / 0.00576) | +0.00077 / 0.00111 / +0.70 | +0.00140 / 0.00091 / +1.55 | no |
| GP (G1 + pace coupling, rho = +0.090) | +0.00091 / 0.00118 / +0.78 | +0.00178 / 0.00097 / +1.84 | no |
| **G3 (per-type Sigma)** | **+0.00553 / 0.00156 / +3.55** | **+0.00553 / 0.00151 / +3.65** | **yes: WINNER** |

- **Winner: G3**, by the pre-registered rule. It is the only eligible arm.
- G1 and GP lose because pooling forces the three-point effect up to the two-point size and forces a positive rim x jumper term, which the data do not have.
- GP's coupling adds about +0.0004 over G1 and is inside the floor.

G3's served Sigma (fold-2 train fit; rim, jump2, three):

    [[ 0.0199, -0.0036, 0.0018],
     [-0.0036,  0.0237, 0.0045],
     [ 0.0018,  0.0045, 0.0043]]

Secondary lines (both folds):
- **Held-out between-team covariance, observed / predicted = 0.73 (fold 2), 0.66 (fold 1).** The training seasons carry MORE shared variance than the test season on both folds (the decline in 1.2). The latent as fitted therefore over-states the test season's shared covariance by about 37% (fold 2). This is reported, not corrected: correcting it would be tuning on the test season.
- **Per-team-game residual variance, observed / predicted** (in points): 0.970 for G3, 0.974 for N. By type: rim 0.965, jump2 1.003, three 0.994. The within-team dispersion is calibrated to within 3.5% in every arm.
- **Segments (fold 2, G3 gain per game):**
  - site: home/away +0.0055, neutral +0.0058;
  - month: Nov +0.0075, Dec +0.0095, Jan +0.0075, Feb +0.0020, Mar -0.0002.
- **Segments (fold 1, G3):** home/away +0.0060, neutral +0.0027; every month positive.
- **Responsiveness** (the standing rule): the latent has no team or prior input and a zero mean, so there is no quintile slope to test. The fg_make predictions it rides on are unchanged.

---

## 4. Engine wiring and parity

**Code.**
- New module `src/cbb_sim/engine/shared_shooting.py` (`ENGINE_SHARED_SHOOTING` = `G1` | `GP` | `G3` | `U1`; unset or `reference` = OFF).
- Parameters in `data/processed/models/shared_shooting/params_v1.json`: the fold-2 TRAIN fit, written by the bake-off and never tuned on sim output.
- Hook in `loop.py`: three small hunks, commit `0ac56fd`. `git diff` showed only this lane's hunks before the commit. Lane A's later `525fabd` added its own hunk elsewhere; mine are intact.
- The draw: one latent per simulated game, from its own stream family `shared_shooting` keyed on (seed, game_id). It is drawn once before the first possession and added to `logit(p_make)` for both teams' FGA of each type.
- `GP` reads the clock's v5b latent normal at `LATENT_ORDINAL` without advancing any stream.

**Parity (off path).**
1. **v6 smoke.** `run_engine.py --seeds 5 --max-games 60 --workers 2` (`laneB_ssl_parity_smoke60x5`) digests to sha256 `0d4ddccc...029f`: **PASS, bit-identical** to `parity_reference_windows_v6.json`.
2. **v3 inputs against the box's S0 rows.** A local run with plain `--input-dir engine_v3` reproduced 119 of 120 games. The exception (401756997) is the overlay trap documented in `engine_inputs_v3_tag_path_2026-09-30.md` section 1: the served adapter reads the v2 round-2 event block from a fixed path, and 265 games' blocks differ.
   - New wrapper `scripts/run_engine_v3evb_v1.py` serves `engine_v3/event_block_F2_2025.npz` in-process, which is the S0 overlay without mounts.
   - With it, the off path reproduces `v3full_S0_s200_o0_off0_n25` **bit-for-bit: 120 games x 25 seeds, 78,000 cells, 0 mismatches**.
3. **Cross-platform.** The box's 200-seed G3 run (Linux) and the local 25-seed G3 run (Windows) agree on seeds 0-24 for all 5,710 games: **3,711,500 cells, 0 mismatches**.
4. **Tests.** `tests/test_engine.py`: 20 passed.

---

## 5. Closed loop: G3 vs S0, v3 inputs, verified truth, full slate, paired seeds

**Runs.**
- Local: `laneB_v3full_G3_off0_n25` (seeds 0-24, 4 workers, 21:02-22:10 EDT, measured 4,885 possessions/s).
- Box (request `docs/ops/box_queue/laneB_1.md`, run by the operator 02:15-02:31Z): `laneB_v3full_G3_s200_o0`, seeds 0-199.
- Reference: the box's S0 read `v3full_S0_s200_o0` (seeds 0-199).

**Graders.**
- `scripts/eval_gates.py` (`CBB_TRUTH=verified_v1`), paired with `scripts/grade_shared_shooting_gatepair_v1.py`.
- `scripts/grade_shared_shooting_loop_v1.py` for the G5 components, the mechanism, segments and the paired game bootstrap. It reproduces eval_gates' G5 lines to 4 decimals.

**Floors (Decision 12).**
- 200 seeds: floor = max(SD across the five 200-seed S0 draws, from `engine_gates_F2_2025_s200_v3_S0_2026-09-30.md`; paired game bootstrap, 200 replicates).
- 25 seeds: floor = max(SD across the five 25-seed draws S0, S0f1..S0f4 `_off*_n25`; bootstrap).

### 5.1 The decisive read: 200 seeds, 5,705 graded games, verified truth

| gate | line | target | S0 | G3 | delta | floor (draws / boot) | floors | status |
|---|---|---|---:|---:|---:|---|---:|---|
| G5 | **home/away score correlation** | 0.2283 | 0.1174 | **0.1350** | **+0.0176** | 0.00073 / 0.00086 | **+20 toward** | FAIL -> FAIL |
| G5 | **total SD ratio** | 1.0 (0.95-1.05) | 0.9306 | **0.9495** | **+0.0189** | 0.00076 / 0.00079 | **+24 toward** | FAIL -> FAIL (0.0005 short of the band) |
| G5 | margin SD ratio | 1.0 | 1.0500 | 1.0494 | -0.0006 | 0.00176 / 0.00141 | -0.3 | PASS -> PASS |
| G5 | PIT K-S p | > 0.1 | 0.0183 | 0.0157 | -0.0026 | 0.0074 | -0.4 | FAIL -> FAIL |
| G1 | possessions/game mean | 67.875 | 69.775 | 69.776 | +0.001 | 0.0047 | +0.2 | FAIL -> FAIL |
| G1 | possessions/game SD | 5.474 | 5.584 | 5.596 | +0.012 | 0.0044 | +2.7 away | PASS -> PASS |
| G2 | PPP cells inside | 9/9 | 3 | 3 | 0 | 0 | 0 | FAIL -> FAIL |
| G3 | 3PA share / FTA per FGA / rim share (pooled) | 0.3906 / 0.3295 / 0.3733 | 0.3873 / 0.3169 / 0.3713 | 0.3872 / 0.3169 / 0.3714 | -0.0001 / 0 / +0.0001 | 5e-5 / 8e-5 / 4e-5 | -1.8 / 0 / +2.2 (one unit in the 4th printed decimal) | PASS (all) |
| G4 | eFG% / OREB% / TOV% / FT rate (pooled) | 0.5086 / 0.2984 / 0.1739 / 0.3295 | 0.5014 / 0.2829 / 0.1755 / 0.3169 | 0.5014 / 0.2830 / 0.1755 / 0.3169 | 0 / +0.0001 / 0 / 0 | 4e-5 to 8e-5 | <= 2.2 | unchanged |
| G6 | home margin non-neutral / neutral | +5.743 / +3.288 | 5.852 / 2.151 | 5.837 / 2.151 | -0.015 / 0 | 0.019 / 0.048 | -0.8 / 0 | PASS / FAIL, unchanged |
| G7 | OT rate | 0.0557 | 0.0301 | 0.0302 | +0.0001 | 0.00016 | +0.6 | FAIL -> FAIL |
| G8 | rotation minutes mean / SD ratio, players used | 29.80 / 1.0 / 9.82 | 30.56 / 1.229 / 8.79 | not graded at 200 (`players.parquet` stayed on the box) | | | | unchanged at 25 seeds (5.6) |
| G9 | margin bias / total bias | 0 / 0 | -0.052 / -0.258 | -0.065 / -0.269 | -0.013 / -0.011 | 0.021 / 0.015 | -0.6 / -0.7 | PASS -> PASS |
| G9 | calibration slope | 1.0 | 0.9169 | 0.9192 | +0.0023 | 0.0015 | +1.5 toward | FAIL -> FAIL |
| G9 | bias cells outside (month / tier / pred-total tercile) | 0 | 5 / 2 / 0 | 5 / 3 / 0 | 0 / +1 / 0 | 0.45 / 0.55 / 0.55 | inside (S0's own five draws read 2-3 tier cells) | FAIL / FAIL / PASS |

**Possessions SD (+2.7 floors).** This is the only line that moves beyond 2 floors away from its target. It is a PASS line that stays PASS (5.596 against a 5.474 +/- 0.75 band), so it is not a veto under guardrails revision (f).
- The likely route, NOT measured: better shooting by both teams produces more made-FG starts, which carry longer clock draws (21.4 s vs DREB 14.4 s), so the count picks up a little variance.
- The same route couples shooting and pace negatively, the direction section 1.6 found in real data.

### 5.2 G5 by component (the guardrails revision asks for G5 by component)

Within-game moments, mean over games, pts^2. The "actual" column holds the actual residual moments against each arm's own per-game sim mean.

| component | actual (vs S0 mean) | S0 | G3 | delta (boot SD) | gap closed |
|---|---:|---:|---:|---:|---:|
| Var(home) | 109.44 | 102.00 | 104.47 | +2.47 (0.13) | 33% of 7.44 |
| Var(away) | 105.65 | 101.29 | 103.80 | +2.51 (0.15) | 58% of 4.36 |
| Cov(home, away) | 38.90 | 25.66 | 28.31 | +2.65 (0.10) | 20% of 13.24 |
| Var(total) = Vh + Va + 2Cov | 292.89 | 254.60 | 264.89 | +10.29 | 27% of 38.3 |
| Var(margin) = Vh + Va - 2Cov | 137.29 | 151.97 | 151.65 | -0.32 | the sim stays 14.7 too wide |

- **The latent behaves as a pure shared effect.** Var(home), Var(away) and Cov each rise by about 2.5-2.65, so total variance rises by about 4 Cov and margin variance does not move (-0.3, inside noise).
- **The brief's "lower margin SD" does not appear beyond noise:** mean sim margin SD -0.013 points, ratio -0.3 floors. That matches the arithmetic in 1.4: a pure shared effect cannot narrow the margin.
- **Margin SD passes by a different route than the brief assumed.** The sim's margin variance (152) is ABOVE actual (137), and the ratio sits at the 1.05 edge. That over-spread belongs to the aggregation object now owned under G9; this latent neither causes nor fixes it.
- **The compensation is not unwound here.** Total variance still rests on the props HANDOFF names (excess possessions, the over-dispersed foul constant). When those fixes ship, total SD falls by their share. The latent replaces part of that variance with a mechanism that exists in the data.

### 5.3 Mechanism: within-game between-team covariance of made FG by type (200 seeds)

| pair (home x away) | S0 | G3 | delta | predicted from Sigma and the sim's own W |
|---|---:|---:|---:|---:|
| rim x rim | 0.389 | 0.786 | +0.397 | 0.570 |
| jump2 x jump2 | 0.115 | 0.365 | +0.250 | 0.270 |
| three x three | 0.308 | 0.383 | +0.076 | 0.110 |
| rim x jump2 (both orders) | 0.210 / 0.211 | 0.093 / 0.109 | -0.117 / -0.103 | -0.066 / -0.063 |
| jump2 x three (both orders) | 0.185 / 0.150 | 0.246 / 0.212 | +0.061 / +0.062 | 0.074 / 0.080 |
| rim x three (both orders) | 0.383 / 0.305 | 0.384 / 0.295 | +0.001 / -0.010 | 0.047 / 0.048 |
| **total** | 2.256 | 2.874 | **+0.618 (boot 0.016)** | 1.071 |

- **The engine realises 58% of the first-order make covariance** (rim x rim 70%, jumpers 93%).
- **A likely dilution path, NOT established:** a rim miss feeds an OREB and a putback rim attempt, so made rim FG covary less than the logit shift alone implies.
- **Net position is short of reality, not past it.** The points covariance actually added (+2.65) is BELOW the 2025 measured shared covariance (3.73, section 1.4). The training Sigma over-states 2025 by about 37% (section 3), but the engine's dilution under-delivers by more.

### 5.4 Levels (the logit-normal mean shift was not corrected, as pre-registered)

| line | S0 | G3 | change |
|---|---:|---:|---:|
| rim make rate | 0.5718 | 0.5713 | -0.055 pp |
| jumper make rate | 0.3915 | 0.3919 | +0.041 pp |
| three make rate | | | +0.004 pp |
| eFG% | | | -0.008 pp |
| mean total | | | -0.011 points (boot 0.008) |

These are the predicted Jensen signs and sizes: rim (p > 0.5) moves down, jumpers move up, and every change is under 0.1 pp.

### 5.5 Segments, per game and per team (200 seeds; G3 vs S0)

| segment | games | total SD ratio | margin SD ratio | corr sim (actual) | Cov(h, a) |
|---|---:|---|---|---|---|
| home/away sites | 4,969 | 0.9301 -> 0.9492 | 1.0418 -> 1.0411 | 0.109 -> 0.127 (0.210) | 25.57 -> 28.27 |
| neutral | 736 | 0.9328 -> 0.9510 | 1.1165 -> 1.1163 | 0.203 -> 0.216 (0.378) | 26.24 -> 28.57 |
| Nov | 1,221 | 0.885 -> 0.904 | 0.962 -> 0.962 | 0.029 -> 0.048 (0.111) | |
| Dec | 915 | 0.986 -> 1.006 | 1.072 -> 1.077 | 0.041 -> 0.059 (0.134) | |
| Jan | 1,422 | 0.924 -> 0.942 | 1.060 -> 1.056 | 0.186 -> 0.202 (0.322) | |
| Feb | 1,365 | 0.948 -> 0.967 | 1.095 -> 1.094 | 0.184 -> 0.204 (0.311) | |
| Mar | 765 | 0.947 -> 0.967 | 1.108 -> 1.110 | 0.210 -> 0.226 (0.367) | |

- **The move is uniform.** Every site and month shifts by +0.018 to +0.020 in total SD ratio and by +0.013 to +0.019 in correlation, with margin flat.
- **Segment power.** Each cell holds about 740-1,420 games: enough for the paired delta, not enough for absolute G5 verdicts.
- **Per game (paired):**
  - within-game Cov: mean +2.65, median +2.67, positive in 64% of games;
  - total SD: mean +0.32, up in 67% of games;
  - margin SD: mean -0.013, down in 51% of games (no change).
- **Per team** (364 teams with 10 or more games): total SD rises for 99.2% of teams (quartiles +0.24 / +0.32 / +0.40); margin SD falls for 55%. These are paired changes; per-team G5 verdicts are UNDERPOWERED.

### 5.6 The 25-seed local read (same games, seeds 0-24) agrees

| line | S0 | G3 | delta | floor (5 draws / boot) | floors |
|---|---:|---:|---:|---|---:|
| home/away corr | 0.1190 | 0.1368 | +0.0178 | 0.0019 / 0.0023 | +7.7 |
| total SD ratio | 0.9058 | 0.9229 | +0.0171 | 0.0036 / 0.0021 | +4.8 |
| margin SD ratio | 1.0216 | 1.0184 | -0.0032 | 0.0019 / 0.0036 | -0.9 |
| Cov(h, a) | 25.80 | 28.43 | +2.62 | 0.17 / 0.29 | +9.2 |

- No other gate line moved by more than 2 floors at 25 seeds. The full table is `results/shared_shooting/grade/gatepair_G3_s25.md`.
- G8 is unchanged at 25 seeds: rotation minutes SD ratio -0.0003, players used 0.
- The PIT p row in that file is mis-parsed: the p values are printed in e-notation. Both arms are FAIL.

### 5.7 Control U1: the same size of effect, but unshared (local, 25 seeds, seeds 0-24)

U1 gives each team its own effect, with G1's variance on every shot type (`laneB_v3full_U1_off0_n25`, 3 workers, 22:11-23:07 EDT).

| line | S0 | G3 (shared) | U1 (unshared) | U1 delta (floors) |
|---|---:|---:|---:|---|
| home/away corr (actual 0.2283) | 0.1190 | 0.1368 | 0.1158 | -0.0032 (-1.4) |
| total SD ratio | 0.9058 | 0.9229 | 0.9170 | +0.0112 (+3.1) |
| **margin SD ratio** | 1.0216 | 1.0184 | **1.0438** | **+0.0222 (+11.8 draws; +6.2 on max floor), AWAY from 1.0** |
| Var(home) / Var(away) | 102.5 / 101.0 | 104.6 / 103.9 | 105.7 / 104.4 | +3.2 / +3.4 |
| Cov(home, away) | 25.80 | 28.43 | 25.71 | -0.09 (-0.4) |
| made-FG between-team cov (mechanism) | 2.345 | 2.900 | 2.239 | -0.11 |
| PIT K-S p | FAIL | FAIL | FAIL (worse, +8 draws on the parsed value) | |

**The contrast is clean.**
- Unshared variance of the same size adds per-team variance, no covariance and no correlation. It pushes the margin SD ratio 11.8 floors further over-spread, which is G9's existing problem.
- The SHARED structure is what moves G5 correlation and total SD without widening the margin.
- This is the round-8 whistle result (`R8bS` vs `R8bU`) repeated on shooting, with points variance this time.

---

## 6. Status, against the pre-registered closed-loop rule (experiments.md 1.7)

At the 200-seed read:
- **Correlation:** +0.0176, 20 floors toward target.
- **Total SD ratio:** +0.0189, 24 floors toward target.
- **No other line regresses beyond floor.** The exception is possession SD (+2.7 floors, a PASS line staying PASS; mechanism named in 5.1). Rim share +2.2 floors is one unit in the printed 4th decimal.

**Status: VALIDATED-PENDING-SHIP-ACTION (Decision 11).** Arm `G3`, default-off, params `params_v1.json`. Nothing is adopted and no default changes. The PM decides. The ship-set considerations are in 7.

---

## 7. What this does NOT establish, caveats, and what the PM should weigh

1. **The G5 lines still FAIL.** Correlation is 0.135 against 0.228; G3 closes 16% of the verified-truth gap (0.111). Total SD ratio is 0.9495 against a 0.95 floor.
   - The remaining correlation gap is the whistle (round 8: refuted as the main channel, partly fixable), the pace x efficiency channel (1.6: NOT shooting), and the mean x residual cross term (lane F's).
   - The make-level mechanism is diluted (5.3), so the engine delivers about 71% of the 2025 measured shooting covariance in points (2.65 of 3.73).
2. **Ship-set interaction (HANDOFF cross-cutting finding 2).** The total SD ratio is still propped by excess possessions (clock `L2` removes them) and by the over-dispersed foul constant (`R8b` removes it).
   - G3 adds about 10 pts^2 of within-game total variance from a source that exists in real data. That is part of what L2 and R8b would take away, but not all: their round reads moved total SD ratio by -0.024 (L2) and about -0.01 to -0.015 (R8b, 25 seeds).
   - G3 should be read inside the Decision-11 set: {L2, K2_Ocell, R8b/R8bS, G3}.
3. **Over-statement of the test season.** The training seasons carry about 37% more shared covariance than 2025 (fold 2), and fold 1 shows the same (0.66 obs/pred). The fit is honest (train only), but a season-level decline may be real; at 1.7 SE it is not established.
   - If it is real, a walk-forward refit each season (2024-25 data for 2025-26) is the honest response, not a scale-down.
4. **What the shared component physically is, is not identified:** venue, officials or shot labelling (1.5). The negative rim x jumper term points at labelling, which would make part of the effect a measurement artefact of the rim / jumper split.
   - Points-weighted it is still a real points covariance in the data: the points-scale residual correlation is +0.04 to +0.07, the eFG residual correlation in the g1/g5 diagnostic is +0.034, and its points share is in 1.4.
5. **FT make:** shared, but worth about 1% of either gap. Not pursued.
6. **GP (pace coupling)** is not eligible offline and has the wrong sign for the pace x efficiency gap. **Pace x efficiency (14% of the gap) remains OPEN**; the clock structure is the suspected owner (1.6).
7. **Not run:**
   - G1 and GP closed loops (not eligible offline);
   - U1 at 200 seeds (box tier 2, status in `aws_launch_chain.md` section 19; the 25-seed local read in 5.7 already separates shared from unshared beyond floor on the margin and correlation lines);
   - G8 at 200 seeds (player rows stayed on the box);
   - a G3 refit on the corrected foundations (event layer v4, E3 features) for the next full retrain.
8. **Resume / reproduce:**

       .venv/Scripts/python.exe scripts/exp_shared_shooting_preds_v1.py         # 2 min, 4 threads
       .venv/Scripts/python.exe scripts/exp_shared_shooting_v1.py measure       # 10 s
       .venv/Scripts/python.exe scripts/exp_shared_shooting_v1.py bakeoff       # 3 s
       # closed loop, local (4 workers, about 68 min per 25 seeds on the full slate):
       ENGINE_SHARED_SHOOTING=G3 CBB_TRUTH=verified_v1 ENGINE_EVENT=round2_s1 ENGINE_CLOCK=v5b_glat_pmean \
         ENGINE_ROTATION=reference ENGINE_FG3=decision8 .venv/Scripts/python.exe scripts/run_engine_v3evb_v1.py \
         --fold F2 --season 2025 --seeds 25 --seed-offset 0 --workers 4 --games-per-block 60 --seeds-per-block 25 \
         --tag laneB_v3full_G3_off0_n25 --results-dir results/engine_v0 --input-dir data/processed/models/engine_v3
       # box: docs/ops/box_queue/laneB_1.md (200 seeds; U1 as tier 2)
       .venv/Scripts/python.exe scripts/grade_shared_shooting_loop_v1.py --arm <arm dir> --ref <S0 dir> [--draws ...] --out <json>
       .venv/Scripts/python.exe scripts/grade_shared_shooting_gatepair_v1.py <arm.md> <ref.md> [<draw.md> ...]

## 8. Incidents caused by this lane

- **Ledger line endings.** Commit `86e3d0f` re-wrote 15 LF line endings of `docs/models/change_ledger.md` as CRLF (my editor normalised the mixed file). The content was unchanged. It was repaired byte-for-byte in `5f2227d` two minutes later.
- **Nothing else.**
  - No process of another lane was touched.
  - Every compute step stayed at 4 cores or fewer: 4 workers x 1 thread, or 3 workers plus 1 grader.
  - No AWS access; the box run went through the operator queue.

## 9. Files

- **Docs:** `docs/models/shared_shooting/experiments.md` (sections 1-2), this doc, the ledger row.
- **Scripts:**
  - `scripts/exp_shared_shooting_preds_v1.py`
  - `scripts/exp_shared_shooting_v1.py`
  - `scripts/run_engine_v3evb_v1.py`
  - `scripts/grade_shared_shooting_loop_v1.py`
  - `scripts/grade_shared_shooting_gatepair_v1.py`
- **Engine:** `src/cbb_sim/engine/shared_shooting.py`, three hunks in `src/cbb_sim/engine/loop.py`, `data/processed/models/shared_shooting/params_v1.json`.
- **Results (gitignored):**
  - `results/shared_shooting/{preds_v1.parquet, measure_v1.json, bakeoff_v1.json, loop_*.json, grade/}`
  - `results/engine_v0/laneB_v3full_{G3_off0_n25, U1_off0_n25, G3_s200_o0}`
