# Clock round 8: league-relative asymmetric team tempo, the in-season term, and outcome-conditioned possession time (lane H, 2026-10-01)

**Nothing is adopted and no default changes.** Every arm is a default-off `ENGINE_CLOCK` mode. The off path is bit-identical to parity v9; this was checked twice, before and after the `loop.py` edit.

**Pre-registrations** (`docs/models/clock/experiments.md`), each committed before the runs it governs:

| section | content | commit |
|---|---|---|
| 36 | tempo arms M1 / M2 / M2D / G2D and the decision rule | `4c98267` |
| 37 | offline result; M2D closed-loop registration | `b9bb713` |
| 38 | PM-ruled amendment: K1 / K2 / K2M (outcome-conditioned time) | `bf51431` |
| 39 | K offline result; a mis-specified gate line disclosed; K closed-loop registration | `7383999` |

**Run log** (EDT, system clock):

| time | step |
|---|---|
| 06:14 | start |
| 06:16-06:20 | over-spread decomposition |
| 06:22 | section 36 committed |
| 06:28-06:32 | tempo arms trained and graded |
| 06:33 | reseed proof |
| 06:34 | latent refit |
| 06:35 | parity PASS |
| 06:39 | section 37; box `d1001_H_1` |
| 06:39-07:18 | M2D tap |
| 06:42 | PM ruling |
| 06:45 | mechanism check |
| 06:50 | section 38 |
| 06:47-06:56 | K arms trained |
| 06:57-07:01 | K offline grade and post-hoc line |
| 07:03-07:04 | K latents |
| 07:05-07:07 | parity PASS after the `loop.py` edit |
| 07:12 | section 39; box `d1001_H_2` |
| 07:13-09:27 | local taps K2 (4 x 25), K2M (3 x 25), K1 (25) |
| 08:06 | box `d1001_H_3` |
| 09:12 | M2D full size back |
| 09:21 | K2M full size back |
| 09:37 | served-stack seed draws back |
| 10:36-10:58 | horn taps, 5 modes x 10 seeds |

## 1. Why A2 over-spread the game level

Script: `scripts/diag_clock_r8_overspread_v1.py` -> `results/clock_r8/diag_overspread.json`. Data: 5,445 games (2025), from round 7's full-size runs.

**The excess is all between games, in the team terms. The game latent does not double-count.** In possessions per team-game, squared:

| | between-game variance | calibrated benchmark corr^2 x Var(actual) | within-game variance | needed within |
|---|---:|---:|---:|---:|
| A2 | 12.40 | 7.98 | 21.81 | 22.99 |
| L2 | 7.15 | 6.19 | 23.14 | 24.78 |

A2's refitted latent was too small, not too large: sigma 0.0433 against 0.0468.

**Game-level elasticity** of log possessions on X = log rel_home + log rel_away (month fixed effects):
- Actual: 0.928.
- L2: 0.719 offline, 0.709 in the sim.
- A2: 1.089 offline, 1.123 in the sim.
- A2 is 1.21x the actual, and the ratio is flat by month.

**Cause.** Round 7 fitted the AFT scale by OLS on log(duration + 0.5), which is a geometric-mean elasticity, and then served it as a scale on the whole law, i.e. on the MEAN. On the same rows, the implied game elasticity is 1.146 on the log scale versus 1.024 on the mean scale (Poisson pseudo-likelihood). That accounts for about 0.12 of A2's +0.19 excess. The rest:
- the fit rows exclude the last minute and composition adds about 0.07;
- the sim adds about 0.03.

**Other checks.**
- Tempo feature train/serve skew: none (99.96% of games exact).
- In-season drift: actual mean duration rises about 3% from November to March, every season from 2022 to 2025.

## 2. Tempo arms, offline

Script: `scripts/grade_clock_r8_offline_v1.py` -> `results/clock_r8/offline_grade.json`.

**Floors.**
- Reseed floor is 0, **proved**: M2D F2 refitted under `--seed 1` gives identical coefficients and a maximum pmf difference of 0.0 (`results/clock_r8/reseed_proof.json`).
- The binding floor is 2 x the paired game-bootstrap SE.

**Fold 2 (selects):**

| arm | d deviance vs C0 (floor) | offence-q slope | elasticity ratio | count cal. slope | between-SD ratio | count gap | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| C0 = L2 | -- | 0.401 | 0.743 | 0.913 | 1.095 | +0.527 | control |
| A2 (r7) | -0.0291 (0.0045) | 0.987 | 1.126 | 0.843 | 1.186 | +0.733 | reference |
| M1 | -0.0301 (0.0045) | 0.915 | 1.006 | 0.920 | 1.087 | +0.767 | REFUSED: count veto +0.240 |
| M2 | -0.0302 (0.0045) | 0.902 | 0.989 | 0.931 | 1.074 | +0.772 | REFUSED: count veto +0.245 |
| **M2D** | -0.0295 (0.0044) | 0.899 | 0.986 | 0.933 | 1.072 | +0.615 | **WINS** every line (count veto +0.088 < 0.10) |
| G2D | -0.0297 (0.0044) | 0.879 | 0.964 | 0.948 | 1.055 | +0.620 | FAILS (c): \|ratio - 1\| vs A2 -0.091, floor 0.110 |

**Fold 1, M2D:**
- deviance -0.0386 (floor 0.0050);
- offence slope 1.025;
- elasticity ratio 1.100 (A2 1.268);
- count gap -0.010 (C0 +0.369);
- signs kept.

**Segments, M2D on F2:**
- Offence slope by month, Nov to Mar: 0.64 / 0.79 / 0.93 / 1.07 / 1.21. Responsiveness rises through the season.
- By site: 0.85 / 0.94 / 0.87.
- By tier: both-power 0.87, neither 0.94, one-power 0.60. The one-power cell has 157 games, at the power line.
- Elasticity ratio by month: 0.89-1.10.
- Month gaps (s): Nov -0.17, Dec -0.07, Jan -0.29, Feb -0.17, Mar -0.04.

**Latent refit on M2D:** sigma 0.0421. It fell, against my stated expectation that it would rise.

## 3. The pace x efficiency channel: it has a clock-structure owner

This section checks lane B's mechanism under the PM ruling. Script: `scripts/diag_clock_r8_chance_time_v1.py`.

**What the real data show:**
- Duration equals the sum of the chance durations in 99.999% of possessions.
- Each extra chance adds 8.2 s. That is a longer first chance (19.5 vs 16.8 s) plus the continuation itself (mean 5.8 s).
- One-chance possessions after a DREB start last 12.4 s when they end in a made FG, 16.5 s for a missed FG, 11.5 s for a TOV and 10.5 s for an FT trip.
- Per possession, Cov(duration, points | start type) = -1.36.

**What the served engine does:** it draws the time before the cascade, marginal over outcomes. Within-game possessions on OREB are **+0.103** at full size; the real value is **-0.011 to -0.033**.

**KD (`ENGINE_CHANCE_TIME`).** It conditions on the drawn whole-possession duration with a whole-duration table, which is internally consistent.
- The bins agree for 87% of chance-1 shots.
- For the other 13% (multi-chance possessions), the table mean elapsed is 21.7 s, against 16.0 s realised and 17.0 s from a chance-1 table.
- The K arms read KD at the first-chance draw. This mild mismatch is left unchanged. A d1-conditioned KD table is the named follow-up.

**Answer.** Yes: the clock's draw-before-outcome structure owns the pace x efficiency channel. The full-size K2M read closes lane B's two rows (section 6).

## 4. K arms, offline

Scripts: `grade_clock_r8_chance_offline_v1.py` and `diag_clock_r8_chance_resid_v1.py`.

**The arms:**
- **K1:** a first-chance law plus a per-OREB continuation time.
- **K2:** K1, plus the first-chance time re-drawn after the cascade at the same uniform, from d1 | chance-1 end class.
- **K2M:** K2 on the M2D law.

On F2, K2's conditional first-chance deviance is 6.708 against 6.821 for its marginal.

**The pre-registered gate line FAILS for every arm.** The line was |possessions-on-OREB slope - actual|:

| arm | change | floor |
|---|---:|---:|
| K1 | +0.016 | about 0.05 |
| K2 | +0.079 | about 0.05 |
| K2M | +0.057 | about 0.05 |

That line is confounded: the actual count carries a reverse pace arrow, so faster games have more of every event.

**Post-hoc line, labelled as post-hoc:** the residual time regressed on per-possession rates, target 0. In seconds per unit rate, F2:

| arm | OREB | FGM | TOV | FTA |
|---|---:|---:|---:|---:|
| C0 | +6.95 | -4.72 | -3.74 | -2.98 |
| K1 | +1.31 | -4.77 | | |
| K2 | -0.98 | -1.63 | +0.21 | -0.58 |
| K2M | -0.91 | -1.31 | +0.83 | -0.50 |

F1 shows the same pattern.

**Status.** By the letter, no K arm reached the loop. They were run as PM-directed reads (section 39).

## 5. Local taps

Setup:
- 476 games, verified sample.
- Control: served `v3full_COMB9GCTKD_s200_o0`, paired.
- Floor: max(4-8 control seed-block draws, 2 x bootstrap SE).
- Files: `results/clock_r8/loop_grade_*_sample_*.json`.

**Direction only (Decision 12). G5 lines are UNDERPOWERED.**

| line (control) | M2D (50 seeds) | K2 (100) | K2M (50) | K1 (25) |
|---|---|---|---|---|
| team slope (0.64) | 0.84 | 0.64 | 0.86 | 0.62 |
| elasticity ratio (0.72) | 1.00 | 0.72 | 1.03 | 0.69 |
| within-game N on OREB (+0.102) | +0.104 | -0.009 | -0.017 | +0.035 |
| G5 total SD ratio (0.91) | 0.89 | 0.965 (+5.4 floors) | 0.94 | 0.89 |
| G5 corr (0.134; actual 0.208) | 0.143 | 0.177 | 0.186 | 0.130 |
| G9 total slope, MC (1.00) | 0.89 | 1.02 | 0.87 | 1.05 |
| G1 count - v4 (+0.85) | +0.85 | +0.79 | +0.67 | +0.87 |

- **K1** (OREB time only) does not move total variance. The outcome-conditioned first chance is the part that matters.
- **K2 alone** shows no visible G9 cost.

## 6. Full size (box, 5,710 x 200)

Requests: `d1001_H_1` (M2D) and `d1001_H_2` (K2M). Graded locally with `scripts/grade_clock_r8_loop_v1.py` and lane B's `diag_g5_channels_v1.py`; outputs `results/clock_r8/loop_grade_{M2D,K2M}_full.json` and `results/clock_r8/g5_channels_full.json`.

**Floors (Decision 12):** max(the 5 served-stack draws `v3full_COMB9GCTKD` + `d1001D_S2f1..4`, 2 x paired bootstrap SE). The sample is 5,445 pbp-complete games for the count lines.

| line | served | M2D (floors) | K2M (floors) |
|---|---:|---|---|
| sim team pace slope | 0.671 | 0.896 (+13.3) | 0.907 (+13.2) |
| game elasticity ratio (target 1) | 0.765 | 1.074 (+14.9) | 1.096 (+15.2) |
| game-prior slope | 0.855 | 1.075 | 1.083 |
| G1 count - v4 | +0.623 | +0.636 (+0.3) | **+0.438 (-4.3)** |
| G1 gate mean / SD (target 67.875 / 5.474) | 68.823 / 5.504 | 68.845 / 5.633 | 68.648 / 5.761, PASS |
| G1 by month (gate) | 4/5 | **2/5** | 4/5 |
| pooled possession SD (v4 truth 5.565) | 5.516 | 5.644 (+4.1) | 5.774 (+8.1, past truth) |
| possession SD ratio | 0.968 | 0.969 (+0.2) | 0.985 (+1.9) |
| count calibration slope | 0.928 | 0.891 (-1.3) | 0.859 (-2.3) |
| within-game N on OREB (actual residual -0.021 / -0.035) | +0.103 | +0.101 | **-0.018** (-96) |
| within corr(N, PPP) (actual residual -0.087 / -0.091) | -0.173 | -0.184 | **-0.107** (+32) |
| pace x makes, corr units (actual +0.004) | -0.074 | -0.075 | **+0.005** (gap 0.078 -> 0.001) |
| pace x OREB (actual -0.048 / -0.052) | -0.008 | -0.007 | **-0.046** (gap -0.040 -> -0.006) |
| **G5 total SD ratio** | 0.926 FAIL | 0.924 FAIL | **0.968 PASS** (+7.0 floors) |
| **G5 home/away corr** (0.228) | 0.126 | 0.139 | **0.178** (corr gap 0.106 -> 0.053) |
| G5 margin SD ratio | 1.043 | 1.044 | 1.046 (+1.0) |
| G9 total bias | -0.280 | -0.240 | -0.293 |
| G9 total MAE | 13.385 | **13.162 (-2.7)** | **13.202 (-2.0)** |
| **G9 total slope** (MC-corrected) | 0.999 | **0.936 (-2.1)** | **0.911 (-2.7)** |
| G9 calibration slope (gate) | 0.948 | 0.949 | 0.945 |
| G9 bias by predicted-total tercile (gate) | 0/6 PASS | **1/6 FAIL** | **1/6 FAIL** (bottom tercile total bias -0.80 -> -1.23 for M2D) |
| G7 OT rate (0.0557) | 0.0305 | 0.0305 | **0.0358 (+11.6)** |

**Reading.**
- **K2M closes both of lane B's clock rows and flips G5 total SD ratio FAIL -> PASS.** The G5 corr gap halves, the G1 count improves, and OT moves toward the truth.
- **The cost comes with the tempo half (M2D).** It is the same in M2D alone and in K2M:
  - G9 total slope -0.06 to -0.09 (2-3 floors);
  - one G9 tercile cell flips PASS -> FAIL;
  - for M2D alone, G1 by month 4/5 -> 2/5 (November and December move just past +1.0);
  - the possession SD overshoots the truth.
- The game elasticity now overshoots by 7-10% (A2: 21%), so a small pace over-spread remains (count calibration slope 0.89 / 0.86).

**Why the G9 total slope still falls when pace is nearly calibrated.**
- Between games, the sim's corr(log pace, log PPP) rises from -0.055 (served) to +0.003 (M2D); the actual is -0.082.
- The efficiency calibration slope stays at 0.72 in both. In the sim, faster teams are more efficient per possession; in reality they are not.
- So the old under-responsive pace was partly hiding an efficiency over-spread. That is the Decision 11 "exposed compensation" pattern.
- The candidate owner is the pace-linked efficiency the transition / chance-time feed adds on top of team ratings that already contain it. **Not tested.**

**K2 alone at full size:** `d1001_H_3` (scp of 49 MB artifacts), **PENDING** at the time of writing.

## 6b. End-of-half possession count (lane L's line, PM message ~10:34) and composition with the late-game laws

**Method.** `scripts/diag_clock_r8_horn_tap_v1.py` (chain `scripts/chain_clk8_horn_v1.sh`) wraps the clock adapter in process and records every possession start: period and seconds left. The wrapper delegates every call, so runs are bit-identical to plain runs. Run size: the 500-game verified sample x 10 paired seeds per mode (`results/clock_r8/horn_*.json`); the actual is `possessions_v4`, same 489 games.

**Possessions started per game, by seconds left at the start:**

| half | bucket (s) | actual | served L2 | M2D | K1 | K2 | K2M |
|---|---|---:|---:|---:|---:|---:|---:|
| H1 | 0-3 | 0.076 | 0.104 | 0.098 | 0.103 | 0.104 | 0.099 |
| H1 | 3-6 | 0.090 | 0.158 | 0.152 | 0.154 | 0.152 | 0.157 |
| H1 | 6-10 | 0.155 | 0.221 | 0.208 | 0.215 | 0.212 | 0.208 |
| H1 | 10-20 | 0.254 | 0.533 | 0.526 | 0.548 | 0.548 | 0.527 |
| H1 | 20-35 | 0.785 | 0.814 | 0.835 | 0.853 | 0.838 | 0.847 |
| H2 | 0-3 | 0.100 | 0.203 | 0.199 | 0.208 | 0.200 | 0.191 |
| H2 | 3-6 | 0.186 | 0.303 | 0.315 | 0.301 | 0.320 | 0.289 |
| H2 | 6-10 | 0.286 | 0.400 | 0.400 | 0.413 | 0.401 | 0.379 |
| H2 | 10-20 | 0.695 | 1.011 | 1.009 | 0.970 | 1.011 | 0.950 |
| H2 | 20-35 | 1.235 | 1.606 | 1.586 | 1.574 | 1.634 | 1.494 |

**Reading.**
- **No round-8 arm fixes or worsens the horn excess.** Every arm sits within about 5% of the served value in every bucket; the change for 10 seeds is about the size of the MC noise.
- The excess is large in both halves. In H1, starts at 10-20 s left run at 2.1x the actual, and at 3-6 s 1.7x. H2 also has a large excess.
- **Owner (clock structure, measured from the code):** the served `srfloor` cell coding (`clock_v3.SR_FLOOR_BUCKET = 5`) maps every clock bucket below 45 s onto the 45-59 s bucket.
  - In H1 the law therefore has no notion of the approaching horn. `eg_regime` covers only the end-game score states.
  - Possessions near the horn get mid-half durations, and a team that should hold for one last shot from about 35 s instead fits in another possession.
  - That is lane L's "first-half horn possession count". The fix belongs in a horn-aware window law (lane L's `clk_Dt` family), not in round 8's team or outcome terms.

**Composition with lane L's late-game laws.**
- **The candidate sent to the box, K2M, uses `cont_mode = "K2"`.** Lane L's reading is correct: under K2 the window laws are bypassed. `LateGameClock` forwards `redraw_end` to the served clock, so the outcome-conditioned re-draw replaces the window draw.
- Under K1, the no-shot law's run-to-the-horn is overwritten by `_clock_cont` unless the guard `used[ns] = left` is added.
- **The hook is kept and documented.** `loop.py` reads `ad.clock.cont_mode` and calls `ad.clock.redraw_end(...)` and `ad.clock.draw_cont(...)`. A late-game wrapper composes by overriding `redraw_end` with an outcome-conditioned window law, and by keeping its no-shot rows at `used = left` after `_clock_cont`.
- I did not edit lane L's adapter.

## 7. Verdict per pre-registered line

| line | verdict |
|---|---|
| Section 36 offline: M2D | **WINS** every line on F2 (deviance -6.7 floors, slope 0.40 -> 0.90, elasticity ratio 0.74 -> 0.99, count veto +0.088 < 0.10). F1 signs hold. M1 and M2 are refused by the count veto. G2D fails line (c). |
| Section 37 closed loop, M2D, full size | team slope **PASS** (+13 floors); game level no longer over-spread to A2's degree (ratio 1.07 vs 1.21) but still over by 7%; G9 total slope **FAILS** (-0.063, 2.1 floors); G1 by month 4/5 -> 2/5; G9 tercile cell PASS -> FAIL; G9 MAE better (-2.7 floors). **Not a ship candidate alone**: it exposes an efficiency over-spread (Decision 11 read, PM rules). |
| Section 38 offline gate, K arms | **FAILS by the letter**, and the line is mis-specified (confounded). The post-hoc confound-free line favours K2 / K2M. |
| Section 38/39 closed loop, K2M, full size | lane B's lines **PASS** (N on OREB +0.103 -> -0.018; pace x makes gap 0.078 -> 0.001; pace x OREB gap -0.040 -> -0.006); G5 total SD ratio FAIL -> **PASS**; G5 corr 0.126 -> 0.178; G1 count better (-4.3 floors); OT better. Carries M2D's G9 total slope cost (-2.7 floors) and the tercile flip. |
| K2 alone, full size | **PENDING** (`d1001_H_3`). Local 100-seed direction: G5 total ratio +0.05, no G9 slope cost. |
| End-of-half possession count (lane L's line) | no arm fixes or worsens it (all within about 5% of served; H1 10-20 s 2.1x actual in every mode). Owner: the `srfloor` cell coding (no horn-aware law in H1) |

## 8. Not run, caveats, incidents

**Not run:**
- K2 alone at full size (`d1001_H_3`, PENDING).
- M2D floor draw (`d1001_H_1` tier 2).
- A d1-conditioned KD table.
- K arms at fold 1 in the sim (no fold-1 engine inputs exist).
- First-half share (no per-period sim score).
- The efficiency-over-spread owner test.

**Artifacts:**
- Tracked: M2D, K1 and K2M F2 schedules and latents.
- **K2 (49 MB per fold) is gitignored and not on HF** (bulk key `model_artifacts`; pushing it alone would push 2,382 other files). It can be rebuilt with `scripts/train_clock_r8_chance_v1.py --arm K2 --fold F2` in 75 s.

**Incidents:**
- My first M2D tap shell was killed at its 30-minute background limit. The python run continued and finished; no other lane was touched.
- A `sed -i` converted `loop.py` to LF in the working tree. It was restored to CRLF before commit, so the diff is only my 40 lines.
- A first fold-1 training pass hit a 0/0 in the Poisson Hessian. It was fixed, and every arm was refitted with the fixed code.
- The core cap of 3 was held throughout (taps at 2 + 1 workers).
