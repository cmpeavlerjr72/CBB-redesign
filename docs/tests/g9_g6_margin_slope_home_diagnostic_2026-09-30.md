# G9 calibration slope and G6 neutral home margin: decomposition by sub-model (Lane F, 2026-09-30)

DIAGNOSTIC ONLY. Nothing is fixed, adopted or re-defaulted here. No new simulation was run.

**Inputs**

- Sim output: the served v5b stack's full 200-seed paired read, `results/engine_v0/F2_2025_s200_v5b_{A,B}_full`, graded in `docs/tests/engine_v1_gates_F2_2025_s200_v5b_full_2026-09-18.md`.
- Fold: fold 2 (2024-25), 5,710 graded games.
- Truth:
  - The grader's own `load_actual_games`, for verified finals.
  - hoopR team box, for the channel identity.
  - The CBBD event layer, used only to split each team's 2PT attempts and makes into rim and jump (221 of 11,400 team-games lack the split and use the league share).
  - The ESPN BET close from `data/processed/lines/lines_close_v1.parquet` (5,375 of 5,700 games).
- 2025-26 was not touched.

**Script:** `scripts/diag_g9_g6_margin_v1.py`, with parts `build`, `analyse`, `extra`, `close` and `boot`. Outputs are in `results/g9g6_diag/*_v1.{json,log}`.

**Cost:** one core, under one minute in total.

## 0. What the two gate lines measure

**G9 calibration slope** (`gates.headline`) is `np.polyfit(sim_margin_mean, margin, 1)[0]`. It regresses the realised final margin (Y) on the sim's per-game mean margin (X) across games, 0.9096 (B stream 0.9117).

- A slope below 1 means that one point of predicted margin difference between games buys only 0.91 points of realised difference.
- So the sim's mean margins are over-spread relative to the information they carry: favourites are over-predicted and underdogs under-predicted. A game the sim puts 1 SD (9.54 pts) above the mean realises about 0.87 pts less than predicted.
- About 0.008 of the 0.090 miss is Monte Carlo attenuation at 200 seeds (MC SD of X is 0.87). The grader's `slope_mc_corrected` of 0.917 still fails.
- Over-spread relative to information is not the same as too-large SD. SD(X) is 9.54 and the close's SD on the same games is 9.44. The close has slope 1.016 on Y; the sim has 0.915 on the lined subset.
- In the joint regression `Y ~ X + close` the coefficients are X -0.003 and close 1.019.

The sim's spread is the right size but part of it carries no signal. That part is identified in section 1.

**G6 "home margin (neutral)"** (`gates.gate_g6`) is the mean of `margin` (listed home minus listed away) over the 736 games flagged `neutral == 1`: sim +2.105 vs actual +3.288, with the actual's SE 0.46.

- It is not a home-court advantage. It is the listed-home team's margin at neutral sites, and the listed-home team is usually the favourite (the sim favours it in 63% of these games).
- The non-neutral line passes (+5.692 vs +5.738), but section 3 shows that this is a cancellation.

## 1. G9: where the slope miss lives

### 1.1 Exact channel identity vs realised (closes to 1e-13)

Per team-side, with P = FGA - OREB + TOV + 0.44 FTA:

PPP = (1 + OREB/P - TOV/P - 0.44 FTA/P) * (points per FGA) + (FTA/P)*FT%

The difference between the two sides is split with Bennet midpoint weights. This is exact for these bilinear forms:

margin = P_ref*sum(channel dPPP) + (Pbar - P_ref)*dPPP [pace] + PPPbar*dP [possession parity], with P_ref = 67.875.

The same formula is applied to every sim seed-row, where the maximum error is 4e-14, and to the actual box. The last channel, "final vs box", is the verified final minus the box identity; it contributes -0.0001.

Contribution of channel c to the miss: k_c = cov(X_c - Y_c, X) / var(X). These sum to 1 - slope exactly.

- Points are k times SD(X), which is 9.54: the overstatement for a game predicted 1 SD above the mean.
- SEs come from a 300-rep game bootstrap.
- The B column is the paired-seed noise floor.

| sub-model channel | k (A) | k (B) | SE | pts at +1 SD |
|---|---:|---:|---:|---:|
| possession_outcome (TOV, FT-trip rate, shot mix) | -0.0127 | -0.0134 | 0.0098 | -0.12 |
| fg_make (rim, jump, 3 makes) | +0.0073 | +0.0066 | 0.0190 | +0.07 (inside noise) |
| rebound (OREB%) | +0.0263 | +0.0256 | 0.0058 | +0.25 |
| rebound chances (derived from misses, no model) | +0.0054 | +0.0056 | 0.0035 | +0.05 |
| free_throw (FT%) | +0.0299 | +0.0301 | 0.0040 | +0.29 |
| clock: pace x efficiency (+0.0205) and possession parity (+0.0144) | +0.0349 | +0.0345 | 0.0036 | +0.33 |
| final vs box | -0.0001 | -0.0001 | - | 0.00 |
| **sum = 1 - slope** | **0.0910** | **0.0888** | 0.018 | **0.87** |

Per-channel game-level slopes of Y_c on X_c:

| channel | slope |
|---|---:|
| TOV | 0.96 |
| OREB% | 0.86 |
| FT trips | 0.94 |
| FT% | 0.65 |
| rim make | 0.85 |
| jump make | 0.75 |
| 3 make | 0.79 |
| pace | 0.72 |
| possession parity | 0.06 |

**This lens mixes two things.** When each channel is split into a season-average team part (team fixed effects plus site terms) and a within-season part (the rest), the result is the table below.

| forecaster | team part k | within-season part k | total |
|---|---:|---:|---:|
| sim vs realised | -0.089 | +0.174 | 0.085 |
| **close vs realised** | -0.088 | +0.072 | -0.016 |

- **An artifact shared with the market.** Every as-of forecaster that updates on same-season results shows this pattern. It comes from the in-sample fixed effect absorbing early-season residuals. The sim's team part is identical to the close's (-0.089 vs -0.088).
- **The real sim defect.** What the sim adds is +0.10 of excess within-season variation.
- **Why a second lens is needed.** Channel attributions in this table, such as FT% +0.030 and clock +0.035, are exact about the realised outcomes. They cannot be taken as owners without subtracting what the close also carries. Hence section 1.2.

### 1.2 Close-referenced attribution (the owner lens)

1 - slope(close on X) = 0.098 (SE 0.006) on 5,375 lined games, where slope(close on X) is 0.902 (B: 0.903). It splits as follows:

- **Season-average team component:** k = 0.007 (SE 0.003; B 0.005).
  - slope(close_team on sim_team) = 0.992 and corr = 0.968.
  - The sim's team strength levels are at the market's scale. They are neither over- nor under-dispersed.
- **Within-season component:** k = 0.091 (SE 0.005; B 0.092).
  - SD of the sim's within-season movement is 3.56 vs the close's 2.28. slope(close_within on sim_within) = 0.33, and corr = 0.52.
  - **Drift** (team x month): SD 2.52 vs 1.83, slope 0.45, k = 0.038.
  - **Game-specific jitter:** SD 2.51 vs 1.35, slope 0.21, k = 0.054. This includes the approximately 0.008 of MC noise.

Split by channel: the close's within component is regressed on the sim's 13 channel within components, and k_c = (1 - beta_c) cov(X_c,w, X)/var(X). This closes exactly.

| owner (within-season) | beta range (close on sim) | k (A) | k (B) | SE | pts at +1 SD (SD(X) 9.62) |
|---|---|---:|---:|---:|---:|
| fg_make (rim 0.46, jump 0.31, 3 0.27) | 0.27-0.46 | 0.048 | 0.050 | 0.004 | 0.46 |
| possession_outcome (TOV 0.28, FT trips 0.48, mix 0.20-0.28) | 0.20-0.48 | 0.020 | 0.021 | 0.002 | 0.20 |
| rebound OREB% | 0.20 | 0.019 | 0.018 | 0.002 | 0.18 |
| free_throw FT% | 0.47 | 0.004 | 0.004 | 0.001 | 0.04 |
| clock (pace 0.49, parity) | 0.49 | 0.003 | 0.003 | 0.001 | 0.03 |
| derived rebound chances | 0.58 | -0.003 | -0.004 | 0.002 | -0.03 |
| **within total** | | **0.092** | **0.092** | 0.005 | **0.88** |
| team component (not channel-identifiable: the A/B channel betas flip, collinear) | | 0.007 | 0.005 | 0.003 | 0.07 |
| **total** | | **0.098** | **0.097** | | **0.95** |

- **Unexplained:** zero arithmetically. Both lenses close exactly, with the close residual orthogonal by construction.
- **Not identified: the mechanism.**
  - Every channel's within-season movement is priced by the market at only 20-50%, in proportion to that channel's variance.
  - That pattern points to a shared input, not to one model's coefficients.
  - Candidates are:
    - the as-of team-rate features, which are raw, not opponent-adjusted, and pooled over home and away;
    - player, rotation and usage inputs;
    - tree matchup interactions;
    - MC noise (about 0.008).
  - This diagnostic cannot separate them without instrumentation. See pre-registration P1.

### 1.3 Team-level responsiveness per sub-model vs realised (offence / defence)

Team fixed effects are fitted on the sim's expected per-game rates and on realised rates, weighted by denominators, with site terms.

- **Realised true-SD:** from split halves, cov(odd games, even games).
- **Points:** margin points for a team 1 SD out on the sim's scale, and the realised gap, which is that times (slope - 1).
- **Caveat:** these are vs realised and carry the same shared as-of artifact. At margin level the close shows a team slope of 1.09 too, so slopes near 1.1 are not a sim defect.

| sub-model | rate | side | SD ratio pred/true | team slope | corr | pts per 1 SD team | realised gap pts |
|---|---|---|---:|---:|---:|---:|---:|
| possession_outcome | TOV% | off | 0.74 | 1.25 | 0.83 | 0.93 | +0.23 |
| possession_outcome | TOV% | def | 0.64 | 1.51 | 0.89 | 0.95 | +0.48 |
| possession_outcome | FTA/poss | off | 0.51 | 1.76 | 0.77 | 0.30 | +0.23 |
| possession_outcome | FTA/poss | def | 0.55 | 1.53 | 0.74 | 0.34 | +0.18 |
| possession_outcome | 3PA share | off | 0.69 | 1.39 | 0.93 | 0.50 | +0.20 |
| possession_outcome | 3PA share | def | 0.56 | 1.57 | 0.83 | 0.28 | +0.16 |
| possession_outcome | rim share | off | 0.63 | 1.51 | 0.90 | 0.64 | +0.33 |
| possession_outcome | rim share | def | 0.62 | 1.53 | 0.88 | 0.55 | +0.29 |
| fg_make | rim FG% | off | 0.78 | 1.23 | 0.88 | 1.49 | +0.34 |
| fg_make | rim FG% | def | 0.62 | 1.57 | 0.86 | 1.00 | +0.57 |
| fg_make | jump FG% | off | 0.51 | 1.96 | 0.70 | 0.38 | +0.36 |
| fg_make | jump FG% | def | 0.66 | 1.41 | 0.70 | 0.49 | +0.20 |
| fg_make | 3P% | off | 0.69 | 1.35 | 0.67 | 0.93 | +0.32 |
| fg_make | 3P% | def | 0.70 | 1.08 | 0.53 | 0.83 | +0.07 |
| rebound | OREB% | off | 0.70 | 1.34 | 0.89 | 1.08 | +0.37 |
| rebound | OREB% | def | 0.78 | 1.11 | 0.78 | 0.77 | +0.08 |
| free_throw | FT% | off | 0.82 | 1.14 | 0.77 | 0.50 | +0.07 |
| free_throw | FT% | def (allowed) | 1.34 | 0.36 | 0.16 | 0.19 | -0.12 (**underpowered**: true SD 0.007) |
| all | PPP | off | 0.84 | 1.19 | 0.95 | 4.64 | +0.89 |
| all | PPP | def | 0.79 | 1.24 | 0.93 | 3.86 | +0.94 |

- **Responsiveness holds.** Team correlations are 0.53-0.95. Nothing is flat at the mean.
- **Team effects: no model is over-dispersed.** At team level, every scoring-stage model is under-dispersed vs realised, offence and defence alike (ratios 0.51-0.84, slopes 1.08-1.96). The one exception is FT% allowed, which is noise.
- **Most of that compression is shared with the market.** Against the close, the team component is at scale (section 1.2).
- **What is over-dispersed is the within-season movement,** in every model.

**Reconciliation with PO G2 (refused 2026-09-18).** G2 shrank possession-outcome team estimates toward a prior-season prior. That compresses the season-level team component, which is already at market scale here (0.99) and is at or below realised scale. So G2 flattened responsiveness and between-game spread, as the closed loop found.

The G9 miss lives in the other component: game-to-game and month-to-month movement of the predictions, 1.6-1.9x the market's SD. The two results agree. The remedy must damp within-season movement and leave the level alone, so it is a reliability or smoothing arm and not a level shrink.

### 1.4 Segments (n games; slope Y on X; slope close on X; bias sim-actual; sim-close bias)

Slope SE per month is about 0.045. B-stream month slopes are within 0.01.

| segment | n | slope Y~X | slope close~X | slope Y~close | bias | sim-close |
|---|---:|---:|---:|---:|---:|---:|
| Nov | 1219 | 0.862 | 0.887 | 0.985 | -1.07 | -0.70 |
| Dec | 915 | 0.917 | 0.935 | 0.983 | -0.17 | -0.02 |
| Jan | 1420 | 0.897 | 0.868 | 1.069 | +0.92 | +0.47 |
| Feb | 1364 | 0.889 | 0.845 | 1.041 | -0.17 | +0.15 |
| Mar | 765 | 0.920 | 0.866 | 1.051 | -0.93 | -0.15 |
| both power conf | 1030 | 0.907 | 0.847 | 1.079 | +0.18 | +0.39 |
| one power conf | 642 | 0.792 | 0.837 | 0.981 | **-3.31** | **-2.10** |
| neither power | 4028 | 0.834 | 0.845 | 0.978 | +0.22 | +0.21 |
| home/away | 4964 | 0.922 | 0.909 | 1.024 | -0.04 | +0.03 |
| neutral | 736 | 0.807 | 0.825 | 0.954 | -1.18 | -0.45 |
| conference game | 3606 | 0.879 | 0.856 | 1.040 | +0.19 | +0.25 |
| non-conference | 2094 | 0.898 | 0.909 | 0.992 | -0.84 | -0.47 |
| abs(X) <3 | 1468 | 0.633 | 0.810 | - | -0.87 | -0.81 |
| abs(X) 3-6 | 1274 | 0.794 | 0.865 | - | -1.04 | -0.39 |
| abs(X) 6-10 | 1202 | 0.956 | 0.908 | - | +0.29 | +0.06 |
| abs(X) 10-15 | 898 | 0.912 | 0.869 | - | +0.78 | +0.63 |
| abs(X) 15+ | 858 | 0.909 | 0.915 | - | +0.57 | +0.98 |

**By month: not an early-season effect.**

- The slope miss is present in every month (0.86-0.92). slope(close on X) is 0.85-0.94, and the close's own slope is 0.98-1.07 in every month.
- The close-referenced within-season k by month is 0.116, 0.085, 0.065, 0.105 and 0.093 (Nov-Mar).
- November does carry the largest within-season movement (sim SD 5.4 vs close 2.9), which is the zero-start as-of noise. But slope is a ratio, and the excess ratio persists all season.

**Spread bands:** these are range-restricted and cannot be read as calibration within a band. The sim-vs-close gap is spread across bands.

**The one-power-conference bias is a separate G9-by-segment miss.** In those games the sim under-predicts the power team by 2.1 pts vs the close. It is a level compression across conferences, not a slope effect, and it feeds G6 (section 2).

## 2. G6: neutral-site margin

**Decomposition of the neutral miss.** Lined subset, n = 734: sim +2.08, close +2.53, actual +3.27.

| piece | sim | close | actual | sim - close |
|---|---:|---:|---:|---:|
| team-strength part (fixed effect theta_h - theta_a), mean over neutral games | 1.96 | 2.14 | 2.63 | -0.19 |
| "listed home at neutral" coefficient | 0.13 | 0.39 | 0.63 (SE 0.41) | -0.26 |
| **total** | 2.08 | 2.53 | 3.27 | **-0.45** |

- **Shared with the close:** -0.74 of the -1.18 miss. The close also misses by -0.74, which is 1.6 SE of the actual mean. It is **underpowered and indistinguishable from sampling noise.**
- **Sim-specific, -0.45 in two parts:**
  - (a) No model can produce a listed-home edge at neutral sites (-0.26 vs the market's pricing). Every site feature is (0,0) at neutral.
  - (b) Team-level compression of listed-home favourites (-0.19). This is the same cross-conference level compression as the one-power bias.
- **Venue:** 633 "elsewhere" games carry the residual (+1.48 +/- 0.44). The 102 games in the listed-home team's own state do not (-0.58 +/- 1.09, underpowered). There is no quasi-home-arena story; only one game was in the listed team's own arena.
- **Season type:** regular season +1.25 +/- 0.44, postseason +0.76 +/- 1.13 (underpowered).

## 3. G6: home advantage by sub-model channel (team-strength-adjusted, non-neutral)

Each margin channel, for sim and for actual, is fitted with the same model: team fixed effects plus a non-neutral home term plus a neutral listed-home term, on the 5,700 games. The columns are points per game. The actual's SE comes from the fixed-effect OLS.

| channel | owner | sim HCA | actual HCA (SE) | sim - actual | sim neutral coef | actual neutral coef (SE) |
|---|---|---:|---:|---:|---:|---:|
| turnovers | possession_outcome | 0.50 | 0.73 (0.07) | **-0.22** | -0.06 | -0.32 (0.17) |
| FT trips | PO foul-trip class + foul accrual (served scalar has no site) | 0.38 | 0.47 (0.04) | **-0.09** | 0.05 | 0.17 (0.09) |
| FT% | free_throw (no site feature) | 0.15 | 0.12 (0.05) | +0.03 | 0.04 | 0.03 (0.11) |
| OREB% | rebound | 0.35 | 0.31 (0.06) | +0.04 | -0.01 | 0.33 (0.15) |
| rebound chances | derived | -0.23 | -0.11 (0.04) | -0.12 | -0.06 | -0.13 (0.09) |
| shot mix (rim/jump/3) | possession_outcome | 0.14 | 0.23 | -0.09 | 0.02 | -0.18 |
| rim make | fg_make | 0.95 | 0.69 (0.10) | **+0.25** | 0.00 | 0.33 (0.25) |
| jump make | fg_make | 0.46 | 0.26 (0.08) | **+0.20** | 0.07 | 0.29 (0.21) |
| 3 make | fg_make | 0.71 | 0.48 (0.15) | +0.23 | 0.10 | 0.27 (0.37) |
| pace | clock (no site) | 0.07 | 0.02 (0.02) | +0.05 | -0.01 | -0.05 (0.04) |
| possession parity | clock / late game | -0.07 | -0.14 (0.03) | +0.07 | -0.01 | -0.07 (0.07) |
| **total** | | **3.41** | **3.06 (0.16)** (close 3.07) | **+0.35** | 0.13 | 0.66 (0.41) |

- **fg_make site terms over-produce home advantage:** +0.68 pts in total.
- **possession_outcome under-produces it:** TOV -0.22 at 3.4 SE, FT trips -0.09 at 2.7 SE, mix -0.09.
- **Rebound is correct** (+0.04). Clock and FT% are small.
- **The sim's home advantage is 0.35 too large vs both actual and close.**
- **The non-neutral G6 pass is a cancellation.** The home team's strength edge is under-predicted: the sim's team part in home/away games is 2.40 vs the close's 2.73, -0.34. That offsets the +0.35-0.37 excess in home advantage. This is the multi-level failure CLAUDE.md warns about.
- **Per-channel neutral coefficients are all underpowered** (SE 0.09-0.37).

**Feature audit (home/away/neutral, served stack):**

| sub-model | site feature | at neutral | notes |
|---|---|---|---|
| event `round2_s1` (possession_outcome) | `site_home`, `site_away` one-hots, offence view only | both 0 | no defence-side site term |
| fg_make `round4_B1` + FGA_3 `decision8` | same | both 0 | |
| rebound `s1_weekly` | same | both 0 | |
| free_throw `s1_conf_aligned` | **none** | - | **CLAUDE.md rule gap** |
| clock `v5b_glat_pmean` | carried in TEAM_COLS, **not consumed** (cell grid has no site) | - | **rule gap** |
| foul accrual | served scalar has **no site**; the site-indexed table (neutral/home/away) is DEFAULT-OFF | - | **rule gap** |
| usage U1 | none | - | |
| as-of team rates (PO builder) | pooled over home and away games, no site adjustment | - | own-ratings ridge is site-adjusted |

Evidence: `possession_outcome.py:436-439`, `build_engine_inputs.py:173,534-537`, `loop.py:115-118,354,356`, `free_throw.py:418-424`, `clock.py:782-785`.

- **Neutral handling:** no model has a "listed home at neutral" or proximity term.
- **Not opened:** the rebound and fg_make as-of rate builders were not opened to check for site pooling.

## 4. Owners

**G9 slope 0.910:**

- **The shared as-of artifact.** Against realised outcomes, the season-level team component behaves exactly as the market's does (-0.089 vs -0.088). It is not a defect.
- **Owner: the within-season movement of the sim's predictions** (month drift 0.038 plus game-specific jitter 0.054, including about 0.008 of MC noise).
  - It is over-varied about 3x relative to what the market prices, in every channel.
  - By channel: fg_make carries 0.46 pts, possession_outcome 0.20, rebound 0.18, free_throw 0.04 and clock 0.03, at +1 SD.
  - The common pattern points to a shared input (as-of team or player features), not to any one model's coefficients. That mechanism is **unexplained**.

**G6 neutral:**

- **Shared noise:** -0.74 of the miss is shared with the close (underpowered).
- **Missing neutral-site edge:** -0.26, with no owner. There is no feature for it in any model.
- **Team-level compression across conferences:** -0.19. Owner: as-of zero start / drift anchor (Lane C territory).

**G6 non-neutral (hidden):**

- The home-advantage excess of +0.35 is owned by the fg_make site terms (+0.68), partly offset by possession_outcome TOV and FT trips (-0.31). The possession_outcome part overlaps the foul-accrual lane.
- It is masked by the -0.34 team-level compression.

## 5. Recommended pre-registrations

**P1 (G9 slope): localize, then damp, within-season movement.** This is a cross-model round on shared inputs.

- **Step 1 (instrumentation, paired seeds, fold 2):**
  - D0: served.
  - D1: team as-of rate inputs frozen at the team's month-start value (removes game-level jitter from team rates).
  - D2: player, rotation and usage inputs replaced by season-to-date typical values (removes roster jitter).
  - D3: seeds raised to 800 on a game subset (MC share).
- **Step 2 (candidates, on whichever input step 1 owns):**
  - C0: served.
  - C1: sample-size reliability weighting of the as-of deviation from the team's own running level (the level is kept; this is not G2).
  - C2: opponent-adjusted as-of rates (the pending Decision 9 arm).
  - C3: C1 + C2.
- **Primary metric:** G9 slope (Y on X) on fold 2, paired against D0/C0. Its noise floor is the A/B seed pair (0.002).
- **Co-primary:** slope(close on X) on the lined subset. Floor 0.0015; the game bootstrap SE of 0.006 is for reference only.
- **Guards:**
  - The close-referenced team-component slope stays within 0.97-1.03.
  - Team-quintile responsiveness does not flatten.
  - Margin MAE does not regress.
  - G5 margin SD ratio stays in the band.
- **Decision rule:** win beyond the floor with no guard broken; ties go to the simpler arm.

**P2 (G6): site effect by channel.**

- **Candidates:**
  - S0: served.
  - S1: fg_make site terms refit with opponent-adjusted as-of rates (site confounded with pooled home/away rates).
  - S2: a site term added to the foul/FT-trip path (the site-indexed foul table, jointly with Lane A).
  - S3: a "neutral listed-home" flag (plus a same-state flag as a second arm) in possession_outcome and fg_make.
- **Primary metric:** the fixed-effect-adjusted non-neutral home advantage, total and per channel, vs actual (target 3.06 +/- 0.16; close 3.07). The per-channel table in section 3 is the multi-level evidence.
- **Secondary:**
  - The neutral coefficient. It is **underpowered** on one season (SE 0.41), so read it pooled with fold 1.
  - The G6 raw lines.
  - The one-power-conference bias vs the close. That one belongs to the drift-anchor round, not to P2.
