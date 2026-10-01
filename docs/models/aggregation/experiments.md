# Cascade aggregation: experiments (append-only)

**Owner:** Lane A (overnight 2026-09-30).

**What this is.** The cascade does not have an "aggregation model": the sim margin is whatever the possession loop produces when possession_outcome, fg_make, rebound, free_throw, clock, usage and rotation are run together on one game's inputs. This folder holds the diagnostics and fix rounds for how those sub-models COMBINE team effects into a margin, as distinct from any one sub-model's own calibration.

- **Origin:** the 2026-09-30 box read (`HANDOFF.md`, "SUMMARY FOR USER, 2026-09-30", AWS operator row and open item 1). S0 (served stack, v3 inputs, verified truth, 5,710 x 200) has G9 slope 0.917. S1 (possession_outcome, fg_make and rebound retrained on E3 v4 team-rate features; every one better calibrated and more responsive offline) has 0.894, worse by 15 floor-SD. The PM's working statement: the served models' compressed team levels were compensating an over-spread in how the cascade combines team effects.
- **Status:** NOTHING in this file adopts anything or changes a served default. No engine module is edited by this round; every counterfactual is an in-memory swap of input arrays in the runner process.

---

## 1. Pre-registration: closed decomposition of the sim margin over-spread by channel (G9 slope)

Written 2026-09-30 20:55-21:20 EDT. COMMITTED BEFORE ANY ARM OTHER THAN THE TIMING SLICE RAN. The timing slice (S0 `FULL` and `PO`, 571 games x 16 seeds, 20:47-21:00 EDT) was used ONLY to measure throughput, bit-identical reproduction of the box rows and the paired Monte Carlo noise of a swap delta; no k or slope from it is read or reported as a result. A code smoke of the Part C harness on S0 `FULL` only (21:01 EDT, 34 s) reproduces the sim's 200-seed mean margin at correlation 0.989, SD 9.64 vs 9.61, slope of X on X_h 0.985; no swap arm was evaluated before this commit.

### 1.1 Stacks (both reproduce the box byte for byte; checked, not assumed)

| stack | inputs | artifacts | reproduction check |
|---|---|---|---|
| S0 | `build_engine_inputs_v3_tag_v1.py --tag S0_laneA` (defaults = `engine_v3`, served artifacts) | served | runner `FULL` rows on 571 games x seeds 0-15 equal `results/engine_v0/v3full_S0_s200_o0_off0_n25` on all 26 columns (9,136 rows, 0 differences) |
| S1 | `--tag S1_laneA --team-rate-table team_rate_features_E3_v4 --variance-table team_rate_variance_O1a_v3 --po-artifacts possession_outcome/round_stageb/T/team_rate_features_E3_v4 --fg-artifacts fg_make/round_stageb/T/team_rate_features_E3_v4/B1 --rb-artifacts rebound/round_stageb/T/team_rate_features_E3_v4/artifacts/s2_F2_A0B0C0_seed0` (the box's `box_builds_v1.sh S1` command; artifacts pulled from HF `model_artifacts`) | Stage B `T` | runner `FULL` on the run sample vs `v3full_S1_s200_o0_off0_n25`; if it does not reproduce, S1 is reported NOT RUN and S0 alone is analysed |

Engine flags: the served defaults (`ENGINE_EVENT=round2_s1`, `ENGINE_CLOCK=v5b_glat_pmean`, `ENGINE_FG_MAKE=round4_B1`, `ENGINE_REBOUND=s1_weekly`, `ENGINE_FREE_THROW=s1_conf_aligned`, rotation `s1`). No draw (`ENGINE_TEAM_RATE_DRAW` off). Truth: `CBB_TRUTH=verified_v1` set explicitly. Close: `data/processed/lines/lines_close_v2_verified.parquet`, ESPN BET, home margin = -spread. 2025-26 is not touched.

### 1.2 Candidate channels (from the brief) and how each is measured

| # | channel (brief) | measured by | kept / dropped |
|---|---|---|---|
| 1 | offence and defence team effects applied on both sides of the same possession | partition L2: swap every feature describing the possession's OFFENCE team (`OFF`) vs its DEFENCE team (`DEF`); the off x def cascade interaction `I2` is the combination term | kept |
| 2 | correlated estimation error of one team's features across sub-models | partition L1 interaction `I1` (all one-at-a-time sub-model swaps vs all-at-once) plus Part A's off-diagonal cross-channel covariance terms | kept |
| 3 | tree non-additivity in matchup cells | `I2` (off x def interaction inside the trees and the cascade) and `I1`; segment table by team-strength quintile gap | kept |
| 4 | usage / rotation concentrating shots on the better players | `PLY`: fg_make shooter skill deviations (`shooter_shrunk_dev_c__{rim,jump2,three}`) to 0, so who shoots no longer changes make probability. Rotation/usage arrays are NOT swapped (a flat rotation is not a "league-average team", it changes possession volume per player and the clock state); free_throw shooter inputs are not swapped (no team features, within-season k 0.004 in the F diagnostic) | kept (reduced form stated) |
| 5 | ratings entering several sub-models alongside rates (double counting) | `RAT` (the four own-rating columns to 0 everywhere: PO, fg_make, rebound, clock) and `RAT_PO` (possession_outcome's copy only); `RAT - RAT_PO` is the fg_make + rebound + clock share | kept |
| 6 | possession count x efficiency coupling | `PACE` (tempo features to neutral: `off/def_tempo_rel` 1.0, `tempo_prior_game` the builder's league fallback 68.70) plus Part A's `pace` and `possdiff` identity channels | kept |
| - | per sub-model team rates | `PO` (8 style columns), `FG` (6 make/allow columns), `RB` (2 columns) | kept |

"Swap to league average" = the centred `_c` feature set to 0 (its own snapshot's league mean), for BOTH teams of every game, in the arrays the engine reads (`team_static`, `slot_static`, and the event adapter's round-2 team block). Everything else (site, date, state, rotation, usage) is untouched, so the all-swapped arm `ALL` is the cascade's site-and-state baseline.

### 1.3 Arms (per stack; one run each; RNG streams paired across arms)

`FULL` (taken from the box rows where reproduction holds), `PO`, `FG`, `RB`, `RAT`, `RAT_PO`, `PACE`, `PLY`, `TEAM` (= PO+FG+RB+RAT), `OFF`, `DEF`, `ALL` (= TEAM+PACE+PLY). Runner `scripts/exp_aggregation_swap_v1.py`.

### 1.4 Decomposition (exact by construction; the residual is reported, never dropped)

Per game g, X_arm(g) = mean over seeds of (home - away) points. D_c = X_FULL - X_c.

- **L1 (sub-model families):** X_FULL = X_ALL + D_PO + D_FG + D_RB + D_RAT + D_PACE + D_PLY + I1.
- **L2 (offence vs defence team):** X_FULL = X_TEAM + D_OFF + D_DEF + I2.
- **L3 (ratings by consumer):** D_RAT = D_RAT_PO + (D_RAT - D_RAT_PO).

For a reference R (realised verified margin Y = the G9 line; or the close C = the owner lens of the F diagnostic), with Z the components of one partition:

- **Slope identity:** 1 - slope(R on X_FULL) = sum_j (1 - beta_j) cov(Z_j, X)/var(X) - cov(e, X)/var(X), where beta is the regression of R on all Z_j jointly (with a constant) and e its residual.
- **Monte Carlo handling:** beta is estimated by just-identified IV with the even-seed components as regressors and the odd-seed components as instruments (and the reverse; the two are averaged), so MC noise does not attenuate beta. The term k_res = -cov(e, X)/var(X) is then the MC share plus anything the partition does not name; its MC expectation is computed from the split-half noise variance and reported beside it.
- **"Closes":** the decomposition is CLOSED when |k_res - k_MC expected| <= 2 floors. Otherwise it is reported as NOT CLOSED with the unexplained amount.
- Per arm also reported: SD(X_arm), slope(Y on X_arm), slope(C on X_arm), margin MAE.

**Part A (offline identity, no new sim):** on the existing full 200-seed reads of S0 and S1 (`v3full_S0_*`, `v3full_S1_*`, seeds 0-199 where on disk), the F diagnostic's exact channel identity (TOV, FT trips, shot mix x3, makes x3, OREB, rebound chances, FT%, pace, possession parity; `scripts/diag_g9_g6_margin_v1.py` functions) is applied and each k_c = cov(X_c - Y_c, X)/var(X) is split into its OWN term cov(X_c - Y_c, X_c)/var(X) and its CROSS terms sum_{d != c} cov(X_c - Y_c, X_d)/var(X). OWN > 0 = the channel is over-spread against its own realisation ("a sub-model is wrong"); CROSS > 0 with OWN <= 0 = the channel moves with the others more than reality does ("each right, the product over-spread"). Exact; residual 0 by algebra (final-vs-box carried as its own row).

**Part C (deterministic expected-points harness; the precise channel algebra):** the pilot showed that a swap delta is too noisy in the closed loop to carry per-channel regressions alone (per-seed paired correlation full-vs-`PO` 0.515; reliability of D_PO across games 0.27 at 16 seeds, about 0.53 at 48). So the same arms are ALSO evaluated in an offline harness with no Monte Carlo: `scripts/diag_aggregation_harness_v1.py` loads each stack's artifacts through the engine's own adapter classes (as `diag_g9_within_season_v1.py` does) and predicts, per (game, offence side) and arm, P(TOV), P(FT trip), shot shares (possession_outcome, first chance, two reference states), P(make) per class (fg_make, weighted over slots by rotation share x usage), P(OREB | miss) (rebound, fixed miss mix) and FT% (free_throw). Expected points per possession: PPP = [p_fga * sum_k s_k v_k p_k + 2 f p_trip] / [1 - p_oreb * p_fga * (1 - sum_k s_k p_k)] (one chance law reused on putbacks); harness margin X_h = P_ref (PPP_home - PPP_away), P_ref the actual league mean team possessions. The L1 / L2 / L3 partitions are taken on X_h exactly (no MC); the sim remainder X_FULL - X_h is carried as its own component `not_harnessed` (clock, state path, cascade details, MC), so X_FULL closes exactly. PACE is not in the harness (no clock); it is measured only in the closed loop and in Part A. Per rate the harness also reports its offline calibration slope against the realised team-game rate (a sub-model "right" on its own) beside its contribution to the margin's over-spread.

**Expected-value non-linearity:** per game and stack, f(E rates) from the sim's seed-summed rates is compared with the seed-mean margin; the gap is the in-sim non-linearity of aggregation (reported, not owned unless > 2 floors).

### 1.5 Sample, seeds, floors

- **Part A and Part C:** all graded fold-2 games of the verified universe (5,710 rows; unplayed rows dropped by the grader's own frame); Part A on the box's 200-seed full reads of S0 and S1 (pulled from HF `results/engine_v0/v3full_S{0,1}_s200_o0`).
- **Closed loop, decisive (box, requested through `docs/ops/box_queue/`):** all 5,710 games x 48 seeds (0-47) per arm per stack; `FULL` from the box's own 200-seed reads (seeds 0-47). Tier 1 arms (both stacks): TEAM, OFF, DEF, ALL, PO, FG, RB, RAT; tier 2: RAT_PO, PACE, PLY.
- **Closed loop, local (only if cores are free after Parts A and C; LOCAL, UNDERPOWERED by construction):** every 5th game (1,142) x 12 seeds, tier-1 arms only. No owner call is made from it alone.
- **Floors (Decision 12):** closed loop: floor = max(SD of the statistic over four disjoint seed quarters / 2, paired game-bootstrap SE (300 reps, games resampled with all their arms and seeds kept together)). Part C has no seed noise: its floor is the game-bootstrap SE plus, for the S1 - S0 difference, the |difference| between the Stage B `R` (served features, retrain seed 0) and `R2` (seed 1) artifact sets evaluated through the same harness when time allows (else stated as missing). Part A: game bootstrap and the 200 seeds split into four draws of 50.

### 1.6 Primary, co-primary, segments

- **Primary:** k_j against Y (the G9 line), per component, per stack, and the S1 - S0 difference Delta k_j on the same games (and seeds), from Part C (deterministic) for the sub-model / offence-defence / interaction components, and from Part A for own vs cross channel terms. The closed loop (box) is the confirmation: per arm SD(X_arm) and MC-corrected slope(Y on X_arm), and the L1 / L2 k's where its IV first stage is strong (split-half reliability of the component >= 0.5); a channel named owner by Part C must not have the opposite sign in the closed loop.
- **Co-primary:** the same against the close (lined subset).
- **Segments (multi-level):** overall; by month (Nov-Mar); by team-strength quintile gap (each team's as-of own-rating net, quintiles over team-games, gap = |q_home - q_away|, 0-4); home/away vs neutral; per channel. Per-team: SD of per-team mean X by arm vs realised. Cells under 300 games are labelled UNDERPOWERED.

### 1.7 What counts as an owner (decision rule; nothing is adopted either way)

- **Owner of the S0 miss:** a component with k_j(Y) >= 0.02 (about a quarter of S0's 0.083), > 2 floors, and the same sign in the close lens.
- **Owner of the S1 deterioration:** a component with Delta k_j(Y) >= 0.01 (about 40% of the 0.023), > 2 floors, same sign in the close lens.
- **"Combination" verdict:** the PM's working statement is CONFIRMED if the interaction terms (I1, I2) or Part A's cross terms carry >= 50% of the S1 - S0 deterioration; REFUTED if the per-sub-model terms (D_PO, D_FG, D_RB) or Part A's own terms carry >= 50% and the interaction terms are inside 2 floors; otherwise MIXED.
- If an owner emerges with time left, the fix round is pre-registered in section 2 of this file as a SUB-MODEL change (never a sim-output shrink, which is the banned pattern) and is not run unless its measured timing fits.

### 1.8 Addendum A (written 21:20 EDT, BEFORE it ran): sub-model replacement factorial and per-sub-model retrain-seed floors

Reason for the addendum: the first Part C read showed that the Stage B `R` artifacts (served features, retrain seed 0) reproduce the served harness margin exactly, while `R2` (seed 1) moves the harness slope by -0.014; so the S1 - S0 comparison needs a retrain-seed floor per sub-model, and the "which sub-model's retrain" question is answered most directly by replacing sub-models one at a time rather than by league-average swaps alone.

- **Factorial (harness, `FULL` arm only, deterministic):** the eight stacks with each of possession_outcome (P), fg_make (F) and rebound (R) either served or Stage B `T` on E3 v4 features: none (= S0), P, F, R, PF, PR, FR, PFR (= S1). Built with `build_engine_inputs_v3_tag_v1.py --tag X_<letters>_laneA` (`--team-rate-table E3_v4 --no-table-for <served ones>` plus the `T` artifact dir of each replaced model).
- **Seed floors (harness):** S0 with ONE sub-model replaced by its `R2` (seed-1) retrain: `Z_P`, `Z_F`, `Z_R`.
- **Read:** the effect of each sub-model's T on 1 - slope(Y on X_h) and SD(X_h), as main effects (average over the other two's states) and interactions; each main effect is compared with |that sub-model's Z effect| (the spec-identical retrain-under-another-seed floor) and the game bootstrap.
- **Rule:** a sub-model owns the S1 deterioration if its main effect is >= 40% of the S1 - S0 harness change, > 2 x max(its seed floor, bootstrap SE), and the same sign in the close lens. Interactions >= 25% of the change make the "combination" verdict at least MIXED.
- **Closed loop:** box request laneA_2 asks for the sim-level retrain floor (`R2` stack `FULL`, 5,710 x 200); the factorial is not run in the closed loop tonight.

### 1.9 Addendum B (written 21:35 EDT, BEFORE it ran): is fg_make T's game-level over-response same-season memorisation through the monthly refits?

Reason: Part C and addendum A (results doc sections 3-4) name fg_make's Stage B `T` retrain as the owner of the S1 deterioration (factorial main effect +0.035 of +0.043 on harness 1 - slope, 4.0 x its retrain-seed floor), and S1's team-game make-rate calibration slopes are BETTER than S0's in November and worse from December on (rim 0.87 vs 0.84 in Nov; 0.62-0.71 vs 0.76-0.90 in Jan-Mar). Hypothesis H_mem: E3 features are smooth within a team-season, so the in-season monthly refits (which train on the same season's earlier games) can fit each team-season's realised shooting at near-constant feature values, and apply it to later games as if it were skill.

- **Test (harness, deterministic):** fg_make scored by its FIRST refit (trained only on seasons before 2024-11-01) for every game (`diag_aggregation_harness_v1.py --fg-first-refit`), all other sub-models as served in each stack; stacks S0, S1 and X_F (fg_make T only). FULL arm.
- **Read:** Delta(1 - slope(Y on X_h)) S1 - S0 and X_F - S0 under first-refit fg_make, vs the same deltas under the served schedule (+0.043 and +0.035); the team-game make-rate calibration slopes by month.
- **Rule:** H_mem SUPPORTED if the first-refit X_F - S0 delta is <= 50% of the served-schedule X_F - S0 delta (+0.035) and the S1 make slopes from January on recover to within 0.05 of S0's; REFUTED if the first-refit delta is >= 80% of it; otherwise PARTIAL. Confound stated in advance: the first refit also lacks the in-season drift information for both stacks; the read is the S1 - S0 (paired) difference, not either level.

### 1.10 Addendum C (written 21:32 EDT, BEFORE it ran): which side of fg_make's team features

Addendum B REFUTED same-season memorisation (results doc section 5). Before a fix round is written, the harness splits fg_make's team-feature response by side: arms `FG_OFF` (offence `off_make_c__*` to 0) and `FG_DEF` (defence `def_allow_c__*` to 0), stacks S0 and S1, plus the same two arms on X_F. Read: k (Y and close lens, joint regression with FULL minus the arm and the remainder) and Delta S1 - S0; descriptive only (no owner rule beyond section 1.7's thresholds).
