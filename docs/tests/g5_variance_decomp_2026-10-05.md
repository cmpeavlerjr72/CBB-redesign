# G5 total-variance decomposition on served stack v2: possessions, shared PPP, team PPP, pace x efficiency (2026-10-05)

Variance worker, 2026-10-05 afternoon. **DIAGNOSIS ONLY.** Nothing was trained or fitted, no default changed, no engine source touched, and no new sim was run. Every number below comes from existing 200-seed runs. 2025-26 stays sealed. The pre-registration draft for the fixes is a separate file, `docs/tests/g5_variance_prereg_draft_2026-10-05.md`, and is NOT in any `experiments.md`.

**Runs read** (all verified truth; 5,700 / 5,632 graded games after dropping 0-0 finals and games without a box):

| tag | run | what it is |
|---|---|---|
| F2 served v2 | `results/engine_v0/v3full_COMB9GCTKD_s200_o0` | fold 2 (2024-25), the adoption read |
| F1 served v2 | `results/engine_v0/f1c_V2_full_s200_o0` | fold 1 (2023-24), V2 confirmation run (static rotation) |
| F2 K2 | `results/engine_v0/laneH_v3full_K2_s200_o0` | served v2 + `ENGINE_CLOCK=v5b_r8K2_glat_pmean` (clock round 8, outcome-conditioned possession time), box `d1001_H_3`, same seeds 0-199 |
| F2 S0, F2 v2-no-G3 | `v3full_S0_s200_o0`, `v3full_COMB9CTKD_s200_o0` | references, used only to date the FT-make coupling |

**Scripts** (committed with this doc; outputs under `results/g5_vdecomp/`, which is gitignored):
- `scripts/diag_g5_variance_decomp_v1.py`: the five-component split, segments, per-team shares, per-rate shared residuals, and the persistent-team estimate.
- `scripts/diag_g5_variance_parts_v1.py`: the channel split of the shared and pace x efficiency terms, a regulation-only view with the OT mixture, and team-block bootstraps.
- `scripts/diag_g5_ftp_coupling_v1.py`: the FT-make coupling probe.

## 0. Verdict

| question | answer |
|---|---|
| Which component is short? | Of the 42.4 pts^2 total-variance gap on F2 (47.0 on F1): **pace x efficiency covariance owns +24.2 (57%)** (F1 +27.7, 59%), **shared PPP owns +22.5 (53%)** (F1 +20.9), possessions own +6.7 (16%), and **team-specific PPP is OVER-dispersed by -10.3 (-24%)**. The sim's per-team PPP variance is right in total (171.7 vs 172.6 pts^2); what is wrong is the split between shared and idiosyncratic, because the cross-team covariance is 11.4 against 17.0 |
| Owner of the pace x efficiency term | **Clock: possession time is drawn before the outcome.** The whole gap sits in the FG-value channel: sim -37.4 vs real +4.2 pts^2. The already-built, already-box-read **K2** arm closes it exactly: -45.6 becomes -23.4 against real -22.1. On K2 the G5 ratio goes 0.925 -> 0.973 (PASS), the corr 0.126 -> 0.170, and G9 is unchanged. K2's status is still TESTING in the ledger: nobody has ruled on that read |
| Owner of the shared-PPP shortfall | **The free-throw block, not FG shooting.** Whistle volume (FTA x FTA and its cross-terms) is short by +15.4 pts^2 (F1 +9.8). An engine-structural **negative cross-team FT-make x FG-value coupling** (sim -13.6 vs real +1.6; present in S0 too) costs +12.9 (F1 +16.4). The FG blocks are slightly OVER-shared (-6.3) |
| Owner of the possessions term | **Not the variance function.** OT rate (G7, 0.031 vs 0.056) is worth about +13.5 pts^2 of total variance. Realised possession residuals carry a persistent team pace error, 2 x sigma2_team = 7.5 pts^2 (11.4 in regulation games). Net of both, regulation possession variance is OVER-dispersed in the sim: 89.2 vs about 72-74 |
| Realised game-level efficiency latent, beyond the matchup | Cov(PPP_h, PPP_a) residual = **17.0 pts^2** (corr 0.197; F1 18.1 / 0.209). That is SD **4.0 pts per team per game (0.059 PPP)**, and 68 pts^2 (23%) of Var(total). Month-centring removes 1%, and persistent team effects at most about 0.6. By rate it is the whistle: FTA/poss corr 0.241 (sim 0.136), FTr 0.202 (0.108), TOV% 0.065 (0.085), eFG 0.036 (0.026), 3P% 0.009 (0.010, none), FT% 0.022 (-0.049) |
| Honest target caveat | 5.7% of realised Var(total residual) (F2; 4.2% on F1) is a **persistent team error**: 16.6 +/- 2.8 pts^2, the same team's games missing in the same direction. It belongs to the as-of mean model, which a within-game variance function cannot and should not carry. An honest within-game sim reads about 0.97 on the G5 ratio, not 1.00. PM to note; no gate change proposed |

**Compensation warning (Decision 11):** on K2, regulation-only total variance matches reality (259.1 vs 262.2). It matches only because regulation possession variance is over (91.8 vs 85.4 raw, about +18 net of the persistent team pace error) while shared PPP is short (41.9 vs 64.5). The candidates in the draft are therefore a set.

---

## 1. Method

All moments are population moments. Per game, `m` is the sim's own 200-seed mean (as-of, no same-game information). The sim deviation is `w = row - m` and the realised residual is `r = actual - m`. Both sides use the same box possession estimator `N = mean over sides of (FGA - OREB + TOV + 0.44 FTA)`, and `e_i = pts_i / N`.

```
dT = P + Q_h + Q_a + rem,  P = (m_e_h + m_e_a) dN,  Q_i = m_N de_i,  rem = nonlinearity
Var(dT) = Var(P)                       (a) possessions
        + 4 Cov(Q_h, Q_a)              (b) shared PPP (game-level efficiency)
        + Var(Q_h)+Var(Q_a)-2Cov       (c) team-specific PPP
        + 2 Cov(P, Q_h+Q_a)            (d) pace x efficiency covariance
        + Var(rem)+2Cov(rem, P+Q)      (e) nonlinearity
```

Each term closes to 1e-12. The channel split goes one level down: `Q_i = m_N (vol*val + fta_pp*ftp)` with `vol = FGA/N` (PO mix, OREB extra shots), `val = FG pts/FGA` (fg_make + shot mix), `fta_pp = FTA/N` (foul / FT trips) and `ftp = FT%` (free_throw), plus an exact remainder `x`.

Realised SEs come from 100-200 game bootstraps. **Persistent team share:** `sigma2_team = mean over same-team game pairs of x_i x_j`, with `x` month-centred; a game carries `2 sigma2_team`. Its SE comes from a 300-draw team-block bootstrap. The sim floor is negligible at 200 seeds x 5,700 games: the 09-30 / 10-01 floor draws put the G5 lines' SD at about 0.005 or less.

## 2. The decomposition (total-variance units, pts^2)

| component | F2 realised (SE) | F2 served v2 | gap | F1 realised (SE) | F1 served v2 | gap | F2 K2 sim (realised) | gap |
|---|---|---|---|---|---|---|---|---|
| Var(total) | 292.8 (6.1) | 250.5 | **+42.4** | 298.2 (6.2) | 251.2 | **+47.0** | 277.7 (293.4) | +15.7 |
| (a) possessions | 109.0 (2.8) | 102.2 | +6.7 | 105.6 (2.5) | 99.9 | +5.7 | 106.9 (110.5) | +3.5 |
| (b) shared PPP 4Cov | 68.1 (5.4) | 45.6 | **+22.5** | 72.2 (4.8) | 51.3 | **+20.9** | 44.3 (67.8) | **+23.5** |
| (c) team-specific PPP | 138.6 (2.7) | 148.9 | **-10.3** | 136.5 (3.0) | 146.3 | -9.8 | 150.5 (138.5) | -12.1 |
| (d) pace x efficiency | -21.4 (3.3) | -45.6 | **+24.2** | -18.0 (3.9) | -45.7 | **+27.7** | -23.4 (-22.1) | **+1.3** |
| (e) nonlinearity | -1.4 (0.9) | -0.7 | -0.7 | +1.9 (0.9) | -0.6 | +2.5 | -0.6 (-1.2) | -0.6 |
| Cov(Q_h,Q_a) / corr | 17.0 / 0.197 | 11.4 / 0.133 | | 18.1 / 0.209 | 12.8 / 0.149 | | 11.1 / 0.128 | |
| gate total SD ratio / corr | | 0.925 / 0.126 (real 0.228) | | | 0.918 / 0.147 (real 0.252) | | 0.973 / 0.170 | |

Realised columns differ slightly by arm because `m` is each arm's own mean. Var(Q_h)+Var(Q_a) on F2 is 172.6 realised and 171.7 sim (K2 172.7). The per-team PPP variance is right; (b) and (c) differ only in how that variance is split between shared and idiosyncratic.

## 3. Channel owners

### 3.1 Pace x efficiency, 2 Cov(P, Q) by channel

| channel | F2 real | F2 served v2 | F2 K2 | F2 S0 | F1 real | F1 served v2 |
|---|---|---|---|---|---|---|
| vol (FGA/N) | -53.3 | -22.3 | -58.8 | -26.3 | -57.1 | -21.5 |
| **val (FG pts/FGA)** | **+4.2** | **-37.4** | **+6.2** | -27.9 | **+9.4** | **-38.7** |
| fta_pp | +23.6 | +15.1 | +29.6 | +22.0 | +28.7 | +15.2 |
| ftp | +3.3 | -1.3 | -1.0 | -1.3 | +1.0 | -1.0 |
| total | -21.4 | -45.6 | -23.4 | -33.2 | -18.0 | -45.7 |

On served v2, a game with more makes has fewer possessions. Reality shows no such link. The served clock draws possession time before the cascade, and the made-FG -> inbound start is slower than the miss -> DREB start. This is the mechanism lane B (10-01, `g5_variance_channels_2026-10-01.md`, pace x makes +42 pts^2) and clock round 8 section 3 found. Here it is confirmed on fold 1: the same -38.7 vs +9.4.

K2 conditions the first-chance time on the chance-1 end class. It moves every channel to reality: val +6.2, vol -58.8, fta +29.6. **The owner is the clock (round 8 K2).** Note that S0 was less wrong (-27.9); the adopted v2 set deepened the coupling (lane B: G3 adds about 10%).

### 3.2 Shared PPP, cross-team blocks (each entry in Var(total) units; the blocks sum to 4Cov exactly)

| block | F2 real | F2 served v2 | F2 K2 | F2 S0 | F2 v2 no G3 | F1 real | F1 served v2 |
|---|---|---|---|---|---|---|---|
| vol x vol | +3.1 | +1.1 | +2.5 | -4.7 | +0.8 | +4.8 | +0.6 |
| vol x val | +28.1 | +39.7 | +37.5 | +41.7 | +43.7 | +31.0 | +40.3 |
| vol x fta | -11.8 | -3.9 | -7.2 | +11.7 | -3.9 | -13.3 | -2.0 |
| vol x ftp | -0.9 | +2.9 | +2.9 | +3.2 | +2.9 | +0.5 | +2.9 |
| val x val (G3's channel) | +12.3 | +9.0 | +8.1 | -10.0 | -10.0 | +14.4 | +14.7 |
| val x fta | +9.9 | -0.2 | +2.7 | +5.4 | +1.2 | +8.7 | -1.4 |
| **val x ftp** | **+1.6** | **-13.6** | **-14.0** | **-13.4** | **-13.7** | **+4.9** | **-12.8** |
| **fta x fta (whistle)** | **+22.6** | **+13.0** | **+14.2** | +1.5 | +12.9 | **+20.7** | **+11.4** |
| fta x ftp | +2.6 | -1.0 | -1.0 | -1.1 | -1.0 | +0.8 | -0.9 |
| ftp x ftp | +0.5 | -1.0 | -1.1 | -1.0 | -1.0 | +0.2 | -0.9 |
| **total 4Cov(Q_h,Q_a)** | 68.1 | 45.6 | 44.3 | 33.0 | 31.5 | 72.2 | 51.3 |

The gap groups by owner as follows (F2 served; F1 in parentheses):

- **Whistle (foul_joint R9ao3 / FT-trip production): +15.4 (+9.8).** This is fta x fta, vol x fta, val x fta and fta x ftp together: real 23.3 vs sim 7.9 (F1 16.9 vs 7.1). R9ao3 already lifted fta x fta from 1.5 (S0) to 13.0, about 58% of the real 22.6.
- **FT-make coupling (free_throw / late-game FT shooter): +12.9 (+16.4).** This is val x ftp, ftp x ftp and vol x ftp: real +1.2 vs sim -11.7 (F1 +5.6 vs -10.8). It is present unchanged in S0, v2 without G3, v2 and K2, so it is **engine-structural and not caused by any v2 flag**.
- **FG blocks: -6.3 (-5.4).** vol x val, val x val and vol x vol are slightly over-shared. G3's val x val is about right on F1 and short by 3 on F2.

**The FT-make coupling, probed** (`diag_g5_ftp_coupling_v1.py`; within-game sim vs realised residual):

| line | F2 real | F2 sim | F1 real | F1 sim |
|---|---|---|---|---|
| corr(FT%_home, FG value_away) | -0.006 | **-0.082** | +0.037 | **-0.076** |
| corr(FT%_away, FG value_home) | +0.029 | **-0.080** | +0.021 | **-0.075** |
| corr(FT%_own, FG value_own) | +0.027 | +0.077 | +0.040 | +0.068 |
| corr(FT%_own, own final margin) | 0.160 | **0.260** | 0.149 | **0.250** |
| corr(FTA_away, away margin) | 0.058 | 0.125 | 0.059 | 0.131 |
| FT% deviation, home trailing at the horn | -0.010 | **-0.027** | -0.001 | **-0.026** |

In the sim, a team's FT% is too tied to the scoreboard. When it trails, its FT% sits 2.7 points below its own mean, against about 1 point or less in reality. Its FT% therefore runs against the opponent's shooting.

Two candidates fit this pattern, and both sit in the free_throw / late-game FT path:
- **(i) Who is fouled in the late-game window.** Reality fouls the leader's worst FT shooter; the sim's pick weakens that and makes trailing-team trips from weak shooters too common.
- **(ii) The served FT model's `score_diff` / pressure term,** read in engine game states it was not trained on.

The game-level box data cannot tell (i) from (ii); a per-trip tap is step 1 of the draft's C2.

### 3.3 Possessions and OT

| | F2 real | F2 served v2 | F2 K2 | F1 real | F1 served v2 |
|---|---|---|---|---|---|
| OT rate | 0.0558 | 0.0305 | 0.0351 | 0.0600 | 0.0309 |
| OT between-group variance of dT | 22.4 | 13.5 | 15.8 | 19.8 | 13.5 |
| regulation-only Var(total) | 261.5 | 234.6 | **259.1** (real 262.2) | 271.7 | 235.2 |
| regulation-only Var(P) | 83.8 | 89.2 | 91.8 (real 85.4) | 83.7 | 87.0 |
| regulation-only shared 4Cov | 64.7 | 43.6 | 41.9 (real 64.5) | 68.3 | 49.3 |
| regulation-only pace x eff | -24.5 | -47.5 | -25.6 (real -25.2) | -19.6 | -47.6 |
| persistent team share of realised Var(P), all / regulation | 7.5 / 11.4 | | | 5.6 / 9.3 | |

- **OT counterfactual (arithmetic).** At the real OT rate with its own conditional moments, served v2 Var(total) is about 264 (+13.5) and K2 about 289, a ratio of about 0.99. **G7 owns the rest of K2's total gap.**
- **Regulation possessions.** Realised regulation Var(P) net of the persistent team pace error is about 72-74. The sim gives 87-92, so within-game possessions are over-dispersed by about 15-25% in variance.
- **Who probably owns that over-dispersion.** The clock's per-game latent sigma (0.0472, fitted by method of moments on training-season residuals, clock experiments section 18) would absorb persistent team pace error if the fit does not net it out. The sim's team pace responsiveness is 0.67-0.77 of actual (round 8), so the latent may be standing in for missing between-game spread. **Hypothesis, not tested here.**

## 4. Segments (F2 served v2; F1 in the results JSON, same pattern)

| segment | n | SD ratio | var_T R/S | (a) R/S | (b) R/S | (c) R/S | (d) R/S |
|---|---|---|---|---|---|---|---|
| site: home/away | 4964 | 0.925 | 293/250 | 109/102 | 65/46 | 141/149 | -22/-46 |
| site: neutral | 736 | 0.924 | 294/252 | 111/104 | **88/45** | 122/149 | -20/-45 |
| as-of pace tercile 1 (slow) | 1900 | 0.936 | 272/238 | 99/94 | 74/43 | 124/144 | -26/-42 |
| as-of pace tercile 2 | 1900 | 0.930 | 288/249 | 103/102 | 59/45 | 141/149 | -14/-46 |
| as-of pace tercile 3 (fast) | 1900 | 0.911 | 318/264 | 124/111 | 70/49 | 151/154 | -23/-49 |
| as-of total quintile 1 | 1141 | 0.942 | 263/234 | 90/89 | 64/44 | 136/142 | -26/-41 |
| as-of total quintile 5 | 1140 | 0.899 | 334/270 | 130/117 | 71/50 | 148/156 | -12/-51 |
| as-of mismatch (abs margin) quintile 1 (close) | 1140 | 0.910 | 309/256 | 127/107 | 74/48 | 137/146 | -29/-44 |
| as-of mismatch quintile 5 (lopsided) | 1139 | 0.926 | 279/239 | 103/94 | 42/38 | 153/156 | -20/-48 |
| team prior (both-team avg quintile <= 1) | 1307 | 0.932 | 289/251 | 117/102 | 75/51 | 131/143 | -29/-44 |
| team prior (1.5-2.5) | 2831 | 0.918 | 294/248 | 108/101 | 62/44 | 143/151 | -18/-46 |
| team prior (>= 3) | 1562 | 0.932 | 293/255 | 105/105 | 72/45 | 137/151 | -22/-45 |
| November | 1219 | 0.880 | 319/247 | 117/98 | 42/42 | **168/153** | -8/-45 |
| December | 915 | 0.982 | 259/250 | 112/101 | 69/43 | 135/153 | -55/-47 |
| January | 1420 | 0.919 | 298/251 | 110/104 | 73/47 | 135/147 | -18/-47 |
| February | 1364 | 0.944 | 283/253 | 101/103 | 77/49 | 126/146 | -18/-44 |
| March | 765 | 0.937 | 287/252 | 105/105 | 78/47 | 121/146 | -18/-45 |

**Reading.**
- **Pace x efficiency and shared PPP: uniform.** Both gaps appear in every site, pace, total, prior and month cell. The sim's (d) is flat at -41 to -51 in every cell; it is structural, not matchup-dependent.
- **Site.** Shared PPP is larger at neutral sites in reality: 88 vs 65 on F2 and 81 vs 71 on F1. Segment SE is about 15, so F2 is about 2.8 SE. Same direction on both folds, but underpowered.
- **Pace.** The ratio is worst in the fast tercile and the top total quintile, on both folds.
- **Close vs lopsided.** The shared shortfall is concentrated in close games: (b) 74/48 in the closest quintile vs 42/38 in the most lopsided. That fits whistle and late-game FT channels.
- **November.** The ratio of 0.880 comes from (c): realised idiosyncratic variance 168 vs 153. That is thin as-of mean error early in the season (the known early-season total bias), not a variance-function defect.
- **Per team** (about 31 games each, underpowered one by one; read the shares only). Served v2 sim Var(total) < realised for 55% of teams on F2 and 59% on F1, and the shared term is short for 62% / 60% of teams. On K2 the first share falls to 42% (median team SD ratio 1.02), while the shared share stays at 62%.
- **Per possession type.** No per-possession tap was run today. The K2 mechanism at possession level is in clock round 8 section 3: duration by end class, and Cov(duration, points | start type) -1.36 real.

## 5. What this adds to earlier work, and caveats

- **Relation to lane B (10-01, `g5_variance_channels_2026-10-01.md`).** That doc owned the gap to pace x makes (+42) partly offset by pace x OREB (-23). This decomposition agrees on the total and adds:
  - the shared vs team-specific partition (team PPP variance right, cross-team covariance short);
  - the FT-block ownership of the shared shortfall, including the structural negative FT-make coupling not reported before;
  - fold-1 replication;
  - the OT and persistent-team pieces;
  - K2's per-component effect.
- **The realised residual includes all as-of mean error.** The persistent part is estimated at 16.6 +/- 2.8 pts^2 (F2) and 12.3 +/- 2.7 (F1); the non-persistent part, such as game-specific injuries, cannot be separated with these data. Shared PPP is robust to it: at most about 0.6 of 17.0 is persistent.
- **The fold-1 run used static rotation** (stated deviation in `retrain_x_adopted_and_fold1_2026-10-05.md`). No K2 fold-1 run exists; the K2 fold-1 artifacts exist under `data/processed/models/clock/r8_K2/F1/` and need the fold-1 overlay.
- **The box possession estimator is the same on both sides.** It differs from the engine count by +0.64 per game in the sim, and it keeps the identities like for like.
