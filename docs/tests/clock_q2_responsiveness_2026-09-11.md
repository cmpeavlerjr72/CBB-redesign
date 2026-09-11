# The clock latent's tempo-quintile-2 responsiveness band -- diagnosis and round 5c

2026-09-11, clock lane. Evidence for `docs/models/clock/experiments.md`
sections 23 (diagnosis), 24 (round-5c pre-registration) and 25 (partial
results). Scripts: `scripts/diag_clk5c_q2.py`, `scripts/diag_clk5c_targets.py`
(both fit nothing and select nothing) and
`scripts/exp_clk5c_dispersion_function.py`. Artifacts:
`data/processed/models/clock/v5c_diag/`, `data/processed/models/clock/v5c_bakeoff/`.

**Nothing in this document changes a served default, softens a gate or waives a
criterion.** `ENGINE_CLOCK` is untouched; B1 `v5b_glat_pmean` remains served
provisionally exactly as commit `e3ccce5` left it.

## 1. The question

Rounds 5 and 5b (sections 16-22) found that the clock's conditional law is right
and its joint law was missing, and closed the gap with a per-game pace latent:
the offline per-game possession-SD ratio moved 0.688 -> 1.009 and, at 25 paired
seeds in the engine, 0.804 -> 1.032. Two pre-registered lines still failed for
B1: the G5 margin SD ratio (1.2 measured floors, out of scope here) and
**criterion 4, the responsiveness band -- `no tempo quintile outside
[0.85, 1.15]` -- against a Q2 reading of 1.174.** Every arm that landed the
primary sat at 1.157-1.178 at Q2, including B3, whose `sigma^2` is linear in
pregame tempo.

Section 17.4 asserted that "Q2's needed dispersion is a genuine 4-SE dip below
the tempo trend, not noise". **The band was pre-registered without a noise
term** while every other line in rounds 5 and 5b is read against a measured
floor. This document measures the noise.

## 2. Method

The grade path's own per-game identity, unchanged:

    Var_arm(Dbar) over a set of games = mean_g[eH_g/M_g^2] + mean_g[vF_g/M_g^2]
                                        WITHIN (iid)         BETWEEN (latent)
    needed                            = Var_g(rbar)
                                      = mean_g(V_iid_g) + implied tau^2
    ratio = sqrt(produced / needed)     -- the bridge constant 3.8509 cancels

Every per-quintile number is therefore an average and a variance over that
quintile's games, and the noise is a **game-block bootstrap over those games**
(2,000 resamples, seed 20260911) -- the same resampling rule the offline floors
of rounds 5 and 5b use. Round 5b's fitted parameters are re-read from disk; no
arm is refitted for the diagnosis.

## 3. Overall

| arm | produced P SD | needed | ratio | bootstrap SE |
|---|---:|---:|---:|---:|
| R `v3c_srfloor_P3_s1` | 2.9430 | 4.2795 | 0.6877 | 0.0109 |
| A1 `v5_glat_shared` | 4.2980 | 4.2795 | 1.0043 | 0.0160 |
| B1 `v5b_glat_pmean` (served) | 4.3194 | 4.2795 | 1.0093 | 0.0160 |
| B3 `v5b_glat_pmean_tempo` | 4.3157 | 4.2795 | 1.0085 | 0.0160 |

Reproduces section 22.2 to four decimals: the diagnosis is reading the same
objects the bake-off read.

## 4. Per game -- the quintile table with its noise

398-399 games and 50,840-57,542 possessions per quintile; all five powered.

| arm | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| R | 0.711 +/- 0.027 | 0.793 +/- 0.029 | 0.732 +/- 0.026 | 0.684 +/- 0.021 | 0.631 +/- 0.021 |
| A1 | 1.035 +/- 0.040 | 1.168 +/- 0.042 | 1.067 +/- 0.038 | 0.997 +/- 0.031 | 0.919 +/- 0.031 |
| B1 | 1.040 +/- 0.040 | **1.174 +/- 0.042** | 1.072 +/- 0.038 | 1.002 +/- 0.031 | 0.924 +/- 0.031 |
| B3 | 1.004 +/- 0.038 | 1.158 +/- 0.042 | 1.075 +/- 0.038 | 1.017 +/- 0.031 | 0.954 +/- 0.032 |

95% bootstrap interval at Q2: B1 **[1.099, 1.263]**, A1 [1.094, 1.257], B3
[1.084, 1.246]. **The band edge 1.15 is inside all three.** Exceedance in units
of the statistic's own SE: **B1 0.56, A1 0.42, B3 0.19.**

## 5. Per component -- within-game vs between-game (B1)

| quintile | needed | iid floor | needed `tau^2` | produced within | produced between | within ratio | between ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 1.2630 | 0.6383 | 0.6248 +/- 0.0950 | 0.6425 | 0.7242 | 1.0067 | 1.159 |
| Q2 | 0.9750 | 0.6137 | **0.3613 +/- 0.0700** | 0.6178 | 0.7254 | 1.0067 | **2.008** |
| Q3 | 1.0805 | 0.5793 | 0.5011 +/- 0.0758 | 0.5831 | 0.6593 | 1.0065 | 1.316 |
| Q4 | 1.1952 | 0.5591 | 0.6360 +/- 0.0726 | 0.5627 | 0.6374 | 1.0064 | 1.002 |
| Q5 | 1.3320 | 0.5297 | 0.8024 +/- 0.0881 | 0.5330 | 0.6047 | 1.0063 | 0.754 |

- **Not a within-game issue anywhere**: the produced iid contribution matches
  the needed iid floor to 0.63-0.67% in every quintile. Section 17.1's channel
  (a) result holds per quintile.
- **The tempo SLOPE of the between component is real**: between ratio 2.008 at
  Q2 against 0.754 at Q5, a difference of 1.25 +/- 0.40 (**3.1 SE**); needed
  `tau^2` Q5 - Q2 = 0.441 +/- 0.112 (**3.9 SE**). A constant-CV latent is nearly
  flat (0.725 -> 0.605) against a need that more than doubles.
- **The Q2-specific dip is 1.4-2.2 SE**: `tau^2` Q2 - Q1 = -0.264 +/- 0.118
  (2.2 SE), Q2 - Q3 = -0.140 +/- 0.103 (1.4 SE). The "4-SE dip" of section 17.4
  is not reproduced.

## 6. Per season -- model-free, and the dip does not replicate

Raw per-team possession-count SD by pregame-tempo quintile, no model, no
residualisation, quintiles cut within each season:

| season | Q1 | Q2 | Q3 | Q4 | Q5 | minimum |
|---|---:|---:|---:|---:|---:|---|
| 2022 | 4.316 | 4.108 | 4.259 | 4.450 | 5.117 | Q2 |
| 2023 | 3.586 | 4.163 | 4.482 | 4.140 | 4.443 | Q1 |
| 2024 | 4.118 | 4.401 | 4.222 | 4.420 | 4.686 | Q1 |
| 2025 (F2 test) | 3.784 | 3.813 | 4.044 | 4.327 | 5.086 | Q1 |

The rising slope replicates in all four seasons; **the Q2 dip does not -- its
location moves season to season, and in the test season the raw Q1 and Q2 SDs
differ by 0.03 possessions.** The dip appears in the NEEDED column only after
the model's conditional mean is removed.

## 7. Per bucketing -- the grid matters as much as the model

- **Deciles (B1):** 1.031, **1.183**, 1.159, **1.205**, 1.074, 1.082, 1.071,
  0.994, 0.965, 0.928. The excess region is D2-D4 and straddles the Q1/Q2 and
  Q2/Q3 boundaries; Q1's passing 1.040 is a passing D1 averaged with a failing
  D2.
- **By the model's own predicted mean duration `mbar` (B1):** 0.906, 1.017,
  1.007, 1.064, 1.070 -- **every bucket inside the band**, same arm, same games,
  same statistic.
- **By each team's as-of possession-count MEAN** (886 games with both teams
  having >= 5 prior games): needed `tau^2` 0.710, 0.565, **0.355**, 0.443, 0.751
  -- a U-shape whose minimum sits in bucket 3, where `tempo_prior_game` on the
  same 886 games puts it in bucket 2 (0.572, **0.265**, 0.371, 0.571, 0.837).
  Underpowered relative to the full slate (45% of it, mid- and late-season
  only) and labelled as such.

## 8. Per team -- is the latent a team property?

Per-game latent estimate `s2_hat = (rc^2 - V_iid)/(3 V_iid + mbar^2)` (round
5b's estimator, unchanged), averaged per offence team, against a label
permutation null (400 permutations, seed 20260911), teams with >= 10 games:

| quantity | value |
|---|---:|
| observed between-team variance | 4.256e-06 |
| permutation null | 2.715e-06 +/- 2.4e-07 |
| **z** | **6.50** (p < 0.0025, 0 of 400) |

**A team-level component exists.** Its size, estimated on the F2 training window
with an empirical-Bayes decomposition (section 25.1): between-team
`tau_b^2 = 2.951e-07` against a within-team estimator variance of 3.626e-05, so
`k = 123` games for half weight and a typical team is shrunk three quarters of
the way back to the global. Caveat, stated: teams differ in tempo, so part of
the permutation signal is the tempo slope of section 5 and not an independent
team effect.

A team's own **as-of possession-count SD** does NOT predict the latent:
`corr = +0.0147` (n = 886, SE about 0.034); needed `tau^2` by as-of-SD quintile
0.504, 0.705, 0.533, 0.601, 0.630 with SEs 0.100-0.163 -- no slope beyond noise.

## 9. Round 5c, partial

Pre-registered in section 24 and committed at `98f8db5` before any arm was
fitted. Arms: B1 (reference), C1 per-team hierarchical, C2 as-of SD (failure
predicted in advance), C4 `sigma^2` quadratic in pregame pace. C3 not fitted.
**CRPS_trunc (N1), the PIT table (N3), M1 and the closed loop did not run, so
NOTHING IS ADOPTED and no default changed.**

| arm | primary ratio | Q1 | Q2 | Q3 | Q4 | Q5 | mean gate gap | between-ratio range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 1.0093 | 1.040 | **1.174** | 1.072 | 1.002 | 0.924 | -0.1251 | 1.25 |
| C1 | 1.0074 | 1.037 | **1.169** | 1.069 | 1.002 | 0.926 | -0.1260 | 1.22 |
| C2 | 0.9845 | 1.022 | 1.148 | 1.046 | 0.976 | 0.893 | -0.1310 | 1.19 |
| **C4** | **1.0100** | 1.061 | **1.139** | 1.038 | 0.990 | 0.979 | -0.1260 | **0.87** |

- **C4 is the first arm whose every quintile is inside the band**, holding the
  primary and the mean gate at B1's values for one extra coefficient. Its fitted
  minimum sits at tempo **67.79** (F2) and **67.32** (F1) -- stable across folds
  -- and it closes 30% of the between-component's tempo spread, raising the fast
  end rather than lowering everything. Its Q2 pass is a 0.26-SE margin:
  **a pass inside noise is no better measured than a failure inside noise.**
- **C2's pre-registered failure is confirmed**: its fitted slope is negative on
  both folds, its feature covers 32% of training games, and it passes the band
  by shrinking the dispersion level 8% -- which costs it the primary
  (|ratio-1| 0.0155 against B1's 0.0093).
- **C1 confirms section 8 both ways**: the team component is real (z = 6.50) and
  too small after honest shrinkage to move a quintile ratio by more than 0.006.

## 10. What this hands the PM

1. **Criterion 4's Q2 failure, which blocks B1, is a 0.56-SE exceedance of a
   band pre-registered without a noise term, on a dip that does not replicate in
   three other seasons, is not the worst bucket at decile resolution, moves one
   bucket when the pace feature changes, and disappears entirely under an
   `mbar` bucketing.** The band stands verbatim and B1 stands as failing it;
   what is new is that the failure cannot be distinguished from sampling error.
   This lane does not decide whether that counts as a failure.
2. **The real, replicating defect is the tempo slope of the dispersion
   function** (3.1-3.9 SE), and C4 closes 30% of it offline at no cost to the
   primary or the mean.
3. **The open line that is NOT noise is criterion 5**: B1's G5 margin SD ratio
   regression, -0.0141 against a measured 0.0115 floor. Nothing in this document
   touches it.
4. Before C4 could be put forward it needs `CRPS_trunc`, the PIT cell table, and
   the 25-seed paired closed loop against B1 on M1, the margin SD ratio,
   `corr(P, eFG%)` and the total bias -- round 5b section 22.6 is the warning it
   must clear.
