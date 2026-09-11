# Clock round 5 -- within-game possession-duration DISPERSION (2026-09-11)

Lane: clock. **Nothing here changes a served default.** The engine default
stays `ENGINE_CLOCK=v3c_srfloor_P3_s1`; an offline winner is a CANDIDATE and
`CLAUDE.md` requires a paired-seed sim run showing no gate regressed before
anything ships. No closed-loop run was possible in this session (section 9),
so **no arm is adopted**.

Pre-registration: `docs/models/clock/experiments.md` section 16, committed at
`28b5f17` BEFORE `src/cbb_sim/models/clock_v5.py` or any fit existed.
Results and verdict: the same file, section 17.

Inputs:

    scripts/diag_clk5_dispersion.py              the measurement (fits nothing)
    scripts/exp_clk5_dispersion_bakeoff.py       the fit + blind grade, one path
    scripts/diag_clk5_mc_check.py                the pre-registered MC cross-check
    src/cbb_sim/models/clock_v5.py               the arms

    data/processed/models/clock/v5_diag/          measurement tables + report
    data/processed/models/clock/v5_bakeoff/       bake-off tables + report

    docs/tests/engine_v1_variance_ot_diag_2026-09-11.md   the finding this answers

Universe: the 2025 (F2 test) clock-complete regulation possessions --
**270,530 rows over 1,991 games** -- the same set round 4 section 15.2 read
`E[min(T,R)]` on. Served arm's own law, routed by S1 refit date. Population
variance (ddof = 0) throughout. Cells under 300 possessions are labelled
UNDERPOWERED and are not read.

---

## 0. The bridge, stated first, because every number below rides on it

Round 4's tiling identity (`duration_s == start_clock - end_clock` on 768,834
of 768,834 rows; clock-complete regulation periods summing to 1199.58 s of
1200) makes the possession count and the realised mean duration **two readings
of one number**:

    P = 1200 / Dbar        exactly, on a clock-complete regulation game
    SD(P) ~= (Pbar / mu) * SD(Dbar) = 3.8509 * SD(Dbar)

The second line is a first-order delta-method step and is the ONLY
approximation in this document. It is checked numerically rather than asserted:

| | value |
|---|---:|
| realised SD of possessions per team-game | **5.1448** |
| the bridge applied to the realised SD(Dbar) = 1.3293 s | **5.1192** |
| **ratio** | **0.9950** |

0.5% low, in the direction the second-order term predicts (`P` is convex in
`Dbar`). Every "possessions" figure below is `3.8509 x` a duration figure and
carries that 0.5%.

---

## 1. The answer: 98.9% of the gap is a missing within-game LEVEL

The engine diagnostic put the defect in the within-game draw: possession SD
3.743 produced against 4.972 needed. The offline mirror of that quantity is the
variance of the per-game mean residual `rbar_g = Dbar_g - mbar_g` around the
SERVED MODEL's own conditional mean, against what independent inverse-CDF
sampling of that same model produces.

| quantity | s2 | s | in possessions |
|---|---:|---:|---:|
| **needed** `Var(rbar)` | **1.23494** | 1.1113 | **4.2795** |
| **produced** by iid draws from the served law | **0.58405** | 0.7642 | **2.9430** |
| ratio | | | **0.6877** |

(the engine's own reading of the same defect is 3.743 / 4.972 = 0.753; the
offline "produced" excludes the engine's across-seed state-composition
feedback and is therefore a LOWER bound on it, which is why the offline ratio
is the more pessimistic of the two. Both say the same thing.)

The gap `0.65089 s2` splits into the three pre-registered channels:

| channel | s2 | share | evidence |
|---|---:|---:|---|
| (a) the conditional law is too narrow | **+0.00731** | **+1.1%** | served conditional SD **8.8731 s** against an actual residual SD **8.8940 s**, ratio **0.99765** |
| (b) a missing within-game correlation | **+0.64359** | **+98.9%** | `tau = 0.8022 s`, CV **4.545%** of the mean duration |
| (c) `prev_end` composition (L34) | **-0.00190** | **-0.3%** | L34's engine mix moves the marginal mixture variance by **-0.258 s2**, i.e. the WRONG WAY |

**The served conditional law is right to a quarter of one percent. The engine
does not draw a game-level duration level at all, and that alone is the
defect.** L34's composition error -- which owns two thirds of the MEAN
shortfall -- is worth -0.3% of the VARIANCE shortfall and has the wrong sign,
so the mean defect and the variance defect are separate problems with separate
owners. Round 4's section 15.6 conclusion ("round 5 for the CLOCK has no target
left") was about the mean and is not disturbed.

### 1.1 How small the correlation has to be

    Var(Dbar) = (sigma2 / M) * (1 + (M-1) rho_bar)
    needed / iid floor = 1.23494 / 0.59135 = 2.0883   at M = 135.88
    =>  rho_bar = 1.0883 / 134.88 = 0.00807

**An average within-game pairwise residual correlation of 0.8% is the whole
defect.** That is why nothing in the per-possession law looks wrong: the
per-pair signal is a rounding error, and only the 135-possession aggregate sees
it. It is also why CRPS and PIT cannot select this round's winner (section 8).

---

## 2. The signature: a LEVEL, not an autoregression

Within-(game, period) autocorrelation of the raw residual. Possessions
alternate offences, so an EVEN lag is the same offence's next possession and an
ODD lag is the opponent's.

| lag | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 20 | 40 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| corr | -0.0195 | **+0.0350** | -0.0072 | **+0.0326** | **+0.0317** | **+0.0304** | **+0.0274** | **+0.0193** | **+0.0114** |
| n | 266,548 | 262,566 | 258,584 | 254,602 | 246,638 | 238,674 | 222,746 | 190,890 | 111,250 |

Two readings, and the second is the one that chose the arms:

1. **The even lags are a near-flat positive floor out to 40 possessions.** An
   AR(1) with the measured lag-2 value of 0.035 would be 1.2e-3 at lag 4 and
   1.5e-6 at lag 8. The measured decay from 0.0350 to 0.0304 over six lags is
   not an autoregression; it is a persistent level plus the slow attrition of
   the games long enough to reach the lag.
2. **The odd lags are zero or slightly negative.** The level belongs to the
   OFFENCE, not to the two teams jointly -- see section 4, which is where that
   turns out to matter less than it looks.

The arithmetic consequence was pre-registered before any arm ran
(experiments.md 16.1 item 2): an AR(1) needs `rho ~= 0.35` to inflate the
variance of a 136-term mean by 2.09, and the measured lag-1 is **-0.0195**.
Arm A4 was run anyway, with that prediction on the record. It came back at
0.713 against the reference's 0.688 (section 8): **the prediction held.**

---

## 3. Multi-level evidence for channel (a): the conditional law is fine almost everywhere

`model_cond_sd` is the served pmf's own SD of the consumed duration;
`resid_sd` is the actual RMS residual around the same conditional mean. A ratio
of 1.00 means the law's width is right.

### 3.1 By possession-type cell (`prev_end`)

| prev_end | n | share | actual mean | model mean | model cond SD | resid SD | **ratio** |
|---|---:|---:|---:|---:|---:|---:|---:|
| DREB | 94,904 | 35.1% | 14.507 | 14.472 | 8.933 | 8.974 | **0.995** |
| made_FG | 99,695 | 36.9% | 21.787 | 21.526 | 8.924 | 8.902 | **1.002** |
| TOV | 45,869 | 17.0% | 14.882 | 14.901 | 9.176 | 9.226 | **0.995** |
| made_FT | 24,681 | 9.1% | 18.588 | 18.100 | 8.172 | 8.252 | **0.990** |
| period_start | 3,981 | 1.5% | 20.848 | 20.467 | 7.998 | 7.887 | **1.014** |
| other | 1,400 | 0.5% | 1.757 | 1.526 | 3.995 | 4.260 | **0.938** |

**Every powered possession-type cell is inside 1.0%.** The conditional law is
not the defect in any cell, including the two cells L34 names as the engine's
composition error (`made_FG` and `DREB`).

### 3.2 By round-2 clock bucket

| bucket | n | model cond SD | resid SD | **ratio** |
|---|---:|---:|---:|---:|
| 300+ | 200,913 | 8.925 | 8.917 | **1.001** |
| 150-299 | 32,945 | 9.082 | 9.113 | **0.997** |
| 90-149 | 13,407 | 9.134 | 9.171 | **0.996** |
| 60-89 | 6,859 | 9.076 | 9.151 | **0.992** |
| 45-59 | 3,787 | 9.482 | 9.662 | **0.981** |
| 30-44 | 4,133 | 9.058 | 10.205 | **0.888** |
| 20-29 | 2,579 | 7.469 | 7.262 | **1.029** |
| 10-19 | 2,777 | 4.749 | 4.439 | **1.070** |
| 5-9 | 1,724 | 2.183 | 2.312 | **0.944** |
| 0-4 | 1,406 | 0.708 | 0.929 | **0.762** |

The one material miss is **30-44 s (1.5% of possessions, ratio 0.888)** and the
end-of-period buckets, which is where L20's horn censoring and round 3's
`srfloor` both live. Weighted by share, the whole table is worth the +1.1% that
channel (a) carries. It is a real, small, named defect and it is NOT this
round's target.

### 3.3 By period, and by terminal event

| period | n | model cond SD | resid SD | ratio |
|---|---:|---:|---:|---:|
| H1 | 134,026 | 8.772 | 8.741 | **1.004** |
| H2 | 136,504 | 8.972 | 9.042 | **0.992** |

| terminal event | n | model cond SD | resid SD | ratio |
|---|---:|---:|---:|---:|
| FGA_3 | 71,511 | 8.848 | 8.489 | 1.042 |
| FGA_rim | 67,691 | 8.931 | 8.618 | 1.036 |
| FGA_jump2 | 47,921 | 8.913 | 8.950 | 0.996 |
| TOV | 46,205 | 8.914 | 9.049 | 0.985 |
| FT_trip_shooting | 18,493 | 8.932 | 9.369 | 0.953 |
| FT_trip_bonus | 15,509 | 8.657 | 8.917 | 0.971 |
| end_period | 962 | 4.407 | 8.313 | **0.530** |
| unknown | 2,238 | 8.628 | 17.361 | **0.497** |

`end_period` and `unknown` are 1.2% of possessions between them and are where
the law is genuinely too narrow (0.53 and 0.50). They are logged, not chased:
0.53 on 1.2% of rows cannot produce 43% of a variance.

---

## 4. Channel (b), per game and per offence

### 4.1 The per-game distribution of the mean residual

| percentile | p1 | p5 | p25 | p50 | p75 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rbar_g` (s) | -2.143 | -1.504 | -0.592 | +0.217 | +0.923 | +2.162 | +2.926 |

The distribution is centred +0.156 s (round 4's known mean shortfall, unchanged
here) and is **5.1 s wide from p1 to p99 -- 19.6 possessions per team-game**.
The standardised per-possession residual `z` is by contrast well behaved
(p1 -2.14, p50 -0.08, p99 +2.83, SD 1.00), which is the same fact from the
other side: the law is right for one possession and wrong for a game.

### 4.2 Shared by the two offences, or the offence's own?

The two offences' mean residuals inside one game:

| | home offence | away offence |
|---|---:|---:|
| `Var(rbar_o)` | 3.0201 | 2.9471 |
| iid sampling floor | 1.1730 | 1.1638 |
| **own latent** (excess over floor, net of the covariance) | **2.3608** | **2.2971** |

    Cov(rbar_home, rbar_away) = -0.5138        corr = -0.1722

and the two decompositions reconcile exactly:

    (2.3608 + 2.2971)/4 + (-0.5138) = 0.6507 = tau2 measured at the game level

**The level is the offence's own and the two offences of a game are NEGATIVELY
coupled.** That reads, at first, as a refutation of `CLAUDE.md`'s "one pace
realisation per simulated game, both teams scaled by it". It is not, and the
reason matters for the decision rule:

On a clock-complete game the possessions alternate and tile 2400 s, so
`n_h*Dbar_h + n_a*Dbar_a = 2400` with `n_h ~= n_a`. The SUM of the two offences'
mean durations is therefore a deterministic function of the possession count,
and the DIFFERENCE is free. Re-expressed in those coordinates:

| component | variance | what it drives |
|---|---:|---|
| `(Dbar_h + Dbar_a)/2` -- the game's pace level | **1.2349** | **the possession count. This round's target.** |
| `(Dbar_h - Dbar_a)/2` -- which offence played slower | **1.7487** | nothing the engine gates: both teams get the same possession count |

So the negative covariance is the arithmetic shadow of the alternation
constraint, and the "offence-own latent" of 2.36 is mostly the
difference component, which is **orthogonal to the possession count and to
every gate G1-G9 reads**. Section 8 records the consequence: the
pre-registration's tie-break on this structure (criterion 4) was
mis-specified, in the same way round 3c's criterion 2 was, and is recorded as
such rather than applied.

### 4.3 Per team-tempo quintile (the responsiveness rule)

Games split on pregame `tempo_prior_game`. Each quintile is 398-399 games and
50,840-57,542 possessions -- all POWERED.

| quintile | mu (s) | Pbar | cond SD ratio | tau (s) | tau CV | needed P SD | produced P SD | **ratio** | ch (b) share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 (slowest) | 18.827 | 63.74 | 0.985 | 0.775 | 4.11% | 3.805 | 2.705 | **0.711** | 96.0% |
| Q2 | 18.098 | 66.30 | 1.008 | 0.608 | 3.36% | 3.617 | 2.870 | **0.793** | 102.2% |
| Q3 | 17.684 | 67.86 | 0.999 | 0.705 | 3.98% | 3.989 | 2.921 | **0.732** | 99.1% |
| Q4 | 17.228 | 69.65 | 0.992 | 0.789 | 4.58% | 4.420 | 3.023 | **0.684** | 97.7% |
| Q5 (fastest) | 16.586 | 72.35 | 1.004 | 0.895 | 5.40% | 5.034 | 3.175 | **0.631** | 99.9% |

Three things, all of them multi-level evidence for the same conclusion:

1. **The conditional law is right in every quintile (0.985-1.008)** and
   channel (b) carries 96-102% of the gap in every quintile. The defect is not
   concentrated in a tier -- exactly the shape the engine diagnostic found for
   the per-team SD ratios (0.855-0.927 across all five team-scoring quintiles).
2. **The NEEDED dispersion slopes hard with tempo (3.805 -> 5.034, +32%) and
   the PRODUCED dispersion barely slopes (2.705 -> 3.175, +17%)**, so the
   served arm is progressively worse the faster the game: 0.711 at Q1 down to
   0.631 at Q5. A fast game has more possessions, so iid sampling averages MORE
   of the conditional variance away, and the missing level is a larger share of
   what remains.
3. **The latent's own CV is not constant across quintiles** (3.36% to 5.40%),
   and Q2 is out of line with the monotone trend of the other four. That is a
   4-SE departure, not noise (the per-quintile SD carries about 4% SE on 398
   games), and it is what arm A6 was pre-registered to attack and what every
   arm still misses -- section 8.

---

## 5. Channel (c): composition moves the variance the wrong way, and negligibly

L34 and round 4 section 15.4 measure the engine's `prev_end` mix error: it
starts **2.50 pp fewer** possessions after a made field goal and **2.10 pp
more** after a defensive rebound, states whose mean durations sit 7.3 s apart
(21.79 vs 14.51). Holding every per-cell law at its actual value and applying
only that mix shift:

| | actual mix | engine mix (L34) | delta |
|---|---:|---:|---:|
| mixture mean duration | 17.6526 | 17.4772 | **-0.1754 s** |
| mixture VARIANCE | 97.197 | 96.939 | **-0.258 s2** |

The mean effect reproduces round 4's finding independently (-0.175 s against
the -0.148 s round 4's engine cell table measured, the residual being the other
levels' pro-rata absorption). The VARIANCE effect enters `Var(Dbar)` divided by
`M`, i.e. **-0.0019 s2, or -0.3% of a +0.651 gap, with the wrong sign.**

**Fixing the upstream `prev_end` mix would make the dispersion marginally
worse, not better.** The two defects are independent and must be
pre-registered separately, which is what round 4 and round 5 have now done.

---

## 6. What the served arm produces, per possession, for completeness

| | actual | served model |
|---|---:|---:|
| marginal mean duration | 17.653 | 17.497 |
| marginal SD | 9.859 | 9.805 |
| conditional SD | -- | 8.873 |
| RMS residual around the model's mean | 8.894 | -- |
| p5 / p25 / p50 / p75 / p95 (actual, s) | 3 / 10 / 17 / 24 / 34 | |

The marginal SD is 0.5% low and the conditional SD 0.2% low. **Nothing about
the one-possession law is 25% short of anything.** The 25% is entirely an
aggregation property.

---

## 7. The arms, and what each one is

Full specification: experiments.md section 16.2, committed before any fit.
Implementation: `src/cbb_sim/models/clock_v5.py`.

| id | arm | fitted on F2 train (2022-2024) | F1 (2022-2023) |
|---|---|---|---|
| R | `v3c_srfloor_P3_s1`, independent inverse-CDF draws | -- | -- |
| A1 | shared per-(seed, game) log-normal latent, `E[A]=1` | `sigma = 0.047248` | 0.046959 |
| A2 | per-(seed, game, offence) latent, independent | `sigma = 0.066741` | 0.066333 |
| A3 | per-offence latent, correlated across the two | `sigma = 0.077680`, `rho_t = -0.2620` | 0.077476, -0.268009 |
| A4 | Gaussian-copula AR(1) within (game, offence) | `rho = 0.036823` | 0.034095 |
| A5 | `v3c_gamma_P3_s1`, a heavier-tailed parametric family | -- | -- |
| A6 | shared latent, `sigma2 = b0 + b1 * tempo_prior_game` | `b0 = -0.002335, b1 = 6.70e-05` | -0.002749, 7.30e-05 |

**F1 and F2 agree to the third decimal on every parameter** (`sigma` 0.04696
against 0.04725; `rho_t` -0.268 against -0.262; `rho` 0.0341 against 0.0368).
The dispersion is a stable property of the sport, not of a season.

No arm contains a multiplier on engine output. `E[A] = 1` by construction in
A1/A2/A3/A6 and the copula preserves every marginal exactly in A4, so no arm
can move a conditional mean in its own favour -- and section 8 reports the
round-4 mean gate for every arm so that claim is checked, not asserted.

---

## 8. The bake-off (F2, the selection fold)

`scripts/exp_clk5_dispersion_bakeoff.py`. Every arm, the reference included, is
fitted by one fitter, scored by the round-3 blind scorer
(`clock_v3.score_arm_v3`, untouched) and reduced to the primary by one piece of
variance algebra. Needed possession SD **4.2795**.

| id | arm | **P SD produced** | **ratio** | CRPS_trunc | censored loglik | PIT worst D | leak cells | `E[min(T,R)]` | mean gap (s) | implied poss delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R | served | 2.9430 | **0.6877** | 4.92746 | -3.51919 | 0.4306 | 20 | 17.4965 | -0.1561 | +0.607 |
| **A1** | shared latent | **4.2980** | **1.0043** | 4.92777 | **-3.50994** | 0.4307 | 20 | 17.5041 | **-0.1485** | **+0.577** |
| A2 | team latent | 4.3103 | 1.0072 | 4.92825 | -3.51112 | 0.4327 | 20 | 17.5196 | -0.1331 | +0.516 |
| A3 | bivariate latent | 4.3330 | 1.0125 | 4.92876 | -3.51106 | 0.4327 | 21 | 17.5159 | -0.1367 | +0.531 |
| A4 | AR(1) copula | 3.0518 | **0.7131** | 4.92746 | -3.51919 | 0.4306 | 20 | 17.4965 | -0.1561 | +0.607 |
| A5 | gamma family | 3.3539 | **0.7837** | **4.97216** | **-3.57150** | **0.4712** | **34** | 17.4892 | **-0.1635** | +0.635 |
| A6 | tempo-dependent latent | 4.3103 | 1.0072 | 4.92776 | **-3.50984** | **0.4295** | 20 | 17.5044 | -0.1482 | +0.576 |

CRPS_trunc is read on the 270,530 clock-complete rows, so its LEVEL is not
comparable to round 4's 4.89946 (which was read on the full 5,319-game F2
slice); the arm-to-arm DIFFERENCES are, and they are what the no-regression
line uses.

### 8.1 The noise floor

Two spec-identical refits on game-block bootstrap resamples of the F2 training
window, seeds 20260911 and 20260912:

| refit | `sigma` | P SD produced | ratio |
|---|---:|---:|---:|
| boot 20260911 | 0.047493 | 4.3217 | 1.0099 |
| boot 20260912 | 0.047107 | 4.2975 | 1.0042 |
| **floor** (larger absolute delta) | 0.000386 | **0.0242** | 0.0057 |

**A1 beats R by 1.3550 possessions = 56.0 floors.** A4 beats R by 0.1088 = 4.5
floors and is still 0.29 away from the target. CRPS_trunc carries round 4's own
measured game-block SE, **0.00684**, as its floor.

### 8.2 The pre-registered Monte-Carlo cross-check

`scripts/diag_clk5_mc_check.py`, 25 replicate draws over the real 2025 state
sequences, sampling exactly the way `clock.sample_from_pmf` does:

| arm | closed form | Monte Carlo | delta |
|---|---:|---:|---:|
| R | 2.9430 | 2.9551 | +0.41% |
| A1 | 4.2980 | 4.2684 | -0.69% |
| A2 | 4.3103 | 4.3434 | +0.77% |

All inside 0.8%, which is the MC's own noise at 25 replicates. **The closed
form is validated; the primary metric is not an artefact of the algebra.**

### 8.3 Responsiveness by pregame-tempo quintile

`produced / needed` per quintile, all five powered:

| arm | Q1 | Q2 | Q3 | Q4 | Q5 | worst deviation |
|---|---:|---:|---:|---:|---:|---:|
| R | 0.711 | 0.793 | 0.732 | 0.684 | 0.631 | **0.369** |
| A1 | 1.035 | **1.168** | 1.067 | 0.997 | 0.919 | 0.168 |
| A2 | 1.039 | **1.172** | 1.070 | 0.999 | 0.921 | 0.172 |
| A3 | 1.044 | **1.178** | 1.076 | 1.005 | 0.926 | 0.178 |
| A4 | 0.737 | 0.823 | 0.759 | 0.709 | 0.654 | 0.346 |
| A5 | 0.826 | 0.900 | 0.838 | 0.774 | 0.706 | 0.294 |
| A6 | **1.004** | **1.157** | 1.071 | 1.012 | **0.958** | 0.157 |

A6, whose dispersion is a fitted function of pregame tempo, is the only arm
that lands inside 5% at BOTH ends (Q1 1.004, Q5 0.958 against A1's 1.035 and
0.919) -- the state-dependence is real and it is doing what it was
pre-registered to do. **But every latent arm is outside the pre-registered
[0.85, 1.15] band at Q2** (1.157 to 1.178). Section 4.3 shows Q2's needed
dispersion is a genuine 4-SE dip below the tempo trend, not noise, so this is a
real miss and it is not waived.

### 8.4 What A1 would be worth in the engine, on the engine lane's own arithmetic

**Imported, not measured here, and not a prediction of a gate read.** The
engine diagnostic's counterfactual (i) widens the possession draw from
`Var_g(P) = 14.010` to its own residual `24.869` -- i.e. per-team-game
possession SD 3.743 -> 4.987, ratio 1.00 -- and prices that ALONE at:

| gate line | measured | counterfactual (i) |
|---|---:|---:|
| G5 total SD ratio | 0.7985 | **0.8862** (closes 40.8% of the gap) |
| `corr(home, away)` | +0.0269 | **+0.1105** (actual +0.2285) |
| G5 pooled pace channel of `Cov(h, a)` | +22.358 | +33.998 (actual +34.449) |
| G1 possession SD (pooled) | 4.567 | ~5.55 (actual 5.474) |

A1 lands the offline ratio at 1.0043, which is that widening. So **if** the
engine reproduces the offline behaviour, A1 is the arm that delivers
counterfactual (i). Three reasons that is a target and not a forecast: the
counterfactual is arithmetic on a decomposition that holds the event mix fixed
while a wider pace draw would change it; the offline "produced" excludes the
engine's own across-seed composition feedback, so the engine may need LESS
latent than `sigma = 0.0472`; and the engine diagnostic's 4.972 is an upper
bound on the required dispersion (section 10 item 3). **A closed-loop run is
what settles it, and this session does not have one.**

---

## 9. Verdict

**NO ARM IS ADOPTED, on two independent grounds, and the served default is
unchanged.**

1. **Criterion 3 fails for every arm that passes criterion 1.** A1, A2, A3 and
   A6 all land the primary inside +/-1.3% (1.0043 to 1.0125 against a +/-10%
   requirement) and all four pass every no-regression line, but all four sit at
   1.157-1.178 in tempo quintile Q2, outside the pre-registered [0.85, 1.15]
   band. The rule says a failing criterion is not softened.
2. **No closed-loop run exists.** `CLAUDE.md` requires a paired-seed sim run
   showing no gate regressed before an offline winner ships, and section 10
   records why one could not run in this session.

**A1 `v5_glat_shared` is the leading CANDIDATE** and the round's result is
otherwise unambiguous:

- it closes the primary from 0.688 to **1.0043**, a 1.3550-possession move
  against a 0.0242 floor -- **56 floors**;
- it regresses NOTHING: CRPS_trunc +0.00031 against a 0.00684 floor, PIT worst
  D +0.0001 and leak cells 20 -> 20, censored log-likelihood **better**
  (-3.5099 against -3.5192);
- the round-4 MEAN gate **improves** rather than holding: `E[min(T,R)]`
  17.4965 -> 17.5041, gap -0.1561 -> -0.1485, implied possessions +0.607 ->
  +0.577. That is the rounding of `round(A*T)` interacting with the horn
  truncation `min(.,R)`, it was measured rather than assumed, and it is small
  and in the right direction. It is NOT why the arm is preferred;
- it costs one fitted scalar, `sigma = 0.0472`, stable to 0.6% across folds.

**The two negative results are as valuable as the positive one and were both
pre-registered:**

- **A4 (AR(1)) fails exactly as predicted.** experiments.md 16.1 item 2 put
  the required `rho` at ~0.35 against a measured lag-1 of -0.0195 and predicted
  failure before the arm ran; it delivered 0.713. **A within-game
  autoregression is not the mechanism.**
- **A5 (a heavier-tailed conditional family) fails on every line.** Primary
  0.784, CRPS_trunc **6.5 floors worse**, PIT leak cells 20 -> 34, worst D
  0.431 -> 0.471, and the mean gap worse (-0.163 against -0.156). **Channel (a)
  is confirmed not to be the defect by an arm built to exploit it.**

### 9.1 A pre-registration defect, recorded rather than applied

Criterion 4 of section 16.7 breaks a primary tie on the S1/S2 structure (the
per-offence latent variance and the offence-pair correlation). A1, A2 and A6
DO tie on the primary inside the floor, so the criterion would fire, and it
would select A2 over A1. Section 4.2 shows why it must not: S1 and S2 are
dominated by the `Dbar_h - Dbar_a` component, which the alternation constraint
makes largely mechanical and which is orthogonal to the possession count and to
every gate. The tie-break therefore discriminates on a dimension that does not
matter for the target. `CLAUDE.md`'s standing rule -- ties go to the simpler
model -- is the one that applies, and it selects A1.

This is the same class of error round 3c's criterion 2 made (experiments.md
13.2) and is logged the same way: the criterion stands in the append-only
pre-registration, and this section records that it was found to be
mis-specified and why, in the same commit as the result.

---

## 10. What this round does NOT establish

1. **There is no closed-loop evidence.** Everything above is offline. Wiring a
   per-(seed, game) latent into the engine needs one extra uniform at
   `loop.py`'s clock call site: the adapter's `draw(team, state, u, gidx)`
   receives no seed and no stream key, so it cannot draw a per-simulation
   latent, and `StreamBook` lives in `loop.py`. That is about 40 lines --
   `book.keys["clock"]` read at a reserved high ordinal (no new RNG family, no
   change to `rng.py`), a `v5_` prefix in `adapters.py::_load_clock`, and a
   mode-table entry -- but `loop.py` is being edited concurrently by the
   rotation lane and this worker will not race it. **The engine numbers in the
   engine diagnostic (3.743 -> a predicted ~4.97) are NOT confirmed here.**
2. **Every counterfactual is arithmetic on a measured decomposition.** Widening
   the possession draw in the engine changes the event mix, which changes the
   state composition, which changes the very conditional variances the algebra
   holds fixed. The closed form was validated against a Monte Carlo that
   resamples the REAL state sequence, not against a re-simulation.
3. **The "needed" side is an upper bound.** `Var(rbar)` absorbs genuine
   between-game pace variation the model cannot predict and the box/pbp
   possession record's own noise. A clock model with more between-game SKILL
   would legitimately need less within-game dispersion. What is not bounded
   away: at its CURRENT skill the served arm produces 69% of what its own
   residual demands.
4. **Q2's dip is unexplained.** Its needed dispersion is 15% below the overall
   and out of line with the monotone tempo trend on either side of it. It is
   the reason nothing is adopted and it is the obvious target for a round 6
   that fits dispersion on more than pregame tempo.
5. **Nothing here touches the possession-count MEAN**, which round 4 closed and
   assigned upstream, or the `end_period` / `unknown` conditional-law cells
   (ratios 0.53 and 0.50 on 1.2% of rows), which are logged for whoever owns
   the horn-censoring boundary.
