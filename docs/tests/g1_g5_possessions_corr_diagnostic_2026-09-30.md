# G1 (possessions +2.0) and G5 (score correlation 0.117 vs 0.253, total SD ratio 0.898): closed decompositions and owners (2026-09-30)

Lane B. **DIAGNOSTIC ONLY.** Nothing is fitted, no served default is changed,
no arm is adopted, no file under `src/cbb_sim/` is touched. Every channel
table below is arithmetic on measured quantities and **closes**: the channels
are differences of measured levels (or exact covariance identities) and the
residual is printed next to them. Counterfactuals are labelled ARITHMETIC; none
is a sim run.

Inputs (read-only):

    results/engine_v0/F2_2025_s200_v5b_A_full        served v5b stack, 200 seeds, 5,710 games (the gate read)
    results/g1g5_diag/tap_full/                      NEW per-possession tap, FULL slate x 6 seeds (20-25)
    results/g4_diag/tap/                             the 09-18 G4 tap (and-one trips, FT-miss OREB rate)
    data/processed/possessions_v2/possessions_2025   the pbp possession layer (the clock's training table)
    data/raw/hoopr team box 2025 via eval.reference  the box the grader reads
    data/processed/games_universe_v2                 finals, pbp_complete, clock_complete_reg

    scripts/diag_g1g5_tap_v1.py      the instrumented tap (clock adapter wrapped in-process only)
    scripts/diag_g1_possessions_v1.py   Object 1: chain, tiling split, and-one convention, counterfactuals, compensation, cuts
    scripts/diag_g5_corr_v1.py       Object 2: gate-corr identity, pace/efficiency split, shared components, segments
    results/g1g5_diag/{g1_report.json, g5_report.json}   machine-readable output (gitignored, results/)

**The tap is proved, not asserted, to be the served engine.** `diag_g1g5_tap_v1.py`
wraps `ad.clock.draw` in its own process and returns the real draw unchanged.
Its `games.parquet` was compared to the served 200-seed run on every shared
(game, seed) row: **34,260 of 34,260 rows, 26 columns, 0 mismatching cells**
(full slate, seeds 20-25; a first 500-game x 9-seed run also matched 4,500 of
4,500). Every regulation half in the tap consumes exactly 1200 s (2400.0 s per
game, 100% of sims), so the engine's count is exactly a tiling of the clock.
The tap ran at commit `5003ce7` with `src/` clean.

---

## 0. A grading-truth defect that touches both objects

`eval.reference.load_actual_games(2025)` grades **five games with a 0-0 final**
(game_ids 401714278, 401722532, 401706691, 401700283, 401716154; `n_periods`
NaN, no box; `game_finals_v2` also has 0-0, `finals_third_source_checked=False`).
They are unplayed games inside the graded set. They do not move G1 (their
`game_poss` is NaN) but they move every score-based line:

| line | as graded (5,710) | without the five 0-0 games (5,705) |
|---|---:|---:|
| G5 actual home/away corr | 0.2532 | **0.2283** |
| G5 sim home/away corr | 0.1170 | 0.1169 |
| G5 total SD ratio | 0.8983 | **0.9233** |
| G5 margin SD ratio | 1.0396 | 1.0392 |
| G9 total bias | -0.862 | **-0.984** (tolerance +/-1.0) |

**Owner: `src/cbb_sim/eval/reference.py` (`load_actual_games` keeps
`home_score` not-null but not "played").** Reported, not fixed (not this lane's
file). It is 18% of the G5 correlation gap and half of the total-SD-ratio
shortfall to 0.95, and it moves G9's total bias to within 0.02 of its tolerance.

---

## 1. Object 1 -- G1 possessions/game: +1.991 per team-game

### 1.1 The closed chain

All per team-game (the contract's `possessions` is the mean of the two sides'
counts; the grader's `game_poss` is the mean of the two sides' box estimates
`FGA - OREB + TOV + 0.44 FTA`). `pc` = the 5,443 graded games with a complete
pbp possession layer and a box; `cc` = the 1,989 of them that are also
clock-complete in regulation.

| channel | possessions | share of +1.991 |
|---|---:|---:|
| **grading source: box estimator vs pbp possession count** (pc) | **+0.932** | **46.8%** |
| subset: all graded -> pc | +0.000 | 0.0% |
| seed sample: 200-seed run - 6-seed tap (pc) | -0.004 | -0.2% |
| **overtime possessions** (pc) | **-0.307** | **-15.4%** |
| subset: pc -> cc (regulation) | -0.004 | -0.2% |
| regulation: tiling slack (pbp periods sum to 1199.58 s, not 1200) | +0.024 | 1.2% |
| **regulation: start-type COMPOSITION** (cc) | **+0.582** | **29.3%** |
| **regulation: duration LAW within start type** (cc) | **+0.795** | **39.9%** |
| regulation: composition x law interaction | -0.029 | -1.4% |
| **closure** | **+1.991** | residual **-0.000001** |

Regulation uses the exact tiling identity: `N_s = 1200/d_s` (engine, exact),
`N_a = S_a/d_a` (pbp), and `N_s - N_a = (1200 - S_a)/d_s + S_a(d_a - d_s)/(d_s d_a)`,
with `d_s - d_a` split by start type (Kitagawa, three terms, residual 1.5e-14).

Like-for-like readings of the same gap: **count vs count +1.058** (engine count
vs pbp count, pc); **estimator vs estimator +1.504** (the grader's formula
applied to the engine's own box). The engine's count exceeds its own estimator
by +0.487 (the `0.44 x FTA` trip approximation; the engine has no eventless
possessions).

### 1.2 Channel 1 (47%): the grader compares a count to an estimator

Exact accounting of `pbp count - box estimator` on pc, by the pbp terminal event:

| term | possessions |
|---|---:|
| eventless: `unknown` terminal, mid-period | **+0.567** |
| eventless: `end_period` terminal (horn, no event) | +0.160 |
| FT: terminal FT-trip possessions - 0.44 x box FTA | +0.236 |
| TOV: terminal TOV possessions - box TOV | +0.007 |
| FG: terminal FGA possessions - (box FGA - box OREB) | -0.037 |
| closure (= +0.932) | residual 2e-15 |

The engine's `possessions` is, by construction, a count on the pbp layer's
definition: the clock is trained on that table and tiles the period with it
(L34, re-confirmed here). **So 0.93 of the G1 miss is the gate grading a count
against a box estimator that under-counts that same definition.**

The biggest term is the `unknown` class (6,405 possessions in 2025, 0.57 per
team-game): median duration **2 s**, 69% start after a made FG, 74% are
followed by a `DREB` start, 99% score 0, only 1.8% are period-final. Whether
these are real possessions (e.g. a missed shot the feed did not log) or a
segmentation artefact is **not established here**; either way they are in the
clock's training table. Owner: the G1 truth definition
(`eval.reference.load_actual_possessions` / `gates.gate_g1`), with the
`unknown` class a question for the `possessions_v2` event layer.

### 1.3 Channels 3-4 (70%): the clock law, and a start-type label skew

| start type | sim share | act share | sim dur (s) | act dur (s) | composition | law | inter |
|---|---:|---:|---:|---:|---:|---:|---:|
| period_start | 0.0144 | 0.0147 | 20.57 | 20.87 | +0.024 | +0.017 | -0.000 |
| DREB | 0.3712 | 0.3508 | 14.43 | 14.51 | -1.162 | +0.115 | +0.007 |
| TOV | 0.1710 | 0.1696 | 14.83 | 14.88 | -0.086 | +0.037 | +0.000 |
| made_FG | 0.3425 | 0.3685 | 21.40 | 21.79 | **+2.222** | **+0.561** | -0.040 |
| made_FT | 0.0971 | 0.0912 | 18.41 | 18.59 | -0.426 | +0.065 | +0.004 |
| other | 0.0037 | 0.0052 | 1.79 | 1.76 | +0.010 | -0.001 | +0.000 |
| total | | | 17.31 | 17.66 | **+0.582** | **+0.795** | -0.029 |

(cc games, regulation; possessions per team-game; about 1.65M engine possessions from 1,989 games x 6 seeds, about 270k pbp possessions.)

**New finding: the engine and its clock's training table label the possession
after an and-one differently.** In the pbp layer the successor of **every**
and-one possession is labelled `made_FG` (all 5,543 and-one possessions with a
successor in the same period on cc, FT made or missed). The engine overwrites the and-one's end code with `made_FT` (FT made)
or the rebound outcome (FT missed), so ~2.2% of its possessions start with a
label the clock never saw for that situation. Re-labelling the engine's and-one
successors to the training convention (arithmetic; same possessions, same
durations, so the sum is unchanged):

| view | composition | law | interaction | sum |
|---|---:|---:|---:|---:|
| as served (engine labels) | +0.582 | +0.795 | -0.029 | +1.349 |
| training labels | **+0.225** | **+1.138** | -0.013 | +1.349 |

Under the training labels **the made-FG share deficit shrinks from -2.6 pp to
-0.5 pp** and the law term grows to 57% of the G1 gap. The `made_FG` training
cell is itself a mixture: clean made-FG starts last **22.16 s**, and-one
successors **15.55 s** (20.29 s if the FT was made, **2.62 s** if missed),
cell mean 21.79 s. The engine asks the clock only clean made-FG questions and
gets 21.40 s back: **-0.76 s** against the clean real value. The law gap is
negative in every start type except the 0.4%-share `other` (-0.06 to -0.39 s as served), i.e. a level defect
(L34's season-drift diagnosis), plus this cell-definition skew.

ARITHMETIC, not a run: if the engine fed `made_FG` to the clock after an
and-one (the training convention), those possessions would draw the made-FG law
and the count would fall by **0.34** per team-game.

### 1.4 The upstream counterfactual: OREB% and FT trips at actual (ARITHMETIC)

A per-chance end model (`pi_k = e_k/(1-c)`, `c` = OREB continuation per chance,
`o = c/(c + e_DREB)`) reproduces the engine's measured count exactly (69.313 vs
69.313) and, with the actual per-chance vector plugged in, reproduces the
measured composition term to 0.014. Replacing one rate at a time with its
actual value, engine durations held (training labels):

| counterfactual | d possessions / team-game | share of +1.991 |
|---|---:|---:|
| OREB share of live boards 0.2825 -> 0.3026 | **-0.099** | 5.0% |
| made-FT ends per chance 0.0719 -> 0.0802 | -0.046 | 2.3% |
| **OREB + FT trips at actual** | **-0.145** | **7.3%** |
| TOV per chance at actual | -0.027 | 1.4% |
| made-FG per chance at actual | -0.066 | 3.3% |
| full actual per-chance vector | -0.211 | 10.6% |

(As-served labels: OREB -0.119, FT +0.032, both -0.088; the served-label FT and
made-FG rows are contaminated by the and-one label skew and are not the
attribution.)

**The coupling the brief suspected is real but small on the count.** In this
engine a missing OREB does not end a possession early: `loop.py` draws the
whole possession's duration BEFORE the chance cascade, marginal over OREBs, so
the OREB and FT shortfalls reach the count only through the next possession's
start type. That path is worth **0.15 possessions (7%)**, not the +2.0.

### 1.5 Compensation: why G9's total passes (count-vs-count, pc, 200 seeds)

`total = 2 N PPP`; exact split, then an exact log split of PPP into chances per
possession x points per chance:

| channel | points per game (both teams) |
|---|---:|
| excess possessions (+1.058 per team, x actual PPP) | **+2.24** |
| PPP deficit (1.0362 vs 1.0585, -2.1%) | **-3.11** |
| of which: chances per possession (OREB continuation) 1.1483 vs 1.1534 | -0.65 |
| of which: FT points per chance 0.1662 vs 0.1740 | **-1.24** |
| of which: FG points per chance 0.7361 vs 0.7437 | -1.20 |
| of which: non-linear remainder | -0.02 |
| **total gap** | **-0.87** |

**The compensation is real and it is in points, not in the count.** G9's total
bias passes because +2.24 points from excess possessions cancel -3.11 points of
PPP deficit, 1.89 of which is the OREB and FT shortfalls G4 already owns. The two
defects are not mechanically coupled through the count (section 1.4: 0.15
possessions); they are independent errors whose point effects offset.
ARITHMETIC: fixing OREB and FT trips alone would move the total bias by about
+1.9 - 0.3 = **+1.6 points** (to about +0.7 as graded, or +0.6 without the 0-0
games); fixing the count alone would move it by about -2.2. **Any G4 fix will
break G9's total unless G1 is fixed with it**, and vice versa.

### 1.6 The other G1 channels

- **Overtime (-0.307, -15%).** The engine goes to OT in 2.9% of games vs 5.6%
  (pc) and plays 0.032 OT periods per game vs 0.068; frequency part -0.325,
  size part +0.018 (9.58 vs 9.01 possessions per OT period per team). This is
  G7's miss seen through G1; it partly MASKS the regulation overshoot.
  Owner: whoever owns G7 (late-game / tie dynamics), not the clock.
- **Period-final (horn) possessions.** Share equal (1.44% vs 1.47%); engine
  duration 11.2 s vs 12.7 s. Inside the law term.

### 1.7 Multi-level evidence (count vs count, pc; cc for clock cuts)

| cut | cells | sim - pbp count |
|---|---|---|
| per game | 5,443 games | mean +1.06, median +1.50, SD 5.10, P(sim > act) 0.62 |
| site | home/away / neutral | +1.05 / +1.12 |
| month | Nov / Dec / Jan / Feb / Mar | +0.95 / +0.91 / +1.19 / +1.24 / +0.92 (Apr n=17 **UNDERPOWERED**) |
| half (cc, regulation) | H1 / H2 | +0.39 / **+0.98** (durations 17.62 vs 17.82 / 17.02 vs 17.50) |
| game minute (cc) | 0-5 ... 30-35 | +0.00 to +0.18 per 5-min bucket |
| game minute (cc) | **35-40** | **+0.56** (41% of the regulation gap; sim 14.80 s vs 15.68 s) |
| team quintile (by team's as-of sim pace; 72-73 teams each) | Q1..Q5 | +1.99 / +1.51 / +0.99 / +0.67 / +0.14 |
| game quintile (by game's as-of sim pace; ~1,090 games each) | Q1..Q5 | +0.59 / +1.31 / +0.69 / +1.07 / +1.64 |
| per team | 364 teams, >= 10 games | median +1.02, 80% of teams sim > act |

Two readings that disagree and are reported as they are: by TEAM the gap
slopes from +2.0 (slowest teams) to +0.1 (fastest), i.e. team tempo is
compressed (sim span 4.2 vs actual 6.1 possessions); by GAME it does not slope
and alternates Q1<Q2>Q3<Q4, which suggests the game-level pace prediction is
stepped (the clock's tempo-tercile cells) rather than smooth. Not decomposed
further here; flagged to the clock owner. The last five minutes carry 41% of
the regulation gap (the late-game lane's window).

### 1.8 G1 owners

| channel | possessions | owner |
|---|---:|---|
| grading source (count vs estimator) | +0.932 | **G1 truth definition** (`eval.reference.load_actual_possessions` / `gates.gate_g1`); `unknown` class -> `possessions_v2` |
| clock law (training labels) | +1.138 | **`clock`** (level drift, L34; made-FG cell skew -0.76 s clean) |
| and-one start-type label skew (moves 0.36 between composition and law) | (inside the two) | **engine `loop.py` end-code bookkeeping vs clock training labels** |
| composition, training labels | +0.225 | upstream: `rebound` (OREB -0.10), `possession_outcome` (FT trips -0.05, TOV -0.03), `fg_make` (-0.07) |
| overtime | -0.307 | G7 owner (late-game) |

---

## 2. Object 2 -- G5 home/away correlation and total SD ratio

Definitions (all population moments; every identity exact): `m` = the sim's
own 200-seed per-game mean (built only from as-of inputs, so it is an as-of
team-strength and matchup adjustment with no same-game information); actual
residual `r = actual - m`; sim within-game deviation `w = sim - m`. Pace `N` is
the box estimator on BOTH sides (like for like), `e_i = pts_i / N`.

### 2.1 The closed chain of the correlation gap (0.1362)

`corr_sim = [Cov(m_h,m_a) + E Cov_w] / D_sim`,
`corr_act = [Cov(m_h,m_a) + Cov(m_h,r_a) + Cov(r_h,m_a) + Cov(r_h,r_a)] / D_act`.

| channel | corr units | share of 0.1362 |
|---|---:|---:|
| **grading truth: five 0-0 unplayed games** (section 0) | **+0.0248** | **18.2%** |
| subset: 5 games without a box | -0.0002 | -0.1% |
| between-game matchup covariance (identical; denominator only) | +0.0009 | 0.7% |
| mean x residual cross terms (per-game mean miscalibration; lane F's object) | +0.0145 | 10.6% |
| **residual covariance: actual Cov(r_h,r_a) vs sim within-game Cov** | **+0.0962** | **70.6%** |
| closure | 0.1362 | residual 0.0000 |

### 2.2 The residual covariance, split on the points scale (exact)

Per row `dpts_i = m_e,i dN + m_N de_FG,i + m_N de_FT,i + xi_i`, one level down
`e_FG = (FGA/N)(FG pts/FGA)` and `e_FT = (FTA/N)(FT%)`. Covariance of the sums
= sum of pairwise covariances, so the table closes. Corr units = points^2 / D.
Actual SE from 100 game bootstraps.

| component | actual | sim | gap | share of 0.0962 | actual SE |
|---|---:|---:|---:|---:|---:|
| pace x pace | 0.1910 | 0.1834 | +0.0076 | 7.9% | 0.0053 |
| pace x efficiency (both directions) | -0.0405 | -0.0599 | +0.0193 | 20.1% | 0.0077 |
| **FT x FT** | **0.0493** | **-0.0014** | **+0.0507** | **52.7%** | 0.0029 |
| FG x FG | 0.0812 | 0.0486 | +0.0325 | 33.8% | 0.0097 |
| FG x FT (both directions) | -0.0045 | 0.0134 | -0.0178 | -18.5% | 0.0076 |
| second-order remainder | 0.0052 | 0.0014 | +0.0038 | 3.9% | 0.0018 |
| **total** | **0.2817** | **0.1855** | **+0.0962** | 100% | residual 4e-15 |

One level down:

| sub-component | actual | sim | gap | actual SE |
|---|---:|---:|---:|---:|
| **FT: whistle x whistle (FTA per possession)** | **0.0427** | **0.0023** | **+0.0404** | 0.0025 |
| FT: FT% x FT% | 0.0010 | -0.0018 | +0.0028 | 0.0005 |
| FT: whistle x FT%, and product remainder | 0.0056 | -0.0019 | +0.0075 | 0.0013 / 0.0007 |
| **FG: value x value (shooting environment, FG pts per FGA)** | **0.0223** | **-0.0182** | **+0.0405** | 0.0087 |
| FG: volume x volume (FGA per possession: TOV, OREB) | 0.0053 | -0.0085 | +0.0138 | 0.0032 |
| FG: volume x value (both directions) | 0.0528 | 0.0764 | -0.0237 | 0.0091 |
| FG: product remainder | 0.0007 | -0.0011 | +0.0019 | 0.0013 |

### 2.3 Which shared components exist in reality beyond pace

Home-vs-away correlation of each team-game rate's residual (actual, as-of
adjusted) against the engine's within-game correlation:

| rate | actual resid corr | SE | sim within corr | gap |
|---|---:|---:|---:|---:|
| **FT rate (FTA/FGA): whistle** | **+0.207** | 0.013 | **-0.012** | **+0.218** |
| points per possession (ln) | +0.185 | 0.013 | +0.094 | +0.091 |
| FT% | +0.029 | 0.013 | -0.045 | +0.074 |
| eFG%: shooting environment | +0.034 | 0.013 | -0.030 | +0.064 |
| FGA per possession | +0.020 | 0.013 | -0.038 | +0.058 |
| 2P% | +0.021 | 0.013 | -0.024 | +0.045 |
| 3P% | +0.013 | 0.013 | -0.009 | +0.022 |
| OREB%: rebounding | +0.039 | 0.013 | +0.021 | +0.017 |
| TOV per possession: turnover environment | +0.066 | 0.014 | +0.080 | -0.014 |
| 3PA share | -0.017 | 0.014 | -0.008 | -0.009 |

**One shared component dominates and the engine has none of it: the whistle.**
Real games share a free-throw environment (+0.21, 16 SE); the engine's two
teams' FT rates are independent (-0.01). Turnovers and rebounding are already
shared about right. Shooting environment is small but real (+0.03 vs the
engine's -0.03). Month-centring the residuals (removing any league-level drift)
changes nothing (FT rate +0.204, points corr 0.3583 vs 0.3588).

### 2.4 Total SD ratio (0.898; 0.923 without the 0-0 games)

`Var(T) = Var(h) + Var(a) + 2Cov(h,a)`, actual residual vs engine within-game
(5,700 games, population moments):

| term | actual | sim | gap | share |
|---|---:|---:|---:|---:|
| Var(home pts) | 111.69 | 100.99 | +10.70 | 24.5% |
| Var(away pts) | 106.10 | 100.47 | +5.62 | 12.9% |
| **2 Cov(home, away)** | **78.12** | **50.78** | **+27.34** | **62.6%** |
| Var(total) | 295.91 | 252.25 | +43.66 | ratio sqrt = 0.923 |

Log split of Var(ln T): pace Var(ln N) 0.00509 vs 0.00474 (19% of the gap);
efficiency Var(ln E) 0.00998 vs 0.00900 (54%); 2Cov(ln N, ln E) -0.00108 vs
-0.00156 (27%). Individual teams' efficiency variances are nearly right
(Var ln e: 0.0161 vs 0.0156 home, 0.0188 vs 0.0186 away); the efficiency gap is
their covariance.

**A second compensation.** Margin SD passes (1.039) because
`Var(margin) = Var(h) + Var(a) - 2Cov`: the missing covariance (+27.3) widens
the margin more than the missing individual variance (-16.3) narrows it.
ARITHMETIC: a pure shared latent adds the same variance to Var(h), Var(a) and
Cov, so it leaves Var(margin) unchanged; closing the covariance gap that way
would take the total SD ratio to about 1.02 and leave the margin ratio at
1.04. A fix that adds only covariance without the shared structure would drop
the margin ratio to about 0.94.

### 2.5 Multi-level evidence (G5)

Residual/within correlations (actual / sim), by segment:

| segment | games | pts | FT rate | ln PPP | eFG | TOV | OREB | power |
|---|---:|---|---|---|---|---|---|---|
| home/away sites | 4,964 | 0.350 / 0.251 | 0.213 / -0.011 | 0.177 / 0.094 | 0.033 / -0.030 | 0.058 / 0.079 | 0.038 / 0.021 | ok (SE 0.014) |
| neutral | 736 | 0.423 / 0.257 | 0.158 / -0.016 | 0.246 / 0.095 | 0.040 / -0.030 | 0.122 / 0.084 | 0.048 / 0.023 | ok (SE 0.037) |
| Nov | 1,219 | 0.288 / 0.235 | 0.216 / -0.006 | 0.077 / 0.084 | -0.030 / -0.028 | 0.072 / 0.080 | 0.003 / 0.014 | ok (SE 0.029) |
| Dec | 915 | 0.320 / 0.232 | 0.223 / -0.008 | 0.203 / 0.079 | 0.032 / -0.031 | 0.074 / 0.077 | 0.051 / 0.022 | ok (SE 0.033) |
| Jan | 1,420 | 0.388 / 0.262 | 0.231 / -0.015 | 0.217 / 0.102 | 0.064 / -0.029 | 0.058 / 0.079 | 0.047 / 0.023 | ok (SE 0.027) |
| Feb | 1,364 | 0.392 / 0.266 | 0.179 / -0.014 | 0.223 / 0.104 | 0.050 / -0.031 | 0.057 / 0.081 | 0.032 / 0.022 | ok (SE 0.027) |
| Mar | 765 | 0.416 / 0.262 | 0.151 / -0.014 | 0.235 / 0.096 | 0.067 / -0.033 | 0.067 / 0.080 | 0.063 / 0.026 | ok (SE 0.036) |
| Apr | 17 | 0.531 / 0.259 | 0.382 / -0.025 | ... | ... | ... | ... | **UNDERPOWERED** (SE 0.27) |
| as-of pace terciles | 1,900 each | 0.362 / 0.360 / 0.355 act vs 0.247-0.258 sim | 0.18-0.24 vs -0.01 | | | | | ok (SE 0.023) |
| as-of home-FTR terciles | 1,900 each | 0.327 / 0.375 / 0.374 vs 0.243-0.260 | 0.19-0.22 vs -0.01 | | | | | ok |
| per team (games involving the team) | 364 teams, median 31 games | median 0.346 vs 0.251; 65% of teams act > sim | | | | | | **UNDERPOWERED per team** (SE ~0.19) |

**The whistle gap is flat**: +0.15 to +0.23 actual, -0.01 sim, in every site,
month and tercile. A level defect (the engine has no shared foul environment
at all), not a tier or season-part effect. The shared-efficiency gap grows
through the season (ln PPP: Nov +0.08 actual, Mar +0.24) while the engine is
flat at 0.08-0.10.

### 2.6 G5 owners

| channel | corr units (share of 0.1362) | owner |
|---|---:|---|
| **shared whistle (FT x FT, 0.0507; whistle x whistle alone 0.0404)** | **+0.051 (37%)** | **foul generation: `possession_outcome` FT-trip classes + the team-foul accrual law** (both teams draw fouls independently; no game-level whistle state) |
| 0-0 unplayed finals in the truth | +0.025 (18%) | `eval.reference.load_actual_games` |
| FG x FG (shooting value x value +0.040, offset by vol x val -0.024) | +0.033 (24%) | `fg_make` (no shared shooting-environment term; value x value is NEGATIVE in the engine) |
| pace x efficiency | +0.019 (14%) | `clock` / start-type mix (the wrong-sign arrow of `pace_efficiency_sign_2026-09-11.md`, still -0.060 vs -0.041 after the pace latent) |
| mean x residual cross terms | +0.015 (11%) | per-game mean calibration (lane F, G9 slope) |
| FG x FT cross | -0.018 (-13%) | engine produces +0.013 where reality has -0.005; not attributed |
| pace x pace | +0.008 (6%, 1.4 SE) | `clock` v5b latent: **already right within noise** |
| remainder, subset, denominator | +0.004 (3%) | -- |

---

## 3. Recommended pre-registrations (PROPOSED; not written into any experiments.md)

### 3.1 Object 1: G1 truth first, then a clock round on start-type labels and level

**Step 0, a grading decision, not a bake-off.** G1 compares an engine count to
a box estimator (+0.93). Candidates for the PM: (a) grade the engine count
against the pbp count on pbp-complete games; (b) grade the engine's own box
estimator against the actual box estimator on every graded game; (c) keep the
current line. (a) and (b) give +1.06 and +1.50 today. Also: drop the five 0-0
games from `load_actual_games`. Owner: eval.

**Step 1, clock round 6 (owner `clock`, with `loop.py` for the feed).**
- Candidates: R = served `v5b_glat_pmean`; **L1** = engine feeds `made_FG` to the
  clock after an and-one (the training convention; no refit); **L2** = training
  table re-labelled with an explicit and-one-successor start type (or the
  engine's convention) and refit, engine feed changed to match; **D1** = R plus
  the season-level anchor lane C is pre-registering (the law's uniform level
  gap); **L2+D1**.
- Primary metric: closed-loop **regulation count vs pbp count on clock-complete
  games**, reported with its composition / law / interaction split by start
  type (so no arm wins by moving the mix); secondary: the chosen G1 line, law
  gap on clean made-FG starts (22.16 s), per-5-minute counts.
- Floor: seed-offset floor at the same seed count (500 games x 25 seeds
  paired) plus a second-fit seed for L2/D1. Decision: adopt only an arm that
  closes the law term beyond the floor with no G1-G9 regression, **G9 total
  checked explicitly** (section 1.5: fixing the count moves the total about
  -2.2 points). OT (-0.31) and the final five minutes stay with G7 / late-game.

### 3.2 Object 2: a shared game-level whistle (owner `possession_outcome`, coordinate with lane A)

- Candidates: R = served; **W1** = one shared game-level latent on both teams'
  foul intensity (FT-trip class logits and the team-foul accrual hazard), sigma
  fitted walk-forward to the actual residual FT-rate correlation; **W0** =
  per-team (unshared) latent with W1's marginal variance (the control that
  separates "more variance" from "shared variance"); **W1+S1** = W1 plus a
  shared shooting-environment latent on `fg_make` logits.
- Primary metric: within-game home/away **residual FT-rate correlation**
  (target +0.207, SE 0.013) and the points-scale **FT x FT covariance term**
  (target 0.049 corr units). Gate checks: G5 correlation and total SD ratio
  (graded without the 0-0 games), **margin SD ratio must stay in band**, G4 FT
  rate must not regress, G9 total checked.
- Folds: fit sigma on fold 1, select on fold 2; floor = seed-offset floor plus a
  second-fit seed; ties to R. Lane A is editing foul accrual today: the
  whistle latent belongs in that joint round's arm list rather than a parallel
  round.

---

## 4. What this document does NOT establish

1. **No counterfactual was simulated.** Sections 1.3-1.5 and 2.4 counterfactuals
   are arithmetic on measured tables; a closed loop moves the mix, the bonus
   state and the covariances they hold fixed.
2. **Whether the `unknown` possessions are real** (0.57 per team-game), and why
   a missed and-one FT is followed by a 2.6 s `made_FG`-labelled possession in
   the pbp layer. Both sit inside the training table the clock learned from.
3. **How much of the actual residual covariance is shared mean-model error**
   rather than shared variance. The as-of expectation is the engine's own
   per-game mean; a better as-of model would shrink the actual residuals. The
   mean x residual cross term (+0.015) shows the means are miscalibrated; the
   size of any shared error inside `Cov(r_h, r_a)` is not identified. Month
   centring removes nothing.
4. **The per-chance end model is a stationary approximation** (one set of
   per-chance rates per side, interior possessions only). It reproduces the
   engine count exactly and the composition term to 0.014.
5. **The team-level vs game-level pace cuts disagree** (section 1.7) and were
   not decomposed.
6. **Tap scope:** 6 seeds on the full slate; the gate-level numbers are from the
   200-seed run; the seed-sample term is -0.004.
7. **Nothing here is a fix.** No multiplier, cap, clip, offset or blend on sim
   output is proposed; the 0-0 finals are reported to their owner, not edited.

---

## 5. Follow-up (same day, 11:37-12:55 EDT): the `unknown` possessions, a like-for-like G1, the and-one labels, and clock round 6 arm L1

PM follow-up to section 4 item 2 and to the G1 grading definition. Sections 0-4
above are unchanged. Nothing is adopted and no default changes; the one
shared-file edit (a default-off hook) is parity-proved in 5.5.

New scripts:

    scripts/diag_g1_unknown_poss_v1.py    re-runs the possession machine (subclass, delegates unchanged) and
                                          records every `unknown` close with its trigger and context
    scripts/diag_g1_unknown_sample_v1.py  mechanical classes, stratified raw-row read of BOTH feeds
    scripts/diag_g1_unknown_box_v1.py     game-by-game regression of (pbp count - box estimator) on each class
    scripts/diag_g1_possessions_v2.py     phantom-corrected pbp table (in memory) and the like-for-like G1 chain
    scripts/diag_g1g5_tap_v2.py           tap v2: passes ENGINE_ANDONE_LABEL through, records halftime score
    scripts/grade_clock_r6_v1.py          one grader for the round-6 closed loop (R, L1, R floor)
    src/cbb_sim/engine/andone_label.py    arm L1, default-off (5.5)

The re-run machine reproduces `possessions_v2` exactly: 0 per-game count
mismatches in 21,969 games (2022-2025) and the same `unknown` totals
(8,573 / 8,601 / 6,803 / 6,405).

### 5.1 What the `unknown` possessions are

The layer writes `unknown` in exactly three places: a defensive rebound
arrives while no possession is open (`_handle_dreb` opens a possession for the
non-rebounder and closes it at once), the mismatch guard closes a possession
that has nothing pending, or a rebound closes a possession opened by an OREB.
Mechanical classes, then rows read from CBBD `plays_{season}` and hoopR
`play_by_play_{season}` at the same moment (3 per class per season, 72
moments; `results/g1g5_diag/unknown_sample_rows.txt`), then the box check.

| class | what the raw rows show | 2022 | 2023 | 2024 | 2025 | median s | box slope (SE) | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **A** DREB after a missed and-one FT | "X made layup / foul / X missed FT / Y defensive rebound": the and-one handler already closed X's possession at the FT, so the rebound opens a 1-s possession for X | 3,719 | 3,914 | 4,042 | 3,754 | 1 | +0.575 (0.022) | **phantom** |
| **E** shooter's own OREB of a missed and-one FT (not `unknown`, same bug) | the layer opens a new possession for X instead of continuing X's | 474 | 504 | 595 | 586 | 2.5 | +0.654 (0.061) | **phantom** |
| **B1** DREB with no open possession right after free throws | stray rebound row at an FT moment, incl. technical-FT sequences at a frozen clock (the 09-18 lookahead pattern) | 469 | 533 | 465 | 419 | 1 | +0.693 (0.063) | **phantom** |
| **B2** DREB with no open possession, elsewhere | <= 3 s: stray or out-of-order rebound row (shot logged after the rebound); > 3 s: the missed shot is absent from both feeds | 2,521 | 2,091 | 536 | 510 | 12 | +0.843 (0.023); -0.12 to +1.27 by season | **mixed**: <= 3 s phantom (22-30% of B2 in 2022-23, 70-73% in 2024-25), > 3 s real |
| **C1** possession opened by a stray OREB (after a make, TOV or FT), closed by the other team | the wrong team gets a possession spanning the other team's real one | 840 | 970 | 1,027 | 1,019 | 16 | +1.184 (0.040) | **phantom** |
| **C2** other mismatch closes | possession ended by an offensive or loose-ball foul, or an unlogged turnover | 690 | 724 | 511 | 506 | 30 | +0.705 (0.054) | **real** (the box cannot see it either) |
| **D** rebound closes an OREB-opened possession with nothing pending | contradictory OREB/DREB pair at one second; the possession is real, its terminal is lost | 334 | 369 | 222 | 197 | 21 | -0.402 (0.082) | **real**, no extra possession |

The box slope is from regressing per-game (pbp count - box estimator) on per-game
class counts per team-game, with all classes in one regression (21,962 games
pooled, season effects). A possession the box cannot see moves the gap
one-for-one (+1). For A and E the phantom is +1 in the count, but the and-one's
own FTA adds 0.44 to the estimator, so a pure phantom reads **+0.56**. It
measures +0.575 and +0.654. **C1 reads +1.18, one-for-one.**

**The second source cannot decide the question.** On every one of the 72
moments read, the hoopR rows were the same as CBBD's, row for row (both come
from ESPN). The box score is the independent check.

The PM's list of candidate classes, answered:
- **End-of-period heaves:** not in `unknown`. Only 1.8% of `unknown`
  possessions are period-final; eventless horn possessions are `end_period`
  and real.
- **Administrative rows at a frozen clock:** inside B1.
- **Jump-ball rows:** jump balls are inert classes, and none appeared in the
  sample.
- **Feed gaps:** B2 > 3 s and C2.
- **Real empty trips:** C2 and D.

**Verdict.** Of the 0.567 `unknown` possessions per team-game (pc, 2025),
the phantoms are A, B1, C1 and B2 <= 3 s; with E they come to **0.548 per
team-game** removed from the count. C2, D and B2 > 3 s are real. Neither
existing count is the right truth:
- the raw pbp count carries about 0.55 phantoms per team-game;
- the box estimator misses real possessions (C2, eventless horn possessions)
  and approximates FT trips (+0.24; the engine's own trip approximation is
  +0.49), so estimator-vs-estimator is not like-for-like either.

**Proposed G1 definition: engine count vs the pbp possession count with
classes A, B1, C1, B2 <= 3 s removed (their seconds given to the next
possession) and E merged into its and-one possession**, on pbp-complete
games. The script is `diag_g1_possessions_v2.corrected_table`, in memory
only: `data/processed` is not touched. Tiling is preserved; 12 s are lost
over 1,989 games, from 96 phantoms with no later possession in their period.

### 5.2 The like-for-like G1

Engine count (200 seeds, pc) 69.837 vs corrected count **68.231** (raw pbp
68.779, box estimator 67.847):

| channel | engine labels | training labels |
|---|---:|---:|
| **like-for-like gap** | **+1.606** | +1.606 |
| overtime (G7 owner) | -0.301 | -0.301 |
| seed sample, subset, tiling slack | +0.011 | +0.011 |
| start-type composition | +0.538 | +0.330 |
| **duration law** | **+1.433** | **+1.620** |
| of which the `made_FG` cell | +0.995 | +1.181 |
| interaction | -0.074 | -0.054 |
| closure | residual -0.000001 | |

The two B2 sensitivities give +1.576 (all B2 kept) and +1.617 (all B2
removed; the set section 28 pre-registered).

**Compared with the gate's +1.991:** the grading-definition term (+0.932)
splits into +0.548 of pbp phantoms and +0.384 of real possessions the box
cannot see. Against a truth under one definition, **the engine is +1.61
high, not +1.06** as the raw pbp count suggested.

Owners of the residual:
- **`clock` law, 1.43-1.62, dominated by the `made_FG` cell.** The engine draws
  21.40 s where the corrected cell is 22.10 s and clean made-FG starts are
  22.16 s. Of that 0.76 s gap, 0.37 s is and-one contamination of the
  training cell (5.3) and about 0.39 s is L34's season-level drift.
- **The and-one labelling.** It moves 0.21 between composition and law, and
  it is also what causes the contamination above.
- **Composition +0.33 (training labels):** the upstream rates of section 1.4.
- **Overtime -0.30:** the G7 owner.

### 5.3 The and-one label, by branch

**Which label is right for the clock: the engine's, in both branches.** On
the floor (2025, clock-complete, pbp layer):
- **FT made** (73% of and-ones): the next possession is an inbound after a
  made FT, lasting 20.29 s (20.85 s mid-game, 300+ s left). That is closer to
  the `made_FT` cell (18.59; 20.27 mid-game) than to clean `made_FG` (22.16;
  22.60 mid-game).
- **FT missed** (27%): the ball is a live FT-miss rebound. The rebounding
  team's possession lasts 19.8 s (18.70 s plus the 1.1 s phantom), like a
  defensive rebound of a regular missed last FT (17.6 s; 19.0 s mid-game).
  That is a `DREB` start, which is also how the training table labels every
  other FT-miss rebound.

The training table is what is wrong: `made_FG` in both branches, plus the
class A phantom or the class E restart in the missed branch.

Sizes, arithmetic (possessions per team-game, K = 3.89 possessions per second
of mean duration; engine and-one rate 0.0222 per possession, FT% 0.709,
P(OREB | FT miss) 0.130):

| branch | channel | engine share of possessions | size |
|---|---|---:|---:|
| made FT | engine feeds `made_FT` (draws 18.41 s vs 20.29 s real) | 1.571% | +0.115 |
| made FT | (if fed `made_FG` instead: draws 21.40 s) | | -0.068 |
| made FT | successors inside the `made_FG` training cell (4.07% of it at 20.29 s) pull the engine's made-FG draws down | 34.3% (made-FG starts) | +0.101 |
| missed FT | engine feeds `DREB` (draws 14.43 s vs 19.8 s real; the DREB cell mixes FG-miss and FT-miss rebounds) | 0.562% | +0.118 |
| missed FT | class A phantoms inside the `made_FG` training cell (1.27% of it at 1.1 s) | 34.3% | **+0.356** |

The largest single and-one effect is **the phantom inside the clock's
`made_FG` training cell (+0.36)**. Fixing the engine label cannot remove it.
Only a corrected training table can (arms L2 / L2a).

### 5.4 Clock round 6 pre-registered BEFORE any run

`docs/models/clock/experiments.md` **section 28**, commit `33e9db5`, pushed
at 11:46 EDT, before the wiring (11:49) and the closed loop (11:56).

| arm | what it is |
|---|---|
| R | served reference |
| L1 | engine fed the training convention (`made_FG` after an and-one in both branches), no refit |
| L2 | clock retrained on the corrected table with the engine's convention (so the two conventions are compared, not assumed) |
| L2a | L2's phantom correction only, without the relabel |
| D1 | L2 plus season-part pooling of the S1 refit, aimed at the mis-fit `made_FG` cell (L34), or lane C's anchor if it lands first |

All arms are default-off.
- **Primary:** count minus like-for-like truth.
- **Floor:** seed offset, plus a second fit seed for the refit arms.
- **Vetoes:** possession SD, G5 SD ratios, first-half share, OT rate.
- The spec notes that the count fix moves the G9 total by about -2.2 on its
  own, so it can ship only together with the G4 fixes.

### 5.5 Arm L1 wired default-off, and its paired closed loop

- **Wiring.** New module `src/cbb_sim/engine/andone_label.py`
  (`ENGINE_ANDONE_LABEL=engine|training`, default `engine`), plus an 8-line
  hook in `loop.py`. `git diff` showed only these hunks; commit `2185b27` at
  11:49 EDT.
- **Parity.** A 60 x 5 default smoke (`laneB_parity_default_smoke60x5`)
  digests to sha256 `0d4ddccc...`: **PASS, bit-identical** to
  `parity_reference_windows_v6.json`.
- **Tests.** `pytest tests/test_engine.py tests/test_clock_adapter_v3.py`:
  57 passed.
- **The loop.** The L1 hook adds no RNG draw; only an integer label changes.
  - Closed loop: tap v2, section-14.4 subset (500 games), 25 paired seeds per
    arm, 3 processes, 11:56 to 12:45 EDT.
  - R seeds 0-24 are bit-identical to the served 200-seed run (12,500 rows,
    0 mismatching cells).
  - Primary truth: the section-28 set (all B2 removed); the B2-split truth is
    reported alongside.

| | R (seeds 0-24) | **L1** (0-24, paired) | R floor (100-124) |
|---|---:|---:|---:|
| **primary: count - like-for-like truth** (subset pc games) | +1.613 | **+1.276** | +1.528 |
| same, B2-split truth | +1.605 | +1.267 | +1.519 |
| split: OT | -0.345 | -0.396 | -0.361 |
| split: composition (cc) | +0.487 | **+0.144** | +0.519 |
| split: law (cc) | +1.984 | +2.030 | +1.902 |
| possession SD (pooled) | 5.561 | 5.495 | 5.533 |
| G5 margin SD ratio | 0.989 | 0.991 | 0.966 |
| G5 total SD ratio | 0.905 | 0.897 | 0.892 |
| first-half points share (actual 0.4778) | 0.4721 | 0.4722 | 0.4721 |
| OT rate (actual on the subset 0.074) | 0.0317 | **0.0282** | 0.0312 |

The closures hold on every arm (primary = OT + regulation, residual under
4e-6; regulation split residual 1e-14).

**Reading, per the section-28 rule:**
- **Primary:** L1 moves the count **-0.337 against a floor of 0.086 (3.9
  floors)**. It matches the arithmetic in section 1.3 (-0.34). It moves
  composition (0.49 to 0.14) and leaves the law term where it was, as designed.
- **Vetoes passed:** possession SD, both G5 SD ratios and the first-half share
  are not worse beyond their floors.
- **Veto that fires: OT rate**, 0.0317 to 0.0282, which is further from the
  actual. The floor for this line from one R pair is 0.0005, which is below
  the binomial SE of an OT rate at 12,500 sims (about 0.0015 per arm). The veto
  is therefore decided on an underpowered floor; the PM should adjudicate it,
  and this lane does not.
- **Under the pre-registered rule L1 is REJECTED on the OT veto.** No default
  changes either way.

What L1 does not do: +1.28 remains, and it is the clock law, dominated by the
`made_FG` cell. L1 also feeds the label that section 5.3 shows is the wrong
class on the floor; it helps only because the training cell carries the same
label. **L2 / L2a (refit on the corrected table) are the arms that test the
right fix. They need a new training-table build and a clock refit, which do
not exist yet. They were not started, so there is no resume command for them.**
ARITHMETIC: at L1's -0.34 possessions the G9 total moves about -0.7 points.

To re-run the L1 loop:

    set ENGINE_ANDONE_LABEL=training & .venv/Scripts/python.exe scripts/diag_g1g5_tap_v2.py 500 25 results/g1g5_diag/tap_r6_L1 0
    (R: ENGINE_ANDONE_LABEL unset -> tap_r6_R 0; floor: tap_r6_Rfloor 100)
    .venv/Scripts/python.exe scripts/grade_clock_r6_v1.py

### 5.6 Still not established

1. **Which of the 2022-23 B2 > 3 s rows are real.** The box slope for B2 swings
   by season (-0.12 to +1.27); the duration split is a reading of the rows, not
   a proof. The headline moves by at most 0.03 across the three B2 treatments.
2. **The OT veto's floor** (5.5).
3. **L2, L2a and D1** (not built).
4. **The corrected table is not written to disk** and no consumer is switched.
   Whether `possessions_v2`'s machine should be fixed at the source (the
   and-one handler and the rebound-with-no-open-possession path) is an
   event-layer decision.

### 5.7 Incident caused by this lane

12:47-12:54 EDT. While appending this section, I passed a text containing
backtick-quoted file names inside a double-quoted shell string, and the shell
executed them as command substitutions. Bash then interpreted
`tests/test_engine.py`, `docs/models/clock/experiments.md` and
`docs/models/DOCUMENTATION_STANDARD.md` line by line as shell scripts. The
nested substitutions spawned a recursive subshell chain (about 740 `bash.exe`
processes at the peak), which loaded the shared box for about 6 minutes.

What was checked and done:
- **Commands:** every executed line failed ("command not found" or a syntax
  error). The captured output has no successful command output. The only
  risky line was a bare `set ...`, which is harmless in a subshell.
- **Files:** no tracked file was modified and no stray file was created
  (`git status` and a find for files newer than 12:44 were checked). Nothing
  was committed or pushed by that command.
- **Clean-up:** I stopped only processes whose command line carried this
  command's own text (`sec5.md` / `Run R14`): 765 in five passes, then 0
  remained. Lane A's `results/foul_joint/run_r8.sh` processes (11708, 38736)
  and every other process were not touched.

This section and section 29 were then appended from a Python file, with no
shell quoting.
