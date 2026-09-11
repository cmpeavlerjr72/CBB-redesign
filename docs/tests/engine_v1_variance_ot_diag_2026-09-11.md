# Engine v1 -- variance and overtime diagnostic, and a paired re-read (2026-09-11)

Lane: engine. **Diagnosis only. Nothing is fixed here, no served default is
changed, and no arm is proposed as adopted.** Every counterfactual below is
arithmetic on a measured decomposition, explicitly labelled as such, and is a
pre-registration target for another lane's round, never a patch.

Inputs:

    results/engine_v0/F2_2025_s200_rewire1                  50 seeds, the gate read
    results/engine_v0/F2_2025_s200_rewire1_seeds00_19       seed subset, no re-sim
    results/engine_v0/F2_2025_s200_rewire1_seeds20_39       seed subset, no re-sim
    results/engine_v0/F2_2025_s20_rewire1_clockv3c          NEW 20-seed run, this session

    scripts/diag_engine_variance_ot.py      the decomposition (new)
    scripts/diag_subset_engine_seeds.py     the seed subsetting (new)
    scripts/diag_pair_gate_reports.py       the paired table + floor (new)

    docs/tests/gates_F2_2025_rewire1_seeds00_19.md            grader, arm A
    docs/tests/gates_F2_2025_rewire1_seeds20_39.md            grader, noise arm N
    docs/tests/gates_F2_2025_s20_rewire1_clockv3c_2026-09-11.md  grader, arm B
    docs/tests/gates_pair_engine_v1_clockv3c_2026-09-11.md    A vs B vs floor
    docs/tests/gates_pair_seedfloor_20_2026-09-11.md          A vs N, the floor
    results/engine_v0/F2_2025_s200_rewire1/diag_variance_ot.txt          full output
    results/engine_v0/F2_2025_s200_rewire1_seeds00_19/diag_variance_ot.txt  20-seed rerun

Truth: `game_finals_v2` (verified, two-source) for scores and `n_periods`;
the box possession estimate (`FGA - OREB + TOV + 0.44*FTA`, averaged over both
teams) for possessions, which is the same definition `build_gate_reference.py`
and every G1 line already use. Population variance (ddof=0) throughout.

---

## 0. A correction to the lane brief, stated first because everything follows from it

**The 50-seed run `F2_2025_s200_rewire1` was ALREADY served by
`ENGINE_CLOCK=v3c_srfloor_P3_s1`.** `run_meta.json` records
`adapter_flags.ENGINE_CLOCK = "v3c_srfloor_P3_s1"` and a `clock` artifact
schedule of six `srfloor_P3_S1_2025_*.pkl` files. The gate read says so too
(its section 1 table, `engine v1` column). The brief's premise -- that the
50-seed run predates the default and used the clock reference -- is wrong.

So there is **no reference-clock-vs-v3c comparison to be had from this pair**,
and none is reported. What the paired run actually measures is different and
still worth having, because the 50-seed run's own section 8 flags it:

> the run started at HEAD `4503c52` with the clock lane's UNCOMMITTED
> `clock_adapter_v3.py` in the tree, so it is **not reproducible from any
> commit** ... `loop.py` and `rotation_adapter.py` were edited (rotation round
> 5) at 01:03 and 01:10, after all 12 workers had imported them.

File mtimes confirm the shape of it. `adapters.py` and `clock_adapter_v3.py`
are both stamped 00:48 and have not moved since; the run started at 00:57:39,
so the workers imported exactly the code now committed at `905b902`. `loop.py`
(01:03) and `rotation_adapter.py` (01:10) were edited AFTER the workers had
imported them, so the 50-seed run executed the PRE-round-5 versions of those
two files while the tree now holds the post-round-5 ones.

**The paired run therefore measures one thing: what rotation round 5's
committed `loop.py` / `rotation_adapter.py` edits do to G1-G9 on identical
seeds.** It is reported as that, not as a clock comparison. Round 5's own
commit message (`32f9f65`) claims those edits "ship UNUSED, reference stays the
default", so the pre-registered expectation is a NO-OP, and the run is read as
two things at once:

1. **a verification of that claim** end to end, on the engine's own default
   path rather than on the rotation lane's harness; and
2. **a reproducibility check of the 50-seed gate read itself.** If the two
   agree bit for bit, the 50-seed run IS reproducible from `905b902` in spite
   of the dirty-tree caveat its section 8 records, and that caveat can be
   closed rather than carried.

Because it is a bit-identity question and not a tolerance question, it is
answered first by `scripts/digest_engine_run.py`, which is the tool this
project already uses for exactly that (hard sha256 equality over every row of
`games.parquet` and `players.parquet` plus `adapter_flags` and
`engine_rules_from_data`; no tolerance, because tolerance is a PM adjudication
and never a script's). The G1-G9 table follows it.

---

## 1. Paired re-read: same 20 seeds, two working trees, plus the seed-offset floor

`run_engine.py --fold F2 --season 2025 --seeds 20 --seed-offset 0 --workers 6
--games-per-block 120 --seeds-per-block 5`, on a clean tree at `905b902`,
written to a NEW directory. Seeds 0-19, the same seed VALUES as the 50-seed
run's first 20, so the `(seed, game_id, family)` streams align row for row.

Wall-clock discipline as the brief required it. A 60-game x 2-seed smoke at
09:56:49 took 17 s and a 240-game x 2-seed smoke took 22 s, which separates a
15.3 s pool startup from 0.0139 s per (game, seed) and puts 5,710 x 20 at about
27 minutes. The box was uncontended (the three other Python processes on it are
from 2026-09-10 15:08-15:10 and were burning no CPU), against the 50-seed run's
contended 447 poss/s/core. Measured, not estimated.

**The seed-offset noise floor, which the gate read carries as PENDING, is
delivered here at 20 seeds and cost no simulation at all**: seeds 0-19 and
seeds 20-39 of the SAME 50-seed run are two spec-identical draws of the same
configuration, which is exactly what CLAUDE.md's floor asks for. Every "moved"
verdict below is read against that band and nothing else.

### 1.1 The seed-offset noise floor at 20 seeds

Full table: `docs/tests/gates_pair_seedfloor_20_2026-09-11.md`
(from `docs/tests/gates_F2_2025_rewire1_seeds00_19.md` and
`..._seeds20_39.md`, both graded by the unedited `eval_gates.py`).

| gate line | seeds 0-19 | seeds 20-39 | **floor \|N-A\|** |
|---|---:|---:|---:|
| G1 possessions/game mean | 69.890 | 69.884 | **0.006** |
| G1 possessions/game SD | 4.570 | 4.569 | **0.001** |
| G3 3PA share | 0.3867 | 0.3869 | 0.0002 |
| G3 FTA/FGA | 0.3173 | 0.3168 | 0.0005 |
| G4 eFG% | 0.4975 | 0.4972 | 0.0003 |
| G4 OREB% | 0.2829 | 0.2826 | 0.0003 |
| G5 margin SD ratio | 0.9996 | 1.0056 | **0.0060** |
| G5 total SD ratio | 0.7793 | 0.7772 | **0.0021** |
| G5 home/away corr | 0.0307 | 0.0217 | **0.0090** |
| G6 home margin, non-neutral | +5.679 | +5.680 | 0.001 |
| G6 home margin, neutral | +2.105 | +2.147 | 0.042 |
| G7 OT rate | 0.0306 | 0.0297 | **0.0009** |
| G8 rotation minutes SD ratio | 1.2258 | 1.2255 | 0.0003 |
| G9 margin bias | -0.2041 | -0.1983 | **0.0058** |
| G9 total bias | -0.8301 | -0.9123 | **0.0822** |
| G9 calibration slope | 0.8488 | 0.8489 | **0.0001** |

Gate-level verdicts are identical across the two seed windows on all nine
gates. Only one count line moves: G9 bias by predicted-total tercile, 1/6
outside against 3/6 outside -- a two-cell swing on a 6-cell count, which is the
floor for a count line and not a finding about either window.

**Two consequences the PM should carry forward.**

1. **The floor is very tight for the slate-level means and rates** (1e-3 to
   1e-4) and looser for the dispersion and bias lines (6e-3 on the margin SD
   ratio, 9e-3 on the home/away correlation, 8e-2 on the total bias). Anything
   below those is not a finding at 20 seeds.
2. **Several G5 and G9 lines are seed-COUNT dependent by construction and must
   never be compared across runs of different seed counts.** The same 50-seed
   run, read at 20 of its own seeds, gives margin SD ratio **0.9996** where the
   full 50 give **1.0265**, total SD ratio 0.7793 against 0.7934, calibration
   slope 0.8488 against 0.8907, and PIT K-S p 2.4e-09 against 8.3e-03 -- all
   from the same rows, because `SD(actual - sim mean)` carries Monte-Carlo
   noise that shrinks with seed count while `mean(sim SD)` does not. G1, G3,
   G4, G6, G7 and G8 are stable across the two counts. **The gate read's
   headline "margin SD ratio 1.0265, PASS" is therefore partly a function of
   having 50 seeds**, and the direction is that more seeds push it further
   above 1, not below. Everything below is 20-seed against 20-seed.

### 1.2 The paired read

**Parity digest: PASS, bit-identical.**

    sha256  be952e62b244c3e361dc29ae5350e54aa1c2dd9b99bcae8f1006e33eac0d9c8d
            114,200 game rows and 1,806,298 player rows on BOTH sides
            A  results/engine_v0/F2_2025_s200_rewire1_seeds00_19  (subset, no re-sim)
            B  results/engine_v0/F2_2025_s20_rewire1_clockv3c     (new run, this session)

`digest_engine_run.py` applies no tolerance: it is a hard sha256 equality over
every row of both parquet files with floats at 6 dp, plus `adapter_flags` and
`engine_rules_from_data`. So **every G1-G9 line is identical, and the full
paired table below is identical line for line** -- it is printed in full
(`docs/tests/gates_pair_engine_v1_clockv3c_2026-09-11.md`) because the brief
asked for it and because an all-zero movement column against a non-zero noise
column is the evidence, not a formality.

| gate | line | A (tree @4503c52 + dirty) | B (tree @905b902 clean) | B-A | N (seeds 20-39) | floor \|N-A\| |
|---|---|---|---|---:|---|---:|
| G1 | possessions/game mean | 69.890 vs 67.875 FAIL | 69.890 FAIL | **0.0000** | 69.884 | 0.0060 |
| G1 | possessions/game SD | 4.570 vs 5.474 FAIL | 4.570 FAIL | **0.0000** | 4.569 | 0.0010 |
| G1 | by month | 0/5 powered inside FAIL | 0/5 FAIL | **0** | 0/5 | 0 |
| G2 | PPP by off x def tercile | 2/9 powered inside FAIL | 2/9 FAIL | **0** | 2/9 | 0 |
| G3 | 3PA share (pooled) | 0.3867 vs 0.3906 PASS | 0.3867 PASS | **0.0000** | 0.3869 | 0.0002 |
| G3 | rim share (pooled) | 0.3715 vs 0.3733 PASS | 0.3715 PASS | **0.0000** | 0.3713 | 0.0002 |
| G3 | FTA/FGA (pooled) | 0.3173 vs 0.3295 PASS | 0.3173 PASS | **0.0000** | 0.3168 | 0.0005 |
| G3 | by team | 0/0 powered NEEDS-INSTR | 0/0 NEEDS-INSTR | **0** | 0/0 | 0 |
| G4 | eFG% (pooled) | 0.4975 vs 0.5086 FAIL | 0.4975 FAIL | **0.0000** | 0.4972 | 0.0003 |
| G4 | TOV% (pooled) | 0.1757 vs 0.1739 PASS | 0.1757 PASS | **0.0000** | 0.1755 | 0.0002 |
| G4 | OREB% (pooled) | 0.2829 vs 0.2984 FAIL | 0.2829 FAIL | **0.0000** | 0.2826 | 0.0003 |
| G4 | FT rate (pooled) | 0.3173 vs 0.3295 PASS | 0.3173 PASS | **0.0000** | 0.3168 | 0.0005 |
| G4 | by team (all four) | 0/0 powered NEEDS-INSTR | 0/0 NEEDS-INSTR | **0** | 0/0 | 0 |
| G5 | margin SD ratio | 0.9996 PASS | 0.9996 PASS | **0.0000** | 1.0056 | 0.0060 |
| G5 | total SD ratio | 0.7793 FAIL | 0.7793 FAIL | **0.0000** | 0.7772 | 0.0021 |
| G5 | home/away score corr | 0.0307 vs 0.2532 FAIL | 0.0307 FAIL | **0.0000** | 0.0217 | 0.0090 |
| G5 | PIT K-S p | 2.38e-09 FAIL | 2.38e-09 FAIL | **0** | 5.52e-09 | -- |
| G5 | mean sim SD, margin | 12.0929 PASS | 12.0929 PASS | **0.0000** | 12.1327 | 0.0398 |
| G5 | mean sim SD, total | 14.0180 FAIL | 14.0180 FAIL | **0.0000** | 13.9172 | 0.1008 |
| G6 | home margin, non-neutral | +5.679 vs +5.738 PASS | +5.679 PASS | **0.0000** | +5.680 | 0.0010 |
| G6 | home margin, neutral | +2.105 vs +3.288 FAIL | +2.105 FAIL | **0.0000** | +2.147 | 0.0420 |
| G7 | OT rate | 0.0306 vs 0.0557 FAIL | 0.0306 FAIL | **0.0000** | 0.0297 | 0.0009 |
| G7 | 1H/2H scoring share | n/a NEEDS-INSTR | n/a NEEDS-INSTR | -- | n/a | -- |
| G8 | rotation minutes mean | 30.57 vs 29.83 PASS | 30.57 PASS | **0.0000** | 30.56 | 0.0100 |
| G8 | rotation minutes SD ratio | 1.2258 FAIL | 1.2258 FAIL | **0.0000** | 1.2255 | 0.0003 |
| G8 | players used per team-game | 8.79 vs 9.80 FAIL | 8.79 FAIL | **0.0000** | 8.79 | 0 |
| G8 | top-1 FGA share | 0.2574 vs 0.2496 NEEDS-INSTR | 0.2574 NEEDS-INSTR | **0.0000** | 0.2571 | 0.0003 |
| G9 | margin bias | -0.2041 PASS | -0.2041 PASS | **0.0000** | -0.1983 | 0.0058 |
| G9 | total bias | -0.8301 PASS | -0.8301 PASS | **0.0000** | -0.9123 | 0.0822 |
| G9 | calibration slope | 0.8488 FAIL | 0.8488 FAIL | **0.0000** | 0.8489 | 0.0001 |
| G9 | bias by month | 4/10 outside FAIL | 4/10 FAIL | **0** | 4/10 | 0 |
| G9 | bias by tier | 3/6 outside FAIL | 3/6 FAIL | **0** | 3/6 | 0 |
| G9 | bias by pred-total tercile | 1/6 outside FAIL | 1/6 FAIL | **0** | 3/6 | 2 cells |

Gate-level verdicts A / B / N: G1 FAIL/FAIL/FAIL, G2 FAIL/FAIL/FAIL, G3
NEEDS-INSTR x3, G4 FAIL x3, G5 FAIL x3, G6 FAIL x3, G7 FAIL x3, G8 FAIL x3, G9
FAIL x3.

**Which lines moved beyond the seed noise: NONE. Every line moved by exactly
zero, against a noise column that is non-zero on sixteen of them.**

Three things follow, and the third is the one worth carrying.

1. **Rotation round 5's "ships UNUSED, reference stays the default" claim is
   verified end to end**, on the engine's own default path rather than on the
   rotation lane's harness, at 5,710 games x 20 seeds.
2. **The 50-seed gate read is reproducible from commit `905b902`.** Its section
   8 records that it "is not reproducible from any commit" because
   `clock_adapter_v3.py` was uncommitted at run time; that caveat is now closed
   by demonstration, not by argument, and the gate read can be re-run by anyone
   from that commit.
3. **This says nothing about rotation round 6.** `85de4b6` and `93b7a19`
   landed while this run was in flight; `src/cbb_sim/engine/` was verified
   byte-identical to `905b902` at 09:56 (run start 09:57:48) and `loop.py` /
   `rotation_adapter.py` were next written at 10:10:30, after all six workers
   had imported them. So the run executed `905b902` and round 6's wiring is
   untested here. It needs its own no-op check before the next gate read.

**Cost.** 5,710 games x 20 seeds = 16.35M possessions in 2,365 s on 6 workers,
1,125 poss/s/core averaged. Throughput fell from 12,970 poss/s in the first
450 s to 6,751 at the end as the possession-outcome v4, rotation v6 and usage
v3 lanes started their own jobs on the same 20 cores. Reported as the contended
rate it is.

---

## 2. Q1 -- the possession SD shortfall is WITHIN-game, not between-game

G1 reads possession SD 4.567 against 5.474. The sim has 50 seeds per game, so
its own variance splits exactly:

| | Var | SD | share of pooled |
|---|---:|---:|---:|
| sim pooled | 20.860 | 4.567 | 100% |
| sim between games (Var of the per-game mean) | 7.130 | 2.670 | 34.2% |
| sim within game (mean across-seed Var) | 14.010 | 3.743 | 67.2% |
| actual pooled | 29.961 | 5.474 | -- |

The actual's own split is NOT separately observed -- one realisation per game --
so it is identified against the engine's own prediction instead, and the
identification is stated rather than assumed:

| quantity | value |
|---|---:|
| slope of actual possessions on the sim's per-game mean | **0.8568** |
| corr(sim per-game mean, actual) | **0.4183** |
| Var(sim per-game mean) | 7.140 |
| SD(actual - sim per-game mean) -- what the engine's own draw must cover | **4.9718** |
| sim's own within-game SD | **3.7430** |

**Verdict: the shortfall is within-game.** Three readings agree.

1. The slope is **0.857, not above 1**. A shrunk between-game pace draw shows
   up as a slope above 1; this one is if anything 17% OVER-spread for the
   correlation it achieves. Widening the between-game draw would make the
   calibration worse, not better.
2. The within-game draw is **short by 25% in SD** (3.743 produced against
   4.972 needed) and **43% in variance** (14.01 against 24.87).
3. Reconstructed, a correctly dispersed engine would carry pooled possession
   Var 30.84 (SD 5.553) against the actual's 29.96 (SD 5.474). The missing
   within-game variance (+10.71) is **118%** of the missing pooled variance
   (+9.10); the between-game term is -1.02, i.e. slightly too much.

On G5's own definition (mean within-game sim SD over SD of the residual), the
possession line reads **0.7420**, alongside total 0.7934 and margin 1.0264.

**Caveat, stated because it bounds the claim.** The 4.972 residual SD absorbs
(a) genuine between-game pace variation the engine cannot predict -- its pace
correlation is only 0.418 -- and (b) the box possession estimate's own
measurement noise. So 4.972 is an UPPER bound on the within-game dispersion a
correct engine should produce, and a clock model with more between-game skill
would legitimately need less of it. What is not bounded away: at its CURRENT
skill the engine's dispersion is 25% too tight, and the between-game channel is
not where the pooled gap lives.

### Per team, per team-scoring quintile (the matchup-specific rule)

364 teams with >= 10 actual games. Median per-team SD ratio sim/actual:
possessions **0.9091** (70.1% of teams below 1), PPP **1.0159**, points 0.9828.

| team scoring quintile | teams | sim P SD | act P SD | ratio | sim PPP SD | act PPP SD | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 73 | 4.271 | 4.607 | 0.927 | 0.1449 | 0.1449 | **1.001** |
| 2 | 73 | 4.344 | 4.914 | 0.884 | 0.1472 | 0.1455 | **1.012** |
| 3 | 72 | 4.354 | 4.896 | 0.889 | 0.1471 | 0.1478 | **0.995** |
| 4 | 73 | 4.352 | 5.092 | 0.855 | 0.1494 | 0.1450 | **1.031** |
| 5 | 73 | 4.295 | 4.974 | 0.863 | 0.1502 | 0.1496 | **1.004** |

**The per-possession efficiency dispersion is right at every team level
(0.995-1.031) and the possession dispersion is short at every team level
(0.855-0.927).** The defect is not concentrated in a tier and it is not shared
by the efficiency models.

### Per month

| month | n | sim P SD | act P SD | ratio | sim total SD | act total SD | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| 11 | 1222 | 4.506 | 5.580 | 0.808 | 15.696 | 18.799 | 0.835 |
| 12 | 916 | 4.532 | 5.596 | 0.810 | 15.843 | 18.294 | 0.866 |
| 1 | 1423 | 4.590 | 5.480 | 0.838 | 16.054 | 19.330 | 0.831 |
| 2 | 1367 | 4.461 | 5.159 | 0.865 | 15.967 | 19.052 | 0.838 |
| 3 | 765 | 4.519 | 5.348 | 0.845 | 16.013 | 18.960 | 0.845 |
| 4 | 17 | 4.190 | 5.499 | 0.762 | 15.600 | 20.370 | 0.766 |

Month 4 is 17 games: **UNDERPOWERED**, not read. The other five are flat at
0.81-0.87 on both lines. **This is a level defect, not a cold-start effect** --
the same conclusion the gate read reached for the possession MEAN, now shown
for the possession VARIANCE.

---

## 3. Q2 -- the total SD ratio and the home/away correlation are two views of the pace draw

### 3.1 First, what G5's SD ratio actually is

`cbb_sim.eval.gates.gate_g5` does **not** compare pooled SDs. It compares
`mean over games of the within-game (across-seed) sim SD` against
`SD(actual - sim per-game mean)`. It is a dispersion-CALIBRATION test: is the
engine's own per-game spread the right size for its own error? The pooled SD
comparison is a different question with different numbers (pooled total ratio
0.8634, pooled margin ratio 1.0605), and the two must not be quoted
interchangeably. Both are reported below; the gate lines are the G5 ones.

### 3.2 The margin gate is structurally blind to the pace model

Within-game variance of the total and of the margin, each split into its three
channels (mean over games; `total = P x (ppp_h + ppp_a)`,
`margin = P x (ppp_h - ppp_a)`):

| | have | need | G5 ratio | channel P | channel R | channel cov |
|---|---:|---:|---:|---:|---:|---:|
| **total** | 201.698 | 316.307 | **0.7985** | 60.203 (29.8%) | 186.570 (92.5%) | -44.952 (-22.3%) |
| **margin** | 150.812 | 141.434 | **1.0326** | 0.306 (**0.2%**) | 151.552 (100.5%) | -0.269 (-0.2%) |

`E[ppp_sum] = 2.0719` against `E[ppp_dif] = 0.0744`, so the possession draw
enters the total's variance with 780x the weight it enters the margin's.
**The possession channel carries 29.8% of the total's within-game variance and
0.2% of the margin's. G5's margin PASS is not evidence about the pace model at
all; only the total line sees it.** That is worth stating plainly because the
gate read's headline is the margin SD ratio landing inside the gate for the
first time in the project, and that result is real but is silent about G1.

### 3.3 Which channel is short in the total

Three counterfactuals, each applied ALONE, each pure arithmetic on the measured
decomposition and NOT a re-simulation:

| counterfactual | Var_g(total) | closes | G5 total ratio |
|---|---:|---:|---:|
| measured | 201.698 | -- | 0.7985 |
| (i) possession draw widened to its own residual (Var_g(P) 14.010 -> 24.869) | 248.409 | **40.8%** | 0.8862 |
| (ii) within-game Cov(ppp_h, ppp_a) set to zero | 184.189 | **-15.3%** | 0.7631 |
| (iii) within-game Cov(P, ppp_sum) set to zero (-0.15503 -> 0) | 246.650 | **39.2%** | 0.8831 |
| (i) + (iii) together | 293.361 | 80.0% | **0.9631** |
| (i) + (ii) + (iii) | 275.852 | 64.7% | 0.9339 |

**(i) and (iii) together carry the total SD ratio from 0.793 to 0.963, inside
the 0.95-1.05 gate, with nothing else touched.** (ii) is reported because it was
pre-registered and it comes back with the OPPOSITE sign to the hypothesis: the
engine's WITHIN-game `corr(ppp_h, ppp_a)` is **+0.1038** (positive, and only
24.0% of games negative), and removing it would make the total worse. There is
no within-game efficiency anti-correlation to fix.

### 3.4 The pooled PPP covariance, and why the margin passes

| | Var(ppp_h) | Var(ppp_a) | Cov | corr |
|---|---:|---:|---:|---:|
| sim | 0.023058 | 0.023482 | **-0.001478** | **-0.0635** |
| actual | 0.022953 | 0.023488 | **+0.000026** | **+0.0011** |

**Each team's own PPP variance is right to 0.5% (1.005 and 1.000). The entire
pooled Var(ppp_sum) / Var(ppp_dif) discrepancy is the COVARIANCE**, and since
`Var(sum) = Vh + Va + 2Cov` and `Var(dif) = Vh + Va - 2Cov`, one negative
covariance simultaneously depresses the total and inflates the margin. It is
the mechanical explanation of "total SD low, margin SD fine" with perfectly
calibrated marginals.

Section 3.3 shows this covariance is NOT within-game (within-game it is +0.104).
It is between-game: across games, teams with a high expected `ppp_h` face
opponents with a low expected `ppp_a`, because team ratings are near zero-sum
in a matchup. Reality has that structure too and offsets it with a game-level
common efficiency factor; the engine has no such factor.

### 3.5 The home/away correlation, decomposed

`corr(home_pts, away_pts)`: sim **+0.0269**, actual **+0.2285**.
`Cov`: sim **+3.325**, actual **+31.690**.

`Cov(P*h, P*a) ~= E[h]E[a] Var(P) + E[P]^2 Cov(h,a) + E[P](E[a]Cov(P,h) + E[h]Cov(P,a))`

| | exact | pace channel | ppp-cov channel | cross channel | rem |
|---|---:|---:|---:|---:|---:|
| sim | +3.325 | **+22.358** | **-7.216** | **-12.041** | +0.224 |
| actual | +31.690 | **+34.449** | **+0.121** | **-2.816** | -0.065 |
| gap | -28.365 | -12.091 (43%) | -7.337 (26%) | -9.225 (33%) | |

Two structural readings:

- sim `corr(home, away)` **BETWEEN games = -0.2447**, **WITHIN game = +0.1444**.
  The defect is entirely between-game. Inside a fixed matchup the engine
  already couples the two teams the right way (through the shared pace draw);
  across matchups it does not, because its pace PREDICTION varies far too
  little (Var 7.14 against the actual's total 29.96, correlation 0.418), so
  what is left is the zero-sum team-rating structure and that is negative.
- The counterfactual, with the assumption stated: if the missing within-game
  possession variance enters as extra SHARED pace noise independent of both
  teams' PPP -- which is precisely what "one pace realisation per simulated
  game, both teams scaled by it" says it should be -- then numerator AND
  denominator move and

  | | measured | counterfactual (i) | actual |
  |---|---:|---:|---:|
  | pooled pace channel | +22.358 | **+33.998** | +34.449 |
  | Cov(home, away) | +3.325 | +14.965 | +31.690 |
  | corr(home, away) | +0.0269 | **+0.1105** | +0.2285 |

  **Fixing the pace draw restores the pace channel to 98.7% of the actual's
  (+34.00 vs +34.45) and moves the correlation from +0.027 to +0.111 -- less
  than half way.** The rest is the -7.34 ppp-covariance gap and the -9.23 cross
  gap, both between-game.

**Verdict on Q2: BOTH, in a stated proportion, and the larger part is pace.**
43% of the covariance gap is the shared-pace channel being too weak (Var(P)
0.696 of actual), 33% is the pace-efficiency cross term, and 26% is a direct
PPP covariance that is negative where reality is zero. The first two are the
clock model's; the third is a between-game structure no sub-model currently
owns.

---

## 4. Which event channel carries the pace-efficiency anti-correlation

Channel (iii) prices `Cov(P, ppp_sum) = -0.155` within game at 22% of the
engine's own within-game total variance and 39% of the gap. This names it
rather than guessing. Within each game, across seeds, the correlation of the
possession count with each rate the L3 sub-models own:

| rate | sim WITHIN-game corr with P | sim pooled corr | actual corr | verdict |
|---|---:|---:|---:|---|
| TOV per poss | +0.0862 | +0.1063 | +0.0964 | matches |
| FTA/FGA | +0.1800 | +0.1999 | +0.1893 | matches |
| OREB% | -0.0950 | -0.0781 | -0.1201 | matches |
| FGA per poss | -0.1905 | -0.2168 | -0.2654 | close |
| **eFG%** | **-0.1976** | **-0.1448** | **+0.0454** | **WRONG SIGN, gap 0.243** |
| PPP sum | -0.2117 | -0.1740 | -0.0310 | the sum of the above |

**Four of the five event channels reproduce the actual's pace dependence to
within 0.03-0.07. eFG% is the one that does not, and it has the wrong sign.**
In the engine, a realisation with more possessions shoots WORSE; in the season,
games with more possessions shoot marginally BETTER. That single channel is the
whole of the excess pace-efficiency anti-correlation, and therefore of
counterfactual (iii).

The mechanism has a name already in the project. L34: 67% of the clock's
duration shortfall is `prev_end` state COMPOSITION -- "the engine starts 2.5 pp
fewer possessions after a made FG and 2.1 pp more after a defensive rebound;
those states sit 7.5 s apart". That link runs in both directions: a seed that
shoots badly produces more defensive-rebound starts, shorter possessions and
more of them. Reality carries the same link AND an offsetting one -- faster,
more transition play raises eFG -- and the two roughly cancel at +0.045. The
engine has the first and not enough of the second.

The engine's state block does carry `is_transition`, `is_transition_f` and
`chance_elapsed_s` into fg_make, so the channel exists; what is not established
here is whether its MAGNITUDE is right. That is a pre-registrable question and
it is left as one.

**The caveat this table needs.** The "actual" column is a BETWEEN-game
correlation -- there is one realisation per game -- and the box possession
estimate carries TOV and FGA in its own numerator, so that column is a bound on
the real dependence, not a clean measurement of it. It is reported as a bound.
The eFG comparison is the one least exposed to that algebra (eFG appears in the
possession estimate only through OREB), which is part of why it is the line
worth acting on.

---

## 5. Q3 -- the OT shortfall is a tie-rate defect, not an OT-handling defect

**A game goes to overtime if and only if the regulation margin is 0.** So the
regulation margin is recoverable from what the contract already writes, on both
sides: it is 0 when `n_periods > 2` and the final margin otherwise. G7's "OT
rate" and "regulation tie rate" are therefore the SAME number, and the question
splits cleanly into a distribution question and a handling question.

| | sim | actual | ratio |
|---|---:|---:|---:|
| OT rate == regulation tie rate | 0.0300 | 0.0557 | **0.539** |
| n | 285,500 rows (8,568 OT) | 5,710 games (318 OT) | |

### 5.1 The regulation-margin distribution

| \|reg margin\| | sim P(=k) | act P(=k) | ratio |
|---:|---:|---:|---:|
| **0** | **0.03001** | **0.05657** | **0.531** |
| **1** | **0.05507** | **0.03660** | **1.504** |
| 2 | 0.05197 | 0.05149 | 1.009 |
| 3 | 0.04869 | 0.05429 | 0.897 |
| 4 | 0.04398 | 0.05534 | 0.795 |
| 5 | 0.04462 | 0.05622 | 0.794 |
| 6 | 0.04151 | 0.04186 | 0.992 |
| 7 | 0.04235 | 0.04606 | 0.920 |
| 8 | 0.04299 | 0.04991 | 0.861 |
| 9 | 0.04356 | 0.04729 | 0.921 |
| 10 | 0.04483 | 0.04308 | 1.040 |

**The actual has a spike at 0 and a trough at 1; the sim is smooth and
monotone from 1 downward.** The actual runs 0.0566 / 0.0366 / 0.0515 at k = 0 /
1 / 2 -- a real dip at one point. The sim runs 0.0300 / 0.0551 / 0.0520. It has
**47% too few ties and 50% too many one-point regulation margins**, i.e. the
missing mass at zero is sitting one point away.

### 5.2 Scale versus shape, separated

| | regulation-margin SD | normal P(tie) at that SD | observed P(tie) | observed/normal |
|---|---:|---:|---:|---:|
| sim | 15.4885 | 0.02435 | 0.03001 | **1.2326** |
| actual | 14.6067 | 0.02554 | 0.05569 | **2.1808** |

The ratio factors exactly:

    0.539  =  (0.02435 / 0.02554)  x  (1.2326 / 2.1808)
           =        0.9534         x        0.5652
              scale (SD 6% wide)      shape (missing pile-up)

The two factors are multiplicative, not additive: the scale factor alone would
leave the OT rate at 95.3% of the actual, the shape factor alone at 56.5%, and
composed they give the observed 53.9%. **Of the 46.1% shortfall, the 6%-too-wide
regulation margin accounts for 4.7 points and the missing pile-up at zero for
43.5 points -- about nine tenths of it.** Excess kurtosis
confirms it independently: sim **+0.294**, actual **+0.740**. The engine's
regulation margin is close to Gaussian; the real one is sharply peaked.

The deficit is present in every predicted-closeness bucket, so it is not an
artefact of one segment:

| \|predicted margin\| quintile | games | sim P(tie) | actual OT rate | ratio |
|---|---:|---:|---:|---:|
| 1 (closest) | 1144 | 0.03895 | 0.07780 | 0.501 |
| 2 | 1147 | 0.03724 | 0.06103 | 0.610 |
| 3 | 1139 | 0.03375 | 0.05795 | 0.582 |
| 4 | 1138 | 0.02761 | 0.05097 | 0.542 |
| 5 (least close) | 1142 | 0.01245 | 0.03065 | 0.406 |

Flat across months too (sim/actual OT ratio 0.50-0.59 for November through
March; month 4 is 17 games and **UNDERPOWERED**).

### 5.3 OT handling, conditional on a tie

No tie-rate defect can touch these numbers.

| n_periods | sim P(.\|OT) | act P(.\|OT) | sim n | act n |
|---:|---:|---:|---:|---:|
| 3 | 0.91772 | 0.82704 | 7,863 | 263 |
| 4 | **0.07586** | **0.13836** | 650 | 44 |
| 5 | 0.00619 | 0.03145 | 53 | 10 **UNDERPOWERED** |
| 6 | 0.00023 | 0.00314 | 2 | 1 **UNDERPOWERED** |

| | sim | actual |
|---|---:|---:|
| total points in OT games | 166.518 | 164.764 |
| \|final margin\| in OT games | 4.966 | 4.412 |
| home win rate in OT games | 0.5326 | 0.6069 |

**Verdict on Q3: a tie-rate defect, not an OT-handling defect.** Once tied, the
OT module produces a plausible scoreline (total within 1.8 points, OT margin
within 0.55). Its one miss -- 0.076 double-OT against 0.138, on 44 actual games,
the last powered cell -- is **the same missing pile-up at zero one level down**,
applied to a five-minute period rather than a forty-minute game, and is
therefore evidence FOR the shape diagnosis rather than a second defect. The
home win rate in OT games (0.533 vs 0.607, actual SE 0.027) is reported and NOT
read: the sim's OT games are a differently-selected set, and there is no
pre-registered tolerance for it.

### 5.4 Where the missing pile-up comes from

Within-game (across-seed) margin dispersion at a fixed matchup is **12.207**
against a residual of **11.892** -- correct in size (G5 margin ratio 1.0264).
So this is not a "too little late-game variance" problem in the SD sense. It is
a SHAPE problem: the engine simulates the last two minutes with the same
shot-selection, foul and clock behaviour as the first two, so it produces no
absorbing behaviour at margin 0 and no repulsion at margin 1. The real
distribution's spike-at-0 / trough-at-1 is the fingerprint of late-game
strategy -- a trailing team down 3 shoots a three to tie rather than a two to
trail by one, a trailing team fouls, a leading team shoots one-and-ones -- and
the engine has no sub-model that represents any of it.

---

## 6. Ranked responsible sub-models, with the mechanism

Ranked by how much measured gate movement each would produce, on the
arithmetic above. Every entry is a pre-registration target for the owning
lane's next round, not a fix.

**1. clock -- the within-game duration draw is 25% too tight in SD.**
Mechanism: the engine's per-game possession dispersion is 3.743 where 4.972 is
required; the between-game draw is fine (slope 0.857, if anything over-spread).
Gates moved: G1 possession SD 4.567 -> ~5.55 pooled; G5 total SD ratio 0.793 ->
0.886 alone; G5 home/away correlation +0.027 -> +0.111 alone. Measured at
sections 2 and 3.3(i) / 3.5. Note this is a DISPERSION target and is separate
from L31/L34's MEAN target (+2.0 possessions, the -1.9% conditional-mean
shortfall) -- fixing the mean does not fix this and the clock lane should
pre-register them as two lines, not one.

**2. fg_make (and, through the `prev_end` mix, possession_outcome) -- the
pace-efficiency link has the wrong sign.** Mechanism: within a game, sim
`corr(P, eFG) = -0.198` against an actual `+0.045`; the other four event
channels match to 0.03-0.07. The engine reproduces L34's `prev_end`
composition link (bad shooting -> more DREB starts -> more, shorter possessions)
and not the offsetting transition-efficiency link, so the two do not cancel as
they do in the season. Gates moved: G5 total SD ratio 0.793 -> 0.883 alone,
0.963 in combination with (1). The pre-registered question is whether
`is_transition` / `chance_elapsed_s` carry enough make-rate lift in the served
`round4_B1` arm, measured as the within-game `corr(P, eFG)` against the
reference. Section 4.

**3. A missing late-game regime -- no sub-model owns it.** Mechanism: the
regulation-margin density at zero is 1.23x its Gaussian value in the sim and
2.18x in the season; the sim has 47% too few ties and 50% too many one-point
margins. Nine tenths of G7's shortfall. The engine's margin SD is correct, so
this is shape, not scale, and cannot be reached by widening anything. Gate
moved: G7 0.030 -> 0.056. Section 5. **This is the one item that needs a new
sub-model rather than a better fit of an existing one**, and it should be
pre-registered as such (late-game shot selection and intentional fouling
conditioned on margin and clock) rather than as a change to the OT module,
which sections 5.3 shows is not the defect.

**4. A game-level efficiency common factor -- no sub-model owns it.**
Mechanism: pooled `corr(ppp_h, ppp_a)` is **-0.064** in the sim and **+0.001**
in the season, entirely between-game (within-game the sim is +0.104, the right
sign). Each team's own PPP variance is right to 0.5%, so this is purely the
coupling. Gate moved: 26% of the home/away covariance gap, the part items 1 and
2 provably cannot reach (+0.111 is where they stop). Lowest rank because it is
the smallest, the least well understood, and the most likely to be an artefact
of grading PPP on a shared box possession denominator -- **that confound must be
ruled out before any arm is fitted for it.**

**Not responsible, and worth recording as such.** The per-possession efficiency
models: each team's PPP variance is 1.005 / 1.000 of actual pooled and
0.995-1.031 across all five team-scoring quintiles, and `Var_g(ppp_dif)` carries
100.5% of the margin's within-game variance at a G5 ratio of 1.033. The
efficiency layer's DISPERSION is calibrated. Its LEVELS are not (G4 eFG% -1.13
pp, OREB% -1.56 pp) and those remain open on their own lanes; this diagnostic
says nothing about them.

---

## 6b. Seed-count robustness of the diagnosis itself

Section 1.1 shows several G5 and G9 gate LINES move with seed count. The
decomposition does not. Re-run on seeds 0-19 alone
(`results/engine_v0/F2_2025_s200_rewire1_seeds00_19/diag_variance_ot.txt`):

| quantity | 50 seeds | 20 seeds |
|---|---:|---:|
| G5 total SD ratio (this script's own recomputation) | 0.7985 | 0.7917 |
| counterfactual (i), share of gap closed | 40.8% | 39.6% |
| counterfactual (iii), share of gap closed | 39.2% | 36.5% |
| counterfactual (ii), share of gap closed | -15.3% | -14.6% |
| within-game corr(P, eFG%) | -0.1976 | -0.1956 |
| pooled corr(ppp_h, ppp_a) | -0.0635 | -0.0616 |
| within-game possession SD produced / needed | 3.743 / 4.972 | 3.757 / 5.021 |
| OT observed/normal (sim) vs (actual) | 1.2326 / 2.1808 | 1.2531 / 2.1808 |
| regulation-margin excess kurtosis (sim) | +0.294 | +0.272 |

Every ranked finding in section 6 holds at both seed counts and none of them
depends on the seed count for its sign or its rough size.

---

## 7. What this diagnostic does NOT establish

1. It is read on a 50-seed run against a 200-seed floor. The decomposition
   quantities are slate-level second moments over 285,500 rows and are far more
   stable than a slate mean, but "far more stable" is not a floor and none was
   run for them.
2. Every counterfactual in section 3.3 and 3.5 is **arithmetic on a measured
   decomposition, not a re-simulation**. Widening the possession draw in the
   engine would change the event mix, which would change PPP, which would
   change the very covariances the arithmetic holds fixed. The numbers are
   upper bounds on what a single-channel fix delivers and are pre-registration
   targets, not predictions of a gate read.
3. The actual possession series is a BOX ESTIMATE. Every "actual" possession
   variance, and every actual correlation involving PPP, carries that
   estimator's own noise and its algebra (TOV and FGA sit in its numerator).
   Section 4's actual column is explicitly a bound.
4. The eFG sign finding is a correlation, not an identified mechanism. The
   `prev_end` account is L34's, imported; this diagnostic shows it is
   consistent with the sign, not that it is the cause.
5. Nothing here attributes anything to one served arm. Six flags moved at once
   in the run being diagnosed.
