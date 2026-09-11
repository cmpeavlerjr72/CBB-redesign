# The wrong-signed pace-efficiency link: which channel, and whose model (2026-09-11)

Lane: engine. **Diagnosis and pre-registration only. Nothing is fitted here, no
served default is changed, no arm is proposed as adopted.** Every counterfactual
is arithmetic on a measured decomposition and is labelled as such.

Question, from `docs/tests/engine_v1_variance_ot_diag_2026-09-11.md` section 4:
within a game, across seeds, the engine's `corr(possessions, eFG%)` is
**-0.1976** where the season reads **+0.0454**. That single channel prices
counterfactual (iii) at 39% of the G5 total-SD gap and part of the home/away
correlation gap. Which mechanism produces it?

**Headline, and it contradicts the variance diagnostic's own ranking.** That
document's section 6 item 2 named **fg_make** ("the pace-efficiency link has the
wrong sign ... whether `is_transition` / `chance_elapsed_s` carry enough
make-rate lift in the served `round4_B1` arm"). Measured here, **fg_make is not
responsible and its transition lift is not short -- it over-delivers its own
design by 1.6x.** 81% of the gap is the **clock**: the engine has no game-level
pace realisation at all, so essentially every unit of within-game pace variation
it produces is CAUSED by the possession outcomes, which forces the correlation
negative. That is the same defect clock round 5 landed independently today
(`docs/models/clock/experiments.md` sections 16-17, arm A1 `v5_glat_shared`),
measured there as a dispersion shortfall and here as its efficiency consequence.

Inputs:

    results/engine_v0/F2_2025_s200_rewire1              50 seeds, the gate read
    results/engine_v0/poss_log_pace_efficiency_2026-09-11   NEW, 60 games x 25 seeds
    data/processed/possessions_v2/{possessions,chances}_2025.parquet
    data/processed/models/fg_make/design_v2_shotshooter.parquet   fg_make's own design
    data/processed/models/engine/                       the served engine inputs (v2)

    scripts/diag_pace_efficiency_actual_v1.py     the season, between games
    scripts/diag_pace_efficiency_actual_v2.py     the season, matchup-residualised
    scripts/diag_pace_efficiency_sim_v1.py        the 50-seed run, within game
    scripts/diag_pace_efficiency_probe_v1.py      served partial dependence + design
    scripts/diag_pace_efficiency_poss_log_v1.py   the per-possession tap

`P` throughout is the possession count summed over BOTH teams (the sim's true
count `st.possessions() * 2`; the season's true count from the possessions
table, never the box estimate). `eFG%` is `(FGM + 0.5*FGM3)/FGA` pooled over
both teams. Regulation only. Population variance (ddof=0).

---

## 0. Making the two sides comparable, which the variance diagnostic could not

The engine's -0.1976 is a **within-game, across-seed** number: the matchup is
held fixed and only the draw moves. The season's +0.0454 is a **between-game**
number and carries every difference between matchups. They are different
objects and the 0.243 "gap" between them is partly that.

The closest honest substitute for a within-matchup read on one realisation per
game is to residualise both `P` and `eFG` on the two teams' own season means
(`observed - (mean(t1) + mean(t2) - league mean)`). The game sits inside its own
two team means, which biases the residual SD DOWN by roughly one part in 29;
stated, not corrected.

| read | corr(P, eFG%) | SD(P) | slope |
|---|---:|---:|---:|
| SIM, within game across 50 seeds | **-0.1976** | 7.411 | -0.001399 |
| season, between games (the diagnostic's number) | +0.0476 | 10.429 | +0.000266 |
| **season, matchup-residualised (like for like)** | **+0.0050** | **7.951** | **+0.000034** |
| season, matchup-residualised, eFG from fg_make's own design rows | -0.0010 | 7.696 | -0.000007 |

**The like-for-like actual number is zero, not +0.045**, and its pace variance
(63.2) is close to the sim's (54.9), so the comparison is not a variance-scaling
artefact. The gap to explain is a slope of **-0.001433** per possession.

---

## 1. The ranked answer

Identically constructed on both sides: split `P` into the part implied by the
game's own `prev_end` composition (`P_comp = live seconds / mix-weighted mean
duration`, per-state durations fixed at the season's values on BOTH sides) and
the remainder (`P_resid`). `P_comp` is the efficiency-to-pace arrow; `P_resid`
is pace that the outcomes did not cause.

| slope of eFG on P, per possession | SIM (within game) | ACTUAL (matchup-resid.) | gap | share of gap |
|---|---:|---:|---:|---:|
| **total** | **-0.001399** | **+0.000034** | -0.001433 | 100% |
| COMPOSITION channel (outcomes -> pace) | -0.001660 | -0.001384 | -0.000276 | **19.3%** |
| RESIDUAL-PACE channel (pace not caused by outcomes) | +0.000262 | +0.001418 | -0.001156 | **80.7%** |

| supporting | SIM | ACTUAL |
|---|---:|---:|
| corr(P_comp, eFG) | -0.4470 | -0.2639 |
| corr(P_resid, eFG) | **+0.0432** | **+0.1761** |
| SD(P_comp) | 3.890 | 6.187 |
| SD(P_resid) | **6.338** | **9.497** |
| Var(P_resid) | **40.17** | **90.19** |

**Mechanism 1 (80.7%): the residual-pace channel barely exists and carries
almost no efficiency signal.** Its variance is 0.45x the season's and its
efficiency slope 0.26x. Named cause below, section 2.

**Mechanism 2 (19.3%): the composition arrow is 1.20x too steep.** Its *lever*
is right (section 3); its *mix* is L34's known defect and is already on the
clock and event lanes' books. This is not a new finding.

### The same split done a second way, inside the engine, with the tap

The per-possession tap records fg_make's own served `p_make` on every attempt,
so the engine's eFG can be split into the part its models PREDICT from state
and the part the Bernoulli draws add:

| within-game slope of ... on P | value | corr |
|---|---:|---:|
| **SERVED eFG** (mean of fg_make's own `p_make`) | **+0.000279** | **+0.2260** |
| REALISED eFG (the box) | -0.001399 | -0.1976 |
| realisation residual, by difference | -0.001678 | -- |

**Every sub-model's state response to pace already has the RIGHT sign.** The
engine's own predicted efficiency rises with pace, at +0.226. The whole of the
negative sign is the realisation channel: a seed that misses more gets more
defensive-rebound starts, those sit 7.1 s shorter than made-FG starts, and the
game fits more possessions. That arrow is real basketball and the season has it
too (-0.001384). **The defect is not that the arrow exists; it is that in the
engine the arrow is nearly the ONLY thing moving pace.**

---

## 2. Mechanism 1, named: there is no game-level pace realisation

`engine/loop.py` draws one duration per possession by inverse CDF on a
per-possession uniform (`ad.clock.draw(team_off, x, book.draw("clock", act),
gidx)`), and `clock_adapter_v3.ClockAdapterV3.draw` is `sample_from_pmf(pmf, u)`
row by row. There is no game-level tempo factor anywhere in the clock path.
CLAUDE.md's modelling rule -- *"One pace realisation per simulated game, both
teams scaled by it"* -- is **not implemented**.

The consequence is exactly what the table above measures. With i.i.d. durations,
the only non-outcome source of within-game pace variation is the clock's own
draw noise, which averages down over ~139 possessions. So:

| within-game Var(P), SIM | 54.92 | = comp 15.13 (27.5%) + resid 40.17 (73.2%) + 2cov -0.38 |
|---|---:|---|
| matchup-residualised Var(P), ACTUAL | 63.23 | = comp 38.27 + resid 90.19 + 2cov -65.25 |

**This is the same defect as the variance diagnostic's ranked item 1** (per-team
within-game possession SD 3.743 produced against 4.972 needed). That document
ranked the dispersion shortfall and the eFG sign as two separate items (1 and
2), with two different owners. **They are one defect with one owner.** Clock
round 5 (`docs/models/clock/experiments.md` sections 16-17) reaches the same
place from the dispersion side and prices the missing within-game correlation at
**98.9%** of its own `Var(Dbar)` gap, with the conditional law itself correct to
**0.24%**. Its arm A1 `v5_glat_shared` -- one shared game-level lognormal
latent, `sigma = 0.047248`, fitted to the third decimal on both folds -- takes
the produced per-team possession SD from 2.9430 to 4.2980 against a needed
4.2795, **56.0 noise floors**. Nothing is adopted there and nothing is adopted
here.

### Counterfactual, arithmetic and NOT a re-simulation

If the engine gains exogenous shared-pace variance taking within-game `Var(P)`
from 54.92 to 98.9 (the level section 2 of the variance diagnostic requires),
and that variance carries the engine's OWN measured state-driven efficiency
response to pace (+0.000279 per possession, the tap's served slope):

| | Cov(P, eFG) | SD(P) | corr(P, eFG) |
|---|---:|---:|---:|
| measured | -0.0768 | 7.411 | **-0.1976** |
| + exogenous pace variance alone | -0.0645 | 9.944 | **-0.1237** |
| + composition arrow also corrected to 1.00x | -0.0372 | 9.944 | **-0.0713** |
| target (matchup-residualised season) | -- | -- | ~0.000 |

**The pace latent alone closes 37% of the distance to zero; with the
composition arrow it closes 64%.** The remaining -0.071 is section 4's open
item and is stated as open rather than attributed. +0.000279 is a LOWER bound on
the slope a shared latent would carry -- a multiplicative latent moves the whole
duration distribution across the 8 s transition threshold together, where the
measured slope comes from today's mixture of composition and i.i.d. noise.

---

## 3. Mechanism 2, and what is NOT the mechanism

### 3.1 The composition lever is right; the mix is L34's

The tap gives the engine's realised duration by `prev_end` against the season's:

| prev_end | SIM dur (s) | ACTUAL dur (s) | SIM share | ACTUAL share |
|---|---:|---:|---:|---:|
| made_FG | 21.6225 | 21.6609 | 0.3398 | 0.3696 |
| DREB | 14.5563 | 14.3869 | 0.3710 | 0.3503 |
| TOV | 14.9843 | 14.7378 | 0.1728 | 0.1704 |
| made_FT | 18.5223 | 18.6437 | 0.0983 | 0.0900 |
| made_FG - DREB spread | **7.066** | **7.274** | | |

**The lever -- how much shorter a possession is after a miss -- is right to
2.9%.** The mix is L34's measured defect unchanged (made-FG starts -3.0 pp,
DREB starts +2.1 pp). So mechanism 2 is a `prev_end`-mix question that the clock
round-4 diagnosis already owns, not a new one.

### 3.2 The transition channel is NOT short. fg_make over-delivers its design

This is the claim the variance diagnostic left as "a pre-registrable question",
and it comes back the other way.

First, a measurement error that has to be cleared before any number is read.
The possessions/chances tables' `is_transition` and `duration_s` are
**POST-OUTCOME**: a chance's duration runs to the *rebound* on a miss and to the
*make* on a make, so misses are systematically pushed into longer buckets. This
is exactly the quantity the change ledger bans at L5 and `fg_make/features.md`
section 4 documents. Read off that column the transition eFG lift is **+32.5
pp**; read off fg_make's own design column `chance_elapsed_s` (time from the
chance-start event to the ATTEMPT, which the attempt's outcome cannot move, with
a unit test enforcing it) it is **+6.5 pp**. **The banned column inflates the
effect five-fold.** Everything below uses the design column.

fg_make's own design rows, 2025 regulation, realised make rate:

| class | design, `is_transition_f` 1 vs 0 | **SERVED** (probe A1) | served / design |
|---|---:|---:|---:|
| FGA_rim | +7.80 pp | **+11.26 pp** | 1.44 |
| FGA_jump2 | +1.98 pp | **+5.29 pp** | 2.67 |
| FGA_3 | +0.60 pp | **+0.79 pp** | 1.32 |

Composed into possession eFG at the engine's own half-court shot mix:

| component of the transition eFG lift | ACTUAL (honest) | SERVED ENGINE | served / actual |
|---|---:|---:|---:|
| shot-class MIX shift (possession_outcome) | +2.40 pp | +2.52 pp | **1.05** |
| WITHIN-class make rates (fg_make) | +3.25 pp | +5.31 pp | **1.63** |
| **total possession eFG lift** | **+6.48 pp** | **+9.40 pp** | **1.45** |

And the transition POPULATION and its pace responsiveness are right:

| | SIM (tap) | ACTUAL (design, matchup-resid.) |
|---|---:|---:|
| mean transition share | 0.1587 | 0.1642 |
| corr(P, transition share) | +0.5793 | +0.5035 |
| d(transition share)/dP | +0.002692 | +0.002440 |

**Candidate (a)/(c) as the variance diagnostic framed them are refuted.
fg_make's transition lift is not too small, it is 1.6x too large; the share of
possessions that get it is right; and its pace responsiveness is right.** The
engine cannot be made to produce the actual's sign by giving fg_make more
transition lift -- it already has more than the data.

### 3.3 A real but small train/serve skew in `chance_elapsed_s`, wrong direction to matter

`loop.py` serves the drawn **possession** duration into `chance_elapsed_s` and
into `is_transition`; the design holds the **chance's own elapsed to the
attempt**:

| | mean (s) | median | P(<= 8 s) |
|---|---:|---:|---:|
| possession duration (what the engine serves) | 17.620 | 17.0 | 0.1992 |
| chance-1 elapsed (what the model was trained on) | 16.749 | 17.0 | 0.2109 |
| all-chance elapsed | 15.279 | 15.0 | 0.2823 |
| **shift, served - trained** | **+0.871 / +2.341** | | -0.012 / -0.083 |

This is a genuine L27-shaped train/serve skew and is logged as an open LEVEL
item (the served slope is -0.317 pp/s at the rim, so a +0.87 s shift costs about
-0.28 pp of rim make rate, and it shrinks the transition population). **It
cannot produce the wrong sign**: its within-game direction is POSITIVE (a seed
with shorter possessions gets smaller served elapsed and therefore HIGHER make
rates), which is the sign the engine already has on its served eFG.

### 3.4 Candidate (d), the rotation / shooter channel: refuted, 0.3%

Within game across seeds on the 50-seed run, regressing eFG on four lineup-shape
variables and taking the part of `Cov(P, eFG)` they explain:

| | corr with P | corr with eFG |
|---|---:|---:|
| FGA Herfindahl | -0.0333 | +0.0108 |
| minutes Herfindahl | +0.0022 | +0.0002 |
| top-1 FGA share | -0.0249 | +0.0050 |
| distinct players used | -0.0008 | -0.0018 |

`Cov(P, eFG) = -0.076563`; the part the four explain jointly is **-0.000262,
0.3%**. The rotation channel is measured, not argued, and it is nil. Consistent
with it: the served rotation is R2 `reference`, whose lineup choice reads
period, clock, margin, `prev_end` and fouls, none of which is the possession
count.

---

## 4. Multi-level evidence, and where it is flat

### 4.1 By shot class, within game (SIM) against matchup-residualised (ACTUAL)

| | SIM make rate | ACTUAL make rate | SIM FGA share | ACTUAL FGA share |
|---|---:|---:|---:|---:|
| rim | -0.0986 | +0.0539 | +0.0159 | +0.1246 |
| jump2 | -0.1002 | -0.0077 | -0.0359 | -0.0702 |
| three | -0.1379 | -0.0521 | +0.0159 | -0.0746 |

The sim's make-rate depression is nearly uniform across the three classes
(-0.099 / -0.100 / -0.138), which is the signature of a whole-game channel
(pace and `prev_end` composition), not of a shooter or lineup channel -- a
lineup channel would track shooter skill and hit the classes unevenly.

### 4.2 By team quintile: flat, i.e. not concentrated in a tier

| quintile | tempo q1 | q2 | q3 | q4 | q5 |
|---|---:|---:|---:|---:|---:|
| SIM within-game corr(P, eFG) | -0.2007 | -0.1917 | -0.2053 | -0.1962 | -0.1964 |

| quintile | scoring q1 | q2 | q3 | q4 | q5 |
|---|---:|---:|---:|---:|---:|
| SIM within-game corr(P, eFG) | -0.2011 | -0.1932 | -0.1966 | -0.2001 | -0.1978 |

**Flat at -0.19 to -0.21 across all ten cells.** A level defect, not a tier
effect, and consistent with a structural cause (a missing latent) rather than a
mis-fitted matchup response. The season's own team-tempo quintiles run
+0.000 / -0.021 / +0.003 / +0.065 / +0.050.

### 4.3 By `prev_end` cell, ACTUAL only

corr(game P, eFG% computed inside that cell), between games: DREB +0.0243,
made_FG +0.0130, TOV +0.0380, made_FT +0.0325. `period_start`
**UNDERPOWERED** (0 games clear the 8-attempt floor). **Inside a fixed
`prev_end` cell the season's pace-efficiency link is uniformly small and
POSITIVE** -- which is the statement "reality's arrow is composition, and
conditioning it away leaves a small positive tempo effect".

### 4.4 Per possession type, SIM against ACTUAL (the diagnostic's own table, unchanged)

The four channels the variance diagnostic found matching (TOV/poss, FTA/FGA,
OREB%, FGA/poss) are not re-read here; nothing in this diagnosis touches them.

---

## 5. Ranked responsible sub-models

**1. `clock` -- 80.7% of the slope gap. No game-level pace realisation.**
Mechanism: durations are drawn i.i.d. per possession, so within-game pace
variance is 73% clock draw noise and 27% outcome composition, with no exogenous
tempo component; `Var(P_resid)` 40.17 against the season's 90.19 and its
efficiency slope +0.000262 against +0.001418. Gate lines moved (arithmetic, not
a re-simulation): corr(P, eFG) -0.198 -> -0.124; and, from the variance
diagnostic's own section 3.3(i), G1 possession SD 4.567 -> ~5.55 and G5 total SD
ratio 0.793 -> 0.886. **The object already exists and has already won its
offline round**: clock round 5 arm A1 `v5_glat_shared`, 56.0 floors, nothing
adopted. Section 6 pre-registers the pace-efficiency gate lines its closed loop
needs and which section 16's pre-registration does not contain.

**2. `clock` / `possession_outcome` -- 19.3%. The `prev_end` mix, i.e. L34.**
Mechanism: the composition arrow is 1.20x too steep because the engine starts
2.1 pp more possessions after a defensive rebound and 3.0 pp fewer after a made
field goal, while the duration lever between those states is right to 2.9%.
Already owned by L34 and clock round 4's closing note ("the residual is now an
EVENT / FG_MAKE defect, not a clock one"). No new pre-registration is proposed;
this diagnosis only prices it at 19.3% of the eFG sign gap.

**3. OPEN, not attributed -- the residual -0.071.** After both of the above the
arithmetic leaves corr(P, eFG) at -0.071 against a target of ~0.000. It
corresponds to the engine's served efficiency response to pace being +0.000279
where the season's non-composition channel is +0.001418. It is **not readable
today**: the engine has almost no exogenous pace variation for its models to
respond to, so the served slope is measured on a mixture that is 73% i.i.d.
noise. **It must be re-measured after a pace latent ships and before any
fg_make or possession_outcome arm is pre-registered against it.** Naming an
owner now would repeat the variance diagnostic's own error.

**Not responsible, recorded as such.** `fg_make` (section 3.2: over-delivers its
design's transition lift by 1.63x; the served response to
`chance_elapsed_s` has the right sign and is if anything too steep);
`possession_outcome`'s transition mix shift (1.05x); the rotation and shooter
channel (0.3%, measured); the free-throw and rebound channels (untouched here,
and the diagnostic's section 4 already matched them to 0.03-0.07).

---

## 6. PRE-REGISTRATION PROPOSAL (not run; appended to clock/experiments.md as PROPOSED)

Five lines, written before anything is run, for the clock lane to accept,
amend or reject. Full text as appended: `docs/models/clock/experiments.md`
section 18.

1. **Candidates.** A1 `v5_glat_shared` (the round-5 winner, one shared
   game-level lognormal latent, `sigma` frozen at its fold-fitted 0.047248)
   against R `v3c_srfloor_P3_s1` (the served reference), plus A2 `v5_glat_team`
   as the per-team variant and A4 `v5_ar1` as the within-game-correlation
   alternative that does NOT add a latent. Paired streams, identical seeds.
2. **Primary metric.** Within-game (across-seed) `corr(P, eFG%)`, target the
   matchup-residualised season value **0.000** (band +-0.05), reported
   alongside the tap's SERVED-eFG slope so the realisation channel and the state
   channel are never conflated again.
3. **Folds and universe.** The engine's F2/2025 slate, closed loop, 500 games x
   25 seeds paired, `ENGINE_*` flags pinned to the 50-seed run's values and the
   engine commit recorded. Offline selection is already done on fold 2 (round
   5); this is a Decision-10 closed-loop gate, not a second selection.
4. **Floor.** A spec-identical refit of A1 under a second fit seed, plus the
   seed-offset floor at the same seed count. The round-5 floor is 0.0242
   possessions; the eFG-correlation floor has never been measured and **must be
   measured first** -- two disjoint seed windows of the same configuration,
   exactly as `gates_pair_seedfloor_20_2026-09-11.md` did for G1-G9.
5. **Decision rule.** A1 ships only if (a) `corr(P, eFG%)` moves toward zero by
   more than the floor, (b) no G1-G9 gate line regresses beyond its own floor,
   (c) the per-team and per-quintile tables stay flat (no tier absorbs the
   move), and (d) the served-eFG slope stays positive. Ties go to R. **A
   multiplier, cap, clip or offset on eFG% or on any make rate is banned as a
   response to any outcome of this round.**

Secondary, report-only and NOT a gate: re-measure item 3 of section 5 (the
residual -0.071) under the winning arm, so the next lane has an honest number.

---

## 7. What this diagnostic does NOT establish

1. Every counterfactual in sections 2 and 5 is **arithmetic on a measured
   decomposition, not a re-simulation**. Adding pace variance changes the event
   mix, which changes the covariances the arithmetic holds fixed. They are
   pre-registration targets and upper/lower bounds, not predictions of a gate
   read.
2. The matchup residualisation uses each team's own season means, which contain
   the game being residualised. The bias is about one part in 29 and shrinks the
   residual SD; no correction was applied and no conclusion here turns on the
   third decimal.
3. The per-possession tap is **60 games x 25 seeds (208,849 possessions)**, not
   the 5,710-game slate, and no noise floor was run for it. Its role is to
   measure cell-level quantities (durations by `prev_end`, transition share,
   served eFG) that are slate-stable, not to move a gate line. The 50-seed
   numbers it is compared against come from the full run.
4. The tap wraps three adapters in one process and runs `simulate_chunk` one
   seed at a time; it changes no file under `src/cbb_sim/` and each wrapper
   returns the real adapter's value unchanged, but **it was not digest-checked
   bit-for-bit against an untapped run**. Its per-game aggregates match the
   50-seed run's cell values (transition share 0.1587, `prev_end` durations
   within 0.2 s), which is consistent with but not proof of bit-identity.
5. The season's `P_comp` / `P_resid` split is not orthogonal
   (corr -0.5552 actual, -0.0078 sim), so the 19.3 / 80.7 attribution is a
   variance-share reading of a correlated decomposition, not an orthogonal one.
   The SERVED-vs-REALISED split in section 1 is orthogonal by construction and
   tells the same story; both are reported for that reason.
6. Nothing here reads the 2025-26 season, and nothing here is graded against
   lines.
