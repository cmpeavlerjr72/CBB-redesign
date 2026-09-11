# Rotation substitution-WAVE audit (round-5 evidence) -- 2026-09-11

Worker: Opus. Inputs: `data/processed/possessions/possessions_{2024,2025}.parquet`
(on-floor sets, `start_reason`, `off_team_fouls`/`def_team_fouls`),
`data/raw/cbbd/pbp/plays_{season}.parquet` (`PersonalFoul`, the three timeout
play types). Scripts: `scripts/diag_rotation_wave_v5.py` (`--mode sweep`,
`--mode reach`). JSON: `data/processed/models/rotation/wave_audit_2026-09-11.json`,
`.../wave_reachability_2026-09-11.json`.

Coverage: **8,502 team-games / 1,170,554 boundaries (2024)** and **10,640
team-games / 1,450,768 boundaries (2025)** -- every game with a complete
on-floor set on every possession, the same universe
`docs/tests/rotation_sub_hazard_audit_2026-09-10.md` measured. Every rate
carries a 95% Wilson interval; cells under 300 rows are UNDERPOWERED and are
labelled, never read as signal or as absence of signal.

---

## 0. What this audit is for

L30 closed round 4 with a mechanism verdict: *"a substitution model needs a
joint dead-ball draw, one Bernoulli per (team, boundary) for whether a wave
happens conditioned on `prev_end`, then a wave size and a composition from the
same per-player hazards, not five independent coins."* Round 4's own diagnosis
(experiments.md 11.11) put numbers on the failure: the fitted hazards reproduce
**every** marginal rate to about 1 pp and the arms built on them still
substitute **43% too often** (0.2058 per boundary against 0.1518), run **19.7
distinct lineups** per team-game against 14.8, and lose 11 pp at the
second-half tip when the reset has to be earned.

This document measures the joint object directly -- does a team substitute at
all, how many move, and who -- and then, per L25, computes **on paper** what a
wave model built from these tables and round 4's own fitted hazards produces,
BEFORE the round-5 pre-registration (`docs/models/rotation/experiments.md`
section 12) was written and before any arm was fitted.

### 0.1 Definitions

At every possession boundary k of a team-game -- the resolution the engine acts
at, and the only resolution CBBD supports (no `Substitution` rows at all in
2024, `features.md` section 3):

* **wave(k)**: the team's on-floor SET at possession k differs from k-1;
* **size(k)**: `|on(k-1) \ on(k)|`, the number of players who leave, which
  under the five-on-the-floor constraint is also the number who enter;
* **cell**: (`prev_end`, time cell, margin band, foul state), where `prev_end`
  is the possession's own `start_reason` and is exactly
  `engine.state.PREV_END_LEVELS`, the time cell is the audit's nine-way split,
  the margin band is `rotation.margin_bucket`, and the **foul state** is 1 when
  any player on the floor carries >= 4 personal fouls.

Descriptive, not predictive: the candidate pool is every player who appears in
the team-game and the foul state is that game's own `PersonalFoul` events. Both
are contemporaneous with the game and are correct for a MEASUREMENT of coaching
behaviour; the bake-off's training path uses the as-of pool and never these.

---

## 1. Does a team substitute at this dead ball?

### 1.1 By `prev_end` -- the dead-ball type (season 2024, 2025 in brackets)

| `prev_end` | P(wave) | 95% CI half-width | n boundaries |
|---|---:|---:|---:|
| **period_start** | **0.8698** (0.8557) | 0.0069 | 9,150 |
| made_FT | **0.3930** (0.4008) | 0.0030 | 104,778 |
| other | 0.1904 (0.1872) | 0.0109 | 5,000 |
| TOV | 0.1715 (0.1694) | 0.0017 | 196,378 |
| DREB | 0.1121 (0.1129) | 0.0010 | 419,232 |
| made_FG | 0.1076 (0.1089) | 0.0009 | 436,016 |

`prev_end` is the single strongest term and it spans **8x** between made_FG and
period_start. The two seasons agree to under 1.5 pp on every level.

### 1.2 By period, time cell, margin band, foul state (2024)

| factor | level | P(wave) | n |
|---|---|---:|---:|
| period | 1 | 0.1482 | 569,478 |
| | 2 | 0.1554 | 589,366 |
| | OT | 0.1439 | 11,710 |
| time cell | H1 20:00-10:00 | 0.1262 | 282,276 |
| | H1 10:00-00:00 | 0.1698 | 287,202 |
| | H2 20:00-16:00 | 0.1352 | 120,322 |
| | H2 16:00-12:00 | 0.1688 | 114,134 |
| | H2 12:00-08:00 | 0.1646 | 114,276 |
| | H2 08:00-04:00 | 0.1486 | 114,578 |
| | H2 04:00-02:00 | 0.1435 | 56,642 |
| | H2 02:00-00:00 | 0.1745 | 69,414 |
| | OT | 0.1439 | 11,710 |
| margin band | \|m\| <= 5 | 0.1425 | 557,574 |
| | \|m\| 6-15 | 0.1620 | 451,496 |
| | \|m\| > 15 | 0.1553 | 161,484 |
| foul state | no 4-foul player on the floor | 0.1504 | 1,088,884 |
| | a 4-foul player on the floor | 0.1707 | 81,670 |
| own team fouls | 0-2 | 0.1214 | 454,109 |
| | 3-5 | 0.1702 | 365,832 |
| | 6-8 | 0.1696 | 235,385 |
| | 9-11 | 0.1731 | 90,488 |
| | 12+ | 0.1898 | 24,740 |

Read together: the state factors are real but SMALL next to `prev_end` -- the
whole margin-band span is 2.0 pp and the foul-state span 2.0 pp against
`prev_end`'s 76 pp. The team-fouls gradient (+6.8 pp from 0-2 to 12+) is mostly
the same thing seen twice: team fouls buy free throws, and `made_FT` is the
high-wave stoppage. The nine time cells are non-monotone (0.126 -> 0.170 ->
0.135 -> 0.169 -> 0.165 -> 0.149 -> 0.144 -> 0.175), which is the media-timeout
grid showing through, so a linear time term cannot represent it and the cell
form is used.

### 1.3 The fitting cell

324 cells = 6 `prev_end` x 9 time cells x 3 margin bands x 2 foul states. In
2024, 243 are non-empty, **153 carry n >= 300**, and those hold **99.39%** of
all boundaries (2025: 249 / 157 / 99.42%). The remaining 0.6% of boundaries sit
in UNDERPOWERED cells and are handled by the two-level shrinkage of section 5,
never by reading a thin cell as if it were a rate.

---

## 2. Timeouts -- the signal exists, and it is the largest non-period one

CBBD pbp carries `OfficialTVTimeOut`, `ShortTimeOut` and `RegularTimeOut`:
**72,285 rows in 2024** and 72,742 in 2025, joined to a boundary on
(cbbd_game_id, period, secondsRemaining).

| | P(wave) | n boundaries | share of all waves |
|---|---:|---:|---:|
| no timeout at this stoppage | 0.1371 | 1,122,918 | 86.7% |
| a timeout at this stoppage | **0.4967** | 47,636 | **13.3%** |

Wave size at a timeout is bigger too: P(size >= 2) is **0.4146** at a timeout
against 0.3165 elsewhere (2024).

**It stays excluded from the model**, for the reason
`docs/models/rotation/experiments.md` 10.2 already recorded and round 4's
diagnosis 11.11 restated: the engine has no timeout model, so a wave
probability conditioned on timeouts could be fitted offline and could not be
evaluated in simulation. What the engine does carry is `prev_end`, and
`made_FT` at 0.393 already carries the part of the timeout effect that arrives
through free-throw stoppages. **The point of round 5 is that the timeout is no
longer needed to make substitutions bunch**: the bunching is now in the wave
draw itself, and section 6 measures how much of the 43% excess that removes.

---

## 3. How many players move

### 3.1 Wave size given a wave (2024; 2025 agrees to under 2 pp on every cell)

| `prev_end` | 1 | 2 | 3 | 4 | 5 | n waves |
|---|---:|---:|---:|---:|---:|---:|
| **period_start** | **0.4112** | **0.4117** | **0.1532** | 0.0227 | 0.0011 | 7,959 |
| DREB | 0.6788 | 0.2456 | 0.0627 | 0.0104 | 0.0025 | 47,005 |
| TOV | 0.6809 | 0.2460 | 0.0610 | 0.0097 | 0.0024 | 33,681 |
| made_FG | 0.6668 | 0.2505 | 0.0667 | 0.0126 | 0.0034 | 46,896 |
| made_FT | 0.7063 | 0.2299 | 0.0524 | 0.0092 | 0.0022 | 41,173 |
| other | 0.6922 | 0.2447 | 0.0525 | 0.0074 | 0.0032 | 952 |

Overall mean wave size **1.4246** (2025: 1.4516) and **29.77 substitutions per
team-game** (2025: 30.43).

| time cell | 1 | 2 | 3 | 4+ | n waves |
|---|---:|---:|---:|---:|---:|
| H1 20:00-10:00 | 0.5950 | 0.2983 | 0.0896 | 0.0171 | 35,611 |
| H1 10:00-00:00 | 0.6759 | 0.2508 | 0.0619 | 0.0114 | 48,776 |
| H2 20:00-16:00 | 0.6009 | 0.2915 | 0.0923 | 0.0153 | 16,266 |
| H2 16:00-12:00 | 0.6639 | 0.2604 | 0.0652 | 0.0105 | 19,262 |
| H2 12:00-08:00 | 0.6766 | 0.2511 | 0.0613 | 0.0110 | 18,806 |
| H2 08:00-04:00 | 0.7430 | 0.2078 | 0.0411 | 0.0081 | 17,021 |
| H2 04:00-02:00 | 0.7605 | 0.1797 | 0.0390 | 0.0208 | 8,129 |
| H2 02:00-00:00 | 0.7725 | 0.1680 | 0.0348 | 0.0247 | 12,110 |
| OT | 0.8843 | 0.1086 | 0.0042 | 0.0030 | 1,685 |

Two facts the round needs. **(1) The period boundary is not a bigger version of
an ordinary stoppage: it is a different distribution.** 0.41 / 0.41 / 0.15 is a
*two-player mean* swap (mean size 1.75) against 1.42 everywhere else, and
that -- not the wave probability alone -- is why the second-half reset is a
near-deterministic joint event. **(2) Waves shrink as the game ends**: 0.595 of
waves are single swaps in the first ten minutes against 0.773 in the last two
and 0.884 in overtime, i.e. late substitutions are surgical, which is the
mechanism behind the close-and-late starter share the gate reads.

---

## 4. Who moves

### 4.1 Starters among the leavers and the entrants, by wave size (2024)

Starters out, given a wave of that size:

| size | 0 starters | 1 | 2 | 3 | 4 | 5 | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3890 | 0.6110 | -- | -- | -- | -- | 119,121 |
| 2 | 0.2511 | 0.3749 | 0.3741 | -- | -- | -- | 44,551 |
| 3 | 0.2134 | 0.2592 | 0.2713 | 0.2560 | -- | -- | 11,561 |
| 4 | 0.2280 | 0.1667 | 0.2042 | 0.1950 | 0.2062 | -- | 1,974 |
| 5 | 0.1656 | 0.0893 | 0.1046 | 0.1285 | 0.2484 | 0.2636 | 459 |

Starters in:

| size | 0 | 1 | 2 | 3 | 4 | 5 | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5155 | 0.4845 | -- | -- | -- | -- | 119,121 |
| 2 | 0.2907 | 0.3650 | 0.3442 | -- | -- | -- | 44,551 |
| 3 | 0.2239 | 0.2139 | 0.2859 | 0.2764 | -- | -- | 11,561 |
| 5 | 0.5054 | 0.1024 | 0.0850 | 0.0871 | 0.0806 | 0.1394 | 459 |

A single swap takes a starter off 61% of the time and puts a starter on 48% of
the time; the net is the slow bleed of starter minutes the occupancy cells read.
Composition is **not** deterministic in either direction -- at size 2 the three
outcomes 0/1/2 starters out are 0.25 / 0.37 / 0.37 -- which is the fact section 6
turns into a model choice.

### 4.2 The leaver's profile, over boundaries where a wave happened (2024)

| quantity | bench players | starters |
|---|---:|---:|
| mean stint of those who LEAVE (min) | 3.919 (n=106,840) | 6.349 (n=146,257) |
| mean stint of those who STAY (min) | 3.457 (n=192,099) | 7.024 (n=443,134) |
| mean fouls of those who LEAVE | 0.804 | 1.035 |
| mean fouls of those who STAY | 0.673 | 0.917 |
| mean rest of those who ENTER (min) | 7.065 (n=125,145) | 3.203 (n=127,952) |
| mean rest of those who STAY OFF (min) | 9.740 (n=429,959) | 3.635 (n=168,370) |

Fouls separate leavers from stayers in the same direction for both classes
(+0.13 / +0.12 fouls), and rest separates entrants from the players left sitting
(-2.7 / -0.4 min). Stint separates them the expected way for bench players
(+0.46 min) and the **wrong** way for starters (-0.68 min): at a boundary where
someone is coming off, the starter who leaves has been on for *less* time than
the starters who stay, because the long-stint starters at that moment are the
ones the coach is protecting. That is a composition fact, not a fatigue fact,
and it is exactly what round 4's fitted hazards already encode -- which is why
round 5 keeps them unchanged as the composition rule rather than refitting.

---

## 5. Do the two teams substitute together?

2x2 over the 580,702 boundaries of 2024 where both teams' on-floor sets are
observed at the same possession index, split on whether the boundary is a
period start:

| | P(home wave) | P(away wave) | P(both) | independent | lift | phi | odds ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| NOT a period start | 0.1433 | 0.1490 | **0.0749** | 0.0213 | **3.51** | 0.430 | 11.59 |
| a period start | 0.8662 | 0.8734 | 0.7801 | 0.7566 | 1.03 | 0.208 | 3.93 |

2025 reproduces it: 0.0766 observed against 0.0219 independent, lift 3.49,
phi 0.433.

Part of that lift is the shared cell -- both teams face the same `prev_end`, the
same clock and the same \|margin\| at the same boundary -- so the number that
matters is the **residual** dependence after conditioning on each team's own
fitted cell probability. Fitting the round-5 wave table on all of 2024 and
comparing:

| quantity | value |
|---|---:|
| observed P(both), non-period boundaries | 0.0749 |
| E\[p_A p_B\] under the fitted cells (independent) | 0.0296 |
| E\[min(p_A, p_B)\] (perfectly coupled) | 0.1451 |
| **residual lift** (observed / independent) | **2.53** |
| **fitted coupling rho** | **0.393** |

So conditioning on the cell explains about a third of the raw lift (0.0213 ->
0.0296) and a real coupling of **rho = 0.39** remains: two fifths of dead balls
are a *shared* opportunity at which both benches move together. The cause is
visible in section 2 -- a timeout is one event for both teams -- and this is the
same information the engine cannot condition on directly. A coupling scalar is
how that information enters without a timeout model, and it is what round 5's
W3 arm tests.

**It changes no single team's marginal**: the construction in section 6 draws
one shared uniform and uses it for both teams with probability rho, and a
uniform shared with the other team is still a uniform.

---

## 6. The L25 reachability check, computed BEFORE the pre-registration

L25: *"before fitting any knob, compute the fitted log-odds separation between
the classes the knob must move apart in the target state and report the target
cell's reachable range over the whole grid; if the target is outside it, the
family is wrong and no fitting will find it."*

The wave family's analogue of "the whole grid" is its **composition rule**. The
wave probability and the size distribution are measured, not chosen; what is
genuinely open is how the `size` leavers are picked out of five and the `size`
entrants out of the bench, given round 4's fitted per-player hazards. There are
exactly four knob-free rules -- rank or draw, on each of the two sides -- so the
grid is enumerable and is enumerated:

* **rank**: the `size` largest `p_out` leave, the `size` largest `p_in` enter;
* **draw**: an Efraimidis-Spirakis race weighted by `p/(1-p)`, i.e. weighted
  sampling without replacement -- round 4's own entry rule.

Probe (`--mode reach`): the audit's 2024 wave and size tables, round 4's
**static** fitted hazards (`rotation_v4_sub_static.json`, unchanged), run
forward on 400 random 2025 games with the REAL starting fives, the REAL
participant pools and the REAL foul sequence, under the five-on-the-floor
constraint. No hard second-half reset, so the H2-tip row below is the *earned*
reset and is not the number the arms will show.

| cell | ACTUAL (same 400 games) | rank/rank (W1) | draw/draw (W2) | rank exit, draw entry (W4) | draw exit, rank entry (W5) |
|---|---:|---:|---:|---:|---:|
| **substitutions per boundary** | **0.1555** | **0.1516** | 0.1528 | 0.1525 | 0.1528 |
| **distinct lineups per team-game** | **15.37** | 10.85 | 17.54 | **14.49** | **14.46** |
| final 8:00, \|m\| <= 5 | 0.7388 | 0.7878 | 0.6896 | 0.7656 | 0.7306 |
| final 8:00, \|m\| 6-15 | 0.7135 | 0.7606 | 0.6685 | 0.7304 | 0.7050 |
| final 8:00, \|m\| > 15 | 0.5120 | 0.5578 | 0.5203 | 0.5037 | 0.5890 |
| H1 20:00-10:00, \|m\| <= 5 | 0.7722 | 0.7761 | 0.7666 | 0.7648 | 0.7783 |
| H2 tip (earned), \|m\| <= 5 / 6-15 / > 15 | 0.9732 / 0.9574 / 0.9508 | 0.819 / 0.797 / 0.776 | 0.808 / 0.809 / 0.751 | 0.805 / 0.806 / 0.780 | 0.812 / 0.809 / 0.805 |

**What the probe establishes.**

1. **The joint draw fixes the over-substitution by construction, and it is the
   whole of the fix.** 0.1516-0.1528 against a real 0.1555 and round 4's
   0.2058: the 43% excess is gone because the number of boundaries carrying a
   change is now a fitted quantity instead of an emergent one. Nothing else in
   the model changed -- the same hazards, the same five-on-the-floor rule.

2. **The target sits strictly INSIDE the composition grid, and the two uniform
   rules are its endpoints.** Distinct lineups: 10.85 (rank/rank) < **15.37
   actual** < 17.54 (draw/draw). Close-and-late starter share: 0.6896
   (draw/draw) < **0.7388 actual** < 0.7878 (rank/rank). A deterministic
   composition recycles the same five and under-churns; a fully drawn one
   over-churns. Both MIXED rules land within 0.9 of the real lineup count, and
   rank-exit/draw-entry lands within 2.7 pp on all three late bands while
   draw-exit/rank-entry misses the blowout band by 7.7 pp.

3. **The opening-ten-minutes close cell is reached by every rule** (0.765-0.778
   against 0.772), against R2's -18.3 pp and H1's -3.9 pp in round 4.

4. **The earned second-half reset still fails**, at 0.78-0.81 against 0.95-0.97.
   The wave draw raises it from round 4's H2 (0.848-0.855) but not to the tip
   value, so the hard reset stays part of every round-5 arm and the earned-reset
   question is **closed**, not re-opened: two families in two rounds have now
   failed it from below.

**What the probe does NOT establish, stated plainly.** It uses the real starting
five and the real participant pool. Round 3 measured what that is worth: the
model's as-of starter set overlaps the real one on 4.58 of 5, and re-grading the
actual sequence with the model's starter set puts the close-late benchmark at
0.7215 against the 0.7491 the gate scores. The probe also uses the real foul
sequence rather than simulated fouls. The arms must still find 2-3 pp the as-of
starter set does not have, and the gate is scored as pre-registered, each side
against its own starting five.

---

## 7. What this audit hands the round-5 pre-registration

1. `prev_end` is the wave model's dominant term (0.108 -> 0.870, an 8x span) and
   is free to the engine; margin band and foul state are real but 2 pp terms.
2. The size distribution is 0.68 / 0.25 / 0.063 / 0.010 / 0.0025 overall and a
   **different distribution** at a period boundary (0.41 / 0.41 / 0.15), which is
   why a period boundary is a joint event and not a big ordinary one.
3. A timeout triples the wave probability (0.137 -> 0.497) and carries 13.3% of
   all waves; it stays out of the model for engine expressibility, and round 5
   is the test of whether the wave draw makes it unnecessary.
4. The two teams' waves are genuinely coupled at **rho = 0.39** after
   conditioning on the shared cell (residual lift 2.53), which is what W3 tests.
5. The composition grid has four knob-free members; the target is strictly
   inside it, with rank/rank and draw/draw as the two endpoints and the two
   mixed rules within 0.9 lineups of the truth. Every arm in section 6's table
   is therefore pre-registered, not only the two endpoints -- and the reason is
   recorded here, before the fit.
