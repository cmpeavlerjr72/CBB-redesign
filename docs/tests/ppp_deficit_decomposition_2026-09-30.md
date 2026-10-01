# PPP deficit in the COMB stack: closed decomposition, owners, and a fed-state defect (lane I, 2026-09-30)

Lane I, overnight 2026-09-30 (23:09-02:30 EDT). **DIAGNOSTIC plus one
pre-registered fix round. Nothing adopted, no served default changed.** COMB =
the Decision 11 set as run at full size: `ENGINE_CLOCK=v5b_r6L2_glat_pmean` +
`ENGINE_SHOT_BLOCK=K2_Ocell` + `ENGINE_FOUL_JOINT=R8b` on the v3 inputs.
Verified truth (`CBB_TRUTH=verified_v1`) everywhere. The same decomposition on
COMB9 (R9ao3 as the foul member, `v3full_COMB9_s200_o0`) gives total -1.476. Its
make channels are the same: rim -1.148, three -0.433, jumper -0.025. The rest:
possessions +0.887, OREB% -0.693, FT% -0.345, FT rate -0.174 with the trip term
+0.116, TOV -0.230, mix -0.189, boards +0.762. Residual 1e-13.

## 0. Answer in five lines

1. The COMB total bias (-1.264 points per game on the 5,700 graded games with a
   box) closes exactly into box channels: **make rates -1.60** (rim -1.15,
   three -0.43, jumper -0.02), **OREB% -0.71**, FT% -0.35, TOV -0.21, shot mix
   -0.19, FT rate +0.03, live boards per possession +0.77 (more misses), and
   **possessions +1.00**. Residual 1e-14.
2. **fg_make is right OFFLINE** on held-out 2024-25 real shots (rim mean p
   0.5840 vs realised 0.5844; three 0.3376 vs 0.3378) **and wrong in the sim**
   (rim 0.5708). The sim's own model, fed the real attempts' features in the
   same (game, offence) cells, closes the rim gap. The carrier is the **state
   the engine feeds it**, not who shoots (shooter mix: -0.24 points in all) and
   not the team inputs (0.00).
3. Inside state, it is the **fed chance timing**: `chance_elapsed_s` at chance 1
   is the whole possession's duration instead of the time to the first shot,
   and at chance 2+ it is a constant median (3 s) instead of the real spread
   (rim putbacks 0/0/1/3/10 s at the 10/25/50/75/90% points). Swapping elapsed
   time and the transition flag to real values moves rim +1.36 pp and three
   +0.36 pp: about **-1.7 points of total per game**, i.e. the whole make-rate
   channel. This defect is in the engine's state feed (`loop.py`), not in a model.
4. The OREB% channel is the rebound model's own level (OFFLINE on real states
   0.2898 vs 0.2992; the anchor `O` / Stage-B `TO` work owns it). The shot mix and
   TOV channels are possession_outcome's own level on real states (jumper share
   +0.57 pp, three -0.49 pp). FT% is right offline (0.7198 vs 0.7212) and
   0.56 pp low in the sim. Swapping the real attempts' features closes 90% of
   that: about half is who goes to the line (prior-season block), about half is
   the foul state (bonus share 0.556 vs 0.456).
5. Fix round `chance_time` was pre-registered in three steps:
   - round 1 (C2, C12) at `44ae240`;
   - addendum 1b (arm `K`, the class-conditional chance-1 time) at `902c2f0`;
   - addendum 1c (arm `KD`, also conditioned on the drawn duration) at `39eb085`.

   Every arm beats `R` on the registered primary (KD by 99.6 floors; F1 confirms).
   Every arm **fails a registered per-row guard** (log loss or calibration deciles)
   that a drawn feed cannot pass by construction (section 6.2). By the registered
   rules: **no eligible arm, `R` stands.** POST-HOC reads for the PM's ruling:
   - **Full size, `COMB9+K` vs `COMB9`** (laneI_1): G9 total bias -1.49 -> **-0.56
     (PASS)**, 32 floors. eFG 0.5009 -> 0.5068 (actual 0.5086). G9 tier and pred-total
     cells improve.
   - The cost of K: the G5 total SD ratio (0.909 -> 0.893) and h/a corr (0.113 ->
     0.092) move AWAY. K removes the served feed's pace-to-make link.
   - **Full size, `COMB9+KD` vs `COMB9`** (laneI_2): G9 total bias -1.49 -> **-0.35
     (PASS)**, +1.14, 39 floors. eFG 0.5081 (actual 0.5086).
   - KD's G5 cost is small: total SD ratio -0.0026 and h/a corr -0.0037, 1.6 and 2.5
     floors. K's was 10 and 14 floors.
   - After KD the deficit left is OREB% -0.75, FT% -0.34, mix -0.21 and TOV -0.18,
     against possessions +0.65.

---

## 1. Inputs and method

    results/engine_v0/v3full_COMB_s200_o0, v3full_S0_s200_o0   full-size box runs, 200 seeds x 5,710 games
    results/ppp_decomp/tap_comb/                               fg_make / FT input tap, COMB, 500 verified stride games x 32 seeds
    results/home_site/fg/preds_S0_F2_s{0,1}.parquet            lane G's reproduction of the served B1 on held-out F2 rows
    results/home_site/{rb,po}_realstate_v1.log                 lane G's real-state replays of rebound and possession_outcome
    results/home_site/ft/preds_FT0_F2_s0.parquet               free-throw held-out predictions (served spec)
    data/processed/models/fg_make/design_v2_shotshooter.parquet  the fg_make training / held-out table

    scripts/diag_ppp_decomp_v1.py        closed box-channel decomposition (any run dir)
    scripts/diag_ppp_tap_v1.py           in-process tap (fg_make + FT inputs); v3 event block swapped in as run_engine_v3evb_v1.py does
    scripts/diag_ppp_tap_grade_v1.py     bit-identity check + the sim-to-real chain per shot class
    scripts/diag_ppp_tap_grade_feat_v1.py   per-feature swaps
    results/ppp_decomp/*.json            machine-readable output (gitignored)

**Decomposition.** Per team side, pooled over a cell, points = sum over
components {rim, jumper, three, FT} of `N * f * s_k * m_k * v_k` (FG) and
`N * f * r * q` (FT), with `N` the box possession estimator on both sides (like
for like; the engine's count is not used here), `f` = FGA/N, `s_k` shot share,
`m_k` make rate, `r` = FTA/FGA, `q` = FT%. LMDI-I splits sim minus actual into
factor channels with zero residual. `f` is split exactly once more with the
estimator identity `f = (1 + o - t)/(1 + 0.44 r)` into FT-trip, TOV and OREB parts,
and the OREB part into OREB% and live boards per possession. The actual rim /
jumper split applies the event layer's per-team-game rim share to the box 2PA
and 2PM. Per game = both teams, so the channels sum to the G9 total bias.

**Tap proof.** The tap's games rows equal the box COMB run's on every shared
(game, seed): **16,000 rows, 26 columns, 0 mismatching cells**. The tap's
recorded p equals the served artifacts re-applied to the recorded rows.

---

## 2. The closed decomposition (full size, COMB, 5,700 games x 200 seeds)

| channel | points per game (sim - actual) | owner |
|---|---:|---|
| possessions (box estimator, both sides) | **+0.995** | clock / G1 (lanes B, H) |
| rim make rate (0.5708 vs 0.5841) | **-1.151** | **engine chance-timing feed** (section 4) |
| three make rate (0.3349 vs 0.3381) | **-0.431** | **engine chance-timing feed** |
| jumper make rate (0.3913 vs 0.3917) | -0.021 | (none) |
| rim share (0.3713 vs 0.3734) | -0.278 | possession_outcome level |
| jumper share (0.2417 vs 0.2360) | +0.521 | possession_outcome level |
| three share (0.3869 vs 0.3906) | -0.428 | possession_outcome level |
| FT points per FGA via FTA/FGA (0.3306 vs 0.3295) | +0.086 | foul (R8b) |
| FT% (0.7123 vs 0.7213) | **-0.346** | FT-trip shooter allocation + foul state (section 5.4) |
| FGA per possession: FT-trip term | -0.058 | foul |
| FGA per possession: TOV (0.1753 vs 0.1739 per poss) | -0.207 | possession_outcome level |
| FGA per possession: OREB% (0.2892 vs 0.2984) | **-0.709** | rebound level |
| FGA per possession: live boards per possession | +0.766 | consequence of the extra misses |
| **sum of channels** | **-1.2603** | |
| sim score minus sim box points | 0.0000 | |
| actual box points minus verified final | -0.0039 | |
| **total bias** | **-1.2641** | residual 1.4e-14 |

(The gate reads -1.274 on 5,705 games; 5 graded games have no box row.)

Net reading: the make-rate channels (-1.60), OREB% (-0.71, partly returned
as +0.77 through more misses to rebound), FT% (-0.35), TOV (-0.21) and mix
(-0.19) are a PPP deficit of about -2.3 points that the +1.0 possessions channel
partly hides.

### 2.1 COMB vs S0 (both full size): what the Decision 11 set moved

| channel | S0 | COMB | move |
|---|---:|---:|---:|
| possessions | +3.014 | +0.995 | -2.019 |
| FT rate (points) + FT-trip term | -0.354 | +0.028 | +0.382 |
| OREB% | -1.196 | -0.709 | +0.487 |
| live boards per possession | +0.578 | +0.766 | +0.188 |
| rim / three / jumper make | -1.068 / -0.422 / -0.010 | -1.151 / -0.431 / -0.021 | -0.083 / -0.010 / -0.011 |
| shot mix (three terms) | -0.178 | -0.185 | -0.007 |
| FT% | -0.370 | -0.346 | +0.024 |
| TOV | -0.237 | -0.207 | +0.030 |
| **total** | **-0.246** | **-1.264** | **-1.018** |

COMB repaired what it targeted (the count, FT trips, OREB%). The make-rate
deficit (-1.5 points) was already there in S0, hidden by +3.0 points of extra
possessions.

---

## 3. Multi-level evidence (full size)

Per-game rows are both teams; site and team-tier rows are one team side per
row (a neutral row carries both sides of a neutral game). Months with fewer than
300 games are **UNDERPOWERED**.

### 3.1 By month

| month | games | total bias | possessions | rim make | three make | OREB% | FT% | TOV | eFG gap (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Nov | 1,219 | -3.34 | +0.88 | -1.23 | -0.24 | -1.00 | -0.91 | -0.44 | -0.98 |
| Dec | 915 | -1.27 | +0.84 | -1.20 | -0.54 | -0.84 | -0.27 | +0.30 | -0.99 |
| Jan | 1,420 | +0.04 | +1.37 | -0.71 | -0.19 | -0.98 | -0.12 | +0.03 | -0.38 |
| Feb | 1,364 | -0.93 | +1.09 | -1.38 | -0.79 | -0.26 | -0.31 | -0.28 | -1.03 |
| Mar | 765 | -0.80 | +0.60 | -1.37 | -0.42 | -0.40 | -0.10 | -0.72 | -0.39 |
| Apr | 17 | -8.51 | | | | | | | **UNDERPOWERED** |

The rim-make channel is negative in every powered month (-0.71 to -1.38).
November is worst overall (-3.34), with a three-share term of -2.34 and an FT
rate term of -1.26 on top (early-season as-of inputs).

### 3.2 By site (per team side)

| site | team-games | bias per side | possessions | rim make | three make | OREB% | eFG gap (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| home | 4,964 | -0.70 | +0.53 | -0.60 | -0.21 | -0.40 | -0.78 |
| away | 4,964 | -0.60 | +0.42 | -0.60 | -0.26 | -0.28 | -0.85 |
| neutral (both sides) | 736 games | -1.06 | +1.33 | -0.86 | -0.25 | -0.87 | -0.47 |

### 3.3 By team strength tercile (season margin tercile, grading only)

Offence tercile, per team side:

| offence tier | team-games | bias per side | rim make | three make | OREB% | eFG gap (pp) | rim make gap (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| bottom | 2,962 | -0.05 | -0.58 | -0.03 | -0.21 | -0.34 | -1.13 |
| middle | 3,064 | -0.70 | -0.73 | -0.26 | -0.52 | -0.75 | -1.39 |
| top | 3,118 | -1.58 | -0.84 | -0.50 | -0.60 | -1.16 | -1.47 |

G9's own cut (home team's tier, per game): bottom -1.27, middle -0.33, top -2.01;
rim make -1.41 / -0.97 / -1.13. The deficit grows with offence quality: the top
tercile loses 1.16 pp of eFG.

### 3.4 Per game, per team, responsiveness

- Per game (5,700): mean -1.26, median -0.52, SD 17.1, P(sim below actual) 0.513.
- Per team (offence, 364 teams with >= 10 games; each cell UNDERPOWERED alone):
  median points bias -0.54 per game, 58% of teams below; median eFG gap
  -0.59 pp, **68% of teams below**.
- Responsiveness, teams by realised season eFG quintile: actual 0.466 / 0.493 /
  0.506 / 0.522 / 0.551, sim 0.476 / 0.492 / 0.499 / 0.508 / 0.524. Span 0.048 vs
  0.085 (ratio 0.56; sorting on realised values alone would give about 0.85).
  The sim's team eFG is compressed, with the top quintile 2.7 pp short.
  The offline offence responsiveness of the served fg_make (predicted over realised span, by
  `off_make_c` quintile, lane G's S0 reproduction) is 1.03 rim, 0.88 jumper, **0.74 three**.
  Part of the tier slope is therefore fg_make's own three-point responsiveness, offline; it is
  not the feed. The rest is not decomposed here.

### 3.5 By predicted-total tercile (per game)

low -1.63, mid -1.47, high -0.70; rim make -1.22 / -1.31 / -0.91.

---

## 4. Offline vs in the sim: the make-rate channel (item 2 of the brief)

### 4.1 Offline, on held-out real 2024-25 shots (served `round4_B1` spec, both seeds)

| class | n | realised | mean p (seed 0) | mean p (seed 1) | sim (COMB, full size) |
|---|---:|---:|---:|---:|---:|
| rim | 235,348 | 0.5844 | 0.5840 | 0.5841 | **0.5708** |
| jumper | 148,773 | 0.3905 | 0.3907 | 0.3905 | 0.3913 |
| three | 246,783 | 0.3378 | 0.3376 | 0.3372 | 0.3349 |

By month the offline rim gap is +0.45 / -0.35 / -0.19 / -0.10 / -0.09 pp (Jan, Feb,
Mar, Nov, Dec). By site it is -0.07 / -0.07 / +0.24 pp (away, home, neutral). By shooter
quintile it is within +/-0.31 pp. There is no season drift in the fg_make level to
anchor: anchor `O` on fg_make would move nothing.

### 4.2 In the sim: the closed chain (500 verified stride games x 32 seeds, same games on both sides)

Each step replaces one feature group of every sim attempt with the values of a
real attempt drawn from the same (game, offence) cell, scored by the same served
artifacts.

| step (rim) | make | move (pp) | points per game |
|---|---:|---:|---:|
| sim realised | 0.5719 | | |
| sim mean p | 0.5715 | -0.04 | (RNG) |
| + shooter feature swapped to the real shooters' | 0.5733 | +0.17 | -0.14 |
| + state swapped (chance number, elapsed, transition, clock, bonus) | 0.5858 | **+1.25** | **-1.08** |
| + team features swapped (serve -> train) | 0.5858 | 0.00 | 0.00 |
| real mean p, real cell weights (offline) | 0.5851 | -0.07 | +0.06 |
| real realised (500-game sample) | 0.5903 | +0.53 | sample noise (full season +0.03) |

Three: shooter +0.08 pp (-0.10), state +0.23 pp (-0.32). Jumper: shooter +0.06,
state -0.17 pp (+0.09). Reverse order (state first): rim state +1.32, then shooter
+0.10. The path dependence is small.

### 4.3 Which state feature (single swaps, matched within chance bucket 1 / 2+)

| swap | rim pp | rim chance 1 / 2+ | three pp | jumper pp | points per game (all classes) |
|---|---:|---|---:|---:|---:|
| `chance_elapsed_s` | **+1.02** | +0.67 / **+2.59** | **+0.36** | -0.04 | **-1.36** |
| `is_transition_f` | -0.25 | -0.31 / 0 | 0.00 | -0.01 | +0.23 |
| elapsed + transition | **+1.36** | +1.09 / +2.59 | +0.36 | +0.01 | **-1.68** |
| period, seconds remaining | -0.23 | | -0.07 | -0.07 | +0.33 |
| in_bonus | -0.06 | | -0.06 | +0.03 | +0.11 |
| chance_number | 0.00 | | 0.00 | 0.00 | 0.00 |
| shooter | +0.12 | +0.05 / +0.42 | +0.08 | +0.06 | -0.24 |

Fed vs real (sim tap vs real rows on the same games):

| quantity | rim sim / real | jumper sim / real | three sim / real |
|---|---|---|---|
| chance-1 elapsed, 10/25/50/75/90% (s) | 4/9/16/23/30 vs 4/7/14/21/26 | 8/13/18/25/31 vs 8/13/18/24/29 | 6/11/18/24/30 vs 5/9/16/22/27 |
| chance-2+ elapsed, 10/25/50/75/90% (s) | 2/3/3/3/3 vs **0/0/1/3/10** | 2/3/3/3/3 vs 1/2/6/11/16 | 2/3/3/3/3 vs 1/2/4/9/15 |
| transition share of attempts | 0.171 vs 0.219 | 0.059 vs 0.089 | 0.102 vs 0.168 |
| mean p, chance 2+ | 0.583 vs 0.610 | 0.403 vs 0.383 | 0.353 vs 0.345 |
| attempt share on chance 2+ | 0.188 vs 0.189 | 0.101 vs 0.105 | 0.100 vs 0.104 |
| mean p, last 5 min of the 2nd half | 0.600 vs 0.604 | 0.399 vs 0.393 | 0.310 vs 0.313 |

**Mechanism (code).** `loop.py` feeds, at chance 1, `chance_elapsed_s = used`
(the whole possession's drawn duration) and `is_transition = (dur <= 8) & prev
in {DREB, TOV}`. The training tables measure chance 1 only: fg_make
(`fg_make.chance_state`) uses chance start to the shot, and possession_outcome's
chance rows (`possessions.py`, chance level) use the chance-1 duration. Every
possession that continues after an offensive rebound therefore feeds a
too-long, too-rarely-transition first shot. At chance 2+ the feed is
`ce_lut`, the training median by chance number (3 s, 2 s), a constant where the
model is strongly non-linear: rim putbacks at 0-1 s make far more often than the
3-s median says. Mean of p at the median is not the mean of p.

**Verdict for item 2:** right offline, wrong in the sim, through the state the
engine feeds. Shooter identity: -0.24 points in all. Team inputs: 0.00.

### 4.4 Who is shooting (the 09-11 bench-heavy finding)

Sim attempts by real shooter-dev quintile: rim 0.212 / 0.201 / 0.210 / 0.190 /
0.187 (real 0.2 each); three 0.212 / 0.202 / 0.206 / 0.187 / 0.193. The sim's
shooters are slightly weaker, in the direction of the bench-heavy finding, but
the swap is worth only rim +0.12 pp, three +0.08 pp and jumper +0.06 pp, about
**-0.24 points per game**. It is not the PPP owner.

---

## 5. The other channels, offline vs sim

### 5.1 OREB% (-0.71 points): the rebound model's own level

Lane G's real-state replay of the served rebound model (`results/home_site/rb_realstate_v1.log`):
mean predicted 0.2898 vs realised 0.2992 on 388,338 live misses. The sim's 0.2892
vs 0.2984 matches that offline level gap. It is the season-drift level that
`season_drift/experiments.md` measured (rebound `R` held-out -1.14 pp, anchor `O`
-0.08 pp) and that Stage B's `TO` arm carries. **Owner: rebound (anchor `O` / `TO`).**
The +0.77 live-boards term is the mechanical return from the extra misses.

### 5.2 Shot mix (-0.19 net) and TOV (-0.21): possession_outcome's own level

Lane G's real-state replay (`po_realstate_v1.log`), pooled over chances: rim
0.3730 vs 0.3738, jumper 0.2417 vs 0.2360, three 0.3854 vs 0.3903. These are
the sim's own shares to 0.0016. **Owner: possession_outcome level** (Stage B's
`TO` carries anchor `O` for PO). The transition flag that `chance_time` C12
repairs also feeds PO `first` (section 6).

### 5.3 Possessions (+1.00): the clock / G1 owners

See `docs/tests/g1_g5_possessions_corr_diagnostic_2026-09-30.md`. COMB's count is
still +1.10 over the box estimator.

### 5.4 FT% (-0.35): right offline, low in the sim; half shooter mix, half state

`scripts/diag_ppp_ft_swap_v1.py`. Real 2024-25 attempts are built with the served features
(`FT.build_ft_design`, as the FT trainer builds them) and scored by the served S1 artifacts. On
the full season the mean p is 0.7198 against 0.7212 realised. The sample is the same 500 games;
neutral-site games are skipped because the tap records the side only through `site_home`. There,
the sim's FT mean p is 0.7121 (553,532 attempts) against 0.7177 on the real attempts, a gap of
-0.56 pp. Swaps come from real attempts of the same (game, shooting team):

| swap | FT mean p move (pp) |
|---|---:|
| shooter block (as-of FT%, attempts, prior-season FT%, has prior season) | +0.28 |
| of which prior-season FT% alone / has-prior-season alone | +0.48 / +0.12 |
| of which as-of FT% alone / as-of attempts alone | -0.00 / -0.10 |
| state (seconds remaining, period, score diff, bonus) | +0.32 |
| of which score diff alone / bonus alone | +0.13 / -0.04 |
| everything (= the real rows' features) | **+0.50** (90% of the gap) |

Sim vs real means at the line:

| feature | sim | real |
|---|---:|---:|
| has a prior season | 0.691 | 0.734 |
| prior-season FT% (centred) | -0.0108 | -0.0082 |
| in bonus | **0.556** | 0.456 |
| score diff (shooting team) | +0.28 | +0.90 |
| seconds remaining | 487 | 465 |

**Reading:**
- The sim sends more newcomers (no prior season) and weaker prior-season shooters to
  the line, at the same in-season as-of FT%. That is the usage / foul-drawer
  allocation for FT trips (about half).
- Its FT trips come too often in the bonus and from the wrong side of the score
  (state, about half). That is the foul-accrual and late-game owners.
- The FT model itself is right.

### 5.5 Second chances, and-ones, late game, end of half

- **Second chances:** the attempt share on chance 2+ is right (rim 0.188 vs 0.189).
  Their make rate is low through the median feed (rim mean p 0.583 vs 0.610).
  Chance-2+ rim attempts carry about +2.6 pp of the rim gap, about -0.45 points.
- **And-ones:** inside the FT-rate channel (r); not separated. Lane C's round 9
  owns their rate.
- **Late game:** the last 5 minutes of the 2nd half carry the same attempt share as
  reality (rim 0.135 vs 0.135), and mean p is close (0.600 vs 0.604). Not a channel.
- **End of half:** not decomposed separately. The clock's horn possessions are in
  the G1 doc.

---

## 6. Fix round: `chance_time` round 1, addenda 1b and 1c

### 6.1 Pre-registration

`docs/models/chance_time/experiments.md` section 1, commit `44ae240` (23:37 EDT),
plus the ledger row `bafee9d`, both BEFORE any table was built or arm run.

- **R:** served.
- **C2:** chance-2+ elapsed drawn from the training distribution by
  (chance bucket x shot class).
- **C12:** C2 plus chance-1 elapsed and transition from a chance-1 time drawn
  conditional on the possession duration and start group.

Tables come from the training seasons only (`scripts/build_chance_time_lut_v1.py`;
F2 2022-2024, F1 2022-2023). Draws come from their own rng family.

**Addendum 1b** (section 3 there, commit `902c2f0`, 00:05 EDT, before the
arm was built) adds:

- **K:** chance-2+ as C2. Chance 1, at the fg_make call only: elapsed drawn from
  the training distribution of the design's chance-1 elapsed by (shot class x
  start group), clipped to the period clock, with fg_make's transition flag
  derived from it. possession_outcome is fed as served.
- Because round 1 showed its two guards were mis-specified, the addendum
  replaced them, labelled POST-HOC: the calibration-decile gate and a symmetric
  slope.

**Addendum 1c** (section 6 there, commit `39eb085`, 00:55 EDT, before KD was built)
adds:

- **KD:** K with the chance-1 table also conditioned on the possession's drawn
  duration (round-1 bins). It is motivated by K's full-size G5 cost (section 6.4).

### 6.2 Offline result (F2, held-out 2024-25 real states, 627,859 of 630,904 rows joined to their possession = 99.5%)

| arm | G_feed (pts/game) | rim gap pp (c1 / c2+) | jumper gap pp | three gap pp | fed - true log loss (rim / jump / three) | calib worst decile pp (rim / jump / three) | slope rim / jump / three |
|---|---:|---|---:|---:|---|---|---|
| R | 1.973 | -1.31 (-1.02 / -2.56) | -0.48 | -0.43 | -0.0187 / -0.0109 / -0.0075 | 12.5 / 8.1 / 6.7 | 1.120 / 0.903 / 0.746 |
| C2 | 1.819 | -0.84 (-1.02 / -0.06) | -0.69 | -0.53 | -0.0182 / -0.0110 / -0.0074 | 9.7 / 8.3 / 6.3 | 1.050 / 0.912 / 0.743 |
| C12 | 1.314 | -0.65 (-0.79 / -0.06) | -0.45 | -0.38 | -0.0168 / -0.0096 / -0.0063 | 9.2 / 7.4 / 5.4 | 1.049 / 0.901 / 0.749 |
| **K** | **0.180** | **-0.18 (-0.19 / -0.10)** | +0.04 | -0.01 | +0.0095 / +0.0023 / +0.0011 | 5.8 / 3.7 / 3.0 | 1.013 / 0.851 / 0.743 |
| **KD** | **0.149** | -0.08 (-0.08 / -0.10) | -0.02 | -0.05 | -0.0150 / -0.0086 / -0.0047 (leak) | 7.8 / 6.8 / 4.6 | 1.048 / 0.895 / 0.767 |

Floors: C2 0.0112 (reseed 0.0071, 2 x boot SE 0.0112), C12 0.0147, K 0.0316, KD 0.0183. Primary:
**C2 beats R by 13.7 floors, C12 by 44.7, K by 56.7, KD by 99.6.** F1 parity for KD:
0.20 s / 0.0025. KD fails the calibration-decile gate (7.8 / 6.8 / 4.6 pp) and passes the
symmetric slope. K's signed gap by month is
-0.39 / -0.25 / -0.03 / +0.04 / -0.12 points (Nov, Dec, Jan, Feb, Mar); R's is -1.8 to -2.1 in
every month and site.

**Registered guards, as written:**

- **Fed log loss not worse than R's by more than 1.1e-4 in any class.** FAIL for
  both arms (rim: C2 +0.00055, C12 +0.0019 vs R).
- **Responsiveness not below R's minus 0.05.** FAIL for both on rim (R 1.120;
  C2 1.050, C12 1.049, threshold 1.070). Jumper and three pass.
- **F1 confirmation (model-free quantile parity).** See section 6.2.1.

**K under the addendum rule:**
- Calibration-decile gate: FAIL in all classes (5.8 / 3.7 / 3.0 pp against 2.0).
- Symmetric slope: FAIL on the jumper by 0.002; passes on rim and three.

**Decision under the registered rules: no eligible arm, `R` stands.**

**Why the two failing guards are mis-specified (stated for the PM, not used to
select).** Every fed log loss is BELOW the true-feature log loss (rim
0.6485 vs 0.6672). The offline replay builds R's and C12's chance-1 feed from the
REAL possession duration, which contains the future: a missed first shot that is
rebounded by the offence has a long possession, so feeding it lowers p exactly
on the misses. R feeds that duration raw and leaks the most, so "not worse
than R's log loss" rewards the leak. In the engine the duration is drawn before
the chance cascade and carries no such information. The one-sided responsiveness
line reads a move from 1.12 toward 1.0 as a loss (the same defect the PM ruled
on for FT technicals, `season_drift/experiments.md` s3 item 2). A draw-based feed
also cannot beat a constant on log loss when neither is informative about the
row; only the mean (the level) can be repaired, and that is the primary.

The addendum's calibration-decile gate has the same flaw. A value drawn
independently of the row's true, engine-unobservable state spreads p without
information about the outcome. That dilutes the deciles while the level stays
right. Every drawn or leaky feed fails the gate (R at 12.5 pp); only the true
features pass. What a sim feed can be held to is the fed value's distribution
given what the engine knows (F1 parity) and the level (G_feed).

#### 6.2.1 F1 confirmation (tables 2022-2023 applied to 2024 real states)

Mean absolute fed-vs-real elapsed quantile gap (3 classes x 2 chance buckets x 5 quantiles) /
chance-1 transition-share gap: **R 3.40 s / 0.055, C2 1.43 s / 0.055, C12 1.03 s / 0.045,
K 0.27 s / 0.005, KD 0.20 s / 0.0025.** Every arm confirms on F1; K and KD are almost exact.

### 6.3 POST-HOC closed loop (NOT a registered selection; outside the rule)

Wired default-off as `ENGINE_CHANCE_TIME=C2|C12|K` (`src/cbb_sim/engine/chance_time.py`,
small `loop.py` hunks, commits `28c03f6` and `1b9bf37`). Default-path parity, checked after each
commit: 60 x 5 smoke digest
`0d4ddccc...` = `parity_reference_windows_v6.json`, **PASS bit-identical**.
`tests/test_engine.py`: 20 passed.

COMB stack, 500 verified stride games x 32 seeds, paired with the R tap (seeds 0-31,
bit-identical to the box COMB rows); floor = |R(seeds 100-131) - R(0-31)|.

| line | R | C2 | C12 | **K** | K move / floors | actual (sample) |
|---|---:|---:|---:|---:|---|---:|
| rim make | 0.5719 | 0.5764 | 0.5785 | **0.5829** | +0.0109 / 7.7 | 0.5901 |
| jumper make | 0.3907 | 0.3892 | 0.3904 | 0.3908 | +0.0002 / 0.1 | 0.3916 |
| three make | 0.3361 | 0.3347 | 0.3363 | **0.3388** | +0.0027 / 6.3 | 0.3377 |
| eFG | 0.5022 | 0.5027 | 0.5049 | **0.5079** | +0.0056 / 11.3 | 0.5107 |
| points per game | 144.74 | 144.78 | 145.09 | **145.67** | **+0.93 / 19.6** | 146.55 |
| total bias (sample) | -1.81 | -1.77 | -1.45 | **-0.87** | +0.93 | |
| possessions (count) | 69.06 | 69.04 | 69.00 | 68.95 | -0.11 / 3.1 | |
| rim share | 0.3716 | 0.3715 | 0.3732 | 0.3714 | 0.0000 | 0.3715 |
| three share | 0.3893 | 0.3892 | 0.3885 | 0.3894 | 0.0000 | 0.3954 |
| FTA/FGA | 0.3317 | 0.3317 | 0.3342 | 0.3322 | +0.0004 / 2.5 | 0.3329 |
| FT% | 0.7110 | 0.7111 | 0.7116 | 0.7114 | +0.0004 / 0.4 | 0.7179 |
| OREB% | 0.2903 | 0.2901 | 0.2899 | 0.2896 | -0.0008 / 2.6 | 0.2985 |
| rim chance-1 transition share | 0.210 | 0.211 | 0.223 | 0.250 | | 0.219 |
| rim chance-1 elapsed 10/25/50/75/90 (s) | 4/9/16/23/30 | same | 4/8/16/22/28 | 4/8/15/21/26 | | 4/7/14/21/26 |
| rim chance-2+ mean p | 0.583 | 0.607 | 0.607 | | | 0.610 |

**KD (addendum 1c), same loop:**
- rim make 0.5846 (+1.27 pp, 8.9 floors); jumper 0.3939 (+0.32 pp, past the actual 0.3916);
  three 0.3389 (+0.28 pp).
- eFG 0.5093; **points per game +1.10 (23.0 floors)**.
- OREB% -0.11 pp (3.8 floors, away).

The within-game G5 internals:

| arm | corr(possessions, eFG) | h/a points corr | total SD |
|---|---:|---:|---:|
| R | -0.087 | 0.241 | 15.27 |
| R floor (seeds 100-131) | -0.096 | 0.235 | 15.32 |
| K | **-0.123** | **0.220** | **15.07** |
| KD | -0.097 | 0.235 | 15.21 |

K moves all three outside R's seed floor; KD keeps them inside it.

Reading:
- **K recovers +0.93 of the ~1.7 points** the tap swaps put on the feed. It moves
  rim make toward actual by 1.1 of the 1.8 pp sample gap, and three make past the
  sample actual. It leaves the shot mix untouched.
- C12 (+0.35) is limited because the engine's possession duration is class-blind:
  chance-1 time still tracks the clock draw. C2 alone nets about zero: chance-2+
  rim goes up, jumpers and threes come down.
- K overshoots the rim transition share by 3 pp.
- The count falls 0.11 (more made-FG starts).
- G5 lines are UNDERPOWERED at 500 x 32 and were not read.

**Multi-level, K minus R, paired** (`scripts/grade_chance_time_loop_cuts_v1.py`). The cell LEVELS
on 500 games carry actual-total noise of about 1.7 points per 100 games, so only the paired moves are
read.

| cut | cells | move in total per game |
|---|---|---|
| month | Nov / Dec / Jan / Feb / Mar | +0.84 / +0.94 / +0.99 / +0.94 / +0.95 |
| site | home-away / neutral | +0.95 / +0.82 |
| G9 home tier | bottom / middle / top | +0.92 / +0.96 / +0.91 |
| per team (190 teams, 3-6 games each; UNDERPOWERED per team) | median offence bias | -0.64 -> -0.19 |

The move is a uniform level shift: the same in every month, site and tier. The top tier's
remaining deficit (-2.9 on this sample) is not addressed by K. That is the compressed team eFG
(section 3.4), a team-responsiveness defect and not the feed.

Expected at full size (ARITHMETIC, not a run): COMB9's total bias -1.48 + 0.9 is about -0.5.
Whatever then remains is the OREB% and FT% channels and the count.

### 6.4 Box reads (full size, POST-HOC diagnostic)

`docs/ops/box_queue/laneI_1.md` (filed 00:20 EDT, not committed per the rules):
`COMB9 + ENGINE_CHANCE_TIME=K` vs `COMB9` at 5,710 x 200, Decision 12 floors,
`scripts/box_chance_time_v1.sh` (commit `38eafdc`). Labelled a POST-HOC DIAGNOSTIC read
for the PM's ruling on the guards, behind registered SHIP-DECISION requests; C12 as
tier B.

**Result, laneI_1 (operator 04:36-04:53Z, clone at `38eafdc`).** The flag check passed: 91.1% of
rows differ from COMB9, and the mean total moved +0.929. Moves are K minus COMB9; floors are the
Decision 12 S0-draw floors from `pair_COMB9K_vs_S0.md`.

| line | COMB9 | COMB9+K | move | floor | reading |
|---|---:|---:|---:|---:|---|
| **G9 total bias** | -1.486 FAIL | **-0.557 PASS** | **+0.929** | 0.029 | toward, 32 floors |
| G9 calibration slope | 0.9462 | 0.9477 | +0.0014 | 0.0044 | inside |
| G9 margin bias | -0.246 | -0.262 | -0.016 | 0.042 | inside |
| G9 cells outside: month / tier / pred-total | 7/10, 4/6, 2/6 | 4/10, 3/6, **0/6 PASS** | | | toward |
| eFG pooled (actual 0.5086) | 0.5009 | 0.5068 | +0.0059 | 0.0001 | toward |
| G1 possessions (actual 67.875) | 68.953 | 68.837 | -0.115 | 0.010 | toward |
| **G5 total SD ratio** | 0.9094 | **0.8933** | -0.016 | 0.0016 | **AWAY** |
| **G5 h/a corr** (actual 0.2283) | 0.1125 | **0.0915** | -0.021 | 0.0015 | **AWAY** |
| G5 margin SD ratio | 1.0408 | 1.0436 | +0.003 | 0.0035 | inside |
| G7 OT rate | 0.0306 | 0.0300 | -0.0006 | 0.0005 | away, 1.3 floors |

Decomposition of COMB9+K: total -0.547. Rim make -1.148 -> -0.144, three -0.433 -> -0.051.
Remaining channels: OREB% -0.74 (boards +0.53), FT% -0.34, mix -0.21, TOV -0.19,
FT rate -0.06, possessions +0.67.

Tier B, `COMB9+C12` (04:50-05:06Z), moves against COMB9:

| line | move | reading |
|---|---:|---|
| G9 total bias | +0.425 (-1.486 -> -1.061) | still FAIL |
| G9 slope | +0.0018 | |
| eFG | +0.0028 | |
| G5 total SD ratio | -0.0013 | inside the 0.0016 floor |
| G5 h/a corr | -0.0022 | 1.5 S0 floors |
| possessions | -0.045 | |

This matches the local loop (+0.35).

**Result, laneI_2: `COMB9+KD` vs `COMB9`** (operator 05:20-05:39Z, clone at `6edbe94`). The flag
check passed: 66.2% of rows differ, and the mean total moved +1.141.

| line | COMB9 | COMB9+K | **COMB9+KD** | KD move | floor | reading |
|---|---:|---:|---:|---:|---:|---|
| **G9 total bias** | -1.486 FAIL | -0.557 PASS | **-0.345 PASS** | **+1.141** | 0.029 | toward, 39 floors |
| G9 calibration slope | 0.9462 | 0.9477 | 0.9452 | -0.0010 | 0.0030 | inside |
| G9 margin bias | -0.246 | -0.262 | -0.251 | -0.005 | 0.042 | inside |
| G9 cells outside: month / tier / pred-total | 7/10, 4/6, 2/6 | 4/10, 3/6, 0/6 | 4/10, 3/6, **0/6** | | | toward |
| eFG pooled (actual 0.5086) | 0.5009 | 0.5068 | **0.5081** | +0.0072 | 0.0001 | toward |
| G1 possessions (actual 67.875) | 68.953 | 68.837 | 68.821 | -0.131 | 0.010 | toward |
| G5 total SD ratio | 0.9094 | 0.8933 | 0.9068 | -0.0026 | 0.0016 | away, 1.6 floors (K: 10) |
| G5 h/a corr | 0.1125 | 0.0915 | 0.1088 | -0.0037 | 0.0015 | away, 2.5 floors (K: 14) |
| G5 margin SD ratio | 1.0408 | 1.0436 | 1.0412 | +0.0004 | 0.0035 | inside |
| G7 OT rate | 0.0306 | 0.0300 | 0.0303 | -0.0003 | 0.0003 | inside |

**Decomposition of COMB9+KD** (5,700 games, residual 6e-14): total -0.336.
- The make channels are closed: rim -0.095, jumper +0.167, three +0.009.
- The PPP deficit that is left: OREB% -0.745 (live boards +0.481), FT% -0.343, shot mix -0.207,
  TOV -0.180, FT rate -0.066.
- Possessions: +0.647.

---

## 7. Owners, summary

| channel | points | offline right? | owner |
|---|---:|---|---|
| make rates (rim, three) | -1.58 | yes (fg_make calibrated) | **engine chance-timing feed** (`chance_time`, this lane). At full size KD closes it: +1.14, G9 total -0.35 PASS, G5 -1.6 / -2.5 floors. The registered guards need a PM ruling |
| OREB% | -0.71 | no (rebound -0.94 pp on real states) | rebound level: anchor `O` / Stage B `TO` |
| FT% | -0.35 | yes (FT model calibrated) | FT-trip shooter allocation (usage: prior-season block, about half) and foul state (bonus share 0.556 vs 0.456, score side; foul accrual / late game, about half) |
| TOV, shot mix | -0.40 | no (PO level on real states) | possession_outcome level: `TO`; transition feed (C12) also touches it |
| shooter mix | about -0.24 (inside make) | n/a | usage / rotation; small |
| possessions | +1.00 | | clock / G1 truth (lanes B, H) |

## 8. Not established / NOT RUN

1. The FT% swap skips neutral-site games, and its state part is joint: the single
   swaps do not add up.
2. And-ones and end-of-half are not separated as channels.
3. The decomposition uses the box possession estimator on both sides. The engine
   count vs pbp count is the G1 doc's object.
4. The chance-1 conditional draw (C12) uses chance duration / possession duration
   from the possession layer as the proxy for time-to-shot. It is not refitted
   against the fg_make design's own elapsed definition. It leaves rim -0.79 pp at
   chance 1 offline.
5. G5 lines are UNDERPOWERED at 500 x 32 and were not read there. The full-size reads are
   in section 6.4.
6. K and KD over-feed the chance-1 transition share of rim attempts: 0.25-0.26 against 0.22
   on the sample. KD's jumper make is 0.32 pp past the sample actual. Neither is decomposed.
7. KD's offline replay conditions on the real duration and inherits round 1's leak, so its
   offline log loss and calibration lines are not interpretable.
8. KD's residual G5 cost (-1.6 / -2.5 floors) is not decomposed. The shared-shooting latent
   (lane B, G3) and the whistle are the registered G5 owners.

## 9. Incidents caused by this lane

1. Commit `bafee9d` (23:37 EDT, the ledger row) also carried another lane's uncommitted edit of
   the ledger: the lane-H "Clock round 7 ... REFUTED 2026-09-30 23:40 EDT" row. Between this lane's
   `git diff` check and its commit, that lane edited the shared file. The content is that lane's
   own, unchanged, and is now on `origin/main` under this lane's commit. Nothing was lost.
2. Commit `247642d` (00:05 EDT) converted
   `change_ledger.md` from CRLF to LF. That made a 269-line diff with no content change. Fixed 10
   seconds later in `c56f427` (CRLF restored; net diff against the prior state is the one row).
   Any lane holding uncommitted ledger edits from those 10 seconds would have seen line-ending
   churn.

No process of another lane was touched. This lane's compute stayed at 4 processes or fewer,
except for a few-second JSON read during the first tap.
