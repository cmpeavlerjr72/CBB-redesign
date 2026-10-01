# chance_time: the within-possession chance timing the engine feeds to fg_make and possession_outcome

Owner of the two STATE inputs the engine computes per chance and feeds to the
served sub-models: `chance_elapsed_s` (fg_make) and `is_transition` /
`is_transition_f` (possession_outcome `first`, fg_make). Append-only. Status
changes go to `docs/models/change_ledger.md` in the same commit.

## 1. Pre-registration, round 1 (lane I, written 2026-09-30 23:40 EDT, COMMITTED BEFORE ANY ARM IS BUILT OR RUN)

### 1.1 Why (evidence already measured; `docs/tests/ppp_deficit_decomposition_2026-09-30.md`)

- Full-size COMB read (5,700 graded games with a box, 200 seeds, verified
  truth): G9 total bias -1.264 points per game, decomposed exactly (LMDI,
  residual 1e-14) into box channels. The make-rate channels are rim -1.15,
  three -0.43, jumper -0.02 points; possessions +1.00; OREB% -0.71; FT% -0.35.
- The served fg_make (`round4_B1`) is calibrated OFFLINE on the held-out
  2024-25 shots: rim mean p 0.5840 vs realised 0.5844, jumper 0.3907 vs 0.3905,
  three 0.3376 vs 0.3378. The sim's rim make is 0.5708.
- An fg_make input tap on COMB (500 verified stride games x 32 seeds; games rows
  bit-identical to the box run, 16,000 rows, 0 mismatching cells) with
  same-(game, offence, chance-bucket) feature swaps attributes the rim gap to
  the STATE inputs (+1.31 pp of the +1.42 pp sim-to-real move; shooter +0.12
  pp; team inputs 0.00 pp) and, inside state, to `chance_elapsed_s`
  (+1.02 pp alone; +1.36 pp with `is_transition_f`). Three: state +0.24 pp,
  `chance_elapsed_s` +0.36 pp. Together about -1.7 points of total per game.
- Mechanism (code read): the engine feeds (a) chance 1 `chance_elapsed_s` =
  the WHOLE possession's drawn duration (`loop.py`, `used`), and
  `is_transition` = possession duration <= 8 s, whereas both training tables
  measure chance 1 only (fg_make: start of chance to the shot; possession_outcome
  chance rows: chance-1 duration); (b) chance >= 2 `chance_elapsed_s` = the
  training MEDIAN by chance number (3 s / 2 s), a constant where the real
  distribution is wide (rim putbacks: 10/25/50/75/90% = 0/0/1/3/10 s) and the
  model is strongly non-linear in it (rim chance-2+ mean p 0.583 sim vs 0.610
  real).

This is a train/serve parity defect in the engine's state feed, not a level
error of any sub-model. No sub-model is refitted in this round.

### 1.2 Arms

Every table is built ONLY from the fold's training seasons (F2: 2021-22 ..
2023-24 = seasons 2022-2024; F1: 2022-2023) by
`scripts/build_chance_time_lut_v1.py`, from `data/processed/possessions_v2/chances_<s>.parquet`
and `data/processed/models/fg_make/design_v2_shotshooter.parquet`. Draws come
from a NEW rng family `chance_time` on its own `StreamBook` (seed, game_id),
so every existing family's stream is unchanged (paired arms stay aligned).

| arm | chance >= 2 `chance_elapsed_s` (fg_make call) | chance 1 `chance_elapsed_s` (fg_make) and `is_transition` (PO first + fg_make) | simplicity |
|---|---|---|---|
| `R` | served: training median by chance number | served: whole possession duration `used`; transition = `dur <= 8` and previous end DREB/TOV | 0 |
| `C2` | inverse-CDF draw from the training empirical distribution of design `chance_elapsed_s` in the cell (chance bucket {2, 3+} x shot class), 1001 quantiles | served (as `R`) | 1 |
| `C12` | as `C2` | chance-1 time `e1 = round(r x used)`, `r` drawn (inverse CDF, 1001 quantiles) from the training distribution of chance-1 duration / possession duration in the cell (possession duration bin x start group {DREB, TOV} / other); fg_make chance-1 elapsed = min(e1, 60); transition = `e1 <= 8` and previous end DREB/TOV, fed to possession_outcome `first` and fg_make | 2 |

Duration bins for `C12`: 0..30 s one per second, 31-35, 36-40, 41-50, 51-60,
61+; a cell with fewer than 200 possessions falls back to the pooled bin of its
start group. Possessions with a single chance enter with `r = 1`.

### 1.3 Offline bake-off (feed parity on held-out REAL states; no sim)

Rows: the fg_make design's held-out fold-test rows (F2: 2024-25), each joined
to its possession in `chances_<s>` / `possessions_<s>` on (game, period,
offence, chance number, chance start clock = shot clock + elapsed). For every
arm, the arm's feed is generated from the REAL state of the row (real
possession duration, real start reason, real chance number, real class), the
served `round4_B1` S1 artifacts score the row with the fed values in place of
the real `chance_elapsed_s` / `is_transition_f`, and the same artifacts score
the row with its real values (`p_true`).

- **Primary (F2, selection):** `G_feed = sum_k v_k A_k |mean p_arm,k - mean p_true,k|`,
  points per game, k in {rim, jumper, three}, `v_k` = 2/2/3, `A_k` = real
  attempts per game in class k (both teams).
- **Secondary:** per class the fed log loss minus the true-feature log loss;
  the mean-p gap split chance 1 / chance 2+, by month, by site, by
  transition / half-court; offence responsiveness (rows bucketed by
  `off_make_c` quintile, span of mean p_arm over span of realised make; must
  not fall below `R`'s by more than 0.05); possession_outcome `first`: share
  of chance-1 rows flagged transition, fed vs real.
- **Noise floor:** max of (|G_feed(arm, draw seed 0) - G_feed(arm, draw seed 1)|,
  2 x paired game-bootstrap SE (200 reps) of G_feed(arm) - G_feed(R)).
- **F1 confirmation (model-free, no F1 fg_make artifacts exist):** tables
  built on 2022-2023, applied to the 2024 real states; the fed
  `chance_elapsed_s` quantiles (10/25/50/75/90) by class x chance bucket and
  the fed transition share must be closer to the real ones than `R`'s (mean
  absolute quantile gap).
- **Decision rule:** an arm is ELIGIBLE iff it beats `R` on G_feed by more than
  the floor, its fed log loss is not worse than `R`'s fed log loss in any class
  by more than the fg_make seed floor (1.1e-4), its responsiveness line holds,
  and F1 confirms. Select the eligible arm with the lowest G_feed; any eligible
  arm within one floor of it and simpler wins. No eligible arm -> `R`.

### 1.4 Closed loop (sanity and direction; local; then the box SHIP-DECISION read)

- Local: the winner wired default-off behind `ENGINE_CHANCE_TIME=<arm>` (module
  `src/cbb_sim/engine/chance_time.py`, one small `loop.py` hunk), parity of the
  default path proved bit-identical to `parity_reference_windows_v6.json`. Tap
  `scripts/diag_ppp_tap_v1.py` on the COMB stack, 500 verified stride games x
  32 seeds, paired with the existing `R` tap (`results/ppp_decomp/tap_comb`).
  Read: make rate per class vs real, sim mean p per class, transition share,
  chance-2+ elapsed quantiles, points per game, possessions, rim / three share,
  TOV%, total bias on the sample. A seed-offset `R` rerun (seeds 100-131) is
  the draw floor. G5 lines are UNDERPOWERED at this size and are not read.
- Box (full size, Decision 12 floors): `COMB9 + ENGINE_CHANCE_TIME=<winner>`
  vs `COMB9` (the PM's current Decision 11 set: L2 + K2_Ocell + R9ao3), every
  gate line. Expected direction, stated in advance: points per game and G9
  total up by roughly +1.0 to +1.7, rim and three make toward actual, shot mix
  and TOV% move through possession_outcome's transition input (C12 only).
  If the repair exposes another compensation (e.g. the possession count
  +1.1, OREB% -0.9 pp), Decision 11 applies: VALIDATED-PENDING-SHIP-ACTION.

### 1.5 What this round may not do

No multiplier, offset, cap or blend on any output. The tables are the
empirical distributions of the models' own training inputs, built from training
seasons only, never fitted to sim output or to the test season. No sub-model is
refitted; no served default changes; nothing is adopted by this lane.

## 2. Round 1 results (lane I, run 2026-09-30 23:38 - 2026-10-01 00:04 EDT)

Scripts: `scripts/build_chance_time_lut_v1.py` (tables, 23:38), `scripts/exp_chance_time_offline_v1.py`
(F2 offline, 23:39-23:40), `scripts/exp_chance_time_f1_parity_v1.py` (F1, 00:03),
`scripts/grade_chance_time_loop_v1.py` (closed loop, 00:03). Engine wiring `28c03f6`
(default-off `ENGINE_CHANCE_TIME=C2|C12`, parity of the default path PASS bit-identical,
digest `0d4ddccc...`). Full tables: `docs/tests/ppp_deficit_decomposition_2026-09-30.md` s6.

### 2.1 Offline, F2 (627,859 of 630,904 held-out rows joined to their possession, 99.5%)

| arm | G_feed | floor | beats R by | rim / jumper / three mean-p gap (pp) | fed - true log loss rim / jump / three | resp. slope rim / jump / three |
|---|---:|---:|---:|---|---|---|
| R | 1.973 | | | -1.31 / -0.48 / -0.43 | -0.0187 / -0.0109 / -0.0075 | 1.120 / 0.903 / 0.746 |
| C2 | 1.819 | 0.0112 | 13.7 floors | -0.84 / -0.69 / -0.53 | -0.0182 / -0.0110 / -0.0074 | 1.050 / 0.912 / 0.743 |
| C12 | 1.314 | 0.0147 | 44.7 floors | -0.65 / -0.45 / -0.38 | -0.0168 / -0.0096 / -0.0063 | 1.049 / 0.901 / 0.749 |

F1 confirmation (2022-23 tables on 2024 real states; mean absolute fed-vs-real elapsed quantile gap /
chance-1 transition-share gap): R 3.40 s / 0.055, C2 1.43 s / 0.055, **C12 1.03 s / 0.045**. Both confirm.

**Registered decision: NO ELIGIBLE ARM; `R` stands.** Both arms fail the fed-log-loss guard (rim
C2 +0.00055, C12 +0.0019 vs R, guard 1.1e-4) and the one-sided rim responsiveness line (R 1.120;
needs >= 1.070; C2 1.050, C12 1.049).

Found while grading, recorded and NOT used to select: every fed log loss is BELOW the true-feature log
loss (rim 0.6485 vs 0.6672). The offline replay builds R's (and C12's) chance-1 feed from the REAL
possession duration, which carries the future (a first shot that is missed and offensively rebounded
sits in a long possession), so the leakiest feed wins the log-loss guard. A draw cannot beat a constant
on log loss when neither is informative about the row. The one-sided slope line penalises 1.12 -> 1.05
(the FT-technicals reading the PM already ruled mis-specified, `season_drift` s3 item 2).

### 2.2 POST-HOC closed loop (outside the rule; COMB stack, 500 verified stride games x 32 paired seeds)

Paired with the R tap (seeds 0-31, bit-identical to the box COMB rows); floor = |R(seeds 100-131) - R|.

| line | R | C2 (move / floors) | C12 (move / floors) | actual (sample) |
|---|---:|---|---|---:|
| rim make | 0.5719 | +0.0045 / 3.2 | **+0.0065 / 4.6** | 0.5901 |
| jumper make | 0.3907 | -0.0015 / 1.5 | -0.0003 / 0.3 | 0.3916 |
| three make | 0.3361 | -0.0015 / 3.4 away | +0.0001 / 0.3 | 0.3377 |
| eFG | 0.5022 | +0.0004 / 0.9 | **+0.0026 / 5.3** | 0.5107 |
| points per game | 144.74 | +0.04 / 0.8 | **+0.35 / 7.4** | 146.55 |
| rim share | 0.3716 | 0.0000 | +0.0016 / 4.2 | 0.3715 |
| three share | 0.3893 | 0.0000 | -0.0008 / 2.3 away | 0.3954 |
| FTA/FGA | 0.3317 | 0.0000 | +0.0024 / 14.7 | 0.3329 |
| possessions (count) | 69.06 | -0.02 | -0.06 / 1.7 | |

C12 recovers +0.35 of the ~1.7 points the tap swaps attribute to the feed. The remaining defect is
visible in the fed values: C12's chance-1 elapsed for rim attempts is 4/8/16/22/28 s against the real
4/7/14/21/26 s, because the engine's possession duration is drawn BEFORE the shot class and is
class-blind, while the real time to the first shot depends strongly on the class (rim attempts come
early). The tap swap that closed the gap drew real elapsed values WITHIN class. C2's chance-2+ draw
lifts rim but lowers jumpers and threes (the median over-fed them), net +0.04 points.

## 3. Addendum, round 1b: class-conditional chance-1 time at the fg_make call (written 2026-10-01 00:10 EDT, COMMITTED BEFORE arm K IS BUILT OR RUN)

POST-HOC relative to round 1: motivated by 2.2. One new arm.

- **`K`**: chance >= 2 as `C2`. Chance 1, at the fg_make call only: `chance_elapsed_s` drawn by inverse
  CDF (1001 quantiles) from the TRAINING distribution of the design's chance-1 `chance_elapsed_s` in
  the cell (shot class x start group {DREB, TOV} / other), clipped to <= min(60, seconds remaining
  at the possession's start); fg_make's `is_transition_f` = (drawn elapsed <= 8 and start in
  {DREB, TOV}). possession_outcome's transition input stays served (as `R`). Same rng family
  `chance_time`; tables from training seasons only (F2 2022-2024, F1 2022-2023), built by a versioned
  sibling `scripts/build_chance_time_lut_v2.py`; flag value `ENGINE_CHANCE_TIME=K`.
- **Evaluation:** the round-1 offline grader unchanged (`G_feed` primary, same floor definition, F1
  model-free parity), plus the round-1 guards REPORTED AS WRITTEN. For K's selection the two guards
  that 2.1 shows mis-specified are REPLACED (labelled POST-HOC): (i) fed log loss -> the fg_make
  calibration gate on the fed predictions (worst decile |mean p - realised| <= 2.0 pp per class,
  `fg_make` round-4 gate); (ii) one-sided slope -> symmetric (|slope - 1| not worse than R's by
  more than 0.05). K is ELIGIBLE iff it beats R on G_feed by more than its floor, passes (i) and
  (ii) in every class, and F1 confirms. Simplicity: K (rank 2, between C2 and C12).
- **Closed loop:** same local loop (COMB, 500 x 32, paired, R-floor 100-131). Then the box read
  `COMB9 + ENGINE_CHANCE_TIME=<K or C12>` vs `COMB9` at full size with Decision 12 floors; the PM
  rules on ship. Nothing is adopted by this lane.
