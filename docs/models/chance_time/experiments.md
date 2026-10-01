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
