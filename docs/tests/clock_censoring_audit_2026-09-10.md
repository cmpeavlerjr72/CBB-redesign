# Clock censoring audit -- L20 evidence (2026-09-10)

Written by `scripts/diag_clock_censoring_v1.py`. Every number below is produced
by that script from `data/processed/possessions/possessions_{season}.parquet`
and `data/processed/games_universe.parquet`; none is typed by hand. No model is
involved. Season 2026 is never loaded (sealed).

This is step 2 of clock round 3: the evidence for `docs/LEARNINGS.md` L20
("horn-truncated possessions must be right-censored") and the source of the two
flag artifacts round 3 trains on.

---

## 0. Definitions (stated before the numbers)

**Horn-ending (right-censored) possession.** `end_clock <= 0`, i.e. the
possession consumed every second that was left in the period. Its observed
`duration_s` is a LOWER BOUND on the duration the offence intended, so the row
is RIGHT-CENSORED at its observed value. Tolerances of 1 s and 2 s are reported
as sensitivity; the 0 s definition is the one round 3 uses.

The flag the clock model used in rounds 1 and 2 was
`terminal_event == "end_period"` -- a possession the possession builder closed
because the period ended with no other terminal event. That is a strict subset:
a buzzer three, a buzzer layup, a turnover as the horn sounds and a bonus trip
that ends the half are all horn-ending and none of them is `end_period`.

**Clock-complete period.** All three of:

| component | test | why |
|---|---|---|
| C1 tail | last logged possession ends within 2 s of the horn | the feed reached the end of the period |
| C2 head | first logged possession starts within 2 s of the period length | the feed started at the opening tip |
| C3 sum | \|sum of possession durations - period length\| <= 4 s | the logged possessions account for the whole period, with no interior gap |

C1 alone is `cbb_sim.models.clock.CLOCK_COMPLETE_TOL_S`, round 2's secondary
read. C2 and C3 are added here so that "complete" means the whole period is
accounted for and not merely its end. A GAME is clock-complete when both
regulation halves are (`clock_complete_reg`); `clock_complete_all_periods` also
requires every OT period.

**Universe ladder.** D-I and not `pbp_truncated`, then CBBD points-complete
(the possession table's own points plus technical FTs equal the schedule's final
score for both teams -- the same definition
`cbb_sim.models.clock.cbbd_complete_games` uses).

| step | rows | games |
|---|---:|---:|
| all possessions 2022-2025 | 3,029,695 | 21,969 |
| D-I, hoopR feed not truncated | 3,029,695 | 21,969 |
| + CBBD points-complete | 2,607,355 | 18,902 |

Everything from section 1 on is on the points-complete universe, which is the
clock bake-off's own universe, so these numbers are directly comparable to
`docs/models/clock/experiments.md`.

---

## 1. How many possessions end at the horn

### 1.1 By season

| season | n_games | n_poss | n_horn | horn_pct | horn_per_game | n_flag | old_flag_per_game |
|---|---|---|---|---|---|---|---|
| 2022 | 4162 | 574997 | 3651 | 0.6350 | 0.8772 | 1173 | 0.2818 |
| 2023 | 4384 | 603144 | 4004 | 0.6639 | 0.9133 | 1250 | 0.2851 |
| 2024 | 5037 | 697933 | 4671 | 0.6693 | 0.9273 | 1481 | 0.2940 |
| 2025 | 5319 | 731281 | 5288 | 0.7231 | 0.9942 | 1709 | 0.3213 |

`n_horn` / `horn_pct` are the correct censoring set; `n_flag` is what rounds 1
and 2 flagged. The correct flag is **3.1x** the old one
(0.6751% of rows against 0.2148%).

### 1.2 By season and period type

| season | period_type | n_periods | n_poss | n_horn | horn_pct | horn_per_period | n_flag_end_period | old_flag_pct |
|---|---|---|---|---|---|---|---|---|
| 2022 | H1 | 4162 | 282169 | 2270 | 0.8045 | 0.5454 | 707 | 0.2506 |
| 2022 | H2 | 4162 | 287762 | 1259 | 0.4375 | 0.3025 | 423 | 0.1470 |
| 2022 | OT | 284 | 5066 | 122 | 2.4082 | 0.4296 | 43 | 0.8488 |
| 2023 | H1 | 4384 | 295551 | 2477 | 0.8381 | 0.5650 | 728 | 0.2463 |
| 2023 | H2 | 4384 | 302400 | 1371 | 0.4534 | 0.3127 | 467 | 0.1544 |
| 2023 | OT | 289 | 5193 | 156 | 3.0040 | 0.5398 | 55 | 1.0591 |
| 2024 | H1 | 5037 | 342519 | 2926 | 0.8543 | 0.5809 | 907 | 0.2648 |
| 2024 | H2 | 5037 | 349315 | 1532 | 0.4386 | 0.3041 | 510 | 0.1460 |
| 2024 | OT | 338 | 6099 | 213 | 3.4924 | 0.6302 | 64 | 1.0494 |
| 2025 | H1 | 5319 | 358457 | 3256 | 0.9083 | 0.6121 | 1045 | 0.2915 |
| 2025 | H2 | 5319 | 366421 | 1830 | 0.4994 | 0.3440 | 592 | 0.1616 |
| 2025 | OT | 356 | 6403 | 202 | 3.1548 | 0.5674 | 72 | 1.1245 |

Read: roughly one possession per period ends at the horn, by construction --
every period that is fully logged ends with one. The point is not the count, it
is WHICH rows they are and what the model does with them: they are the rows that
define the conditional law of duration in the last seconds of a period, and
treating them as completed observations is what made rounds 1 and 2 draw short
there.

### 1.3 Share of periods whose last logged possession reaches the horn

| season | period_type | n_periods | last_poss_at_horn | last_start_under_35 | mean_last_duration |
|---|---|---|---|---|---|
| 2022 | H1 | 4162 | 0.7787 | 0.9039 | 16.8032 |
| 2022 | H2 | 4162 | 0.4383 | 0.8587 | 10.6913 |
| 2022 | OT | 284 | 0.6620 | 0.9894 | 7.7606 |
| 2023 | H1 | 4384 | 0.7694 | 0.9040 | 16.5689 |
| 2023 | H2 | 4384 | 0.4423 | 0.8748 | 10.3301 |
| 2023 | OT | 289 | 0.7197 | 0.9619 | 8.0450 |
| 2024 | H1 | 5037 | 0.7917 | 0.9023 | 16.7675 |
| 2024 | H2 | 5037 | 0.4346 | 0.8553 | 10.8084 |
| 2024 | OT | 338 | 0.7840 | 0.9941 | 7.6923 |
| 2025 | H1 | 5319 | 0.8026 | 0.9041 | 17.1807 |
| 2025 | H2 | 5319 | 0.4651 | 0.8566 | 16.3544 |
| 2025 | OT | 356 | 0.7640 | 0.9775 | 7.9354 |

---

## 2. Clock completeness

### 2.1 By season and period type, with each component separately

| season | period_type | n_periods | c1_tail | c2_head | c3_sum | complete | mean_unaccounted_s | p90_unaccounted_s |
|---|---|---|---|---|---|---|---|---|
| 2022 | H1 | 4162 | 0.7787 | 1.0000 | 0.8943 | 0.7787 | 1.7857 | 5.0000 |
| 2022 | H2 | 4162 | 0.4383 | 1.0000 | 0.5324 | 0.4383 | 7.7842 | 23.0000 |
| 2022 | OT | 284 | 0.6620 | 1.0000 | 0.7676 | 0.6620 | 3.2430 | 9.7000 |
| 2023 | H1 | 4384 | 0.7694 | 1.0000 | 0.8905 | 0.7694 | 1.8495 | 5.0000 |
| 2023 | H2 | 4384 | 0.4423 | 1.0000 | 0.5299 | 0.4423 | 7.7751 | 23.0000 |
| 2023 | OT | 289 | 0.7197 | 1.0000 | 0.8304 | 0.7197 | 2.8028 | 9.0000 |
| 2024 | H1 | 5037 | 0.7917 | 1.0000 | 0.9081 | 0.7917 | 1.6986 | 4.0000 |
| 2024 | H2 | 5037 | 0.4346 | 1.0000 | 0.5227 | 0.4346 | 8.0802 | 23.0000 |
| 2024 | OT | 338 | 0.7840 | 1.0000 | 0.8639 | 0.7840 | 2.0207 | 6.0000 |
| 2025 | H1 | 5319 | 0.8026 | 1.0000 | 0.9092 | 0.8018 | 0.6902 | 4.0000 |
| 2025 | H2 | 5319 | 0.4651 | 1.0000 | 0.5454 | 0.4625 | 0.8176 | 23.0000 |
| 2025 | OT | 356 | 0.7640 | 1.0000 | 0.8596 | 0.7640 | 2.2669 | 7.0000 |

`mean_unaccounted_s` is period length minus the summed possession durations: the
seconds of game time the CBBD event stream never logged.

### 2.2 By season, per game

| season | n_games | share_reg_halves_complete | share_all_periods_complete |
|---|---|---|---|
| 2022 | 4162 | 0.3486 | 0.3328 |
| 2023 | 4384 | 0.3453 | 0.3337 |
| 2024 | 5037 | 0.3490 | 0.3389 |
| 2025 | 5319 | 0.3743 | 0.3636 |

---

## 3. Duration distribution: horn-ending vs the rest

Rows with `duration_s > 90` are excluded exactly as the clock model
excludes them (a CBBD feed gap, not a possession).

| group | n | mean | sd | q10 | q25 | q50 | q75 | q90 | q99 |
|---|---|---|---|---|---|---|---|---|---|
| horn-ending | 17601 | 13.082 | 11.259 | 1.000 | 4.000 | 9.000 | 21.000 | 30.000 | 43.000 |
| not horn-ending | 2589591 | 17.500 | 9.749 | 5.000 | 10.000 | 17.000 | 24.000 | 30.000 | 44.000 |

### 3.1 By the seconds-remaining band the possession STARTED in

| band | n | horn_rate | old_flag_rate | mean_dur_all | mean_dur_uncensored |
|---|---|---|---|---|---|
| [0, 5) | 8140 | 0.6111 | 0.2111 | 1.5522 | 0.8857 |
| [5, 10) | 10887 | 0.3638 | 0.1155 | 4.4414 | 3.1408 |
| [10, 20) | 21336 | 0.1729 | 0.0545 | 7.6138 | 6.2500 |
| [20, 30) | 23714 | 0.1314 | 0.0357 | 12.0897 | 10.2175 |
| [30, 45) | 41753 | 0.0415 | 0.0128 | 15.3494 | 14.5480 |
| [45, 60) | 38914 | 0.0030 | 0.0017 | 14.5890 | 14.4852 |
| [60, 90) | 69657 | 0.0002 | 0.0001 | 16.5455 | 16.5337 |
| [90, 1201) | 2392791 | 0.0000 | 0.0000 | 17.8359 | 17.8359 |

This table is L20 in one place. With fewer than 10 seconds left,
**47.0%** of possessions end at the horn, and the mean observed
duration in that band collapses toward the clock that was left -- not because
offences behave differently by that much, but because the clock ran out. The
old flag catches almost none of it (`old_flag_rate` column).

### 3.2 By terminal event

| terminal_event | n | horn_rate | n_horn |
|---|---|---|---|
| FGA_3 | 674100 | 0.0083 | 5622 |
| FGA_jump2 | 475984 | 0.0047 | 2256 |
| FGA_rim | 653963 | 0.0030 | 1977 |
| FT_trip_bonus | 142800 | 0.0042 | 597 |
| FT_trip_shooting | 172764 | 0.0018 | 315 |
| TOV | 458531 | 0.0021 | 968 |
| end_period | 5600 | 0.9936 | 5564 |
| unknown | 23450 | 0.0129 | 302 |

`end_period` is ~99% horn-ending, as it must be. Everything else in this table
is a horn-ending possession the old flag missed: buzzer threes are the largest
single group.

---

## 4. What clock-completeness does to the G1 target

G1's actual quantity is possessions per team-game over regulation, counted from
the logged possessions. In a half whose feed stops early, that count is short by
the possessions the feed never logged, so the ACTUAL is biased DOWN on exactly
the games where the sim (which always runs its clock to zero) is unbiased. This
is why round 3 re-bases G1 on clock-complete games.

| season | n_games | mean_all | sd_all | n_cc_games | cc_share | mean_cc | sd_cc | mean_shift |
|---|---|---|---|---|---|---|---|---|
| 2022 | 4162 | 68.4684 | 5.2350 | 1451 | 0.3486 | 68.3463 | 5.1613 | -0.1221 |
| 2023 | 4384 | 68.1970 | 5.2314 | 1514 | 0.3453 | 68.1902 | 5.1259 | -0.0067 |
| 2024 | 5037 | 68.6752 | 5.1795 | 1758 | 0.3490 | 68.4872 | 5.1586 | -0.1880 |
| 2025 | 5319 | 68.1404 | 5.0892 | 1991 | 0.3743 | 67.9410 | 5.1422 | -0.1995 |

### 4.1 2025 (the selection fold's test season) by month

| month | n_games | mean_all | n_cc | mean_cc | mean_shift |
|---|---|---|---|---|---|
| 1 | 1358 | 67.6550 | 520 | 67.4096 | -0.2454 |
| 2 | 1203 | 67.3583 | 440 | 67.2364 | -0.1219 |
| 3 | 734 | 67.5940 | 305 | 67.3918 | -0.2022 |
| 4 | 16 | 69.0625 | 7 | 70.8571 | 1.7946 |
| 11 | 1125 | 69.4907 | 395 | 69.0038 | -0.4869 |
| 12 | 883 | 68.6699 | 324 | 68.9090 | 0.2391 |

Read: `mean_shift` is the size of the grading-truth correction, and it is small
and NEGATIVE (-0.01 to -0.20 possessions per team-game). Two effects run against
each other: dropping a half's unlogged tail removes possessions from the actual
count (which biases the all-games actual DOWN), while the games whose feeds are
complete are also slightly slower than average (which biases the clock-complete
actual DOWN). The second wins, narrowly.

The consequence for round 3 is worth stating plainly: re-basing G1 on
clock-complete games does NOT close round 2's +1.5 to +2.6 emergent overshoot --
it widens it by about 0.2. Re-basing is done because the all-games actual is a
count of LOGGED possessions in halves that stop early, which no correct sim can
reproduce; it is a grading-truth fix, not a gap-closing one. The censoring fix
is what has to close the gap.

---

## 5. Artifacts written

| path | what | how it avoids clobbering another worker |
|---|---|---|
| `data/processed/games_universe_v2.parquet` | `games_universe` plus `clock_complete_reg`, `clock_complete_all_periods`, `points_complete`, `n_periods_logged`, `n_horn_poss`, `unaccounted_s_total` | VERSIONED SIBLING. `games_universe.parquet` is read-only here, so the engine worker's reader is untouched |
| `data/processed/clock_censoring/half_clock_completeness_v1.parquet` | one row per (game_id, period): the three components and the flag | new directory, new file |
| `data/processed/clock_censoring/censoring_v1_{season}.parquet` | one row per (game_id, period, poss_index): `censored_horn` and the tolerance variants | SIDE TABLE. `data/processed/possessions/` and `data/processed/possessions_v2/` are read-only here |
| `data/processed/clock_censoring/censoring_audit_v1.json` | every table above, machine-readable | new file |

`possessions_v2` carries byte-identical `duration_s`, `start_clock` and
`end_clock` to `possessions` (checked on 2025: 0 of 768,834 rows differ), so the
censoring side table joins to either. It is keyed on (game_id, period,
poss_index), which is unique in both.
