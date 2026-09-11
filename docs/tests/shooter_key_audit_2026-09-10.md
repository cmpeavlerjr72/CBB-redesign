# Shooter-key audit: `participant_1_id` vs `shot_shooter_id` (CBBD pbp)

Author: Opus worker, 2026-09-10. Reproducer: `scripts/diag_shooter_key_v1.py`
(writes `data/processed/models/usage_v2/shooter_key_audit.json` and
`.../shooter_key_audit_perclass.json`). Seasons 2022-2025 only; **2026 is sealed
and is never read here**.

Trigger: the attribution worker's finding at the 2026-09-10 shutdown
(`HANDOFF.md`, TOP PRIORITY ON RESUME) that CBBD's `participants` array is not
ordered shooter-first, so `participant_1_id` is the ASSISTER on about half of
assisted made field goals. `cbb_sim.models.event_stream.build_stream` sets
`player_id = participant_1_id` on every row and
`cbb_sim.models.usage.build_usage_events` credits the field-goal event to that
column, so the adopted usage bake-off's shooter labels are contaminated.

---

## 1. What the defect actually is

It is not a missing-data problem and it is not random noise. On a made field goal
with an assist, CBBD emits exactly two participants -- the shooter and the
assister -- and **the order is a coin flip**.

Measured over all 2022-2025 rows of the modelling universe (D-I, non-truncated,
`pbp_complete`), on assisted made FGAs where `participant_1_id` and
`shot_shooter_id` disagree:

| statement | share of disagreeing rows |
|---|---:|
| `participant_1_id` == `shot_assisted_by_id` (p1 IS the assister) | **100.000%** |
| `participant_2_id` == `shot_shooter_id` (p2 IS the shooter) | **100.000%** |

Zero exceptions in 247,000 disagreeing rows. So the two ids are the right pair in
the wrong order, roughly half the time, and no heuristic on `participant_1_id`
alone can recover the shooter. `shot_shooter_id` is a dedicated column and is the
only correct key.

---

## 2. Agreement by event type (2022-2025 pooled, modelling universe)

`agree%` is over rows where **both** ids are present. `sh%` is
`shot_shooter_id` coverage.

| event bucket | rows | p1 present % | `shot_shooter_id` present % | agree % | p1 = assister, of disagreements |
|---|---:|---:|---:|---:|---:|
| **FGA made, assisted** | 504,279 | 100.000 | 99.858 | **51.017** | 100.0% |
| FGA made, unassisted | 483,068 | 99.993 | 99.855 | 99.994 | n/a (~27 rows; p1 is NOT the assister on any of them) |
| FGA missed | 1,253,848 | 99.992 | 99.992 | **100.000** | -- |
| FTA (`FT_made` + `FT_missed`) | 717,129 | 99.999 | 99.999 | **100.000** | -- |
| TOV | 470,851 | 94.967 | **0.000** | n/a | -- |
| OREB | 448,075 | 73.651 | **0.000** | n/a | -- |
| DREB | 964,479 | 92.809 | **0.000** | n/a | -- |
| DeadBallReb | 44,194 | 0.000 | 0.000 | n/a | -- |
| foul | 653,027 | 99.892 | **0.000** | n/a | -- |
| steal | 251,869 | 99.950 | 0.000 | n/a | -- |
| block | 127,564 | 99.994 | 0.000 | n/a | -- |
| technical | 7,061 | 79.691 | 0.000 | n/a | -- |
| other (end_period / end_game / unknown) | 499,810 | 37.149 | 0.000 | n/a | -- |

**Two things follow, and both matter.**

1. The contamination is confined to **assisted made field goals**. Missed FGAs
   and free throws agree on 100.000% of rows -- there is only one participant on
   those, so the ordering bug cannot fire.
2. `shot_shooter_id` is populated on **shooting rows only** (0.000% on rebounds,
   turnovers, fouls, steals, blocks). It is a strict *replacement* for the FGA
   shooter and is **not** a replacement for `participant_1_id` anywhere else.
   Any model keying a rebounder, a fouler, a stealer or a charged turnover player
   must keep reading `participant_1_id`.

## 2.1 By season (agree % / `shot_shooter_id` missing %)

| bucket | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| FGA made, assisted | 51.158 / 0.156 | 50.993 / 0.122 | 50.982 / 0.142 | 50.962 / 0.148 |
| FGA made, unassisted | 100.000 / 0.013 | 100.000 / 0.009 | 99.986 / 0.280 | 99.994 / 0.229 |
| FGA missed | 100.000 / 0.009 | 100.000 / 0.007 | 100.000 / 0.013 | 100.000 / 0.002 |
| FTA | 100.000 / 0.000 | 100.000 / 0.001 | 100.000 / 0.000 | 100.000 / 0.001 |
| every non-shooting bucket | -- / 100.000 | -- / 100.000 | -- / 100.000 | -- / 100.000 |

The defect rate is **flat across all four seasons** (50.96-51.16% agreement on
assisted makes). It is a stable property of the feed, not a season-specific
drift, so nothing about the fold structure changes.

## 2.2 Season roll-ups

| season | universe rows | all-row agree % | FGA-row agree % | made-FGA agree % |
|---|---:|---:|---:|---:|
| 2022 | 1,355,557 | 91.672 | 89.140 | 75.091 |
| 2023 | 1,442,360 | 91.605 | 88.953 | 74.925 |
| 2024 | 1,679,385 | 91.747 | 89.050 | 75.184 |
| 2025 | 1,947,952 | 91.603 | 88.831 | 74.742 |

Tie-out with `HANDOFF.md` (which quoted raw, un-universe-filtered 2025 made field
goals): raw 2025 made-FGA agreement **74.639%**, raw 2025 assisted-made
agreement **50.968%**. The handoff's "74.5% overall / 50.9% assisted" reproduces.

---

## 3. Per-class impact inside the usage model's own window (2024-2025)

The usage model lives on 2024+ (L13: no on-floor ids before then). Per its own
three field-goal classes, with the v2 rim override applied:

| season | class | events | assisted % | **mislabelled %** | mislabel % of assisted | `shot_shooter_id` missing % |
|---|---|---:|---:|---:|---:|---:|
| 2024 | FGA_rim | 225,021 | 24.756 | **11.865** | 47.926 | 0.048 |
| 2024 | FGA_jump2 | 153,702 | 10.237 | **4.965** | 48.383 | 0.270 |
| 2024 | FGA_3 | 225,731 | 28.175 | **14.083** | 49.985 | 0.037 |
| 2025 | FGA_rim | 235,458 | 25.186 | **12.044** | 47.819 | 0.047 |
| 2025 | FGA_jump2 | 149,087 | 10.264 | **4.926** | 47.941 | 0.211 |
| 2025 | FGA_3 | 246,890 | 28.066 | **14.082** | 50.172 | 0.043 |

`TOV` and `FT_trip`, the other two usage classes, are **unaffected**: FTA rows
agree on 100.000% and TOV rows have no `shot_shooter_id` at all, so
`participant_1_id` remains the only and the correct key for both.

**The contamination is not uniform across classes and that is the damaging
part.** Three-point allocation is mislabelled at 14.1% and mid-range at 4.9% --
a factor of 2.9. A model trained on these labels therefore learns a systematically
flattened three-point shooter profile (the credit for a spot-up three is handed to
the passer about half the time it is assisted) while the mid-range profile is left
nearly intact. That is a *differential* bias across the very classes the
allocator is supposed to distinguish, not a wash.

### 3.1 The `in_five` coverage filter did not catch it

On mislabelled rows, the wrong man (`participant_1_id`, the assister) is inside
the resolved offensive five on **95.31% (2024) / 98.59% (2025)** of rows -- the
same rate as the true shooter (94.72% / 98.34%). The assister is a teammate on
the floor, so he is always a legal alternative. `usage.usable_events`'
`in_five` filter passed the contaminated rows through untouched, which is why the
round-1 coverage table (`experiments.md` section 2.1, 95-98% modelled) showed
nothing wrong.

### 3.2 Per team

Agreement of `participant_1_id` with `shot_shooter_id` on FGA rows, over teams
with >= 200 such rows in the season:

| season | teams | min % | p05 % | median % | p95 % | max % | SD (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 358 | 80.83 | 84.67 | 89.52 | 93.05 | 96.05 | 2.48 |
| 2023 | 363 | 81.59 | 84.55 | 89.37 | 92.97 | 95.08 | 2.44 |
| 2024 | 362 | 80.59 | 84.76 | 89.20 | 92.95 | 94.61 | 2.50 |
| 2025 | 364 | 77.64 | 84.93 | 88.87 | 92.92 | 94.75 | 2.47 |

Mislabel rate per team (the complement) spans **5.25% to 22.33%** with an SD of
about 2.5 pp. It tracks assist rate, so the bias is a *team-level* one: a
high-assist team's usage profile is corrupted three to four times as heavily as a
low-assist team's. This is exactly the shape of defect the standing
matchup-specific rule exists to catch, and an aggregate agreement number (91.6%
over all rows) hides it completely.

### 3.3 Drop rate if rows with no shooter are dropped (never imputed)

| season | per-team FGA rows with `shot_shooter_id` missing: min / median / p95 / max |
|---|---|
| 2024 | 0.000% / 0.052% / 0.327% / 3.612% |
| 2025 | 0.000% / 0.000% / 0.283% / 5.632% |

Pooled, 0.04-0.27% of FGA rows per class. Two teams sit above 3%; every other
team is under 1%. Dropping them costs less than a third of a percent of the
training set and is reported per season and per team by the v2 trainer rather
than imputed.

---

## 4. Every sub-model that keys a player off a CBBD participant column

Grep basis: `participant_1_id`, `participant_2_id`, `participant_`,
`shot_shooter_id` across `src/cbb_sim/` and `scripts/`.

| sub-model | file / line | what it keys | column read | affected? | why |
|---|---|---|---|---|---|
| **usage (L4)** | `models/usage.py` via `event_stream.build_stream:222` | the credited player of `FGA_rim` / `FGA_jump2` / `FGA_3` | `participant_1_id` | **YES** | 4.9-14.1% of rows per class carry the assister as the shooter. FIXED IN THIS TASK. |
| usage (L4), `TOV` class | same | the charged player | `participant_1_id` | no | `shot_shooter_id` is 0% populated on TOV rows; p1 is the correct and only key. (Separate known gap: p1 is blank on 5.0-5.3% of TOV rows -- team turnovers -- already reported in `experiments.md` R11.) |
| usage (L4), `FT_trip` class | same | the fouled shooter, trip position 1 | `participant_1_id` | no | FTA agreement is 100.000% in every season. |
| **fg_make (L4)** | `models/fg_make.py:361` (`"shooter_id": st["player_id"]`) | the shooter of every FGA | `participant_1_id` | **YES** | Same rows, same rate. The shooter as-of make rate (`shooter_make_c`, `shooter_att_c`, `shooter_games_asof`, `shooter_fga_asof`, `prior_season_make_c`) is built by accumulating each attempt onto the wrong player on 4.9-14.1% of attempts, and the make/miss outcome is attributed to him. Feature set `B_plus_shooter` and everything above it is contaminated. NOT changed here -- PM to dispatch a rerun. |
| free_throw (L3) | `models/free_throw.py:349` | the free-throw shooter | `participant_1_id` | no | FTA rows have one participant; agreement 100.000% in all four seasons. |
| rebound (L3) | `models/rebound.py:194` (`nxt_pid`) | the rebounder | `participant_1_id` | no | `shot_shooter_id` is 0% populated on OREB/DREB/DeadBallReb. p1 is correct. Known separate gap: p1 blank on 26.3% of OREB and 7.2% of DREB rows (team rebounds), and 100% of DeadBallReb -- reported, never imputed. |
| rotation (L5) | `models/rotation.py:241-244` | the fouler on `PersonalFoul` rows (foul-out / availability) | `participant_1_id` | no | foul rows have one participant and no `shot_shooter_id`. |
| attribution (L6) | `models/attribution.py:380-390` | shooter, rebounder, charged player, stealer, assister | `shot_shooter_id` for the shooter; `participant_1_id` elsewhere | no | **already correct** -- this is the model that found the defect. |
| possession_outcome (L3) | `models/possession_outcome.py` | -- | none | no | no player key at all (team-level). |
| clock (L3) | `models/clock.py` | -- | none | no | no player key. |
| pace (L2) | `models/pace.py` | -- | none | no | no player key. |
| `pbp/possessions.py` | -- | -- | none | no | `PLAY_COLUMNS` carries no `participant_1_id`; the state machine is team-level. |
| engine adapters | `engine/adapters.py:340-358`, `scripts/build_engine_inputs.py:81-83` | reads `usage/asof_v2.parquet` + `usage_params_v1.json` | (downstream of usage) | **YES, indirectly** | the engine's `usage_rate` table is built from the contaminated as-of rates. Needs repointing at the v2 artifacts once the PM adopts them. Not touched here. |

**Summary: two sub-models read the wrong column -- `usage` and `fg_make` -- and
both read it for exactly the same population (FGA rows).** Everything else that
reads `participant_1_id` reads it for an event type where `shot_shooter_id` does
not exist and `participant_1_id` is the correct and only key.

`event_stream.build_stream`'s `shooter_key` parameter (added 2026-09-10) defaults
to `participant_1_id`, i.e. to the existing behaviour, so `free_throw`,
`rebound` and `rotation` are byte-identical after the fix and no concurrently
running worker is disturbed. `usage.build_usage_events` defaults to
`shot_shooter_id`; `fg_make` has to opt in explicitly and has not been changed.

---

## 5. What this does NOT establish

- It does not measure how much the usage bake-off's *decision* changes. That is
  the round-2 rerun in `docs/models/usage/experiments.md` section 8.
- It does not measure the fg_make impact. `fg_make`'s target is the make/miss
  outcome, not the identity, so a mislabelled shooter degrades a *feature* rather
  than the label; the magnitude has to be measured by a rerun, not asserted.
- It says nothing about 2026, which was not read.
