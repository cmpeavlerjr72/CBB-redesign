# fg_make shooter-key defect: quantification (round 3, step 1)

Author: Sonnet worker, 2026-09-10. Reproducer:
`scripts/diag_fg_make_shooter_key_v1.py` (writes
`data/processed/models/fg_make/shooter_key_audit_v1.json`). Seasons
2022-2025 only, fg_make's own universe (D-I, non-truncated, `pbp_complete`,
`FG.DEFAULT_UNIVERSE`) and fg_make's own three shot classes under the **v2**
possessions build (L16 rim-location override, 2.27 ft) -- the exact population
`fg_make.build_design` trains on. **2026 is sealed and is never read here.**

Trigger: `docs/models/change_ledger.md`'s "CBBD's `participant_1_id` is the
ASSISTER..." row and `docs/tests/shooter_key_audit_2026-09-10.md`, which
measured this defect for `usage` (2024-2025, usage's own window) and flagged
fg_make as "STILL ON THE WRONG COLUMN and needs a re-run" without ever
measuring fg_make's own 2022-2025 window at the class/assisted split, or how
much fg_make's own shooter as-of FEATURES move. This doc is that measurement,
and is the evidence round 3's pre-registration (`experiments.md` section 17)
is built on.

---

## 1. Quantification

### 1.1 Mismatch rate, by class, pooled 2022-2025 (fg_make's own universe, v2 classes)

`participant_1_id != shot_shooter_id`, on rows where both ids are present:

| class | n attempts | assisted % of class | mismatch % (all) | mismatch % of ASSISTED | mismatch % of UNASSISTED | `shot_shooter_id` missing % |
|---|---:|---:|---:|---:|---:|---:|
| `FGA_rim` | 817,509 | 25.24 | **12.077** | **47.894** | 0.000 | 0.038 |
| `FGA_jump2` | 572,643 | 10.39 | **4.964** | **47.734** | 0.005 | 0.141 |
| `FGA_3` | 851,043 | 28.02 | **14.061** | **50.239** | 0.000 | 0.047 |

Confirms `docs/tests/shooter_key_audit_2026-09-10.md` section 3 exactly on the
two seasons that doc covered (2024-2025) and extends it back to 2022-2023,
which move by less than 0.3 pp from the 2024-2025 numbers on every class (1.3
below). **The defect is confined to assisted makes** (mismatch on unassisted
attempts is 0.000-0.005%, i.e. feed noise, not the ordering bug) and is a
factor of 2.4-2.9x larger on threes/rim than on mid-range jumpers, which is
exactly the differential-bias shape the standing matchup-specific rule exists
to catch: a model trained on these labels learns a flattened three-point
shooter profile while the mid-range profile is left nearly intact.

### 1.2 By season

| season | class | n | mismatch % | assisted mismatch % | `shot_shooter_id` missing % |
|---|---|---:|---:|---:|---:|
| 2022 | `FGA_rim` | 171,648 | 12.165 | 47.818 | 0.022 |
| 2022 | `FGA_jump2` | 132,095 | 4.844 | 46.721 | 0.020 |
| 2022 | `FGA_3` | 184,916 | 13.948 | 50.281 | 0.078 |
| 2023 | `FGA_rim` | 185,382 | 12.280 | 47.825 | 0.030 |
| 2023 | `FGA_jump2` | 137,759 | 5.093 | 47.669 | 0.036 |
| 2023 | `FGA_3` | 193,506 | 14.105 | 50.411 | 0.035 |
| 2024 | `FGA_rim` | 225,021 | 11.870 | 48.007 | 0.048 |
| 2024 | `FGA_jump2` | 153,702 | 4.979 | 48.435 | 0.270 |
| 2024 | `FGA_3` | 225,731 | 14.088 | 50.049 | 0.037 |
| 2025 | `FGA_rim` | 235,458 | 12.049 | 47.898 | 0.047 |
| 2025 | `FGA_jump2` | 149,087 | 4.936 | 47.982 | 0.211 |
| 2025 | `FGA_3` | 246,890 | 14.088 | 50.246 | 0.043 |

Flat across all four seasons on every class (max spread 0.4 pp on the mismatch
rate, 1.7 pp on the assisted-conditional rate) -- a stable property of the
feed, not a season-specific drift, exactly as the usage-window audit already
found. Nothing about the F1/F2 fold structure changes.

### 1.3 Drop rate if rows with no `shot_shooter_id` are dropped (never imputed), by season and team

Pooled per-team distribution (teams with >= 50 class rows in the season;
`min_pct` = 0.0 in every row because most teams never lose a row):

| class | teams | min % | median % | p95 % | max % | SD (pp) |
|---|---:|---:|---:|---:|---:|---:|
| `FGA_rim` | 366 | 0.00 | 0.00 | 0.10 | 2.91 | 0.20 |
| `FGA_jump2` | 366 | 0.00 | 0.08 | 0.50 | 1.52 | 0.20 |
| `FGA_3` | 366 | 0.00 | 0.00 | 0.09 | 2.82 | 0.26 |

By season (p95 / max, pp):

| season | `FGA_rim` | `FGA_jump2` | `FGA_3` |
|---|---|---|---|
| 2022 | 0.00 / 1.81 | 0.00 / 1.41 | 0.20 / 8.94 |
| 2023 | 0.00 / 2.42 | 0.23 / 3.02 | 0.00 / 6.14 |
| 2024 | 0.16 / 4.49 | 1.04 / 3.30 | 0.00 / 7.21 |
| 2025 | 0.00 / 6.90 | 1.00 / 1.94 | 0.00 / 6.66 |

Pooled over 2022-2025 and all three classes, dropping rows with no
`shot_shooter_id` costs **1,385 of 2,241,195 attempts (0.062%)** at the
design-row level (`n_design` under `participant_1_id` 2,241,063 vs under
`shot_shooter_id` 2,239,678 on the identical event population -- see 1.4). A
small number of teams sit above 5-9% in a single season (always `FGA_3`, the
class with the thinnest per-team volume), every other team is under 2%; this
matches usage's own drop-rate finding (`change_ledger.md` row 63) and is
reported per season and per team here rather than imputed, exactly as
`event_stream.build_stream(shooter_key="shot_shooter_id")` and
`usage.build_usage_events` already do it.

### 1.4 How much the shooter as-of features move when re-keyed

`fg_make.build_fg_events`/`_season_events` gained a `shooter_key` parameter
this round (mirroring `event_stream.build_stream`'s and
`usage.build_usage_events`'s existing parameter of the same name and default),
defaulting to `participant_1_id` (byte-identical to every existing artifact)
and threading straight through to `event_stream.build_stream`. Neither function
filters or reorders rows on `shooter_key` -- only the `shooter_id` VALUE on FGA
rows changes -- so the event tables built under the two keys are the same
length (2,241,195) in the same row order, which makes an exact attempt-for-
attempt join possible (`scripts/diag_fg_make_shooter_key_v1.py::feature_movement`).

98,691 / 28,386 / 119,612 attempts (rim/jumper/three) get a **different
shooter id outright**. But the ripple is much bigger than the relabelled rows
alone, because every OTHER attempt by the true shooter or the wrongly-credited
assister also has its cumulative make/attempt tally shift:

| class | corr(old, new) `shooter_make_c` | mean \|delta\| (all attempts) | mean \|delta\| (reassigned rows only) | % of ALL attempts moved >= 1 pp | % moved >= 3 pp | `shooter_att_c` corr | `shooter_games_asof` corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| `FGA_rim` | **0.694** | 7.68 pp | 14.21 pp | 78.9% | 63.1% | 0.905 | 0.988 |
| `FGA_jump2` | **0.802** | 5.11 pp | 15.96 pp | 64.0% | 46.1% | 0.967 | 0.995 |
| `FGA_3` | **0.335** | 12.97 pp | 18.79 pp | 85.4% | 77.8% | 0.894 | 0.989 |

The three-point shooter feature is the least correlated between keyings
(0.335) and the most differentially contaminated in 1.1 (14.1% mismatch, 50.2%
of assisted threes), which is internally consistent. Round 2b's own realised
`shooter_make_c` quintile span was 16.08 pp on `FGA_rim` (`round2b/run_report.json`
`s1_detail.resp_decision8.by_driver`); a 7.68 pp mean absolute move against
that span is **roughly half the feature's own realised range**, which is why
this is a data fix expected to move the Decision-8 shooter-slope reading, not
a cosmetic relabelling. `shooter_att_c` (attempts-to-date) and
`shooter_games_asof` (exposure) move far less (corr 0.89-1.00) because
attempt COUNTS are conserved in aggregate even when the credited player
changes -- only the make/miss CONTENT of the credited player's history moves,
which is exactly `shooter_make_c` and nothing else in the bundle.

---

## 2. Round 3 pre-registration and results

See `docs/models/fg_make/experiments.md` section 17 (pre-registration) and
section 18 (results, appended after `scripts/train_fg_make_v3_shooter.py` ran).
