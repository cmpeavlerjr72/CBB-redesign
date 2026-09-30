# Event layer v4: phantom possessions removed at the handler that makes them (2026-09-30)

Lane B, event-layer job (PM ruling 2026-09-30: the phantom possessions found in
`docs/tests/g1_g5_possessions_corr_diagnostic_2026-09-30.md` section 5 are an
EVENT-LAYER defect). **A versioned sibling only.** `possessions`, `_v2` and
`_v3` are untouched, no consumer is switched, nothing is retrained, nothing is
adopted.

## 1. Design, written and committed before the code (commit `ef7a0cd`)

Three new switches on `cbb_sim.pbp.possessions._GameMachine` /
`segment_season`, **all default False**. With every switch off, the machine's
code path is the one that wrote `possessions_v2`, and it must reproduce it
bit-identically (section 3.1). v4 = `tech_lookahead=True` (the 09-18
technical-FT fix, v3) plus all three switches.

Class letters are those of the diagnostic (section 5.1 there).

| class | what creates it today | v4 rule (switch) | effect |
|---|---|---|---|
| **A** missed and-one FT -> defensive rebound | `_handle_fga`'s and-one branch closes the shooter's possession at the free throw (`reason_next = made_FG`) whether the FT was made or missed. The rebound of a MISS then finds no open possession, and `_handle_dreb` opens and immediately closes a 1-s possession for the shooter (`unknown`). | `andone_live_miss`. And-one FT **made**: close at the FT with `reason_next = made_FT`, which is what happens on the floor (a dead-ball inbound after a made FT). And-one FT **missed**: do NOT close. The chance stays open awaiting the rebound (`pending_terminal` = the made FGA's class), exactly as a missed last FT of a trip does. | phantom removed; the possession ends at the real rebound; the next possession starts `DREB` |
| **E** shooter's own OREB of a missed and-one FT | the same premature close: the OREB finds no open possession and opens a NEW possession for the shooter | fixed by the same switch: the OREB now continues the open possession as chance 2 (`start_reason = OREB`) | restart removed |
| after an and-one, start-type label | always `made_FG` | same switch: `made_FT` (FT made) / the rebound outcome (FT missed), per the floor evidence (diagnostic 5.3) | label follows the floor; this is the engine's convention |
| **B1** stray DREB row at a free-throw moment | `_handle_dreb` with no open possession opens and closes a possession for the non-rebounder | `stray_reb_guard`: a DREB that arrives with no open possession **within 3 s** of the last close (`prev_end_clock - secondsRemaining <= 3`) is treated as an administrative row and ignored. It changes no possession, no score, and no foul count. | phantom removed; its seconds stay with the real next possession |
| **B2** DREB with no open possession, elsewhere | same path | same switch, same 3-s rule. **<= 3 s** (a stray or out-of-order rebound row): ignored. **> 3 s**: unchanged. It stays a real possession whose missed shot the feed did not log (terminal `unknown`). | mixed class split at the rule's own threshold |
| **C1** possession opened by a stray OREB | `_handle_oreb` with no open possession calls `_ensure`, which opens a possession for the rebounder, although the rebounder's team just ended its own possession (made FG, made FT, TOV) | `stray_oreb_guard`: an OREB (not the administrative FT-reset kind, which is already skipped) arriving with no open possession, **by the team whose possession was the last one closed in this period**, is ignored | phantom removed; the other team's real possession keeps its seconds |
| **C2** mismatch close (offensive or loose-ball foul, unlogged turnover) | real possession whose terminal event is not in the feed | **unchanged** (terminal `unknown`) | real, kept |
| **D** contradictory OREB/DREB pair | real possession, terminal lost | **unchanged** | real, kept |

What stays **unresolved**:
- **B2 > 3 s.** Whether each of these is a real possession: rows read in
  2022-23 say mostly a missing shot, but the box slope swings by season.
- **B1 > 3 s** (10-30% of B1): kept as real, by the same threshold.
- **C2's terminal.** It is often an offensive foul the box counts as a
  turnover, but the layer cannot label it without the row.
- **The 3-s threshold.** It is the diagnostic's measured split (A and B1 are
  97-98% and 68-92% at 3 s or less; C2 is 95% longer), not a fitted value.
- **Out-of-order rows** where the rebound precedes its own shot at the same
  second: ignoring the rebound lets the later shot row be processed normally,
  but the shot's own rebound is then missing, so that possession closes on
  the next event.

Fouls, scores, points, technical free throws and every event count are
untouched by all three rules: an ignored row is a rebound row that credits
nothing. Section 3 checks it.

## 2. Implementation and the default-path proof

- **Code.** `src/cbb_sim/pbp/possessions.py`, commits `58bd906` and the
  follow-up in this section's commit:
  - `EVENT_FIX_SWITCHES`, `VERSION_EVENT_FIXES` (`v4` = all on) and
    `STRAY_REB_MAX_S = 3`;
  - the three switches on `_GameMachine` / `segment_season`;
  - `andone_made_next` (default `made_FT`), used only when `andone_live_miss`
    is on, so that the clock arm L2a can keep the old `made_FG` label.

  Handlers changed: the and-one branch of `_handle_fga`, the head of
  `_handle_dreb`, and the head of `_handle_oreb` (after the existing
  administrative FT-reset check). No foul, free-throw, technical or turnover
  handler is touched.
- **Build.** `scripts/build_possessions_v4.py` (`--variant l2a` writes
  `possessions_v4a`; not built today).
- **Tests.** `tests/test_event_layer_v4.py` has one sequence per class, read
  from real rows: default machine (phantom present) vs v4 (phantom gone,
  events equal). With `tests/test_technical_lookahead.py`: **26 passed**.
- **Default path.** The build script rebuilds the default machine for every
  season and compares it with `possessions_v2` using `DataFrame.equals`
  (same columns, same dtypes). **Possessions and chances are BIT-IDENTICAL in
  all five seasons (2022-2026).** Re-checked for 2025 after the
  `andone_made_next` addition. Lane A's builder constructs `_GameMachine(gm,
  sub)` with defaults, so nothing under it changed.
- **v4 tables:** `data/processed/possessions_v4/{possessions,chances}_{2022..2026}.parquet`
  plus `build_report.json`. Gitignored (`.gitignore`, beside v3).

| season | v4 possessions | v4 chances |
|---|---:|---:|
| 2022 | 722,036 | 826,346 |
| 2023 | 754,397 | 864,255 |
| 2024 | 764,030 | 878,464 |
| 2025 | 762,309 | 879,021 |
| 2026 (sealed; row counts only) | 784,449 | 904,852 |

**HF sync key (registered here, NOT synced, `hf_sync_data.py` NOT edited):**
bulk key `possessions_v4` -> `data/processed/possessions_v4/` (and
`possessions_v4a` -> `data/processed/possessions_v4a/` once built), to be added
to `BULK_DIRS` / the path map in `scripts/hf_sync_data.py` by whoever syncs.

## 3. Verification (seasons 2022-2025; `scripts/diag_event_layer_v4_verify_v1.py` -> `results/g1g5_diag/event_layer_v4_verify.json`)

### 3.1 (a) Real events are conserved

Per team-game, 2025 (the other seasons show the same pattern; every figure is
in the json):

| event | v2 | v4 | games changed v2->v4 | games changed v3->v4 (the phantom switches alone) | box |
|---|---:|---:|---:|---:|---:|
| FGA | 57.941 | 57.941 | 0 | 0 | 58.006 |
| FGM | 25.648 | 25.648 | 0 | 0 | 25.678 |
| 3PA | 22.644 | 22.644 | 0 | 0 | 22.657 |
| 3PM | 7.657 | 7.657 | 0 | 0 | 7.660 |
| turnovers | 11.783 | 11.783 | 0 | 0 | 11.784 |
| FTA | 18.931 | 18.874 | 320 | **0** | 19.117 |
| FTM | 13.653 | 13.608 | 295 | **0** | 13.801 |
| points (excl. technical) | 72.606 | 72.561 | 295 | **0** | -- |
| technical points | 0.141 | 0.187 | 295 | **0** | -- |
| OREB (chance continuations) | 10.542 | 10.434 | 982 | 982 | 10.349 |
| DREB-started possessions | 24.053 | 23.984 | 646 | 646 | 24.349 |

- **Field goals, threes and turnovers are identical in every game of every
  season** (0 games changed).
- **FTA, FTM and points** move only through the 09-18 technical-FT lookahead
  (v3). The phantom switches change none of them in any game.
- **Rebounds are the one event family the layer's own tallies move**
  (-0.07 to -0.11 per team-game), because the stray rebound rows are no
  longer turned into possessions. Their mean is further from the box, and
  the per-game MAE vs box worsens (OREB 0.235 -> 0.275, DREB 0.452 -> 0.507).
  The box counts those stray rows as rebounds; it is compiled from the same
  ESPN feed, so it is not independent here.
- **The rebound MODEL is unaffected:** it reads raw rows through
  `models.event_stream` (section 4).
- **Fouls:** not a column of the possession table. No foul handler changed,
  and the and-one foul is still counted in `_handle_fga` before the new
  branch. They are identical by construction, but this was **not measured on
  output**.

### 3.2 (b) Possessions vs the box estimator, game by game

Mean (pbp count - box estimator) per team-game:

| season | v2 | v4 |
|---|---:|---:|
| 2022 | +0.914 | +0.296 |
| 2023 | +0.830 | +0.206 |
| 2024 | +1.014 | +0.388 |
| 2025 | +0.883 | +0.300 |

What remains (+0.2 to +0.4) is the real possessions the box cannot see (C2,
eventless horn possessions) and the 0.44-FTA trip approximation.

Slope of the per-game gap on each former class's per-game count (same
regressors for both versions; SE about 0.03-0.15):

| former class | 2022 v2 -> v4 | 2023 | 2024 | 2025 | expected for a removed phantom |
|---|---|---|---|---|---|
| A missed and-one FT rebound | +0.54 -> **-0.46** | +0.57 -> **-0.42** | +0.56 -> **-0.44** | +0.61 -> **-0.38** | **-0.44** (the estimator still carries 0.44 per and-one FTA) |
| E own-OREB restart | +0.66 -> -0.37 | +0.60 -> -0.38 | +0.69 -> -0.33 | +0.65 -> -0.36 | -0.44 |
| C1 stray OREB | +1.13 -> **+0.15** | +1.30 -> **+0.29** | +1.05 -> **+0.06** | +1.20 -> **+0.19** | 0 |
| B1 stray DREB at FTs | +0.21 -> -0.38 | +0.69 -> +0.00 | +1.14 -> +0.21 | +0.99 -> +0.09 | 0 |
| B2 (mixed) | +1.27 -> +0.67 | -0.12 -> -0.26 | +1.22 -> +0.53 | +0.39 -> -0.27 | kept part stays |
| C2 real (kept) | +0.60 -> +0.70 | +0.67 -> +0.66 | +1.02 -> +1.07 | +0.93 -> +0.97 | unchanged, as it should be |

**The phantom slopes land on their expected values**:
- A and E sit near -0.44, which is zero once the estimator's own 0.44 per
  and-one FTA is accounted for.
- C1 goes from about 1.2 to about 0.1-0.3.
- B1 goes to about 0.
- C2, the real class, is unchanged.

### 3.3 (c) The `unknown` class table before and after (same classifier)

| class | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| A | 3,719 -> 0 | 3,914 -> 1 | 4,042 -> 0 | 3,754 -> 1 |
| B1 | 469 -> 152 | 533 -> 171 | 465 -> 40 | 419 -> 43 |
| B2 | 2,521 -> 1,770 | 2,091 -> 1,636 | 536 -> 146 | 510 -> 153 |
| C1 | 840 -> 44 | 970 -> 33 | 1,027 -> 20 | 1,019 -> 18 |
| C2 (real) | 690 -> 692 | 724 -> 731 | 511 -> 510 | 506 -> 508 |
| D (real) | 334 -> 330 | 369 -> 367 | 222 -> 222 | 197 -> 195 |
| **all `unknown`** | **8,573 -> 2,988** | **8,601 -> 2,939** | **6,803 -> 938** | **6,405 -> 918** |
| same-team consecutive possessions | 13,787 -> 7,998 | 13,676 -> 7,990 | 12,321 -> 6,139 | 11,790 -> 5,940 |

- The B1 and B2 remainders are the > 3 s cases the rule keeps as real.
- The small C1 remainder is stray OREBs by the team that did not just have
  the ball; the rule deliberately does not touch those.

### 3.4 (d) Duration by start type, clock-complete regulation, 2025

| start type | share v2 -> v4 | mean s v2 -> v4 | share of possessions <= 2 s, v2 -> v4 |
|---|---|---|---|
| period_start | 0.0147 -> 0.0148 | 20.87 -> 20.88 | 0.23% -> 0.23% |
| DREB | 0.3508 -> 0.3527 | 14.51 -> 14.54 | 6.30% -> 6.20% |
| TOV | 0.1695 -> 0.1710 | 14.88 -> 14.94 | 4.13% -> 3.86% |
| **made_FG** | **0.3685 -> 0.3509** | **21.79 -> 22.19** | **2.81% -> 1.54%** |
| made_FT | 0.0912 -> 0.1068 | 18.59 -> 18.95 | 3.82% -> 3.01% |
| other | 0.0052 -> 0.0038 | 1.76 -> 1.69 | 77% -> 80% |
| all | | 17.656 -> 17.809 | 67.94 -> 67.36 per team-game |

- **The made-FG cell loses its 1-2 s phantoms.** Its share of possessions at
  2 s or less halves.
- Its mean rises to 22.19 s, the clean value the diagnostic measured
  (22.16).
- The made-and-one successors move to `made_FT` (+1.6 pp share).

The other seasons move the same way (made_FG 21.66 -> 22.10 in 2022, 21.70 ->
22.13 in 2023, 21.57 -> 22.00 in 2024).

**The accepted like-for-like G1 against the BUILT v4** (`scripts/diag_g1_lfl_v4_v1.py`,
pbp-complete graded 2025 games): engine 69.837 vs v4 count 68.196 =
**+1.641** per team-game. The in-memory correction gave 68.231 (+1.606); the
difference is v4's rules also catching stray rows that did not produce an
`unknown` possession, plus the technical lookahead.

## 4. Consumer delta (read-only; nothing rebuilt or retrained)

`scripts/diag_event_layer_v4_consumer_delta_v1.py` ->
`results/g1g5_diag/event_layer_v4_consumer_delta.json`. Rows are matched v2
vs v4 on a version-stable identity (game, period, offense, end clock and an
occurrence counter), because `poss_index` renumbers after every removed
phantom. It changes on about 35% of rows, but it is an identifier and no
consumer uses it as a feature.

2025 (2022-2024 in the json):

| consumer (served) | input today | rows v2 -> v4 | removed / added | matched rows whose used columns change | regenerate |
|---|---|---:|---:|---|---|
| **clock** `design_v2` (feeds v3c S1 + v5b) | `possessions` **v1** (hard-coded `clock.DEFAULT_POSS_DIR`; same segmentation as v2) | 768,834 -> 762,309 | 6,547 / 22 | start_reason 15,581; start_clock and duration 4,708; start_score_diff 3,974; terminal_event 2,700; off_in_bonus 303 | `scripts/exp_clk6_r6_arms_v1.py --arm L2 --step design` (below) |
| **possession_outcome** round-2 design (first + cont) | `possessions_v2` chances (`POSSESSIONS_VERSION="v2"` constant) | 876,408 -> 876,076 (design-eligible terminals) | 3,613 / 3,281 | start_reason 13,470; start_clock 1,545; start_score_diff 1,422; is_transition 294; off_in_bonus 57; fta 29; points 21; terminal_event 14 | `scripts/train_possession_outcome_v2.py` after its version constant becomes an argument (a versioned sibling; not written) |
| **rotation** hazard set (in memory) | `possessions` v1 (hard-coded `rotation.DEFAULT_POSS_DIR`) | 768,834 -> 762,309 | 6,547 / 22 | duration 4,706; start_clock 4,708; start_score_diff 3,974 | `scripts/train_rotation_v3b_s1.py` after a poss-dir argument (sibling; not written) |
| **late_game** design (not served) | PO round-2 design + `possessions_v2` chances (hard-coded) | 886,756 -> 879,021 | 8,477 / 742 | start_score_diff 3,959; points 2,574 | `scripts/build_late_game_design_v1.py` after the PO design is rebuilt and its chances path becomes an argument |
| fg_make, rebound, free_throw, usage | raw CBBD pbp via `models.event_stream` (does not run the possession machine) | unchanged | 0 / 0 | none from the phantom switches; free_throw's technical-FT sibling already exists (`trips/attempts_v1_era_techfix.parquet`, 09-18) | -- |

Every consumer that reads the possession layer picks its version by a
hard-coded path or constant. **Switching any of them to v4 is a one-line
change in a versioned sibling trainer.** That is deliberately not done today.

## 5. Join point for lane A's corrected foul state

Lane A's state (`data/processed/models/possession_outcome/round6/foul_accrual_poss_v2.parquet`,
built by `scripts/build_foul_accrual_design_v2.py`) is keyed
**(game_id, season, poss_index)**. The chance-level overlay
(`round8/in_bonus_overlay_v2state_v1.parquet`) is keyed
**(game_id, poss_index, chance_number)**. Its columns:
- `def_team_fouls_true`, `off_team_fouls_true`: the counter at open, minus
  the possession's own pre-open fouls;
- `off_in_bonus_true`;
- `def_pre_open`, `off_pre_open`, `def_trip`, `def_silent`;
- the overlay's `in_bonus`.

**`poss_index` is only a valid join key between tables built by the same
machine configuration**, and v4 renumbers it. The join point:

1. **Replay.** Lane A's replay builds `_GameMachine(gm, sub)`. For v4 it must
   build `_GameMachine(gm, sub, tech_lookahead=True,
   **possessions.VERSION_EVENT_FIXES["v4"])`: the same four arguments
   `build_possessions_v4.py` passes. Its (game_id, season, poss_index) keys
   are then those of `possessions_v4`, one to one. The v4 switches touch no
   foul handler; `_handle_fga` still counts the and-one foul before the new
   branch. So the replayed foul counts per game equal v2's, and only the
   possession boundaries move.
2. **Key and columns:** keys (game_id, season, poss_index), plus
   chance_number for the overlay. Expected columns are the ones listed above,
   plus a `machine_version = "v4"` column.
3. **Checks before use:**
   - an anti-join on (game_id, season, poss_index) against
     `possessions_v4/possessions_{season}.parquet` must be empty in both
     directions;
   - `period`, `offense_team_id` and `start_clock` must be equal on every
     joined row;
   - the per-game foul totals must equal the v2 build's.

With that, **`possessions_v4` + `chances_v4` + the v4-keyed foul state is the
single input of the next full retrain.**

## 6. Resume commands (nothing below was run today)

Clock round 6 (`docs/models/clock/experiments.md` section 28). The arm roots
are new directories; no served artifact is touched.

    # L2 (v4 training table, engine labels)
    .venv/Scripts/python.exe scripts/exp_clk6_r6_arms_v1.py --arm L2 --step design
    .venv/Scripts/python.exe scripts/exp_clk6_r6_arms_v1.py --arm L2 --step s1
    .venv/Scripts/python.exe scripts/exp_clk6_r6_arms_v1.py --arm L2 --step latent
    # L2a (phantoms removed, made-and-one label kept at made_FG)
    .venv/Scripts/python.exe scripts/build_possessions_v4.py --variant l2a --seasons 2022 2023 2024 2025
    .venv/Scripts/python.exe scripts/exp_clk6_r6_arms_v1.py --arm L2a --step design
    .venv/Scripts/python.exe scripts/exp_clk6_r6_arms_v1.py --arm L2a --step s1
    .venv/Scripts/python.exe scripts/exp_clk6_r6_arms_v1.py --arm L2a --step latent

- **Wrapper status.** `exp_clk6_r6_arms_v1.py` runs the served trainers
  (`train_clock_v3c_s1.py`, `exp_clk5b_mean_consistent.py`) unedited, with
  their path constants pointed at `data/processed/models/clock/r6_<arm>/`.
  It has passed `--dry-run` only; its real steps have not been executed.
- **Engine wiring (the clock lane's, in an engine file):** a default-off
  `ENGINE_CLOCK` mode that reads `clock/r6_<arm>/` (a `V3C_MODES` entry plus
  the refitted sigma), parity-proved like `2185b27`.
- **Closed loop.** `scripts/diag_g1g5_tap_v2.py` with that mode, then
  `scripts/grade_clock_r6_v1.py`. Its truth should then be read from
  `possessions_v4` directly (as `diag_g1_lfl_v4_v1.py` does), not from the
  in-memory correction.
- **D1:** season-part pooling needs a new trainer option; not written.

Table rebuilds for the next full retrain (each needs its version switch first;
see section 4): possession_outcome (`train_possession_outcome_v2.py`), rotation
(`train_rotation_v3b_s1.py`), late_game (`build_late_game_design_v1.py`), and
lane A's foul state (section 5). The event-stream models need no rebuild for v4.

## 7. What is NOT established

1. **Whether the box or v4 is right about the stray rebound rows.** v4
   treats them as administrative; the box counts them as rebounds. They share
   the same feed, so the box cannot referee.
2. **Fouls.** Conservation is by construction, not measured on output.
3. **B2 > 3 s and B1 > 3 s** stay as real possessions by the 3-s rule.
4. **The L2 / L2a wrapper** has not been executed.
5. **No consumer reads v4 yet**, nothing is retrained, and nothing is adopted.
