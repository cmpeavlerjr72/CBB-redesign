# Engine inputs v3 (live-path replay) and what it moves -- fold 2, 2024-25

2026-09-30, Lane G. Nothing is adopted and no default changed. v2 files were never written to.
Code state at the run: HEAD `ef7a0cd`; before the simulation `git status` showed no foreign edit under `src/cbb_sim` except
`src/cbb_sim/pbp/possessions.py` (another lane's `v4` event-layer switches, all default OFF). Lane G's own
uncommitted edits were limited to the live-path files. The default path was re-checked: the served engine on v2 inputs reproduces
`po4b_R_s25` BIT-IDENTICALLY on 8 of the 500 games x 3 seeds (26 numeric columns), so the paired deltas below are read against a
reference the current tree still reproduces.

## 1. The v3 build

`scripts/build_engine_inputs_v3_replay.py`: for each of the 151 fold-2 game dates (5,710 games, v2 game order) the live builder was run
with as-of = first tip minus 30 min and `created_at` = that cutoff (replay), by `diag_live_parity_v1.run_date`, then assembled into
`data/processed/models/engine_v3/{games,arrays,names}_F2_2025.*` plus `event_block_F2_2025.npz` (the round-2 event team block for the same rows).
Backtest and live are one code path by construction. About 85 minutes on 2 cores. 30 team-game sources across the season had not finished
at a date's cutoff (late-night games dated the day before; date-keyed features count them, a wall-clock cutoff would not): the replay
records the count instead of raising (`strict_finish=False`); the live default still raises.

hf_sync_data.py bulk key to register (not synced): `engine_inputs_v3` -> `data/processed/models/engine_v3` (14 MB; the directory is untracked,
not committed). Per-date counts: `docs/tests/engine_inputs_v3_replay_per_date_2026-09-30.csv` (games, cells, cells differing by cause, per date).

Totals over the 151 dates (cells = array cells compared against v2; 6,710,912 cells, 1,435,899 (21.4%) differ):

| cause | cells differing from v2 | what it is | share of the object |
|---|---:|---|---|
| stale slot rows | 1,374,856 | backtest `merge_asof(backward)` carried an earlier (often last season's) row | `shooter_att_c__rim` differs in 39.7% of real slots, `shooter_make_c__three` 36.8%, `shooter_shrunk_dev_c` 34.3%, FT shooter 47.8%, usage FGA_3 29.0%, rebound 27.2% (134,707 real slots) |
| whole-season fill constants | 41,319 usage slot-class cells at the constant (5 classes) plus 784 team-games at the clock `tempo_prior_game` constant | median over the full season table | usage 6.1% of slot-class cells; clock 6.9% of team-games |
| `has_prior_season_fg` | 713 (first-row-class dependence); 3,590 differ in total (2.7% of slots) | class of the player's first shot in the game | 0.5% of slots |
| all-zero possession-outcome block | 5,170 cells = 235 team-games (2.1%, about 117 games) | `season_idx = 0`, site 0, days 0 | 2.1% of team-games |
| anonymous rotation | 4,141 cells = 1,144 team-games (10.0%) | game's own on-floor data incomplete | 10.0% of team-games |
| round-2 event block not in design / clock fallback | 7,539 / 1,653 | back-filled from an earlier game or left at 0.0 | 530 team-games |
| fg team form of pbp-incomplete games (new cause, found here) | 1,767 | backtest backward-fills games not in the fg design | |
| residual, not attributed | 620 (0.009%): 21 fg team cells, 290 shooter, 133 usage, 77 shot-mix, 94 rebound, 5 other | the four pilot dates had none; this is the tail across 151 dates; not chased | |

## 2. Training-table trace (read-only; nothing rebuilt or retrained)

The four defects are confined to the ENGINE-INPUTS builder. The training tables do not share them, with three smaller items that are
not the same defects:

| table | column | defect | affected share |
|---|---|---|---|
| fg_make design v2 (`design_v2_shotshooter`) | shooter block | stale carry: none, one exact row per attempt | 0% |
| same | `has_prior_season` | none in training (class-specific per row, correct). SERVE MISMATCH: the engine's shared slot column is class-independent; the two definitions disagree in 1.39% of player-games | 1.39% of player-games |
| same | `lg_make_asof`, `off_make_raw` `fillna(whole-table median)` | defect 2 in training | 0.02% / 0.12% of rows |
| fg design, `free_throw` design, usage as-of | opening-day league/position rate `bfill` from the next date (forward-looking) | 1.98% of fg rows and 1.82% of usage rows are on a season's first date |
| usage as-of v2 | `fillna(median)` of league/position rates | defect 2 (size bounded by the first-date rows above) | <= 1.8% |
| clock design | `off_tempo_rel`/`def_tempo_rel` fill 1.0, `tempo_prior_game` and `days_since_start` fill with the whole-table median | defect 2 in training | 0.57% of rows (14,442 of 2,607,192) |
| possession_outcome design (round 1 and round 2) | zero blocks | none: games with no possession table are DROPPED, `season_idx` is 0 for no 2023+ row; 3.4% of rows sit at 0.0 legitimately (no history) | 0% all-zero |
| rotation (`load_team_possessions`, `build_asof_player_features`) | anonymous rotation | none: incomplete games are dropped from training | 0% |

So training on covered games only, serving on all games is a population mismatch, not a training-table defect: 2.1% (zero block) and
10.0% (anonymous rotation) of served team-games come from data the sub-models were never trained on.

## 3. Effect on the gates (500 games, 25 seeds 0-24, paired; `po4b_R_s25` reference, `po4b_R_s25_floor` = seeds 1000-1024)

Same grader path as 2026-09-18 (`eval_gates.py` per run, `diag_pair_gate_reports.py`, `grade_po4b_closed_loop.py`). Full tables:
`docs/tests/engine_inputs_v3_replay_pair_v3_vs_v2_2026-09-30.md` and `..._pair_hybrid_vs_v2_...md`. "x floor" = |delta| / |floor run - reference|.
"hybrid" = v3 team, roster, rotation and event arrays with the v2 slot arrays, which separates the slot defects from the rest.

| gate | line | v2 | v3 | v3 - v2 (x floor) | hybrid - v2 (x floor) |
|---|---|---|---|---|---|
| G1 | possessions mean (target 68.328) | 70.019 | 69.918 | -0.101 (1.0x) | -0.040 (0.4x) |
| G1 | possessions SD | 5.609 | 5.609 | 0.000 | +0.026 (0.5x) |
| G5 | margin SD ratio | 0.9695 | 0.9711 | +0.0016 (0.1x) | +0.0151 (1.1x) |
| G5 | total SD ratio | 0.8355 | 0.8286 | -0.0069 (7.7x) | +0.0026 (2.9x) |
| G5 | home/away correlation (target 0.2374) | 0.1063 | 0.1023 | -0.0040 (0.3x) | -0.0009 (0.1x) |
| G5 | margin SD / total SD (points) | 12.226 / 15.842 | 12.143 / 15.799 | -0.083 (1.6x) / -0.043 (3.6x) | +0.020 (0.4x) / -0.042 (3.5x) |
| G9 | margin bias | +0.112 | +0.321 | +0.210 (0.9x) | +0.127 (0.6x) |
| G9 | total bias | -0.962 | -0.297 | +0.665 (1.9x) | -0.073 (0.2x) |
| G9 | calibration slope | 0.8920 | 0.8858 | -0.0062 (0.2x) | +0.0165 (0.4x) |
| G4 | eFG% (target 0.5086) | 0.4990 | 0.5029 | +0.0039 (6.5x) | +0.0001 (0.2x) |
| G4 | oreb% / tov% / ft_rate | .2836 / .1762 / .3195 | .2841 / .1762 / .3185 | +.0005 (2.5x) / 0 / -.0010 (0.6x) | +.0004 (2.0x) / +.0002 / -.0004 |
| G3 | 3PA share / rim share | .3894 / .3721 | .3897 / .3718 | +.0003 / -.0003 (0.4x) | +.0001 / +.0002 |
| G6 | home margin non-neutral (target +5.669) | +5.989 | +6.225 | +0.236 (1.0x) | +0.158 (0.7x) |
| G7 | OT rate (target 0.068) | 0.0312 | 0.0311 | -0.0001 (0.0x) | -0.0014 (0.4x) |
| G8 | rotation minutes SD ratio | 1.2304 | 1.2353 | +0.0049 (16x; floor is 0.0003) | +0.0032 (11x) |
| G8 | top-1 FGA share / players used per team-game | .2574 / 8.83 | .2572 / 8.82 | -.0002 / -0.01 (0.5x) | -.0015 / -0.01 |
| G2 | powered PPP cells inside tolerance | 2/9 | 3/9 | +1 cell | 0 |
| responsiveness slopes (off_3pa / rim / tov) | | 1.031 / 0.888 / 1.084 | 1.041 / 0.892 / 1.052 | -0.7 / -0.6 / +2.75 floors (tov toward 1) | |

No gate-level verdict changes (G1 FAIL, G2 FAIL, G4 FAIL, G5 FAIL, G6 PASS, G7 FAIL, G8 FAIL, G9 FAIL, in every run). G5 PIT K-S p 0.455 -> 0.290 (both PASS).
Per-game paired movement (mean over 25 seeds within a game, then over games): mean |delta margin| 1.53, mean |delta total| 1.43 points
(hybrid vs v2: 0.25 and 0.35; v3 vs hybrid: 1.34 and 1.24). For scale, a different set of 25 seeds on the SAME v2 inputs moves a game's mean margin by 2.83 and
total by 3.83 points, so v3 moves an individual game about half as much as a reseed. Mean shift v3 minus v2: margin +0.21, total +0.67.

Reading: nearly all of the movement is the stale-slot family (hybrid vs v2 is 0.25 points per game). It raises totals (+0.67, G9 total bias moves toward 0),
raises eFG% by 0.4 pp (toward the target, 6.5 floors), slightly lowers the total SD ratio (away from 1). The team/roster/rotation/event defects
move G8 minutes SD ratio and G5 total by amounts that clear the tiny floors but are small in absolute terms. A single pair of runs cannot split the slot family into stale rows, fill constants and
`has_prior_season_fg`; by count the last is 0.5% of slots and cannot carry a 0.4 pp eFG move.

## 4. Earlier conclusions that are exposed, by defect

Every gate read before today was graded on v2 inputs. The defect applies to both arms of any PAIRED comparison, so paired deltas (clock, event, fg rounds) are less exposed than absolute levels,
but an arm whose effect interacts with the shooter or rotation inputs is not covered by that argument.

1. Stale slot rows (largest). Lines moved: G4 eFG, G9 total bias, G5 total SD ratio, G6 home margin (slightly). Direction measured: v2 under-stated eFG by 0.4 pp, under-stated totals by 0.67 points (about three quarters of the v2 G9 total bias of -0.96 was this defect: v3 minus hybrid is +0.74), and over-stated total dispersion slightly (v2 total SD ratio 0.8355 vs v3 0.8286; v3 is further from 1). Any conclusion of the form "the engine is x points low on totals / shoots y low" that cited v2 absolutes is exposed by that much; fg_make, usage and rebound adoptions judged on eFG or total bias inherit it. The stale values also cross the season boundary (up to 130 attempts of last season's counts in early November), so early-season readings are the most exposed; the 500-game stride subset has 0 powered early months anyway.
2. Whole-season fill constants. Reach: 6.1% of usage slot-class cells, 6.9% of team-games on the clock tempo constant. Lines: G1 possessions mean/SD and G8 usage lines. Direction: the tempo constant (68.70) sat up to 1.9 possessions from the as-of league mean on the pilot dates, so it could lean G1 possessions either way by date; the measured G1 mean movement is -0.10 (1.0x floor), so no standing G1 conclusion moves by more than the floor. Not separable from defect 1 in a single run.
3. `has_prior_season_fg`. 0.5% of slots; game-day-dependent but binary. Could not move a gate line beyond the floor; listed for completeness. Not separately measured.
4. All-zero possession-outcome block (2.1% of team-games) and anonymous rotation (10.0%). Lines: G6 home margin (zero site flags lose home court in consumers of `team_static`: v3 raises home margin +0.16 in the hybrid, 0.7x floor), G8 minutes SD ratio and players-used (anonymous rotations have no named players: +0.003 on the SD ratio), G5 total (-0.04, 3.5x). All small in absolute terms. Conclusions about G8 rotation lines (which failed in every run and still do) are not changed.

## 5. Remaining work

- Adopt v3 as a served input only after a PM decision; the paired arms of rounds 2 to 5 could be re-read on v3 (25 seeds, 12 to 13 min on 2 cores each).
- Chase the 620 residual cells (mostly fg shooter block and usage; need per-cell traces on the dates in `dates/*.json`): 2 to 3 h.
- Close the population mismatch (zero block, anonymous rotation): train-on-served-population question for the PM.
- Fix `has_prior_season_fg` definition (class-specific slot or any-class), the round-4 shared columns being rim-row-only, and the fill constants in the training tables (bfill, whole-table medians): 4 to 6 h plus paired sim.
- Everything from `docs/ops/live_slate_path_2026-09-30.md` section 5.6 still stands (2027 artifacts, ratings entry point, ingestion, day-1 priors).

Commands: `scripts/build_engine_inputs_v3_replay.py shard|assemble`, `scripts/diag_v3_replay_summary_v1.py`, `scripts/diag_v3_posthoc_v1.py`,
`scripts/run_engine_live.py --tag F2_2025 --input-dir data/processed/models/engine_v3 --season 2025 --seeds 13 --seed-offset 0 --replay --subset-po4b --plain-out --out-dir results/engine_v3_replay/sim`
(second half `--seeds 12 --seed-offset 13`), `scripts/exp_v3_merge_runs_v1.py`, then `eval_gates.py`, `diag_pair_gate_reports.py`, `grade_po4b_closed_loop.py --ref po4b_R_s25 --floor po4b_R_s25_floor --arms v3_replay_s25`.
