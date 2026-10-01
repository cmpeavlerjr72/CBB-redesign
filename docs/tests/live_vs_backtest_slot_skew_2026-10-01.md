# Live vs backtest slot skew (lane F, 2026-10-01). Fold 2 (season 2025) only; nothing sealed read; no served default changed.

## Answer

The "more known slots on 2024-12-10 and 2025-02-11" are NOT a live-builder quirk. The shot-block backtest table `data/processed/models/engine/shot_block_K2_Ocell_F2_2025.npz` was built against the v2 engine inputs' roster slots, while the adopted full-size read (and the live chain) use the v3 inputs. In 334 of 5,710 games the v2 inputs have NO named players (anonymous fallback on both teams) and the v3 inputs name them. The live-built inputs equal the v3 inputs array for array, so live is consistent with v3; the shot-block table is the one input that still carried v2 slots. The adopted read therefore served `K2_Ocell` with "unknown shooter" shooter features in those 334 games and named slots everywhere else. Live is right as-of; the backtest table is stale. It is a table-vintage defect, not a leak.

## Cause, traced

1. Slot names come from the as-of rotation priors (`ROT.build_asof_player_features` + `build_priors`). The v2 builder (`scripts/build_engine_inputs.py:636-684`) builds priors from the season's on-floor table only. A team-game that has no row in that table (a pbp-incomplete game: no possessions, so no on-floor rows) is absent from the prior ordering, gets `priors.get(...) is None`, and takes the anonymous fallback (`roster_cbbd = -1..-15`). The live builder (`src/cbb_sim/live/features.py:361-366`, `rotation_priors`) adds a stub row (pid -1, zero minutes) per slate team-game "so the team-game exists in the ordering", so those games get priors from their teams' EARLIER appearances. The v3 inputs were built by the live-replay path, so they carry the stub behaviour.
2. Measured on all 5,710 games (v2 `engine/arrays_F2_2025.npz` vs v3 `engine_v3/arrays_F2_2025.npz`): roster ids differ in 334 games; in all 334 BOTH team-games are absent from the on-floor table (334 of 334). v2 has 1,144 anonymous team-games, v3 507. By month: 88 games in 2024-11, 71 in 12, 98 in 01, 68 in 02, 9 in 03. 2024-12-10 and 2025-02-11 contain such games; 2025-02-25, 2025-03-04 and 2024-11-04 do not.
3. AS-OF CHECK (which side is right): of the 8,123 players the v3/live inputs name and v2 does not (all in the 334 games), 100% have an appearance for that team on an EARLIER date and 0% appear in the game's own box. Nothing is named from the game itself, so neither side leaks. v2 is the conservative one (it loses roster information that was knowable before tip, for games whose own feed happens to be incomplete); live / v3 is correct.
4. The shot-block table `shooter` and `known` arrays equal the v2 roster (`known` mismatches vs the v2 inputs: 0; vs the v3 inputs: 8,123 slots in 334 games). `engine/shot_block.py:48-52` checks only that the game-id order matches, not the slots, and finds the file by tag (`F2_2025`), so the v3 inputs load the v2-aligned table.

## Who reads a slot-keyed table

- Slot-keyed LOOKUP FILES outside the inputs: only the shot-block table (`engine/shot_block.py`). Every other `np.load` in the engine is team-keyed (foul joint and R9ao3 LUTs, event team block, chance time, season anchor) or per-game (`team_rate_draw`, shared shooting). R9ao3's team-prior table is team-keyed.
- Slot-keyed arrays INSIDE the inputs directory (`slot_static`: fg_make shooter blocks and free-throw shooter blocks; `usage_rate`; `reb_rate`; `rot_share / rot_srank / rot_start / rot_fpm / rot_pavail`; `roster_cbbd`) are built per inputs version. Live-built vs the v3 arrays for the same games, dates 2024-12-10 (23 games), 2025-02-11 (37) and control 2025-02-25 (35): zero elements differ for every one of these arrays on every date (`roster_cbbd`, the five `rot_*`, `usage_rate`, `reb_rate`, `slot_static`, `team_static`; max abs 0.0). So fg_make, free throw, usage, rebound and rotation see identical slots live and in the adopted read. Against the v2 arrays they would differ (v2 vs v3 over the whole season: `usage_rate` 856,470 elements, `reb_rate` 145,915, `roster_cbbd` 9,555, `slot_static` and the rest, all in the 334 games plus their downstream rows), but v2 is not what the adopted read used.
- So the skew reaches exactly one served model: shot block (`K2_Ocell`, adopted 02:06 EDT 2026-10-01).

## Size of the difference (shot block; the 334 affected games of 5,710)

Fixed table = the sibling built from the v3 slots (below). Through the fitted model (coefficients per standard deviation: `shooter_known` +0.028, `shooter_blocked_c` +3.08):
- 8,123 of the 10,020 slot-sides in those games change (the rest are anonymous in both).
- Logit change per changed slot: mean abs 0.057, min -0.191, max +0.487.
- Weighted by rotation share (who takes the shot), per team-game: mean +0.029 logit, mean abs 0.032. At a block rate of order 0.1 per miss that is about +0.003 on P(block | miss), a few percent relative, in 5.9% of games; the game-level points effect is negligible. Direction: slightly more blocks where the table previously said "unknown shooter".
- Per date: 2024-12-10, 92 of 690 slot-sides (known 0 to 1); 2025-02-11, 25 of 1,110 (identical shooter rates on ids common to both: max abs 0.0); 2025-02-25, 2025-03-04, 2024-11-04: 0.
This is below any gate resolution; it is a vintage bug to close, not a result that moves the adoption read.

## Fix (applied as a versioned sibling; nothing served changes)

`scripts/build_shot_block_lut_v3in_v1.py` builds the K2_Ocell table from the v3 slots with the live builder's own function (`build_shot_block_lut_live_v1.build_table`, full-season events, per-date as-of) and writes `data/processed/models/engine_v3/shot_block_K2_Ocell_v3in_F2_2025.npz` (untracked data file; regenerate with the script, about 35 s). Evidence, `results/shot_block_v3in_sibling.json`:
- Bit-identity on unaffected games: `team`, `anchor`, `shooter`, `known` are bit-identical to the v2 table on all 5,376 games whose slots agree; `coef`, `mu`, `sd` equal.
- Affected games: `team` and `anchor` diff 0.0; `shooter` max abs 0.149, `known` max 1.0 (the intended change).
- Builder agreement: the live builder run per date (`as_of` = the evening clock) equals the sibling with max abs 0.0 on all four arrays on 2024-12-10 and 2025-02-11; on 2024-11-04, 2025-02-25 and 2025-03-04 the live table equalled the v2 table (no affected games).

To make the adopted read and the chain agree, the engine must read the sibling when it runs on v3 inputs. That changes served numbers in 334 games, so it is a Decision-11 style adoption and is not applied. Proposed patch (not applied) in `engine/shot_block.py` `ShotBlock.__init__`: before the `LUT_DIR / ...` fallback, if the loaded inputs directory holds `shot_block_{arm}_v3in_{slate}.npz`, use it (inert until that file sits in the served inputs dir; the daily chain builds its own per-slate table and is unaffected). Decision read: the adopted stack with and without the sibling on the 5,710 x 200 box run; expect no gate to move (about +0.003 on P(block | miss) in 5.9% of games).

## Not done

No full-size sim of the sibling (box). The block-rate level behind the +0.003 figure is an order-of-magnitude value, not recomputed from the design.

---

# Update (same day, task 4): engine flag, tap, and the live-vs-v3in check over many dates

**Engine flag (default off).** `ENGINE_SHOT_BLOCK=K2_Ocell_v3in` serves `data/processed/models/engine/shot_block_K2_Ocell_v3in_F2_2025.npz` (tracked, 0.7 MB; built by `scripts/build_shot_block_lut_v3in_v1.py`). The off path is bit-identical to parity v9: 60 games x 5 seeds, plain default, digest compare PASS on this tree with the flag code in place (and again after the `ENGINE_OT_STATS` loop edit). Registered in `shot_block.ARMS`; `DEFAULT` stays `K2_Ocell`; `tests/test_shot_block_v3in.py`.

**Local tap, direction only** (`scripts/run_laneF_v3in_tap_v1.sh`: default vs the flag, 400 games x 5 seeds, 2 workers, v3 inputs, fold 2): the 393 unaffected games (1,965 rows) are bit-identical between arms; the 7 affected games (35 rows) differ in 1 row; mean total points on the affected rows +0.17 (sd of the home-points difference 0.68). UNDERPOWERED: 7 affected games cannot decide anything (Decision 12); the full-size paired read is `docs/ops/box_queue/d1001_F_1.md`.

**Live chain table = v3in table, many dates** (`scripts/diag_shot_block_live_vs_v3in_dates_v1.py --auto 24`, `results/shot_block_live_vs_v3in_dates.json`): the live-built `K2_Ocell` table (`build_live` inputs + `build_table` as of the evening clock) vs the v3in sibling over 24 fold-2 dates: the 8 dates with the most affected games, 12 random other dates that have affected games, and 4 dates with none; 1,523 games (27% of the 5,710), 109 of the 334 affected games. On every date the roster slots equal the v3 inputs and `team`, `anchor`, `shooter`, `known` differ by max abs 0.0 (no date with any difference). 24 of the 151 fold-2 dates were checked (about 24 min of one core); the other 127 were not run (about 2.5 core-hours for all 151); resume command: `.venv/Scripts/python.exe scripts/diag_shot_block_live_vs_v3in_dates_v1.py <date> [<date> ...]`.
