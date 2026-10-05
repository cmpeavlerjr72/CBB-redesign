# Player layer day-1 sizing, Phase 1 (2026-10-05)

Question: do game-level outputs depend on the player layer on opening day? (readiness gap "Player layer day-1 priors": `build_live` gives 0 of 222 opening-day team-games a rotation prior.)

**Arms.** (a) SERVED: the v3 inputs as built (live replay; a team's first game is already anonymous, later games name players from earlier games this season). (b) ANON: every game of the opening window rebuilt by `build_live` with no rotation prior (the 2026-27 opening-day state: league-mean rotation, no named slots, no-history shooter / usage / rebound / FT blocks). Window = first 14 days: 2024-25 (fold 2, 599 games, 598 graded) and 2023-24 (fold 1, 643 games). Served stack v2 defaults (fold 1: the fold-1 artifacts via `overrides_V2.json`, `ENGINE_ROTATION_SCHEME=static` as in `chain_fold1_v1.py`). 50 seeds each, paired streams; reseed floor = SERVED at seed offset 1000. Verified truth.

**Checks.** Unpatched rebuild of 2 window dates per fold equals the v3 arrays on every array. `F2_srv_o0` equals box `v3full_COMB9GCTKD_s200_o0` (seeds 0-49, window games) bit for bit on all 28 game columns. Games where SERVED has no named slot (181 F2, 182 F1) are identical between arms, as they must be.

| metric (all window games) | F2 SERVED | F2 ANON | F2 ANON-SERVED [95% game bootstrap] | F2 reseed floor | F1 SERVED | F1 ANON | F1 ANON-SERVED | F1 reseed floor |
|---|---|---|---|---|---|---|---|---|
| margin MAE | 10.43 | 10.40 | -0.03 [-0.18, 0.12] | +0.05 [-0.16, 0.25] | 9.91 | 9.92 | +0.02 [-0.15, 0.19] | -0.03 [-0.21, 0.15] |
| total MAE | 14.73 | 15.17 | **+0.45 [0.18, 0.72]** | -0.06 [-0.31, 0.18] | 14.29 | 15.05 | **+0.76 [0.49, 1.01]** | -0.07 [-0.29, 0.15] |
| total bias (sim - actual) | -4.94 | -7.52 | **-2.58 [-2.76, -2.41]** | +0.05 [-0.19, 0.30] | -5.22 | -7.98 | **-2.76 [-2.94, -2.58]** | +0.11 [-0.14, 0.35] |
| home-win Brier | 0.147 | 0.146 | -0.001 | +0.002 | 0.158 | 0.160 | +0.001 | 0.000 |
| gate verdicts G1-G9 | identical except G8 (player gate: FAIL -> NEEDS-INSTRUMENTATION, no named players); window-sized, UNDERPOWERED for verdicts |||||||

Games where SERVED names players (417 F2 / 461 F1): total shift -3.70 / -3.85 per game, total MAE +0.64 / +1.05. Margin is unaffected in every segment (both teams lose equally).

**Mechanism (per possession type, named games, pooled).** ANON lowers every make rate: rim 0.583 -> 0.574, jump2 0.382 -> 0.372, three 0.331 -> 0.315, FT 0.681 -> 0.646 (F2; F1 the same within 0.3 pp); volumes move < 1%. The no-history shooter state is fitted on real no-history players (freshmen, walk-ons), so an all-anonymous roster plays like a bench of unknown freshmen. eFG (pooled, window) 0.497 -> 0.487.

**Informational (not a bake-off read).** Lane F's seeded inputs (returners + transfers named, league role-profile shares, `engine_v3_seed`, F2 only): total MAE -0.22 [-0.31, -0.13] vs SERVED, -0.58 on the 181 true first games, total bias +0.67.

**Verdict.** MATERIAL game-level gap on totals (both folds, 9-12x the reseed floor on total bias, CI excludes 0 on total MAE); none on margin. Phase 2 (pre-registered bake-off of day-1 player priors) is warranted.

Scripts: `scripts/build_engine_inputs_anon_window_v1.py`, `scripts/run_engine_window_v1.py`, `scripts/grade_player_day1_v1.py`. Results: `results/player_day1/` (phase1_F{1,2}.json, gates_*.md; gitignored).
