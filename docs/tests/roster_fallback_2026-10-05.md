# Fallback roster for teams with no season-S roster (2026-10-05)

Pre-registered in `docs/models/player_day1/experiments.md` section 4 (commit 463ab25, before any arm ran); results in section 5.

Setting: A3 day-1 priors on the opening 14 days; a seeded random 20% of teams (73 in 2024-25, 72 in 2023-24) have their season-S roster withheld. R0 anonymous; R1 S-1 roster minus final-year players (proxy S-1 - start_season >= 3) ordered by S-1 minutes; R2 = R1 minus S-1 players listed on another team's season-S roster (observed outgoing transfers). Games with at least one treated team, 50 seeds, verified finals, 2025-26 not read.

| arm | fold | total MAE | bias | margin MAE | d total MAE vs R0 (CI) | reseed floor | rotation-minutes MAE (unnamed = 0) | named share of actual minutes |
|---|---|---|---|---|---|---|---|---|
| R0 | F2 | 14.56 | -6.44 | 10.51 | - | 0.335 | 22.98 | 0 |
| R1 | F2 | 14.20 | -5.03 | 10.29 | -0.353 [-0.60, -0.12] | | 15.43 | 0.46 |
| R2 | F2 | 14.26 | -5.50 | 10.23 | -0.298 [-0.50, -0.11] | | 14.82 | 0.46 |
| R0 | F1 | 15.45 | -8.22 | 10.09 | - | 0.119 | 23.08 | 0 |
| R1 | F1 | 15.04 | -6.42 | 9.78 | -0.408 [-0.68, -0.13] | | 14.64 | 0.50 |
| R2 | F1 | 15.07 | -6.71 | 9.99 | -0.373 [-0.62, -0.13] | | 14.27 | 0.50 |

Verdict: R1 passes the registered rule (beats R0 beyond the F2 floor by 0.018 only; F1 confirms with a wide margin); R2 fails the F2 floor. The F2 floor is a single noisy reseed (R0 at another offset gives 14.559, R1 at offset 1000 gives 13.980, paired deltas -0.24 to -0.36). Remaining bias is the opening-day total bias, a different object.

Use: `python scripts/build_engine_inputs_day1prior_v1.py serve --arm A3 --fallback R1 ...` (default off). Teams that used it are listed in the build diag `d1p_fallback_teams`. Runs: `results/player_day1/fb/` (gitignored); grader `scripts/grade_player_day1_fallback_v1.py`.
