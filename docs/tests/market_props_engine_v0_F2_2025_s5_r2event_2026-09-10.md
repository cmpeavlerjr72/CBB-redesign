# Player-prop market scorecard -- engine_v0/F2_2025_s5_r2event (season 2025, settle=book)

Generated 2026-09-10 by `scripts/grade_market_props.py`. Results: `results/engine_v0/F2_2025_s5_r2event`.

## Status: NEEDS-INSTRUMENTATION

No props-lines file at `data\raw\cbbd\props_lines_2025.parquet` (none exist yet for this project -- CLAUDE.md bans scraping the retail props books directly). Schema this script expects the day one does: see `src/cbb_sim/eval/props.py` module docstring (`cbbd_game_id, athlete_id, stat, provider, snapshot, line, over_odds, under_odds`). `tests/test_eval.py::test_props_synthetic_grading` proves the grading logic below against a synthetic lines fixture in the meantime.

