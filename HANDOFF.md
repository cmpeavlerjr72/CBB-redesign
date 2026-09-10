# HANDOFF.md

## SESSION 2026-09-10 -- READ THIS FIRST

What happened (numbered):
1. Reviewed CBB-Monte (code, 2025-26 results, data, tournament fork) and cfb-props-sim methodology via five workers; reports in `docs/postmortem/`.
2. Wrote `docs/FRAMEWORK_PLAN.md`; user decided: free odds only, grade on scores/stats first, no Odds API yet, keep KenPom arm, remote CBB-redesign, game markets first then props, social out of scope.
3. Verified data sources; CollegeBasketballData API works with the CFBD key (shared quota ~30k calls/month; ~29.5k remaining after day 1 minus the pbp pull).
4. Repo initialised, venv, HF dataset `mvpeav/cbb-sim-data` (nothing pushed yet).
5. Built L0: universe, crosswalk, gate references, KenPom snapshots, leak harness. Pre-registered Control engine.

6. Control engine, pace bake-off, pbp pull, possessions layer, coaches, variance decomposition, L3 bake-off all landed (see PROJECT_STATUS.md). Commits through 5a1ef86.

Still running / resume-safe: shot-classification diagnostic -> docs/tests/shot_classification_diag_2026-09-10.md; clock bake-off -> docs/models/clock/; rotation bake-off -> docs/models/rotation/ (+ player crosswalk src/cbb_sim/data/player_ids.py). Each writes its own docs and artifacts; PM commits after review.

Refuted this session: game-level pace as a sampler (L14); independent count draws (L10); CBBD season ratings as pregame features (L7); CFB's "team-beyond-coach = 0" for CBB (L15).

Standing rules recap: PM/worker split; bake-off before any choice; no hand tuning on output; bottom-up; matchup-specific; multi-level evidence; profitability frame with accuracy-first phase; sealed 2026; created_at < tipoff; leak test every external feature. Full text in `CLAUDE.md`.

Watch items: FTA/FGA trend 0.305 -> 0.352; November pace +3 poss; hoopR event vocabulary drift; player_box `active` placeholder before 2026; CBBD `/recruiting/portal` filter is `year` not `season`; git push prints a harmless "Key not valid for use in specified state" credential warning.

Refuted this session: nothing yet.

Evidence trail: `docs/tests/data_audit_hoopr_2026-09-10.md`, `docs/tests/data_audit_cbbd_2026-09-10.md`, `docs/tests/gate_reference_2026-09-10.md`, `docs/tests/leak_test_kenpom_2026-09-10.md`.
