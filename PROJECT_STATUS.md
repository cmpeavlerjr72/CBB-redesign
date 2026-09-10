# PROJECT_STATUS.md

Last updated: 2026-09-10 (day 1). Season tips Nov 1-3, 2026.

Reading order: `CLAUDE.md` -> `docs/FRAMEWORK_PLAN.md` -> `ARCHITECTURE_DECISIONS.md` -> `docs/SIM_GUARDRAILS.md` -> `docs/LEARNINGS.md` -> `docs/models/README.md` -> `HANDOFF.md`. Postmortem of last year: `docs/postmortem/01-06`.

## Where we are

Week 1 of 8 (`docs/FRAMEWORK_PLAN.md` section 7). Done on day 1:
- Postmortem of CBB-Monte (no residual signal vs close; root causes: uncentred KenPom on a drifting scale, no home court, independent team draws, unfitted dispersion, stale models, no player layer).
- Data on disk: hoopR pbp/player_box/team_box/schedules/shots/rosters/player_core 2022-2026 (+2027 schedule); CBBD lines 2013-2026, games 2013-2026, season ratings (end-of-season, banned as features), lineup samples; CBBD pbp with on-floor players 2022-2026 (bulk pull in progress).
- Foundations: game universe (31,103 games, D-I flag, truncation flag, sealed flag), team crosswalk (367 teams, 100% ESPN/CBBD/KenPom), per-season gate reference tables, KenPom point-in-time snapshots (centred), leak-test harness (KenPom as-of join passes; CBBD season ratings fail as expected).
- Docs: framework plan, decisions 1-6, guardrails with G1-G10, learnings L1-L8, control_engine pre-registration.

Done later on day 1:
- Control engine built and gated (docs/tests/control_engine_F2_2026-09-10.md): trails close by 0.36 margin / 0.54 total MAE, no ATS edge; own ratings tie KenPom; dispersion fails for structural reasons (L10). This is the bar.
- L2 pace bake-off: all 32 arms fail PIT; multiplicative formula adopted as a prior only; pace is emergent (Decision 7, L14).
- CBBD pbp 2022-2026 pulled (13.4M rows; onFloor from 2024 only, L13). Canonical possession/chance tables built (src/cbb_sim/pbp). Raw data and Control results mirrored to HF.
- Head-coach table 2022-2027 (Wikipedia API, 99.5%). Variance decomposition: player identity 10-52% of player rates; team-beyond-coach real at team level (L15).
- L3 possession-outcome bake-off: NO winner. Every arm fails calibration on 2025 via a +1.8 pp FGA_jump2 level shift; 2025 putbacks labelled differently at source. Diagnostic running.

In progress: shot-classification diagnostic (artifact vs real shift), L5 clock-consumption bake-off, L4 rotation bake-off.

## Production stack right now

Nothing shipped. No live runs.

## Next in queue

1. Resolve the 2025 jump2 shift: if ARTIFACT, fix events.py mapping, rebuild possessions, rerun L3 grid; if REAL, pre-register L3 round 2 with in-season adaptive refit and recency-weighted arms.
2. Pre-register possession_make (shot make/miss by class, FT%) with player terms keyed per L15.
3. Clock and rotation results -> engine v0 assembly (possession loop) -> gates G1-G7 vs Control.
4. `pbp_complete` flag in the universe from the CBBD feed-completeness check.
5. Seed-noise study to replace provisional tolerances.
