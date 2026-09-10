# PROJECT_STATUS.md

Last updated: 2026-09-10 (day 1). Season tips Nov 1-3, 2026.

Reading order: `CLAUDE.md` -> `docs/FRAMEWORK_PLAN.md` -> `ARCHITECTURE_DECISIONS.md` -> `docs/SIM_GUARDRAILS.md` -> `docs/LEARNINGS.md` -> `docs/models/README.md` -> `HANDOFF.md`. Postmortem of last year: `docs/postmortem/01-06`.

## Where we are

Week 1 of 8 (`docs/FRAMEWORK_PLAN.md` section 7). Done on day 1:
- Postmortem of CBB-Monte (no residual signal vs close; root causes: uncentred KenPom on a drifting scale, no home court, independent team draws, unfitted dispersion, stale models, no player layer).
- Data on disk: hoopR pbp/player_box/team_box/schedules/shots/rosters/player_core 2022-2026 (+2027 schedule); CBBD lines 2013-2026, games 2013-2026, season ratings (end-of-season, banned as features), lineup samples; CBBD pbp with on-floor players 2022-2026 (bulk pull in progress).
- Foundations: game universe (31,103 games, D-I flag, truncation flag, sealed flag), team crosswalk (367 teams, 100% ESPN/CBBD/KenPom), per-season gate reference tables, KenPom point-in-time snapshots (centred), leak-test harness (KenPom as-of join passes; CBBD season ratings fail as expected).
- Docs: framework plan, decisions 1-6, guardrails with G1-G10, learnings L1-L8, control_engine pre-registration.

In progress: Control engine (F1/F2, three anchor arms, gates G1/G5/G6/G9/G10 vs ESPN BET closes).

## Production stack right now

Nothing shipped. No live runs.

## Next in queue

1. Control engine results -> record in experiments.md; anchor decision per pre-registered rule.
2. L2 pace bake-off spec (families and feature sets), L1 anchor bake-off formalised from the Control results.
3. Variance decomposition (coach / team / player) on four factors, tempo, usage; needs coach table (not yet sourced) and CBBD pbp.
4. CBBD pbp audit -> event dictionary -> L3 possession-outcome model spec.
5. HF push of data/raw once the pbp pull completes.
