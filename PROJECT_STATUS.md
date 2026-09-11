# PROJECT_STATUS.md

Last updated: 2026-09-11 03:25 EDT (end of the overnight session). Season tips Nov 1-3, 2026.

## Overnight 2026-09-10 21:50 -> 2026-09-11 04:00 EDT: read `HANDOFF.md` "SUMMARY FOR USER" first

Engine v1 exists and is gated (provisionally, 50 seeds): margin variance fixed, margin bias inside tolerance, sim margin correlates 0.91 with the close; possessions still +2.0, OREB% and FTA/FGA low, total variance low, OT rate low, rotation spread wrong. Adopted tonight: usage shooter fix + S1; fg_make engine-safe state + shrunk shooter (round4_B1) on inputs v2; rebound S1_weekly; free throw S1_conf_aligned; attribution round 2 (S1 on two targets). Nothing adopted for clock (best arm served provisionally) or rotation (R2 served; two new families characterised). Lines source accepted (CBBD ESPN BET). AWS parity proven, instance terminated. Decisions 9 (pending evidence) and 10 added; learnings L22-L35. Next: full 200-seed read with the noise floor on AWS; fix OREB%/FTA upstream; rotation round 6 (composition); possession-outcome early-season shrinkage arm.


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

- Shot-classification diagnostic: 2025 putback artifact is upstream ESPN (L16); fixed by a data-derived rim override (2.27 ft) in the v2 event layer; pbp_complete flag added (79-98% by season).
- Rebound bake-off: winner LightGBM team-level (L17). Free-throw bake-off: winner shooter-keyed LightGBM; no bonus-rule era boundary (L18).
- Clock round 1: no arm adopted (end-of-period bend smoothed). Rotation round 1: no arm eligible (state dependence). Round-2 pre-registrations issued for both.
- Eval harness (engine-agnostic gates G1-G9, market and props graders) reproduces Control to 4 decimals. Preseason 2027 pull, roster continuity, fault-tolerant daily chain (dry run OK).

In progress: L3 possession-outcome round 2 (event fix + S0/S1/S2 training schemes), clock round 2, rotation round 2, field-goal make bake-off, shot-allocation (usage) bake-off.

## Production stack right now

Nothing shipped. No live runs.

## Next in queue

1. Collect the five running bake-offs; if the L3 training scheme (S1/S2) wins, it becomes the default for every sub-model.
2. Engine v0 assembly: GameState (with bonus_era from free_throw), possession loop wiring clock -> event -> allocation -> make/FT/rebound -> rotation; lookup-table export of tree winners; results contract output.
3. Gates G1-G7 vs Control on 2025 (F2); paired-seed confirms for each tree winner (G2/G4).
4. Seed-noise study to replace provisional tolerances.
5. Player attribution for rebounds/assists/steals/blocks (props layer) after usage lands.
