# PROJECT_STATUS.md

Last updated: 2026-09-11 12:30 EDT (end of the morning session). Season tips Nov 1-3, 2026.

## Morning 2026-09-11 09:53 -> 13:30 EDT: read `HANDOFF.md` "SUMMARY FOR USER, MORNING" first

Engine v1 read at the full 200 seeds with a seed-offset noise floor on AWS (23 min compute, ~$1.80; box throughput 1,030-1,200 poss/s/core, last night's 83 was a startup artefact): every gate verdict from the 50-seed read holds and every miss is far outside seed noise. The two dispersion defects (possession SD 25% short; eFG% anti-correlated with pace) were traced by two independent lanes to ONE cause: the engine had no per-game pace realisation, a CLAUDE.md rule the served clock violated. Clock rounds 5/5b built the shared per-game latent; the mean-preserving form B1 wins the primary (possession SD ratio 0.80 -> 1.03 closed loop, total SD ratio 0.74 -> 0.82, home/away corr 0.00 -> 0.10) and is SERVED PROVISIONALLY (e3ccce5), not adopted (tempo-quintile-2 responsiveness 1.17 vs 1.15; margin SD ratio -1.2 floors). OT shortfall is margin shape (half the ties, too many 1-point games), a late-game regime sub-model nobody has built. Rotation rounds 6-8: nothing adopted; entry composition (K1) is the best MAE in eight rounds, exit-side share now matches reality, but a count over state is a level not a rate (L36-L37); round 8 (exit rate by starters on the floor) slopes correctly but shrinks toward the wrong parent; nothing adopted, R2 served. Usage: score_diff post-outcome (fixed, harmless offline); Decision-10 gate says do not wire the tree; the state-free tree loses to U1 on truth-gap; U1 stays. Possession-outcome round 4: early-season shrinkage wins weeks 0-3 but moves error to weeks 4-7; reliability counters (G4) improve without cost on first-shot only; nothing ships; Decision 9 amended (conf-aligned cont win did not reproduce; tree alignment cells need the box). Mean-based market lean at 200 seeds: negative and inside noise, bands flat, market beats both engines on MAE (as expected for an engine failing 8 of 9 gates). 36+ commits, HF mirrored.

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

Nothing shipped to production. Engine v1 served stack (all provisional): event round2_s1; fg_make round4_B1 on inputs v2; rebound S1_weekly; free throw S1_conf_aligned; usage U1; rotation R2 under S1; clock v5b_glat_pmean (shared per-game pace latent, mean-preserving; provisional); attribution round 2. Last full read: `docs/tests/engine_v1_gates_F2_2025_s200_aws_2026-09-11.md` (on the v3c clock; the v5b stack needs its own 200-seed read).

## Next in queue (2026-09-11 12:30)

1. AWS box session (~1 h, ~$5): 200-seed paired read of the v5b stack with the offset floor; the two possession-outcome tree alignment cells (minutes there, 5-6 h here); rotation round 8/9 Decision-10 freezes.
2. Clock round 5c: the tempo-quintile-2 responsiveness band (per-team or tempo-conditioned latent SD); then close L34's prev_end mix from the event/fg_make side.
3. Late-game regime sub-model (tie pile-up, fouling, clock stops): pre-register from `docs/tests/engine_v1_variance_ot_diag_2026-09-11.md` section on OT.
4. Rotation round 9: exit rate shrunk toward the powered composition marginal; write the round-7/8 engine adapter so the Decision-10 freeze can run.
5. Possession-outcome round 4b: G4 + G3 + second-seed floor; ship decision for first-shot only.
6. OREB% -1.6 pp and FTA/FGA -1.2 pp (G4): still unowned; pre-register a rebound/free-throw-trip diagnostic.
7. Rebound early-conference calibration: alignment arms after the box runs the PO cells.
8. HF token rotation (user).

Superseded queue from day 1 follows.

1. Collect the five running bake-offs; if the L3 training scheme (S1/S2) wins, it becomes the default for every sub-model.
2. Engine v0 assembly: GameState (with bonus_era from free_throw), possession loop wiring clock -> event -> allocation -> make/FT/rebound -> rotation; lookup-table export of tree winners; results contract output.
3. Gates G1-G7 vs Control on 2025 (F2); paired-seed confirms for each tree winner (G2/G4).
4. Seed-noise study to replace provisional tolerances.
5. Player attribution for rebounds/assists/steals/blocks (props layer) after usage lands.
