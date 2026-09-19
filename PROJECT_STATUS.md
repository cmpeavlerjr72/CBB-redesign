# PROJECT_STATUS.md

Last updated: 2026-09-18 21:40 EDT (end of the evening session). Season tips Nov 1-3, 2026.

## Evening 2026-09-18 ~17:30 -> 22:00 EDT: read `HANDOFF.md` "SUMMARY FOR USER, EVENING 2026-09-18" first

Nothing adopted; no served default changed (only the `provisional_clock` label retired for the adopted v5b clock, with a new Windows parity reference v6). The served v5b stack is now read at the full 200 paired seeds on AWS (spot, ~$4.87): no verdict moved from the 75-seed read; possession SD and margin SD ratio PASS, possession mean (+2.0), total SD ratio (0.898), home/away corr (0.117 vs 0.253), G9 slope (0.910), G4 OREB%/FTA-FGA, G7 OT still FAIL. Gate G4's two misses are TRACED with closed decompositions: OREB% = rebound season drift (73%) + the engine hard-feeding `blocked_f = 0` for lack of a shot-block model (47%); FTA/FGA = the bonus-trip rate, because the silent team-foul accrual is one whole-game constant (bonus too often early, too rarely late), plus technicals (23%) and the final 2:00 (21%). Rounds run on each owner, all offline: rebound round 3 (trend+carry leads 13.5 floors but the trend is an extrapolation; a DRAWN block flag closes the blocked channel), shot_block round 1 (new sub-model folder; fails on the same drift), foul accrual (GBM accrual law 36 floors, team-keyed conditional bonus 23 floors), free-throw technicals 1/1b (nothing eligible; target verified), late-game round 1 (the 09-11 role-conditioning headline WITHDRAWN as a tap defect; the real object is the clock's 45-second floor bucket; window-gated floor removal is round 2). Possession-outcome G2, the 09-11 offline winner, was REFUSED at the ship gate: shrinkage compressed between-game spread 22 floors and flattened responsiveness. An event-layer bug (one-row technical-FT lookahead, ~1,200 mis-tagged attempts/season) was found and fixed behind `tech_lookahead` (default off, `possessions_v3` siblings; no consumer switched). Rotation round 10 repaired wave arrival/size but lost every veto line again: rotation is PARKED behind the game-gate work. Season drift is now a cross-model defect needing one designed answer. Decision 9 arms were null or negative in three more sub-models.

## Morning 2026-09-11 09:53 -> 13:30 EDT: read `HANDOFF.md` "SUMMARY FOR USER, MORNING" first

Engine v1 read at the full 200 seeds with a seed-offset noise floor on AWS (23 min compute, ~$1.80; box throughput 1,030-1,200 poss/s/core, last night's 83 was a startup artefact): every gate verdict from the 50-seed read holds and every miss is far outside seed noise. The two dispersion defects (possession SD 25% short; eFG% anti-correlated with pace) were traced by two independent lanes to ONE cause: the engine had no per-game pace realisation, a CLAUDE.md rule the served clock violated. Clock rounds 5/5b built the shared per-game latent; the mean-preserving form B1 wins the primary (possession SD ratio 0.80 -> 1.03 closed loop, total SD ratio 0.74 -> 0.82, home/away corr 0.00 -> 0.10) and was served provisionally (e3ccce5) then ADOPTED at 13:45 after rounds 5c/5d showed its Q2 blocker unpowered and the v5b 75-seed AWS read showed no margin-SD regression at scale (G1 possession SD now PASSES; total SD ratio 0.80 -> 0.89). OT shortfall is margin shape (half the ties, too many 1-point games), a late-game regime sub-model nobody has built. Rotation rounds 6-8: nothing adopted; entry composition (K1) is the best MAE in eight rounds, exit-side share now matches reality, but a count over state is a level not a rate (L36-L37); rounds 8-9 repair the exit RATE (94% of the real span by starters on the floor) but the floor composition stays 25% bench-heavy and Decision 8 worsens; nothing adopted in rounds 6-9, R2 served; the wave side conditioned on starters is next, after the overdue engine adapter. Clock 5c: B1's Q2 "failure" is inside its own noise; a pace-quadratic latent SD passes every band offline (no closed loop yet). Late-game regime measured and pre-registered (70% of the tie shortfall is end-game role behaviour the engine lacks). Usage: score_diff post-outcome (fixed, harmless offline); Decision-10 gate says do not wire the tree; the state-free tree loses to U1 on truth-gap; U1 stays. Possession-outcome round 4: early-season shrinkage wins weeks 0-3 but moves error to weeks 4-7; reliability counters (G4) improve without cost on first-shot only; nothing ships; Decision 9 amended (conf-aligned cont win did not reproduce; tree alignment cells need the box). Mean-based market lean at 200 seeds: negative and inside noise, bands flat, market beats both engines on MAE (as expected for an engine failing 8 of 9 gates). 36+ commits, HF mirrored.

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

Nothing shipped to production. Engine v1 served stack (all provisional): event round2_s1; fg_make round4_B1 on inputs v2; rebound S1_weekly; free throw S1_conf_aligned; usage U1; rotation R2 under S1; clock v5b_glat_pmean (shared per-game pace latent, mean-preserving; ADOPTED 2026-09-11); attribution round 2. Last full read: `docs/tests/engine_v1_gates_F2_2025_s200_aws_2026-09-11.md` (v3c clock, 200 seeds) and `engine_v1_gates_F2_2025_s200_v5b_full_2026-09-18.md` (served v5b stack, full 200 paired seeds; supersedes the 75-seed read of 2026-09-11). Parity reference for the served stack: `docs/ops/parity_reference_windows_v6.json`.

## Next in queue (2026-09-18 21:40)

1. Foul accrual: the accrual law and the trip-foul production JOINTLY (the standalone constant fix exposes hidden compensation; see the ledger's round-6 closed-loop row), plus a pre-registered offensive-foul mechanism. Largest owned gate miss.
2. Season-drift anchor: one pre-registered cross-model round (rebound, shot_block, FT technicals; possession_outcome as control).
3. Rebound stage 2 (`S1_weekly`, ~3 h/cell, box job after the image's lightgbm-threads/joblib fix) and the drawn-block closed loop once shot_block has a level-passing arm.
4. Late-game round 2: DEFAULT-OFF window-gated clock floor removal (`C2_clk` / `D_clk`), paired closed loop, primary P(0)/P(1); state-enriched events (B_L3) as the second arm.
5. Possession-outcome: a reliability arm that preserves the spread of team estimates; build a weeks-0-7 closed-loop game sample so early-season cells are powered; the A2 weekly-cadence cell (run 2026-09-18: primary inside floor, weeks 0-3 gap 3.83 -> 2.51 pp) under seed 1 plus its `cont` cell is the cheapest next step; Decision 9 ruling (recommended: close against opponent adjustment / conference flag / alignment, keep cadence open).
6. Event layer v3: re-grade FT technical arms, verify FT make / PO / clock immaterial, measure usage and late_game deltas, then switch consumers in one commit with a new parity reference.
7. FT technicals 1c (team rate relative to the as-of league level; blended shooter rule): low priority.
8. Rotation: PARKED. When resumed: a team-indexed Decision-8 round with the close-late (cell x n_st) interaction. Box: 25-seed K1/Z1 freezes.
9. HF token rotation (user). G3 instrumentation.

Superseded queue from 2026-09-11 12:30 follows.

1. AWS box session (spot was unavailable in us-east-2 at 13:08; on-demand ~$10/h needs the user's OK): finish the v5b read to 200 paired seeds; the two PO tree alignment cells; the PO G2/G3 paired closed loop; rotation 25-seed freezes.
2. Clock 5d: wire the pace-quadratic latent SD (C4) behind a flag and run the paired closed loop vs the served B1; then L34's prev_end mix from the event side.
3. Late-game regime round 1: run the pre-registered arms in `docs/models/late_game/experiments.md` (fouling/FT supply first).
4. Rotation: write the round-7/8/9 engine adapter (scope in rotation experiments.md 21.15), run Decision 10 on K1 and Z1, then round 10 on the wave side conditioned on starters on the floor.
5. Possession-outcome: G2 (prior-season shrink) is the decided offline winner for both models, blocked on the paired-seed sim run; run it, then ship if no gate regresses and the quintile slope holds.
6. OREB% -1.6 pp and FTA/FGA -1.2 pp (G4): still unowned; pre-register a rebound/free-throw-trip diagnostic.
7. Rebound early-conference calibration: alignment arms after the box runs the PO cells.
8. HF token rotation (user).

Superseded queue from day 1 follows.

1. Collect the five running bake-offs; if the L3 training scheme (S1/S2) wins, it becomes the default for every sub-model.
2. Engine v0 assembly: GameState (with bonus_era from free_throw), possession loop wiring clock -> event -> allocation -> make/FT/rebound -> rotation; lookup-table export of tree winners; results contract output.
3. Gates G1-G7 vs Control on 2025 (F2); paired-seed confirms for each tree winner (G2/G4).
4. Seed-noise study to replace provisional tolerances.
5. Player attribution for rebounds/assists/steals/blocks (props layer) after usage lands.
