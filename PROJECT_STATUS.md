# PROJECT_STATUS.md

Last updated: 2026-10-07 (day session). Season tips Nov 2, 2026.

## SESSION 2026-10-07 (day; PM Opus 5.5): K2 clock ADOPTED (served stack v3). Read this block first.

Usage reading 25% of weekly allowance by Wed midday (user: "plenty of room"). AWS this week ~$11 of $60.
- **SERVED STACK v3:** clock round-8 K2 `v5b_r8K2_glat_pmean` is the engine default (ce25650). F1 read confirmed the F2 read: G5 SD ratio 0.918 -> 0.961 PASS, corr 0.147 -> 0.188, no gate status regressed (`docs/tests/adoption_clock_K2_2026-10-07.md`). `adapters.SERVED_V2` reproduces parity v9; new parity reference v10. Retrain chain stage `clock_k2` (default on). 2026-27 serves the 2025-04-01 refit, carried forward.
- **OT foul carry:** now the retrain-chain default (`--no-ot-foul-carry` opts out). Suite 748 passed.
- **Oct 10 runbook rehearsal** (`docs/ops/runbook_rehearsal_2026-10-07.md`): 19 stages, 14 OK, 4 SEAL_OK, 1 MANUAL, 0 FAIL. `parity_v10` is stage 1. 2027 rosters 307/365 (R1 covers 58). On the day: seal flag (user ruling 10-01: lift in the Oct 10-17 window), roster re-pull, switch `preseason_dir.json`.
- **Early-season totals** (d0-14 -4.5 / -5.2) are not touched by K2 (`docs/tests/early_total_bias_K2_2026-10-07.md`). They are broad (about 2/3 of teams), not prior-concentrated.
- **Owner found** (`docs/tests/early_fta_rate_diag_2026-10-07.md`): the foul accrual LUT A2 (R9ao3) has no calendar input. The real H1 in-bonus share is 0.28-0.31 in week 0 vs 0.20 from d46, every season, within team. The bonus channel is about 74% of the FTA/P gap.
- **Calendar-term round** (spec 3337d13; `docs/tests/foul_accrual_calendar_2026-10-07.md`): NOT ADOPTED. `A2dbk` wins the primary but fails the d0-14 calibration and F1 responsiveness gates, and is worth only +0.15 pts in the loop. A2 over-predicts late season by 10-14% and ignores team foul style.

Next (in order):
1. Oct 10: run `ops_seal_week_v1.py` (seal lift per user ruling).
2. Seal-week build to the first 2026-27 live build, target Oct 17.
3. Register the scheduler; at least 5 unattended passes before Nov 2.
4. Accrual round 2: add an as-of level term and team foul priors to A2, plus a calendar term. Pre-register; it rides the post-freeze retrain if it wins.
5. Lines re-probe Oct 26.

## SESSION 2026-10-05 (day; PM Opus 5.5): readiness track + pre-freeze modeling. Read this block first.

Deploy target 2026-11-02. Freeze 2026-10-10. Seal lifts in the Oct 10-17 audit window (user ruling 10-01). AWS approved this week ($60 cap; ~$3.8 used). Weekly token allowance resets Friday mornings; ~20% used by Mon morning.

Done today: leftover edits committed; readiness audit `docs/ops/readiness_gaps_2026-10-05.md` (with PM rulings: KenPom skipped for serving, scheduler registration after first 2026-27 live build, OT team-foul mismatch queued with G7); Smart App Control was blocking pyarrow (user turned it off); daily chain dry-run OK on non-sealed stages; tip_times_2027 + `ops_register_tasks_v1.ps1` (not registered); 2027 stage_inputs path to a named hard stop; ESPN roster puller (296/365 teams) + injury player-out parser + default-off `availability` hook in build_live (parity v9 PASS); day-1 player priors A3 SELECTED for 2026-27 serving.
Not adopted (ledger): F_R / F_T retrains; PO TOV season anchor T1 (pre-registered veto on F2 d15-45); FT X1 (G9 total bias regresses at full size). Served v2 retained; fold-1 confirmation holds except pre-existing total bias.
Main model risk for launch: early-season totals low ~4 pts (days 0-14), owners TOV level (F1), FT rate/make, anonymous shooters (A3 helps). Decomposition: `docs/tests/total_bias_decomp_2026-10-05.md`.

Afternoon 10-05: Oct-10 runbook is run-only (`scripts/ops_seal_week_v1.py`, 17 stages: 12 OK, 4 SEAL_OK, 1 MANUAL; builders for R9ao3 v2 and shot-block prior validated bit-identical on past seasons; A3 wired into the 2027 inputs stage; shot-block live LUT season-generic; preseason dir is one setting `data/overrides/preseason_dir.json`). Daily TOV-level monitor in grade stage (report only). Early as-of shrinkage hypothesis REFUTED (Phase 1). FT shooter-sample priors P1-P3 NOT ADOPTED; anonymous-slot prior under A3 worth only ~0.2-0.3 pts, no round before Nov 2. NEW LAUNCH RISK: 69 of 365 teams have no 2027 ESPN roster (100% anonymous); re-check coverage Oct 12 / 19 / 26; a prior-roster-minus-seniors fallback needs its own pre-registered check in seal week. First Oct-10 step: parity v9 smoke.

Late 10-05: roster fallback R1 (prior-season roster minus final-year players) SELECTED by PM for teams without a 2027 roster at serve time (pre-registered, beats anonymous F2 -0.35 MAE vs floor 0.335 [thin margin], F1 -0.41 vs 0.119); default-off `--fallback R1` in the A3 path; TODO wire `--fallback R1` into the 2027 inputs stage / ops_seal_week before Oct 10. OT team-foul mismatch: the TRAINING side is wrong (tables reset fouls at each OT; rule and engine carry second-half fouls); fix = `chain_full_retrain_v1.py --ot-foul-carry`; full-size read `docs/ops/box_queue/d1001_F_2.md` (b49f99d) NOT RUN: no spot capacity 14:25-14:55; retry spot (no on-demand); any adoption is post-freeze retrain of served models.

Evening 10-05: lines: primary CBBD /lines (Bovada, DK; no capture time), fallback = own ESPN + CBBD hourly snapshots with captured_at (`pull_espn_odds_snapshot_v1.py`, `build_lines_open_close_v1.py`, chain stages; hourly task written, not registered); verify ESPN spread sign on first real odds (Oct 26). G5 decomposition (`docs/tests/g5_variance_decomp_2026-10-05.md`): per-team PPP variance right, shared share wrong; owners = clock pace x efficiency coupling (existing K2 arm: total SD 0.973, corr 0.170, G9 unchanged on F2 full read; NEEDS F1 full read, then PM rules before the Oct 10 freeze) and the FT block (shared whistle; FT% vs opponent FG coupling); honest SD-ratio target ~0.97. Test suite: 742 pass (stale spy fixed). Tomorrow: box run = K2 fold-1 full read + d1001_F_2 OT carry (if not done today).

Next (in order): 1. Oct 10 runbook rehearsal on non-sealed steps; 2. seal-week build (ratings, R9ao3 priors, shot-block prior `engine/shot_block_prior_2027_v1.parquet`, A3 builder, first 2026-27 live build), target Oct 17; 3. register scheduler, >=5 unattended passes before Nov 2; 4. early-season shrinkage candidate (thin as-of samples) pre-registered, not a calendar term; 5. daily TOV-level monitor (report only); lines re-probe Oct 26.

## RESULT OF THE OVERNIGHT 2026-09-30 -> 10-01: read `HANDOFF.md` "SUMMARY FOR USER, OVERNIGHT" first

SERVED STACK v2 ADOPTED 02:06 EDT 2026-10-01 (PM decision under the user's delegation): `ENGINE_CLOCK=v5b_r6L2_glat_pmean`, `ENGINE_SHOT_BLOCK=K2_Ocell`, `ENGINE_FOUL_JOINT=R9ao3`, `ENGINE_SHARED_SHOOTING=G3`, `ENGINE_CHANCE_TIME=KD` are now the engine defaults (old stack: `adapters.SERVED_V1`; parity reference v8). Full-size read vs the old stack: G1 possession mean and G4 OREB% FAIL -> PASS, total bias stays PASS (-0.34), slope 0.917 -> 0.948, home/away corr 0.117 -> 0.126, no verdict regressed (`docs/tests/engine_gates_F2_2025_s200_v3_COMB9GKD_full_2026-10-01.md`, `docs/tests/adoption_served_v2_2026-10-01.md`). E3's Stage C refutation WITHDRAWN (a train/serve skew in the fg_make retrain; E3 is G9-neutral). Truth default flipped to verified. Full retrain chain is one command and ran on the box (not adopted). Daily chain v3 built. AWS: $44.38, instance terminated 02:04 EDT.

Next in queue (2026-10-01): 1. full retrain (F_R, F_T) with the adopted set on, full-size read vs served v2; 2. (done 02:42: v3 event team block is the default, parity reference is now v9, daily chain serves the adopted stack) 2026-27 tables for R9ao3 priors and the shot-block builder, behind the seal decision; 3. remaining PPP channels (OREB% anchor `TO`, FT% who-shoots, TOV / mix); 4. G9 slope: PO / fg_make team-rate response; 5. G5 corr and total SD (pace x efficiency unowned); 6. clock tempo input redesign; 7. late-game round 3 (G7); 8. team FT slope 0.617, K2_Ocell team slope, G4 site arm read; 9. ops: live shot-block table status, seal-lift decision (USER), 2027 rule constants, tip-time schedule, day-1 player priors. Rotation parked.

## Overnight 2026-09-30 20:39 EDT -> 2026-10-01 04:00 EDT (user away; wall clock only)

Seven lanes launched 20:41 EDT on the home box. At ~20:45 the user APPROVED AWS for this session and delegated adoption to the PM where the data supports it; an eighth lane, the AWS operator (Opus; spot c7a.48xlarge, $60 cap, instance terminated by 03:00), launched 20:47: full-size Decision 11 set (K2_Ocell, R8b, L2, combined) vs S0, then the lanes' box queue (`docs/ops/box_queue/`), then lane D's full retrain. Compute stops 02:30, reports due 02:45, PM wrap-up 03:00-04:00. Rules: `docs/ops/worker_rules_overnight_2026-09-30.md`. Core caps sum to 20.

| lane | worker | object | cores | output |
|---|---|---|---|---|
| A | Opus | Aggregation over-spread: closed decomposition of sim margin spread by channel (owns G9 slope after E3's Stage C refutation) | 4 | `docs/models/aggregation/experiments.md`; `docs/tests/aggregation_overspread_decomposition_2026-09-30.md` |
| B | Opus | G5 shared variance: per-game shared shooting latent + pace x efficiency; measure, bake-off, default-off flag, paired loop | 4 | `docs/tests/shared_shooting_latent_2026-09-30.md` |
| C | Opus | Foul round 9: first-half trip production on the corrected foul state; late-game round 3 pre-registration if time | 3 | `docs/tests/foul_round9_first_half_2026-09-30.md` |
| D | Opus | Full retrain on the clean foundation: sibling trainers, one-command chain, smoke, as much of the real run as measured timing allows | 3 | `docs/ops/full_retrain_chain_2026-09-30.md`; `scripts/chain_full_retrain_v1.py` |
| E | Sonnet | Truth default flip, parity reference v7, verified sample default, re-reads of earlier paired rounds on v3 | 2 | `docs/tests/truth_flip_and_v3_rereads_2026-09-30.md` |
| F | Sonnet | Daily chain v3: sim / publish / grade / bias-CLV stages, replay-verified on fold-2 dates | 1 | `docs/ops/daily_chain_v3_2026-09-30.md`; `scripts/chain_daily_v3.py` |
| G | Opus | Home/away/neutral site terms: audit, fg_make +0.68, offline bake-offs for free_throw / clock / foul channel | 2 | `docs/tests/home_site_terms_2026-09-30.md` |

## Session 2026-09-30 (started 10:50 EDT): restart after a 12-day gap; 33 days to tip-off

Six lanes launched 11:00 EDT, reports due 15:30 (E at 14:00). Core caps sum to 20.

| lane | worker | object | cores | output |
|---|---|---|---|---|
| A | Opus | Foul accrual + FT-trip production JOINT round (tap instrumentation, pre-registration, offline, paired closed loop 500 x 25) | 6 | PO experiments.md new section; `docs/tests/foul_joint_round_2026-09-30.md` |
| B | Opus | DIAGNOSTIC: G1 possession mean +2.0 and G5 home/away corr 0.117 vs 0.253 / total SD 0.898; both were unowned. Tests whether the passing total bias is a compensation between +2.0 possessions and low OREB% / FT rate | 3 | `docs/tests/g1_g5_possessions_corr_diagnostic_2026-09-30.md` |
| C | Opus | Season-drift anchor: one cross-model offline round (rebound, shot_block, FT technicals; PO control) | 4 | `docs/models/season_drift/experiments.md`; `docs/tests/season_drift_anchor_round_2026-09-30.md` |
| D | Opus | Late-game round 2: window-gated clock floor removal + B_L3, default-off, paired closed loop | 4 | late_game experiments.md new section; `docs/tests/late_game_round2_2026-09-30.md` |
| E | Sonnet | Ops readiness audit for 2026-27: lines source liveness, schedule, rosters, daily chain gaps, day-1 as-of features, HF sync | 1 | `docs/ops/readiness_2026-27_2026-09-30.md` |
| F | Opus | DIAGNOSTIC: G9 calibration slope 0.910 and G6 home margin; both were unowned and both sit on the margin side, the committed deliverable | 2 | `docs/tests/g9_g6_margin_slope_home_diagnostic_2026-09-30.md` |

Mid-session state (PM, 13:50 EDT; full summary in HANDOFF.md at wrap-up). NOTHING ADOPTED, no served default changed. Decision 9 CLOSED; Decisions 11-12 added; guardrails revised (G1 like-for-like, TOV count, multi-draw floors, G5 decomposition, verified truth). Foundations built today as versioned siblings: honest engine inputs v3 (live-path replay; the backtest builder had same-game leaks), event layer v4 (phantom possessions, and-one labels), corrected pre-open foul state overlay, verified-finals truth (`data/reference/verified_v1/`, `CBB_TRUTH=verified_v1`), live-slate path, daily ingestion, own-ratings entry point. Main fix round: `team_rate_estimator` (E3 state-space rate selected; Stage B retrains and Stage C paired 200-seed loops RUNNING ON AWS from ~13:45, operator lane). Pending-ship fixes under Decision 11: drawn block flag (`ENGINE_SHOT_BLOCK=K2_Ocell`, OREB% +38 floors), foul round-8 trip repair (`ENGINE_FOUL_JOINT` R8b), both vetoed only on G5 lines whose 25-seed floor was shown to be understated fivefold. Refuted today: shared whistle as the main G5 channel; late-game duration arms (add possessions); L1 relabel. Local lanes running: clock round 6 on v4, anchor serving hook, day-1 ratings priors, chain v2 wiring.

Re-plan against `docs/FRAMEWORK_PLAN.md` section 7 (PM, 2026-09-30). The plan had G1-G7 passing by Oct 8; the served stack passes possession SD, margin SD ratio and both G9 biases, and fails the rest. Remaining calendar:

| dates | phase | exit |
|---|---|---|
| Sep 30 - Oct 9 | Game-gate fix sprint: FT rate, OREB% (drift anchor + drawn block), possession mean, score correlation / total SD, calibration slope, OT rate. Each fix pre-registered, default-off, paired closed loop | Every game gate either passes or has a closed decomposition and a dated decision to ship without it |
| Oct 10 | FREEZE the game stack; full 200-seed read on the box (needs the user's AWS approval) | Gate table for the frozen stack |
| Oct 10 - 17 | Audit: full walk-forward 2024-25, then unseal 2025-26 with lines; market scorecard; leak detector | G9 / G10 reported honestly |
| Oct 17 - 27 | Daily ops for 2026-27: chain stages, day-1 as-of features with prior-season carry, lines capture, grading, bias monitor | End-to-end dry run on the exhibition slate |
| Oct 28 - Nov 3 | Paper-trade opening week; go / no-go | First live-week grading report |

Player props stay the stretch goal (rotation parked 2026-09-18); game markets validated honestly are the committed deliverable.

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

SERVED STACK v2 since 2026-10-01 02:26 EDT (commit 7a1cfcd): the v1 stack below with five adopted defaults: clock `v5b_r6L2_glat_pmean`, shot block `K2_Ocell` (drawn), foul joint `R9ao3`, shared shooting `G3`, chance time `KD`. Parity reference `docs/ops/parity_reference_windows_v9.json` (v3 event team block, default since bcf6bb7; v8 is the same stack on the old block); gate table `docs/tests/engine_gates_F2_2025_s200_v3_COMB9GKD_full_2026-10-01.md`; old stack reachable as `adapters.SERVED_V1` (parity v6 / v7). Inputs v3; truth verified by default.

Nothing shipped to production. Engine v1 served stack (all provisional): event round2_s1; fg_make round4_B1 on inputs v2; rebound S1_weekly; free throw S1_conf_aligned; usage U1; rotation R2 under S1; clock v5b_glat_pmean (shared per-game pace latent, mean-preserving; ADOPTED 2026-09-11); attribution round 2. Last full read: `docs/tests/engine_v1_gates_F2_2025_s200_aws_2026-09-11.md` (v3c clock, 200 seeds) and `engine_v1_gates_F2_2025_s200_v5b_full_2026-09-18.md` (served v5b stack, full 200 paired seeds; supersedes the 75-seed read of 2026-09-11). Parity reference for the served stack: `docs/ops/parity_reference_windows_v6.json`.

## Next in queue (2026-09-30 15:40; full list with evidence in HANDOFF.md "Open items, in order")

1. Read the box results (HANDOFF operator row): Stage B primaries; the S0 full read on v3 inputs + verified truth is the NEW BASELINE gate table; S1 vs S0 on the G9 slope; the pending fixes at 200 seeds with four floor draws. Rule under Decisions 11-12.
2. The full retrain on the clean foundation (event layer v4 + corrected foul state + E3 v4 features + ratings C + anchor O arms; sibling trainers for the v4 consumers; v3 tag inputs; verified truth), then the combined closed loop of Decision 11. Box job.
3. Flip the truth default, regenerate the parity reference, switch to the verified sample, re-read earlier paired rounds on v3.
4. G5 total-variance owner: shared shooting latent + pace x efficiency (the whistle is refuted).
5. Foul round 9 (first-half trips on the corrected state), then late-game round 3, G7.
6. Home-advantage site terms (free_throw, clock, foul channel) and fg_make's +0.68.
7. Clock: pace responsiveness through E3-style features; remaining count gap.
8. Ops, ~25-35 h: chain sim / publish / grade / bias-CLV stages; 2027 adapters and rule constants; day-1 decisions file; tip times; rosters; lines re-probe Oct 26 / Nov 2; KenPom-covers decision and HF token (user).
9. Rotation PARKED; props the stretch goal.

Served stack: UNCHANGED today (see "Production stack right now"). Pending under Decision 11: clock `L2`, `ENGINE_SHOT_BLOCK=K2_Ocell`, `ENGINE_FOUL_JOINT` R8b, team_rate_estimator E3 (Stage B/C on the box), own_ratings C (retrain dimension).

## Superseded queue (2026-09-18 21:40)

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
