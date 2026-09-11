# SUMMARY FOR USER, MORNING 2026-09-11 09:53 -> 13:30 EDT

Written by the PM 12:20-12:45 EDT. Every number is in a committed doc; paths in brackets. Twelve worker lanes ran; reports were due 12:45; all delivered.

## What was decided (with evidence)

| lane | decision | evidence |
|---|---|---|
| AWS 200-seed read + noise floor | DONE: runs A (seeds 0-199) and B (offset) both complete, 23 min compute, instance 10:00-10:48, ~$1.80, terminated (CLI-verified). Box throughput 1,030-1,200 poss/s/core; last night's 83 was a pool-startup artefact (never size off a <1 min probe). Every gate verdict from the 50-seed read holds and every miss is an order of magnitude outside the A-B band; the 50-seed PROVISIONAL label is retired. Scorecard calibration unlocked: corr(sim margin, close) 0.918; sim colder than market at low p, hotter at high (slope 0.909). ROI still refused (<2,000 seeds; ~4 h, ~$10 on the box). Fixes on the way: posix path normalisation in the digest (parity v5), G8 grader memory blow-up at 200 seeds | `docs/tests/engine_v1_gates_F2_2025_s200_aws_2026-09-11.md`; `docs/ops/aws_launch_chain.md` section 13; commit 5a792ed |
| Engine variance/OT diagnostic | The 50-seed run already served the v3c clock (my brief was wrong). Paired 20-seed rerun bit-identical; a free 20-seed offset floor; G5/G9 dispersion lines are seed-count dependent by construction (never compare across seed counts). Possession SD 25% short and WITHIN-game; eFG% anti-correlated with pace (-0.198 vs ~0); together 80% of the total-SD gap. OT shortfall = margin shape (ties 47% short, 1-point margins 50% over); OT module itself fine | `docs/tests/engine_v1_variance_ot_diag_2026-09-11.md`; commit 4ea6237 |
| Pace-efficiency sign | ONE cause for both dispersion defects: `loop.py` had no per-game pace realisation (CLAUDE.md rule violated); 73% of within-game pace variance was i.i.d. clock noise. fg_make cleared (transition lift 1.63x its design; served eFG moves positively with pace); rotation 0.3%. Measurement rule: possessions-table `is_transition`/`duration_s` are post-outcome (5x inflated transition lift); use fg_make's `chance_elapsed_s` | `docs/tests/pace_efficiency_sign_2026-09-11.md`; commit 386a3ee |
| Clock rounds 5 / 5b | Channel decomposition: 98.9% of the possession-SD gap is a missing within-game level, 1.1% the conditional law (heavier tails failed everything). Shared per-game latent A1: SD ratio 0.688 -> 1.004 (56 floors) but G1 mean +0.23 (Jensen). 5b: mean-preserving B1 (`v5b_glat_pmean`) removes the mean cost (G1 +1.704 -> +1.649, inside floor); closed loop 25 paired seeds: possession SD ratio 0.804 -> 1.032, total SD ratio 0.736 -> 0.823, corr(home,away) 0.003 -> 0.101, corr(P,eFG%) -0.211 -> -0.122. NOT adopted (Q2 responsiveness 1.174 vs [0.85,1.15]; margin SD ratio -1.2 floors, still inside gate). PM DECISION: SERVED PROVISIONALLY (same footing as v3c); `reference` and `v3c` selectable; old arm bit-identical (parity v5). Round 5c = the Q2 band | clock experiments.md sections 16-22; `docs/tests/clock_duration_dispersion_2026-09-11.md`, `clock_mean_consistent_latent_2026-09-11.md`; commits 28b5f17..3ef1abd, e3ccce5 |
| Rotation rounds 6 / 7 / 8 | r6: entry composition; K1 (entry class count) best MAE in six rounds (+63 floors over R2) but 4/8 cells and Q2 veto; temperature refuted by its own likelihood (tau = 1.00); defect moved to the EXIT side (starter off on 0.46 vs 0.587). r7: exit count P(k_out|size,state) closes the exit share (0.561 vs 0.558) and reproduces the 4-foul diagnostic, but loses 46-58 floors of MAE and collapses close-late cells: a count over state is a LEVEL not a RATE (real exit rates 0.37-0.66 by starters on floor). r8: exit rate conditioned on n_starters_on_floor (Y1): slopes correctly at every level (X1 was flat at 0.58 vs a real 0.31-0.68) and carries 46% of the span, repairs both G8 cells r7 broke, MAE 9.385 (+14 floors over X1, still -39 vs K1), close-late gradient -17.5 pp vs -21.0; fails because the shrinkage PARENT is the level (data weight 0.34 at one starter). Nothing adopted. Next: invert the hierarchy onto the powered composition marginal. Decision-10 freeze two rounds overdue (no round-7/8 adapter written). R2 served throughout | rotation experiments.md sections 12-19; `docs/tests/rotation_composition_audit_2026-09-11.md`, `rotation_exit_audit_2026-09-11.md`, `rotation_exit_rate_2026-09-11.md`; commits 4380a08..8599142, b6a18ec..8e378b5 |
| Usage Decision-10 gate + round 3 | `score_diff` post-outcome on 99.7%+ of rows (L27 pattern); pre-outcome builder mode added; offline winner unchanged (leak harmless for an identity target). Closed loop 25 seeds: live != frozen (z 7-8, real loop signature) and live does not beat refit-without-state: DO NOT WIRE. Round 3: state-free tree vs U1: passes Decision-8 slope (0.955 vs 0.922) and G1-G9 no-regression but is 0.5/0.9 pp (12-14 floors) FURTHER from truth on top-1/top-3 share: NOT ADOPTED. U1 stays | usage experiments.md sections 10-12; `docs/tests/usage_state_confound_2026-09-11.md`, `usage_decision10_gate_2026-09-11.md`, `usage_nostate_bakeoff_2026-09-11.md`; commits bbc079e, d8a1fb8, c41ec25, 520ea6b |
| Possession-outcome round 4 | Early-season shrinkage: G1 (EB to league mean) wins the pre-registered spec (weeks 0-3 gap 3.83 -> 2.77 pp) but pushes weeks 4-7 across the 2.0 gate (1.37 -> 2.69); G4 (reliability counters as features) improves weeks 0-3 without cost and improves week 8+ on `first`, fails cont calibration. Nothing ships; round 4b = G4 + G3 + second-seed floor. Round 3's conf-aligned cont win DID NOT REPRODUCE (0.237 pp); tree alignment cells NOT RUN twice (5-6 h here, minutes on the box). Decision 9 amended (still pending, leaning against) | PO experiments.md sections 8-9; `docs/tests/possession_outcome_early_season_2026-09-11.md`; commits 4f6b237..61c81fd; ARCHITECTURE_DECISIONS.md Decision 9 amendment |
| Market lean, mean-based, 200 seeds (user question) | ATS 0.499 (ROI -5.2%), O/U 0.496 (-5.9%); always-home 0.502, always-over 0.500, Control 0.496/0.494; bands by |sim-line| flat; split-half win rate stable to 0.1 pp. Margin MAE: v1 9.25, Control 9.11, close 8.74. No edge, as expected for an 8/9-gate-failing engine. Probability-based ROI still needs 2,000 seeds (3.5 pp seed error on a cover probability at 200) | `docs/tests/market_lean_mean_based_s200_2026-09-11.md`; commit 26b7516 |
| HF mirror | model_artifacts 832 files / 1.9 GB, results 332 files / 724 MB synced 12:10; rotation round8 and clk5b_B2_s25 skipped as live, re-sync next session | sync lane report |

## Rules and learnings added

L36-L37 (rotation: conditional rule vs arrival frequency; level vs rate). Decision 9 amended. Measurement rules: never compare G5/G9 dispersion across seed counts; never size a box run off a sub-minute probe; possessions-table transition/duration columns are post-outcome. Clock served arm is now the per-game latent (rule compliance restored).

## Worker incidents (disclosed)

Pace-sign worker stashed five other lanes' uncommitted files for ~30 s during a rebase, then restored them intact (verified by the affected lanes' clean commits). Clock 5b lane ran ~18 engine workers for ~2 min, over its cap. AWS and HF lanes each parked on a background monitor once and had to be told to wait in the foreground. Both PM briefs that named the "reference clock" in the 50-seed run were wrong (it was v3c).

## Open items, in order

1. Box session: 200-seed paired read of the v5b stack; the two PO tree alignment cells; rotation Decision-10 freezes. ~1 h, ~$5.
2. Clock 5c (Q2 band); then L34 prev_end mix from the event side.
3. Late-game regime sub-model (tie pile-up).
4. Rotation round 9: shrink the exit rate toward the powered composition marginal, not the level; write the round-7/8 engine adapter so Decision 10 can run.
5. PO round 4b (G4, G3, floor).
6. G4 OREB%/FTA-FGA: unowned; pre-register.
7. HF token rotation (user).

---

# SUMMARY FOR USER, 2026-09-10 21:50 -> 2026-09-11 04:00 EDT

Written by the PM 03:05-03:48 EDT; final. Every number below is in a committed doc; paths in brackets.

## What was decided (with evidence)

| lane | decision | evidence |
|---|---|---|
| Usage r2/r2b | Shooter re-keyed on `shot_shooter_id` (CBBD orders shooter/assister at random on assisted makes; 51% agreement). LightGBM winner unchanged on all 5 classes; U1 top-3 usage gap -2.17 -> -1.79 pp; S1 monthly adopted (+0.44-0.82 floors, no gate lost). Engine still runs U1; tree blocked on a Decision-10 gate. L28 | `docs/tests/shooter_key_audit_2026-09-10.md`, `docs/models/usage/experiments.md` r2-2b, commit 2a7f624 |
| fg_make r2/2b/3/4 | `score_diff` was the score AFTER the shot (post-outcome leak; 62-84% of its effect manufactured; tripled sim margin variance). Winner: engine-safe state, no margin term, S1 (L27). r3: the shooter label defect had manufactured the shooter signal (FGA_3 quintile span 35.9 -> 2.8 pp; L29); corrected-label arm served interim. r4: one shrunk shooter column (B1) wins every gate, D8 slope 0.31 -> 0.92-1.04; a join-coverage leak found and repaired (L32); engine inputs must be keyed on `shot_shooter_id` (skew worth 1 pp of 3P%). Served: `round3_shooter_S_C_s1` until inputs v2 carry the shrunk column, then `round4_B1` | `docs/tests/fg_make_state_confound_2026-09-10.md`, `fg_make_shooter_key_2026-09-10.md`, `fg_make_shooter_skill_2026-09-10.md`; fg_make experiments.md sections 14-20; commits 9ae38fa, 2b83b94, 40a7be6, 46f54bc |
| Attribution r1/r2 | r1 (static, F1 selects): REB_off cond_logit, REB_def lgbm, steal proportional, block lgbm, assisted/blocked aware_ridge; assist and stolen adopt nothing. Table-builder defect (all-zero team features) fixed (L22). r2: the score leak reached 2 of 8 targets; REB_def and block move to S1 on calibration; assisted within-2025 miss 6.1 -> 4.3 pp, still fails | `docs/tests/attribution_team_asof_lg_defect_2026-09-10.md`, `attribution_score_diff_leak_2026-09-10.md`; commits 75e09f6, 9f585e9 |
| Clock r3/3b/3c | Horn censoring fix confirmed (~0.45 poss; L26). Nothing adopted offline (0/9, 0/6) or closed-loop (0/6). State question closed: P2 = P3 beat P1 by 0.42 poss in the engine; cell-based family preferred (tempo slope inside the D8 band). A freeze ablation detects but cannot size a loop (L31). Best arm `srfloor|P3|S1` +1.16 poss vs incumbent +2.70; residual is a uniform -1.9% duration shortfall. r4: the clock call site is now game-indexed and `ENGINE_CLOCK` serves `v3c_srfloor_P3_s1` (provisional; 1a5acef). Round 4 adopted nothing (0/5): the shortfall is 67% state composition from upstream make rates (`prev_end` mix), only 32% the clock's own law; train/serve quantity closed four ways; recency weighting makes it worse because duration rises through the season (L34). Clock round 5 not warranted; the residual is an event/fg_make make-rate defect | `docs/tests/clock_censoring_audit_2026-09-10.md`; clock experiments.md sections 8-14; commits 9961890, 15f812a, 421b97b |
| Rotation r3/3b/4/5 | r3: the override family cannot reach the close cell (L25); S1 adopted as scheme; R2 is 2/8 cells under S1. r4: per-player hazards identified, 5/8 cells, beat R2 by 61-76 floors on minutes MAE, but over-substitute 43% (L30); loop.py `push_lineups` order defect fixed (6431772). r5: joint wave draw fixes the sub rate and close-late keep; composition (WHO) is the next joint structure (L33). No arm adopted; R2 served | `docs/tests/rotation_close_game_audit_2026-09-10.md`, `rotation_sub_hazard_audit_2026-09-10.md`, `rotation_wave_audit_2026-09-11.md`; rotation experiments.md sections 6-11; commits 1e6bb3e, 6431772, 32f9f65 |
| Rebound / free throw S1 | Rebound: S1_weekly adopted (+15.6 floors); the early-conference calibration gate fails for every scheme (open). Free throw: S1_conf_aligned adopted (first-4-conference-weeks gap 0.94 vs 1.27 pp), the first data point for conference-aligned refits | rebound / free_throw experiments.md sections 7-8; commit a0810d8 |
| Possession-outcome r3 (your conference-regime question; Decision 9) | ANSWERED, against the hypothesis for this model: residual structure is a season-start effect (weeks 0-3), and bucketing by each team's own conference boundary explains LESS than plain calendar week; weeks-since-refit explains nothing. Conference flag, one-pass and iterative opponent adjustment all inside the floor (best +0.000225 vs floor 0.000804); non-conference calibration fails (2.49 pp) under every arm. Reference stands; Decision 9 stays PENDING EVIDENCE. Alignment cells (conf-aligned, weekly) on `first` NOT RUN by the 03:15 stop; `cont` shows a lead for conf-aligned. Next: shrink as-of style rates in the thin-sample early season (L35) | `docs/tests/possession_outcome_conference_regime_2026-09-10.md`; PO experiments.md round 3; pre-reg 0e55ca6 |
| Truth tables v1/v2 | Team shots, player games, finals, two sources each. The FT disagreement (7-10% of team-games) was technical free throws excluded from the event layer on purpose, not a defect (L24); with them carried, FTA matches the box on 99.3-99.8%. Four flagged 2025 finals settled by a third source | `docs/tests/truth_tables_v1_2026-09-10.md`, `truth_tables_v2_2026-09-10.md`, `ft_trip_reconciliation_2026-09-10.md`; commits 06c2a69, c35b154 |
| Engine contract | FGM by class and FTM added; the eFG% gate reads (smoke: 0.474 vs 0.509 actual under the leaked fg_make; 0.500 after r2) | commit 827503f |
| Lines source | CBBD ESPN BET 2023-2025 ACCEPTED: coverage 93-100%, overround 4.6%, spread MAE 8.81 vs an honest KenPom approximation at 9.00 on the same 16,076 games (corr 0.966), close beats open by 0.06. No line-level timestamps: CLV is open-to-close only, 2025 only. The old 10-11 pt MAE band was wrong for CBB | `docs/tests/lines_cbbd_validation_2026-09-10.md`; commits 4306c50, b369936 |
| Market scorecard v2 | Built and guarded: calibration at >= 200 seeds, ROI/Brier refused below ~2,000, per-row artifact `max_train_date < tipoff` on every family, open-to-close leak check | `docs/tests/market_scorecard_pipeline_2026-09-10.md`; commit ad92ef9 |
| Engine rewiring v1 | Inputs v2 (shooter block on `shot_shooter_id`, shrunk shooter column), run_meta with per-family `max_train_date` and `engine_commit`, parity digest v3; rotation R2 under S1, rebound/FT manifests. The 200-seed read ran 00:57-03:08 locally: 50 complete seeds (PROVISIONAL vs the 200 floor); gate table below; noise floor PENDING | commits 493a818, 3f0b7d9, 4503c52, 0cfd68a; `results/engine_v0/F2_2025_s200_rewire1` |
| AWS | Spot c7a.48xlarge launched 22:04, Linux parity PASS on every simulated value, terminated 00:22, ~2h18m, ~$5-6. The throughput read was unreliable (short run); the hf pull path bug found there is fixed (230e302). Recommend rotating the HF token (it appeared in a worker's process listing) | `docs/ops/aws_launch_chain.md` section 12; commits bfe807a, 230e302 |

## Rules and learnings added

Decision 9 (opponent adjustment, conference flag, refit cadence and alignment as MANDATORY BAKE-OFF ARMS, pending evidence; amended at your request from a standing rule to a test). Decision 10 (closed-loop gate for every engine-produced state feature; both frozen and refit-without arms). Learnings L22-L35. Data rule: model-artifact directories over 20 MB are gitignored and HF-synced. Token-discipline rule for workers (no polling, report once, write docs once).

## Event: weekly API limit

At 23:50 EDT the worker models' weekly limit terminated five workers mid-run; it reset at midnight and every lane was resumed from its on-disk state. Cost: about 30 minutes and one restarted engine run.

## Engine gate table (engine v1, F2 2025, 5,710 games, 50 complete seeds of 200; PROVISIONAL against the 200-seed floor; noise floor PENDING)

Run `F2_2025_s200_rewire1`, 00:57-03:08 local, 12 workers contended (447 poss/s/core; a full 200-seed slate needs ~8 h here, ~1 h on the box once its throughput is measured properly). Served sub-models: event round2_s1, fg_make round4_B1 on inputs v2, rebound S1_weekly, free throw S1_conf_aligned, rotation R2 under S1, clock reference (the run predates the v3c default). `docs/tests/engine_v1_gates_F2_2025_s200_2026-09-11.md`, commit b14d039.

| gate line | engine v1 (50 seeds) | engine v0 (5 seeds, yesterday) | Control | actual / tolerance | v1 |
|---|---|---|---|---|---|
| G1 possessions per game | 69.88 (+2.00) | 72.12 (+4.24) | 68.27 | 67.88, +/-1.0 | FAIL |
| G1 possession SD | 4.57 | 4.76 | 5.70 | 5.47, +/-0.75 | FAIL |
| G2 PPP cells passing | 2 of 9 | 1 of 9 | 3 of 9 | 9 of 9 | FAIL |
| G3 3PA / rim / FTA-FGA shares | .387 / .372 / .317 | n/a | n/a | .391 / .373 / .330 | PASS |
| G4 eFG% / TOV% / OREB% / FTr | .497 / .176 / .283 / .317 | n/a | n/a | .509 / .174 / .298 / .330 | FAIL |
| G5 margin SD ratio | 1.027 | 1.598 | 1.678 | 0.95-1.05 | PASS |
| G5 total SD ratio | 0.793 | 0.726 | 1.318 | 0.95-1.05 | FAIL |
| G5 home/away score corr | +0.03 | -0.62 | +0.08 | +0.25, +/-0.05 | FAIL |
| G5 PIT K-S p | 0.008 | ~0 | ~0 | > 0.1 | FAIL |
| G6 home margin, non-neutral / neutral | +5.67 / +2.11 | +3.91 / +1.54 | +6.06 / +2.23 | +5.74 / +3.29 | PASS / FAIL |
| G7 OT rate | 3.0% | 1.2% | 1.8% | 5.6% | FAIL |
| G8 minutes mean / SD ratio / players used | 30.6 / 1.23 / 8.8 | 29.9 / n/a / 8.8 | n/a | 29.8 / 1.0 / 9.8 | PASS / FAIL / FAIL |
| G9 margin bias | -0.21 | -1.82 | +0.15 | 0, +/-0.5 | PASS |
| G9 total bias | -0.89 | +0.01 | -2.83 | 0, +/-1.0 | PASS on cancelling errors, read as FAIL |
| G9 calibration slope | 0.89 | 0.18 | 0.96 | 0.95-1.05 | FAIL |
| Market: sim margin MAE / corr vs close | 9.29 / 0.908 | 16.03 / 0.329 | n/a | close itself 8.74 | n/a |

Reading: gate-level PASS 0 / FAIL 8 / needs-instrumentation 1 (v0: 0 / 6 / 3; worse only because G4 and G8 went from unmeasured to measured). Line-level, v1 passes seven lines v0 failed: margin variance is fixed (the fg_make leak), home/away correlation is no longer negative, margin bias is inside tolerance, the team-quintile slope is 0.84 (v0 0.56), and the sim now tracks the close at 0.91 correlation. Still open, in order: possessions +2.0 (clock composition from upstream make rates, L34; the served clock arm was not in this run), OREB% -1.6 pp and FTA/FGA -1.2 pp (upstream of the total), total SD ratio 0.79 (too little total variance, consistent with the same two), OT rate 3.0 vs 5.6%, home/away correlation +0.03 vs +0.25, rotation minutes spread and players used (rotation round 6). ROI and Brier are refused by the grader at 50 seeds. New open item: round4_B1's shooter slope is monotone 2 of 4 on the engine's roster-slot population vs 4 of 4 on the attempt-weighted design population.


## Open items and the recommended next step

1. Clock: done tonight (game-indexed call site, `v3c_srfloor_P3_s1` served). The remaining +1.0 to +1.7 possessions is an upstream make-rate defect (L34): re-read G1 after `round4_B1` and inputs v2 are served.
2. Rotation round 6: composition conditioned on who left (L33). R2 served meanwhile.
3. fg_make: serve `round4_B1` once inputs v2 are the default; per-class serving is open (BR wins the rim class by 65 floors).
4. Usage: a Decision-10 gate before wiring the tree; audit its `score_diff` with the own-row delta test.
5. Decision 9 stays pending: possession-outcome found adjustment and alignment inside the floor, free throw found conference-aligned refit wins on calibration. Next PO round: shrinkage of as-of style rates in the thin-sample early season, plus the two alignment cells not run.
6. Full 200-seed read plus the seed-offset noise floor, on AWS after a proper throughput measurement (runbook ready, parity proven, hf pull fixed); then the market scorecard's calibration section at 200 and ROI at 2,000.
7. Rebound: early-conference calibration fails under every scheme; the Decision-9 arms are the candidate fix.
8. Rotate the HF token.

## Left out and why

No unsealing of 2026. No ROI numbers (seed floor). No per-class fg_make serving (adapter). No rotation round 6 launch (03:02 stop). PROJECT_STATUS.md is updated at 03:47.

---

# HANDOFF.md

## SESSION 2026-09-10 -- READ THIS FIRST

What happened (numbered):
1. Reviewed CBB-Monte (code, 2025-26 results, data, tournament fork) and cfb-props-sim methodology via five workers; reports in `docs/postmortem/`.
2. Wrote `docs/FRAMEWORK_PLAN.md`; user decided: free odds only, grade on scores/stats first, no Odds API yet, keep KenPom arm, remote CBB-redesign, game markets first then props, social out of scope.
3. Verified data sources; CollegeBasketballData API works with the CFBD key (shared quota ~30k calls/month; ~29.5k remaining after day 1 minus the pbp pull).
4. Repo initialised, venv, HF dataset `mvpeav/cbb-sim-data` (nothing pushed yet).
5. Built L0: universe, crosswalk, gate references, KenPom snapshots, leak harness. Pre-registered Control engine.

6. Control engine, pace bake-off, pbp pull, possessions layer, coaches, variance decomposition, L3 bake-off all landed (see PROJECT_STATUS.md). Commits through 5a1ef86.

7. Later on day 1: rebound, free throw, usage, fg_make bake-offs decided (all LightGBM winners; Decision 8 amended the responsiveness gate to test slope). Eval harness, preseason 2027, daily chain landed. Commits through 43cc21e.

Still running / resume-safe (2026-09-10 evening): clock round 2, rotation round 2, possession-outcome round 2 (S1 walk-forward refit winning), engine v0 assembly (src/cbb_sim/engine/, results/engine_v0/), player attribution bake-off (docs/models/attribution/). Earlier: shot-classification diagnostic -> docs/tests/shot_classification_diag_2026-09-10.md; clock bake-off -> docs/models/clock/; rotation bake-off -> docs/models/rotation/ (+ player crosswalk src/cbb_sim/data/player_ids.py). Each writes its own docs and artifacts; PM commits after review.

Refuted this session: game-level pace as a sampler (L14); independent count draws (L10); CBBD season ratings as pregame features (L7); CFB's "team-beyond-coach = 0" for CBB (L15).

Standing rules recap: PM/worker split; bake-off before any choice; no hand tuning on output; bottom-up; matchup-specific; multi-level evidence; profitability frame with accuracy-first phase; sealed 2026; created_at < tipoff; leak test every external feature. Full text in `CLAUDE.md`.

Watch items: FTA/FGA trend 0.305 -> 0.352; November pace +3 poss; hoopR event vocabulary drift; player_box `active` placeholder before 2026; CBBD `/recruiting/portal` filter is `year` not `season`; git push prints a harmless "Key not valid for use in specified state" credential warning.

Refuted this session: nothing yet.

Evidence trail: `docs/tests/data_audit_hoopr_2026-09-10.md`, `docs/tests/data_audit_cbbd_2026-09-10.md`, `docs/tests/gate_reference_2026-09-10.md`, `docs/tests/leak_test_kenpom_2026-09-10.md`.

Watch item (2026-09-10): CBBD /teams/roster season=2027 returns 0 players for every team; roster continuity is an upper bound until rosters populate. Recheck weekly; fallback is hoopR rosters_2027 or ESPN team roster endpoint (light use).

## SHUTDOWN 2026-09-10 ~17:00 ET -- RESUME CHECKLIST (read before doing anything)

State at shutdown: repo at the last commit on main (run `git log -1`). Five workers were told to checkpoint by 16:50 ET:
1. Clock round 2: NO ARM ADOPTED. ROOT CAUSE FOUND (L20): horn-ending possessions are truncated, not censored, in training; flag them right-censored, model intended duration, engine truncates at the horn. Add a CBBD clock-completeness flag to games_universe and re-base the end-of-half gate on clock-complete halves. This is round 3 and it is the first thing to run on resume; the engine's +4 possessions is this bug.
2. Rotation round 2: NO ARM ADOPTED (docs/models/rotation/experiments.md section 4). R5 hybrid fixes lineup concentration (K-S D 0.03) and blowouts but under-keeps starters late in close games by 4-7 pp; R2 fails only blowouts. Round 3 = R5 with a close-game keep-starters override fitted like the other two.
3. Possession-outcome round 2: winners lgbm+S1 (first) and cascade+S1 (cont), calibration 0.98 / 1.86 pp; noise floor may be PARTIAL (see experiments.md section marked PARTIAL). S1 (in-season monthly walk-forward refit) is the default training scheme for all sub-models per pre-registration.
4. Engine v0: in progress under src/cbb_sim/engine/ and results/engine_v0/; see docs/models/engine/RESUME.md if written. Provisional adapters flagged in run_meta.json.
5. Player attribution: in progress; see docs/models/attribution/RESUME.md and the PARTIAL marker in its experiments.md.

TOP PRIORITY ON RESUME (found 16:45 ET by the attribution worker): CBBD `participant_1_id` is the ASSISTER, not the shooter, on ~49% of assisted made FGAs (matches `shot_shooter_id` only 50.9% of the time on those rows; 74.5% overall). `src/cbb_sim/models/usage.py` reads the shooter off `participant_1_id`, so the usage bake-off's shooter labels are contaminated on assisted makes. Fix: usage must key the shooter on `shot_shooter_id` (98.7% coverage), then RERUN the usage bake-off (same pre-registration, note the label fix as a data fix), and re-check fg_make/free_throw shooter keys for the same defect. Also: CBBD blanks the rebounder on 18-25% of offensive-rebound rows (team rebounds); attribution reports it, never imputes.

Engine v0 note: the F2 gate run never returned (block size too large); use the smoke command in docs/models/engine/RESUME.md first. The engine's 120-sim smoke read: possessions +4.65, PPP -6.3%, cancelling in the total; clock model is the first suspect. Verify `winner_FGA_3.joblib` is the LightGBM re-export (the engine worker saw the stale one).

On resume: (a) `git status` and commit anything the workers left uncommitted, excluding files > 50MB and anything under data/raw or results; (b) read the five docs above; (c) launch FRESH workers for clock round 3, rotation round 3, engine v0 completion and attribution completion, each briefed with the relevant experiments.md (do not resume the old agents; their contexts are too large); (d) then the seed-noise study and gates G1-G7 engine vs Control.

Decided today and not to be reopened: Decisions 1-8; learnings L1-L19; winners for rebound, free throw, usage (5 classes), fg_make (3 classes), possession outcome (S1 arms).

## QUEUED 2026-09-10 evening: S1 scheme confirmation passes (PM)

S1 (in-season monthly walk-forward refit) is the standing default for every sub-model (docs/models/README.md "Standing result", L21), but rebound, free throw, fg_make, usage round 1-2 and attribution were decided under static fits. Each needs a pre-registered "S1 scheme confirmation" round: winner refit under S1 vs static on the selection fold, same gates and floor, adopt S1 unless a gate regresses beyond the floor; persist per-month artifacts in a versioned directory with a manifest (refit_date -> path) for the engine's generic per-game selector. Usage (round 2b), rotation (round 3b) and clock (round 3b) workers were told on 2026-09-10 evening. Still to dispatch, after the current workers finish and the machine is free: rebound, free throw, fg_make (one Sonnet runner), attribution (second, since its assisted binary already shows a 4-7 pp within-season calibration miss under the static fit, L22).

Watch items added 2026-09-10 evening (from truth tables v1, `docs/tests/truth_tables_v1_2026-09-10.md`): (1) RESOLVED same evening (L24): the FTA/FTM disagreement is technical free throws, excluded from the event layer on purpose and included in the box; `ft_trip_ambiguous` refuted as a cause. Truth tables v2 add technical FTA/FTM columns and reconcile with them added back; the engine needs a technical-FT rule (0.14-0.22 pts/team-game low), queued. (2) Player crosswalk to CBBD ids is 0% for 2022-23 (no CBBD rosters pulled for those seasons); any player-level truth for those seasons is box-only. (3) 2025's 4 flagged finals settled by ESPN game pages (`data/processed/truth/diag_finals_resolution_2025.json`): hoopR right on 3, CBBD on 1 (401722537, where hoopR carries the side-flip); truth tables v2 apply the resolution.

## OVERNIGHT PLAN 2026-09-10 21:50 EDT -> 2026-09-11 04:00 EDT (PM, user away)

User decisions at 21:40 EDT: AWS box approved (spot, terminate when idle); CBBD API free to use without waste; CBBD lines acceptable as the free lines source; PM prioritises; maximise parallelism; real wall clock only.

Running at 21:50 (nine workers): usage round 2/2b (shooter fix + S1); possession-outcome round 3 (conference regime diagnostic, cadence x opponent-adjustment cross); rotation round 4 (substitution-hazard family); clock round 3c (closed-loop state parametrisation in the engine); fg_make round 3 (shooter re-key); rebound + free-throw S1/cadence confirmation; attribution round 2 (S1 + score_diff leak fix); CBBD lines pull/validation + 2022-23 rosters; AWS bring-up with Linux parity then terminate.

Plan-vs-actual (PM fills the "actual" column as reports land):
| window (EDT) | planned | actual |
|---|---|---|
| 22:00-00:00 | usage, fg_make r3, rebound/FT, attribution, lines, AWS parity report; commit each; engine adapters pointed at every adopted manifest | LANDED: usage r2/2b (2a7f624), fg_make r3 (40a7be6; interim served model round3_shooter), attribution r2 (9f585e9), lines accepted (b369936), market scorecard v2 (ad92ef9), truth v2, HF mirrored. NOT LANDED: rebound/FT S1, AWS parity. EVENT 23:50 EDT: weekly API limit (Opus and Sonnet) terminated five workers mid-run: PO r3 (stages 7-11 pending), clock 3c (screening done, deciding read pending), rotation r4 (runs in flight), fg_make r4 (B1 passing D8 on all classes), rebound/FT (rebound job in flight). Their background training processes are gone. Limit resets 12am ET; PM relaunches finishing-only workers, max 3 concurrent, to conserve the new week's quota. AWS instance i-02092cafa1d72fdde found running since 22:04 EDT with its worker possibly dead; time-boxed to 01:05 EDT then terminated. |
| 00:00-01:30 | clock 3c and rotation r4 decide; possession-outcome r3 decides cadence and opponent adjustment; engine rewired with all winners; 200-seed gate read G1-G9 vs Control on AWS if parity held, else local | LANDED: clock 3c (nothing adopted; P2/P3 state closed; best arm +1.16 poss; residual = uniform -1.9% duration; L31; straddled runs re-run, 15f812a); rotation r4 (nothing adopted; hazard family identified, 5/8 cells, beats R2 by 61-76 floors; L30; loop.py push_lineups order fixed 6431772); fg_make r4 (B1 shrunk shooter wins; join-coverage leak found; L32; blocked on engine inputs v2); rebound S1_weekly + free throw S1_conf_aligned (a0810d8); AWS parity PASS, instance terminated, ~$5-6 (bfe807a); hf pull fixed (230e302); rewiring commits landed (inputs version, run_meta max_train_date per family, engine_commit, parity v3). NOT LANDED at 01:33: the 200-seed read (results dir created, no run process visible; rewiring worker not yet reported); possession-outcome r3 still running (resumed 00:20 with a 6.5 h budget, told to finish by 03:15); clock r4 diagnosis+pre-registration committed (421b97b), arms not fitted; rotation r5 pre-registered (a14a569), running. |
| 01:30-03:00 | gate report reviewed; worst failing gate gets its next pre-registered round; market scorecard on CBBD close lines 2023-2025 if lines validated (calibration vs de-vigged market, edge buckets, no ROI below 2,000 seeds) | Rotation r5 decided (nothing adopted; L33; 32f9f65). Clock r4 decided (nothing adopted; call site game-indexed; v3c arm served; L34; 0f0029b). PO r3 stopped at 03:15 and decided (reference stands; L35; c032636). 200-seed run ended 03:08 with 50 complete seeds; graded by 03:19 (b14d039). Market scorecard v2 ran on the read (calibration section at 50 seeds is provisional; ROI refused). No new rounds launched after the 03:02 stop. |
| 03:00-04:00 | PROJECT_STATUS.md and this file updated; all results committed and pushed; HF synced; instance terminated; summary for the user at the top of this file | Done by 03:23: summary at the top of this file (gate table filled), PROJECT_STATUS updated (542054a), tree clean, remote verified, HF fully mirrored (raw 919, results 239, engine_inputs 25, model_artifacts 768; 0 missing), no EC2 instance running, memory updated. Large S1 artifact dirs (rebound 524 MB, free throw 175 MB) gitignored and on HF, not in git. |

Not to be done without the user: unseal 2026; reopen Decisions 1-10; paid data; scraping; history rewrites; killing others' processes.

## SESSION 2026-09-11 MORNING, 09:53 -> 13:30 EDT (PM, user present; hard stop 13:30)

Launched 09:58 EDT, five workers, reports due 12:45, each capped at 6 threads / 6 engine workers, commits own files only:
1. Rotation round 6 (Opus): composition conditioned on who left (L33); pre-registers section 12, offline + closed-loop (Decision 10) paired with rot5 seeds.
2. Possession-outcome round 4 (Opus): early-season shrinkage of as-of rates (L35) plus the two alignment cells round 3 did not run; bearing on Decision 9 reported, not decided.
3. Engine v1 paired re-read (Opus): served stack with ENGINE_CLOCK=v3c on the first 20 seeds of F2_2025_s200_rewire1 -> results/engine_v0/F2_2025_s20_rewire1_clockv3c; variance decomposition (possessions vs PPP vs covariance, home/away) and OT tie-rate diagnostic on the 50-seed run; diagnose only.
4. Usage Decision-10 gate (Sonnet): own-row delta test on every state feature of the usage tree; closed-loop arms U1 / tree live / tree frozen / tree refit-without-state.
5. AWS 200-seed read + seed-offset noise floor (Sonnet): proper throughput measurement first; runs A/B to results/engine_v0/F2_2025_sN_v1_clockv3c_{A,B}; terminate by 12:45 regardless; budget guard $25.

Not launched: rebound early-conference calibration (waits on lane 2's shrinkage result); HF token rotation (user action; token appeared in a worker's process listing last night).
