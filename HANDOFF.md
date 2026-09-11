# SUMMARY FOR USER, 2026-09-10 21:50 -> 2026-09-11 04:00 EDT

Written by the PM at 03:05 EDT (draft; the 03:47 wrap-up fills the PENDING items). Every number below is in a committed doc; paths in brackets.

## What was decided (with evidence)

| lane | decision | evidence |
|---|---|---|
| Usage r2/r2b | Shooter re-keyed on `shot_shooter_id` (CBBD orders shooter/assister at random on assisted makes; 51% agreement). LightGBM winner unchanged on all 5 classes; U1 top-3 usage gap -2.17 -> -1.79 pp; S1 monthly adopted (+0.44-0.82 floors, no gate lost). Engine still runs U1; tree blocked on a Decision-10 gate. L28 | `docs/tests/shooter_key_audit_2026-09-10.md`, `docs/models/usage/experiments.md` r2-2b, commit 2a7f624 |
| fg_make r2/2b/3/4 | `score_diff` was the score AFTER the shot (post-outcome leak; 62-84% of its effect manufactured; tripled sim margin variance). Winner: engine-safe state, no margin term, S1 (L27). r3: the shooter label defect had manufactured the shooter signal (FGA_3 quintile span 35.9 -> 2.8 pp; L29); corrected-label arm served interim. r4: one shrunk shooter column (B1) wins every gate, D8 slope 0.31 -> 0.92-1.04; a join-coverage leak found and repaired (L32); engine inputs must be keyed on `shot_shooter_id` (skew worth 1 pp of 3P%). Served: `round3_shooter_S_C_s1` until inputs v2 carry the shrunk column, then `round4_B1` | `docs/tests/fg_make_state_confound_2026-09-10.md`, `fg_make_shooter_key_2026-09-10.md`, `fg_make_shooter_skill_2026-09-10.md`; fg_make experiments.md sections 14-20; commits 9ae38fa, 2b83b94, 40a7be6, 46f54bc |
| Attribution r1/r2 | r1 (static, F1 selects): REB_off cond_logit, REB_def lgbm, steal proportional, block lgbm, assisted/blocked aware_ridge; assist and stolen adopt nothing. Table-builder defect (all-zero team features) fixed (L22). r2: the score leak reached 2 of 8 targets; REB_def and block move to S1 on calibration; assisted within-2025 miss 6.1 -> 4.3 pp, still fails | `docs/tests/attribution_team_asof_lg_defect_2026-09-10.md`, `attribution_score_diff_leak_2026-09-10.md`; commits 75e09f6, 9f585e9 |
| Clock r3/3b/3c | Horn censoring fix confirmed (~0.45 poss; L26). Nothing adopted offline (0/9, 0/6) or closed-loop (0/6). State question closed: P2 = P3 beat P1 by 0.42 poss in the engine; cell-based family preferred (tempo slope inside the D8 band). A freeze ablation detects but cannot size a loop (L31). Best arm `srfloor|P3|S1` +1.16 poss vs incumbent +2.70; residual is a uniform -1.9% duration shortfall. r4: the clock call site is now game-indexed and `ENGINE_CLOCK` serves `v3c_srfloor_P3_s1` (provisional; 1a5acef). Round 4 adopted nothing (0/5): the shortfall is 67% state composition from upstream make rates (`prev_end` mix), only 32% the clock's own law; train/serve quantity closed four ways; recency weighting makes it worse because duration rises through the season (L34). Clock round 5 not warranted; the residual is an event/fg_make make-rate defect | `docs/tests/clock_censoring_audit_2026-09-10.md`; clock experiments.md sections 8-14; commits 9961890, 15f812a, 421b97b |
| Rotation r3/3b/4/5 | r3: the override family cannot reach the close cell (L25); S1 adopted as scheme; R2 is 2/8 cells under S1. r4: per-player hazards identified, 5/8 cells, beat R2 by 61-76 floors on minutes MAE, but over-substitute 43% (L30); loop.py `push_lineups` order defect fixed (6431772). r5: joint wave draw fixes the sub rate and close-late keep; composition (WHO) is the next joint structure (L33). No arm adopted; R2 served | `docs/tests/rotation_close_game_audit_2026-09-10.md`, `rotation_sub_hazard_audit_2026-09-10.md`, `rotation_wave_audit_2026-09-11.md`; rotation experiments.md sections 6-11; commits 1e6bb3e, 6431772, 32f9f65 |
| Rebound / free throw S1 | Rebound: S1_weekly adopted (+15.6 floors); the early-conference calibration gate fails for every scheme (open). Free throw: S1_conf_aligned adopted (first-4-conference-weeks gap 0.94 vs 1.27 pp), the first data point for conference-aligned refits | rebound / free_throw experiments.md sections 7-8; commit a0810d8 |
| Possession-outcome r3 (your conference-regime question; Decision 9) | PENDING at 03:05: Stage A diagnostic and the cadence x opponent-adjustment cross; hard stop 03:15 | `docs/tests/possession_outcome_conference_regime_2026-09-10.md` (if written); PO experiments.md round 3 |
| Truth tables v1/v2 | Team shots, player games, finals, two sources each. The FT disagreement (7-10% of team-games) was technical free throws excluded from the event layer on purpose, not a defect (L24); with them carried, FTA matches the box on 99.3-99.8%. Four flagged 2025 finals settled by a third source | `docs/tests/truth_tables_v1_2026-09-10.md`, `truth_tables_v2_2026-09-10.md`, `ft_trip_reconciliation_2026-09-10.md`; commits 06c2a69, c35b154 |
| Engine contract | FGM by class and FTM added; the eFG% gate reads (smoke: 0.474 vs 0.509 actual under the leaked fg_make; 0.500 after r2) | commit 827503f |
| Lines source | CBBD ESPN BET 2023-2025 ACCEPTED: coverage 93-100%, overround 4.6%, spread MAE 8.81 vs an honest KenPom approximation at 9.00 on the same 16,076 games (corr 0.966), close beats open by 0.06. No line-level timestamps: CLV is open-to-close only, 2025 only. The old 10-11 pt MAE band was wrong for CBB | `docs/tests/lines_cbbd_validation_2026-09-10.md`; commits 4306c50, b369936 |
| Market scorecard v2 | Built and guarded: calibration at >= 200 seeds, ROI/Brier refused below ~2,000, per-row artifact `max_train_date < tipoff` on every family, open-to-close leak check | `docs/tests/market_scorecard_pipeline_2026-09-10.md`; commit ad92ef9 |
| Engine rewiring v1 | Inputs v2 (shooter block on `shot_shooter_id`, shrunk shooter column), run_meta with per-family `max_train_date` and `engine_commit`, parity digest v3; rotation R2 under S1, rebound/FT manifests. The 200-seed read launched 00:57 locally with a time budget; expected ~40-55 complete seeds (PROVISIONAL vs the 200 floor). Gate table PENDING | commits 493a818, 3f0b7d9, 4503c52, 0cfd68a; `results/engine_v0/F2_2025_s200_rewire1` |
| AWS | Spot c7a.48xlarge launched 22:04, Linux parity PASS on every simulated value, terminated 00:22, ~2h18m, ~$5-6. The throughput read was unreliable (short run); the hf pull path bug found there is fixed (230e302). Recommend rotating the HF token (it appeared in a worker's process listing) | `docs/ops/aws_launch_chain.md` section 12; commits bfe807a, 230e302 |

## Rules and learnings added

Decision 9 (opponent adjustment, conference flag, refit cadence and alignment as MANDATORY BAKE-OFF ARMS, pending evidence; amended at your request from a standing rule to a test). Decision 10 (closed-loop gate for every engine-produced state feature; both frozen and refit-without arms). Learnings L22-L34. Data rule: model-artifact directories over 20 MB are gitignored and HF-synced. Token-discipline rule for workers (no polling, report once, write docs once).

## Event: weekly API limit

At 23:50 EDT the worker models' weekly limit terminated five workers mid-run; it reset at midnight and every lane was resumed from its on-disk state. Cost: about 30 minutes and one restarted engine run.

## Engine gate table

PENDING at 03:05; filled at 03:47.

## Open items and the recommended next step

1. Clock: done tonight (game-indexed call site, `v3c_srfloor_P3_s1` served). The remaining +1.0 to +1.7 possessions is an upstream make-rate defect (L34): re-read G1 after `round4_B1` and inputs v2 are served.
2. Rotation round 6: composition conditioned on who left (L33). R2 served meanwhile.
3. fg_make: serve `round4_B1` once inputs v2 are the default; per-class serving is open (BR wins the rim class by 65 floors).
4. Usage: a Decision-10 gate before wiring the tree; audit its `score_diff` with the own-row delta test.
5. Possession-outcome r3 decides whether opponent adjustment and conference alignment become standing rules (Decision 9).
6. 200-seed and 2,000-seed reads on AWS after a proper throughput measurement (runbook ready, parity proven).
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
| 01:30-03:00 | gate report reviewed; worst failing gate gets its next pre-registered round; market scorecard on CBBD close lines 2023-2025 if lines validated (calibration vs de-vigged market, edge buckets, no ROI below 2,000 seeds) | |
| 03:00-04:00 | PROJECT_STATUS.md and this file updated; all results committed and pushed; HF synced; instance terminated; summary for the user at the top of this file | |

Not to be done without the user: unseal 2026; reopen Decisions 1-10; paid data; scraping; history rewrites; killing others' processes.
