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
