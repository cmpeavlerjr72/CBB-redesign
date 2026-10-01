# Worker rules, overnight session 2026-09-30 20:40 EDT -> 2026-10-01 04:00 EDT

PM-owned. Every lane brief points here. The user is away; nobody can answer questions. Decide, record the choice and its reason in your doc, and keep going.

## Clock: the real wall clock only

Neither the PM nor any worker can estimate elapsed time. Never reason about how long something "has taken" or "will take" from feel.

- Run `Get-Date -Format "yyyy-MM-dd HH:mm:ss K"` at the start, before launching every compute step, and after every compute step. Put those timestamps in your session log.
- Size a run from a MEASURED timing (a timed slice of at least 3 minutes of the same job at the same core count), never from a guess or a sub-minute probe.
- 02:00 EDT (2026-10-01): launch no new compute that the measured timing says will not finish by 02:30.
- 02:30 EDT: stop computing. Kill only your own PIDs. Write the docs with whatever is complete; mark anything unfinished as NOT RUN or PARTIAL with the resume command.
- 02:45 EDT: your final report is back with the PM, committed and pushed. A late report is worth less than a partial one on time.

## Compute

- Home box, 20 cores, shared by seven lanes. Your core cap is in your brief; it is a hard cap on concurrent worker processes x threads. Pin `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` and LightGBM `n_jobs` so the cap holds.
- AWS: APPROVED by the user for this session (message after 20:42 EDT, "approval and even encouragement"). Only the OPERATOR lane touches AWS. Other lanes do not launch, start, stop or touch any instance; they hand jobs to the operator through the box queue below.
- BOX QUEUE: to get a job run on the box (full-size 5,710 x 200 paired loops with four floor draws, big retrains), commit and push your code first, then write ONE request file `docs/ops/box_queue/<lane>_<n>.md` containing: the git commit hash, the exact command line(s), the flags / env, the inputs and artifacts needed (and their HF bulk key or path), the expected output paths, and the one line you need decided. Do not commit request files. The operator picks requests up between jobs, runs them, syncs outputs back to the stated local paths and writes `<lane>_<n>.done.md` next to the request (or `.failed.md` with the error). Do not poll for it more often than every 10 minutes. Requests arriving after 01:30 EDT may not run; the operator stops accepting at 02:00 and the instance is terminated by 03:00.
- No paid data. Do-not-scrape list in CLAUDE.md stands.
- Use `.venv/Scripts/python.exe`; set `PYTHONIOENCODING=utf-8`.
- Multi-step shell goes in a script FILE (in `scripts/` if it is worth keeping, otherwise your scratch dir), never a long ad-hoc line with backticks or globs that could execute file names.
- Never kill, restart or signal a process you did not start. Never `Stop-Process python`, never `pkill`. Track your own PIDs.
- No polling under 10 minutes. Long commands: one foreground call with a long timeout, or background and wait for the completion notification.

## Foundations to read on (unless your brief says otherwise)

- Engine inputs v3 (the live-replay inputs), never v2.
- Grading truth: set `CBB_TRUTH=verified_v1` explicitly in every run and use the verified same-rule samples. Do not rely on the default (lane E is flipping it tonight).
- Closed-loop floors follow Decision 12: at least 4 seed-offset draws and a paired game bootstrap, the floor is the max. A 500 x 25 loop cannot decide a G5 ratio line; say "underpowered" rather than pass or fail.
- Decision 11: a fix that exposes a compensation is VALIDATED-PENDING-SHIP-ACTION, not adopted and not refused. You adopt nothing and change no served default. New behaviour goes behind a default-off flag whose off path is bit-identical (prove it with the parity reference).

## Standing rules that bind you (CLAUDE.md has the full text)

- Pre-register before running: the spec (candidates, features, folds, primary metric, segments, noise floor, decision rule) is APPENDED to the relevant `docs/models/<model>/experiments.md` and committed BEFORE the experiment runs. Append-only. Status changes go to `docs/models/change_ledger.md` in the same commit.
- Folds: fold 1 trains through 2022-23 and tests 2023-24; fold 2 trains through 2023-24 and tests 2024-25 and is the selection metric. 2025-26 stays SEALED.
- No hand tuning on engine output: no multipliers, caps, clips, offsets, calibration curves or blends on sim output. A fix is a sub-model change.
- Multi-level evidence: overall, per-game, per-team, per-possession-type (and per-player where relevant). Label underpowered cells as underpowered.
- Responsiveness: predictions by prior quintile must slope with actuals.
- Winner must beat a spec-identical retrain under another seed. Ties go to the simpler model.
- Trainers are versioned filenames, never overwritten. Never overwrite a data file another lane may read: write a versioned sibling.

## Git, shared working tree

- All lanes share one working tree on `main`. Stage ONLY your own files by explicit path (`git add <path> ...`). Never `git add -A`, `git add .`, `git commit -a`, `git stash`, `git pull --rebase`, `git reset`, `git checkout -- <file>`.
- If you must edit a file another lane may also edit (`engine/loop.py`, shared `src/cbb_sim` modules), keep the hunk small and behind your flag, re-read the file immediately before editing, and commit that file right after the edit with `git add -p`-free explicit staging: check `git diff <file>` first and if it contains hunks that are not yours, do NOT commit it; say so in your report.
- Push after each commit (`git push origin main`; on a non-fast-forward, `git pull --no-rebase` is also banned: use `git fetch` then `git merge --ff-only origin/main` or, if that fails, plain `git merge origin/main` with no local uncommitted changes of your own at risk; if unsure, leave the commit local and report it). The credential helper prints `Key not valid for use in specified state`; pushes still land.
- Tracked files under 50 MB; model-artifact directories over 20 MB are gitignored. Never commit `.env`.
- PM-owned, do not edit: `HANDOFF.md`, `CLAUDE.md`, `PROJECT_STATUS.md`, `ARCHITECTURE_DECISIONS.md`, `docs/SIM_GUARDRAILS.md`, `docs/LEARNINGS.md`. Propose text for them in your report.

## Reporting

- Write each doc once, when final. Report to the PM once, at the end, in under 400 words: verdict per pre-registered line (with the number, the floor and the doc path), what was NOT run, any incident you caused, the commits, and the next step you recommend. Plain statements; do not present a partial or underpowered result as a finding.
- Report every interruption you caused to another lane.
