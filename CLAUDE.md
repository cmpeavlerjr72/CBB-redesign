# CLAUDE.md

Possession-level Monte Carlo simulation engine for NCAA men's basketball: game scores, game markets, player props.
Clean-sheet rebuild of `C:\Users\devuser\CBB-Monte` (2025-26 season), following the methodology of `C:\Users\devuser\cfb-props-sim`.

Read in order: `PROJECT_STATUS.md` -> `docs/FRAMEWORK_PLAN.md` -> `ARCHITECTURE_DECISIONS.md` -> `docs/SIM_GUARDRAILS.md` -> `docs/LEARNINGS.md` -> `docs/models/README.md` -> `HANDOFF.md`.
Postmortem evidence for every rule below: `docs/postmortem/01-06`.

## Standing rule: PM / worker split (user rule, 2026-09-10)

Fable 5.1 is the project manager: writes specs, reviews bake-off tables and gate reports, makes decisions, writes the decision docs. Worker agents (Opus for engine core and cascade models, Sonnet for pipelines, graders and bake-off runners, Haiku for lookups and summaries) do the reading, coding, and analysis. Keep tool-heavy work out of the PM context.

## Standing rule: bake-off before any choice (user rule, 2026-09-10)

No model class, distribution family, sampling scheme, feature bundle, or anchor rating is adopted because it is convenient or familiar. Every choice is a pre-registered comparison of multiple candidates on walk-forward holdouts:

- The spec (candidates, features, folds, primary metric, segment breakdowns, noise floor, decision rule) is written to `docs/models/<model>/experiments.md` and committed BEFORE the experiment runs.
- Folds: fold 1 trains through 2022-23 and tests 2023-24; fold 2 trains through 2023-24 and tests 2024-25. Fold 2 is the selection metric. The 2025-26 season (with lines) is sealed until a sub-model has been selected on fold 2.
- One grading script scores every arm blind. A winner must beat a spec-identical retrain under another seed (the noise floor). Ties go to the simpler model.
- An offline winner ships only after a paired-seed sim run shows no gate regressed.
- `experiments.md` is append-only. Status changes go to `docs/models/change_ledger.md` in the same commit.

## Standing rule: no hand tuning on engine output (user rule, 2026-09-10)

No post-hoc multipliers, caps, clips, offsets, calibration curves, or blends on sim output. If the engine's output has to be adjusted to resemble reality, the engine is wrong and the responsible sub-model must be found and fixed. Exact wording and the cautionary inventory: `cfb-props-sim/docs/SIM_GUARDRAILS.md` core principle and section 5. Last year's tournament fork added a post-hoc win-probability calibration curve; that is the pattern this rule bans. Decision-layer probability calibration is allowed only walk-forward, clearly labelled, and never to hide a sim-level defect.

## Standing rule: bottom-up the cascade

Build and gate the smallest unit first: possession outcome, shot make, rebound, foul, then clock, then game aggregates, then player attribution, then markets. Never accept a downstream stage that compensates for a known upstream bias. A passing aggregate over a biased sub-model is a failure.

## Standing rule: matchup-specific, not league-average

Every sub-model must show responsiveness: predictions bucketed by team or player prior quintile must slope with actuals, not sit flat at the mean. Check explicitly when evaluating any new model.

## Standing rule: multi-level evidence

Every finding and every fix shows overall, per-game, per-team, per-possession-type, and per-player (when relevant) evidence. Aggregate-only proof is insufficient; last year's engine passed a total-variance look while offence and defence errors cancelled underneath. Underpowered cells are labelled underpowered, never presented as signal or absence of signal.

## Standing rule: profitability frame, with an accuracy-first phase (user decision 2026-09-10)

The objective is profitability against lines. Until a free lines source is validated, grade primarily on score and stat accuracy (margin/total MAE, per-stat calibration, distribution shape), and add the market scorecard (calibration vs de-vigged market, ROI by edge bucket at real odds, CLV cross-check) wherever lines exist. An edge that beats the close but cannot predict line movement is presumed leaked. Grade against verified finals, never against pbp-accumulated totals. No paid data without asking the user again.

## Standing rule: backtests must be honest

- Every backtest row must satisfy `created_at < tipoff`, enforced in code. Last year's two "improved" variants were generated after the games were played and are worthless as evidence.
- Every external feature (KenPom, any rating feed) passes the leak test before entering a feature table: change-form correlation with own-week margin, |corr| <= 0.15, honest baseline 0.04-0.08.
- Grading truth (finals, OT accumulation) is verified against a second source before any grade is trusted.
- A seed-count study fixes the minimum seeds before any ROI number is read.

## Modeling rules

- Every rating feature is expressed relative to its own snapshot's league mean. Raw levels are banned (KenPom's league-mean AdjO drifted 100 -> 109.3 across last year's data and inflated every model).
- Home/away/neutral is a first-class feature in every scoring-stage model; audit each model's feature list for it.
- Every team and player rate feature is opponent-adjusted as-of (Decision 9); raw-centred rates are a reference arm only. Conference-game flag is first-class like home/away. Refit cadence and conference alignment are bake-off dimensions, not assumptions.
- One pace realisation per simulated game, both teams scaled by it. Dispersion comes from the model's own variance function and is validated against realised residual SD.
- Rates per possession, not counts.
- RNG seeded on (seed, game_id, family). Paired bake-off arms share aligned streams.
- Ties resolve through an overtime model, never discarded.
- Rule-era flags live in GameState, not baked into sub-models.
- One engine. Variants live behind flags over a shared core. No forks.
- Sim loop uses lookup tables and vectorized NumPy, never live model calls. Pin thread-count env vars in any container.

## Data rules

- Primary source: sportsdataverse/hoopR-mbb-data (CC BY 4.0). Embedded ESPN market columns in pbp are stripped from features.
- Do not scrape sports-reference.com (terms ban ML use), barttorvik.com or masseyratings.com (robots disallow AI crawlers), PrizePicks, Underdog, DraftKings.
- KenPom snapshots from last year stay on disk and are one bake-off arm; our own ratings from hoopR box data are the other and the compliant fallback.
- Bulk data (`data/raw`, `results`) is gitignored and synced to the private HF dataset `mvpeav/cbb-sim-data` via `scripts/hf_sync_data.py`. `data/processed` and `data/reference` are tracked.

## Git practice

Remote: `https://github.com/cmpeavlerjr72/CBB-redesign.git`, branch `main`. Commit and push after every meaningful change. Tracked files stay under 50MB. Verify pushes with `git ls-remote origin main`. Never commit `.env`.

## Scripts convention

`scripts/` is flat and prefix-named: `pull_ build_ train_ run_ exp_ diag_ grade_ leak_ chain_ inv{N}_`. Trainers are versioned filenames (`train_x_v2.py`), never overwritten. `src/cbb_sim/` is the small tested core (game state, engine, usage allocation, lookups, variance decomposition, clients).

## Worker discipline

Workers run concurrently on one machine. Never kill, restart, or signal a process you did not start (`Stop-Process python` is banned); restart only your own PID. Never overwrite a data file another worker may be reading; write a versioned sibling and let the PM switch. Report any interruption you caused.

## Windows

Set `PYTHONIOENCODING=utf-8` before running scripts that print non-ASCII. Use `.venv/Scripts/python.exe`.
