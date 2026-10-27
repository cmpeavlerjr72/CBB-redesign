# Shot-location stage (L3, between possession outcome and make / foul / rebound)

Status: ROUND 1 PRE-REGISTERED 2026-10-27 (`experiments.md` section 1). Nothing fitted, nothing served, freeze ON.
Companion docs: `features.md`, `experiments.md`. Origin and evidence trail: `docs/tests/coordinate_audit_2026-10-27.md`,
`gamezone_*_2026-10-27.md`, `sportradar_*_2026-10-27.md`, `docs/research/shot_coordinate_sources_2026-10-27.md`.

## 1. Purpose
Given offence, defence, site, game state and the possession-outcome class (`FGA_jump2` or `FGA_3`; the rim class is
ONE band by PM ruling 11 of 2026-10-27), sample the location band of the attempt so that make, and-one draw and
offensive rebound can be conditioned on it. Candidate stage; exists only if round 1's Reads B and C pass.

## 2. Target variable
Band within class (section 1.2 of `experiments.md`): jump2 -> short2 / long2; three -> corner3 / wing3 / top3.
Population: fg_make design population, CBBD located attempts, seasons 2022-2025, folded frame.

## 3. Methodology at a glance
Walk-forward folds (F1 train 2022-23 test 2024; F2 train 2022-24 test 2025, selection). S1 monthly schedule for the
Read A wrappers. Three reads: A ceiling (attempt level), B predictability (multinomial band models B0-B3), C implied
game aggregates. Primary decision on B + C. Registered measurement ceiling: split-half reliability of team band
profiles 0.89 / 0.79 / 0.67 / 0.80 offence, 0.69 / 0.76 / 0.52 / 0.75 defence.

## 4. Winner
Not yet run.

## 5. Robustness check
Not yet run (segments registered in `experiments.md` 1.3).

## 6. Decisions log
- Rim excluded (ruling 11): outcome-dependent rim placement in ESPN 2022-25 and Sportradar 2013-18.
- CBBD 2022-2025 only; no vendor union (rulings 3, 13).
- Foul channel read as and-one rate only (feed logs no FGA on missed fouled shots).

## 7. Consumption from the sim
None. No engine column, no flag. Serving needs its own paired-seed sim round.

## 8. Artifacts
None at this commit. Round 1 writes under `data/processed/models/shot_location/round1/` (gitignored, HF `model_artifacts`).

## 9. Known gaps / followups
`experiments.md` section 1.10.
