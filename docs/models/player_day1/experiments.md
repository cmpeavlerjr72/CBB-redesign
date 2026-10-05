# Player layer day-1 priors -- experiments

Append-only. Section 1 is the pre-registration, committed BEFORE any arm runs (CLAUDE.md "bake-off before any choice").

---

## 1. Pre-registration (2026-10-05, player-layer worker; PM decides adoption)

**Motivation.** Phase 1 (`docs/tests/player_day1_phase1_2026-10-05.md`): an all-anonymous opening-day roster lowers simulated totals by 2.6 / 2.8 points per game (fold 2 / fold 1) and worsens total MAE by 0.45 / 0.76 (CIs exclude 0; reseed floor about 0.06). The cause is the no-history shooter state on every slot. Margin is unaffected. In 2026-27, `build_live` gives every opening-day team-game that state.

**Object.** A day-1 seeding rule, applied through the existing default-off `build_live(..., seed_fn=...)` hook. It acts only on team-games with no in-season rotation prior, i.e. all-anonymous slots. It names players in those slots. The live builder's own as-of joins then supply each named player's fg / FT shooter blocks, usage and rebound rates. Those joins already read the previous season, and each served sub-model keeps its own shrinkage. No sub-model is refitted. No arm has a fitted parameter.

**Candidates** (slots are filled from slot 0 in the stated order; unfilled slots stay anonymous; `rot_srank`, `rot_start`, `rot_fpm`, `rot_pavail` stay as the anonymous fallback):
- **A0** anonymous (current day-1 state).
- **A1n** every player with minutes for THIS team in season S-1, ordered by S-1 minutes at this team. No current roster, so departures are included. `rot_share` = league role profile.
- **A1** A1n restricted to players on this team's season-S roster (CBBD `roster_{S}`), i.e. true returners. `rot_share` = league role profile.
- **A2** A1 plus a roster-turnover-aware team-level shrink of `rot_share`. Each named returner keeps his S-1 fraction of this team's minutes, p_i (summing to r, the returning-minutes fraction). The departed mass 1-r is spread over the anonymous slots in proportion to the league role profile. High turnover moves the team toward the anonymous profile. The rule has no free parameter.
- **A3** A1 plus transfers in: players on this team's season-S roster who logged S-1 minutes for ANOTHER D-I team, ordered by S-1 minutes, after the returners. `rot_share` = league role profile.

Complexity order for ties (simplest first): A0 < A1n < A1 < A3 < A2.

**As-of.** S-1 minutes come from the previous season's on-floor table, complete before season S tips. The season-S roster file is an end-of-season snapshot. Stated caveat: mid-season additions and early departures are mislabelled. A3's requirement of S-1 D-I minutes limits the exposure. In serving, the season-S roster is the preseason CBBD roster pull. Every source table is cut as `build_live` cuts it.

**Setting.** Opening window = first 14 days of the season. Every window game is rebuilt with no in-season rotation prior (`scripts/build_engine_inputs_anon_window_v1.py` recipe), then the arm's seed is applied. Games outside the window are not simulated. The shot-block slot table is held at the anonymous state (shooter 0, known 0) for window games in ALL arms. This is a stated deviation: the fold-1 table builder does not exist, and the term is second-order.

**Folds.** Fold 2 SELECTS: S = 2024-25, S-1 = 2023-24, 599 window games, served stack v2. Fold 1 CONFIRMS: S = 2023-24, S-1 = 2022-23, 643 games, fold-1 artifacts via `fold1_v1/overrides_V2.json` with `ENGINE_ROTATION_SCHEME=static` as in `chain_fold1_v1.py`. 2025-26 is SEALED and not read.

**Runs.** 50 seeds at offset 0, paired streams (`scripts/run_engine_window_v1.py`). Noise floor = the fold-2 winner re-run at seed offset 1000. Verified truth. One grader scores every arm: `scripts/grade_player_day1_v1.py` plus the per-type and player tables in `scripts/grade_player_day1_v2.py`.

**Primary metric.** Fold-2 total MAE over all window games (per-game sim mean total vs verified final).

**Decision rule.**
1. An arm beats A0 only if its paired delta in total MAE is below -floor AND its 95% game-bootstrap interval excludes 0. The floor is |total MAE(winner, o0) - total MAE(winner, o1000)|, with a minimum of 0.06 (the Phase 1 reseed).
2. Among arms that beat A0, the lowest total MAE wins. A simpler arm within the floor of it wins instead (ties go to the simpler arm).
3. Vetoes on fold 2 (paired vs A0):
   - margin MAE worse by more than its own reseed floor;
   - home-win Brier worse by more than 0.003;
   - |total bias| larger than A0's.
4. Fold 1 must not reverse the sign of the winner's total MAE delta vs A0. If it does, report it and make no recommendation.

**Segments reported** (no selection on them):
- true first games (no in-season history in the served inputs) vs later window games;
- site (home / away / neutral);
- per possession type pooled make rates (rim, jump2, three, FT) and volumes vs window truth;
- responsiveness: actual vs predicted total by predicted-total quintile;
- player level, for prop readiness: the share of actual box minutes played by named slots, and minutes / points MAE on named player-games.

Underpowered cells are labelled.

**Output.** The winner is implemented as `make_seed_fn(arm)` in `scripts/build_engine_inputs_day1prior_v1.py`, default off. It runs for 2026-27 with one command and HARD-STOPS if the S-1 on-floor table or the season-S roster is missing. The PM decides adoption.
