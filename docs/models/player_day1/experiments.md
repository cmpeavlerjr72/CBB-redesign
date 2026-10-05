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

---

## 2. Fold-1 source deviation (2026-10-05, before any fold-1 arm was built or run)

`possessions_2023.parquet` has no on-floor columns: lineups start in 2023-24. So the registered S-1 minutes source does not exist for fold 1 (S-1 = 2022-23). Fold 1 instead reads S-1 minutes from the hoopR player box, mapping ESPN athlete ids to CBBD ids through the CBBD rosters' `source_id` (`--minutes-source box`). Fold 2 keeps the registered on-floor source. On fold 2 (S-1 = 2023-24) the box source maps 99.8% of player-game rows. Over 355 teams it shares 97.5% of the top-8 seeds with the on-floor source and the same top player for 83%. No fold-2 arm had been graded when this was written; fold-2 arm runs had started.

---

## 3. Results (2026-10-05). Fold 2 selects, fold 1 confirms. 50 seeds, paired, verified truth

Total-MAE floor = |A1n o0 - A1n o1000| = 0.122. Margin-MAE floor (same pair) = 0.007. Deltas are vs A0, with 95% game-bootstrap intervals.

| arm | F2 total MAE | F2 d total MAE | F2 d margin MAE | F2 total bias | F1 d total MAE | F1 d margin MAE | props: F2 minutes MAE / pts MAE (named rotation player-games) |
|---|---|---|---|---|---|---|---|
| A0 | 15.173 | - | - | -7.52 | - | - | - |
| A1n | **14.620** | -0.553 [-0.79, -0.31] | +0.054 [-0.11, 0.20] | -4.94 | -0.911 | -0.073 | 8.37 / 5.05 |
| A1 | 14.719 | -0.454 [-0.64, -0.27] | +0.015 [-0.13, 0.16] | -5.84 | -0.699 | -0.144 | **6.20** / 4.92 |
| A2 | 14.785 | -0.389 [-0.55, -0.23] | -0.019 [-0.15, 0.11] | -6.03 | -0.643 | -0.139 | 6.53 / **4.72** |
| A3 | 14.622 | -0.551 [-0.77, -0.33] | -0.001 [-0.15, 0.15] | -5.27 | -0.789 | -0.094 | 8.65 / 5.31 |
| (served, reference only) | 14.725 | -0.448 | +0.027 | -4.94 | -0.755 | -0.018 | 5.60 / 4.47 |

Brier is within 0.0035 of A0 for every arm, and none is worse.

**Rule applied as registered.**
- Rule 1: all four arms beat A0.
- Rule 2: A1n has the lowest total MAE, A3 is within the floor of it, and A1n is the simpler of the two.
- Rule 3: A1n is VETOED. Its margin MAE is +0.054 against its own margin floor of 0.007.
- With the vetoed arm removed, rules 1-2 are re-applied. A3 is now lowest, and A1 is within the floor and simpler. A1 is VETOED: margin +0.015 > 0.007.
- Re-applying again leaves A3: margin -0.001, Brier -0.003, |bias| 5.27 < 7.52, so no veto.
- Rule 4: fold 1 has the same sign (-0.789).

**WINNER: A3.** Stated weakness: the margin floor is one reseed draw, and 0.007 is well under the Phase 1 margin reseed of 0.05-0.06. Every margin delta here sits inside its bootstrap interval, and all arms improve margin MAE on fold 1. So the vetoes of A1n and A1 rest on a tight floor. They are not evidence of harm. A3 and A1n tie on total MAE.

**Segments (F2 / F1).**
- First games (181 / 182; no in-season history at all): total bias A0 -9.5 / -10.1, A3 -8.8 / -8.7, A1n -8.2 / -8.3. The player layer recovers 0.7-1.8 points. Most of the opening-day under-prediction is NOT the player layer and needs its own owner.
- Later window games: A3 total MAE 14.48 / 13.83 vs served 14.47 / 13.71.
- Neutral-site cells (46 F2) are UNDERPOWERED.
- Responsiveness slope (actual on predicted total) is 0.59 / 0.76 for A3 vs 0.62 / 0.88 for A0. The window is too short to read it.

**Per possession type (pooled make, sim / truth, F2).** FT: A0 0.646, A1 0.668, A3 0.677, A1n 0.692, truth 0.707 (served 0.670). Three: 0.317 -> 0.326 (truth 0.329). Rim and jump2 move by 0.3-0.5 pp. The lever is the no-history shooter state, FT above all. It sits well below the true make rate of opening-day rosters. A1n scores best partly because it names departed players, which removes more no-history slots. That identity is wrong, and its props are the worst. Under the bottom-up rule, A1n compensates for an upstream bias.

**Props.** A3 names players who play 72% / 76% of actual minutes (A1: 42% / 53%). Its transfers get league role-profile shares in slot order, so minutes MAE on named player-games is 8.65 (A1: 6.20). Props need a share prior for named day-1 players. That is a separate object, not run here.

**Recommendation to the PM.**
1. Ship A3 behind the hook (default off; `make_seed_fn("A3")`) only together with the season-2027 CBBD roster. It hard-stops without that roster.
2. Open two upstream objects: the no-history FT / fg shooter prior, and the opening-day total bias.
3. Do not use A1n.

---

## 4. Pre-registration: fallback roster for teams with no season-S roster (2026-10-05 ~14:50 EDT, committed BEFORE any arm is built or run)

**Problem.** 69 of 365 teams have no 2026-27 ESPN/CBBD roster (`roster_2027.report.json`). A3 needs the season-S roster; such a team-game is 100% anonymous. If coverage is still incomplete near 2026-11-02 a fallback is needed.

**Design.** On each fold's opening window (first 14 days, no in-season rotation prior, anonymous shot-block slot held as in section 1), a seeded random 20% of the teams that play in the window (numpy default_rng(20261005), sorted team ids) are "treated": their season-S roster is withheld, as if missing. All other teams keep A3 with the real roster. Treated teams get one of:
- **R0** nothing: anonymous (what A3 does today for a team with no roster).
- **R1** the S-1 roster (players with S-1 minutes for this team, ordered by S-1 minutes, `rot_share` = league role profile) minus players in their final year of eligibility. Eligibility is NOT recorded in the historical rosters (`experience` exists only in 2027), and `end_season` is hindsight (leak), so the pre-season proxy is `S-1 - start_season >= 3` (4th or later D-I season by the roster file's `start_season`), taken from `roster_{S-1}`; these are dropped as senior/grad. Players with unknown start_season are kept. Stated caveat: redshirt and COVID years make the proxy imperfect.
- **R2** R1 minus observed outgoing transfers: S-1 players of the treated team that appear on a DIFFERENT team's season-S roster among the NON-treated teams (knowable in 2026-27: the 296 rostered teams are visible while the 69 are missing). Caveat: historical roster_S is an end-of-season snapshot (same caveat as A3). Incoming transfers to a treated team are unknown in every arm (slots stay anonymous).
The three arms differ only in the treated teams' slots. Complexity order: R0 < R1 < R2.

**Folds.** Fold 2 selects: S = 2024-25 (S-1 = 2023-24, on-floor minutes), 599 window games. Fold 1 confirms: S = 2023-24 (box minutes, as section 2). 2025-26 is SEALED and not read. Only window games with at least one treated team are simulated (identical across arms elsewhere by construction). 50 seeds, offset 0, paired streams; served stack v2 (fold 1 via the same overrides as section 3). Noise floor: R0 re-run at seed offset 1000 on the same games; floor = |R0 o0 - R0 o1000| total MAE, minimum 0.06.

**Primary metric.** Total MAE over the simulated games (at least one treated team). Reported: total bias, margin MAE, and for the treated teams' player-games: share of actual box minutes played by sim-named players (coverage), minutes MAE on named rotation (>=10 min) player-games, and precision (share of named sim players with >= 10 actual minutes). Segment: games with both teams treated vs one (underpowered cells labelled). Verified truth, one blind grader (`scripts/grade_player_day1_fallback_v1.py`).

**Decision rule.** R1 (or R2) is adopted as the fallback only if on fold 2 its paired total-MAE delta vs R0 is below -floor AND its 95% game-bootstrap interval excludes 0, AND |total bias| does not exceed R0's, AND on fold 1 its delta is not positive (does not lose). Between R1 and R2 both passing, the lower fold-2 total MAE wins unless within the floor, then the simpler (R1). If neither passes, R0 stands (anonymous). Margin MAE is reported and vetoes only if worse than R0 by more than 0.05 (the Phase 1 margin reseed scale; section 3 documented that 0.007 was too tight).

**Output.** The winner ships as a default-off option `fallback=R1|R2` of `make_seed_fn` in `scripts/build_engine_inputs_day1prior_v1.py` (CLI `--fallback`); teams using it are listed in `diag["d1p_fallback_teams"]` and flagged in the build diag. With the option off, behaviour is bit-identical.

---

## 5. Results of section 4 (2026-10-05). Roster-less-team fallback, 50 seeds, paired, verified truth

Treated: 73 teams (F2) / 72 teams (F1), seeded 20% of window teams (seed 20261005). Games simulated: those with at least one treated team (219 F2, 231 F1 sims; 218 / 231 graded; 23 / 28 have both teams treated, rest one). `docs/tests/roster_fallback_2026-10-05.md` has the table. Floor = |R0 o0 - R0 o1000| total MAE, min 0.06.

| arm | fold | total MAE | total bias | d total MAE vs R0 [95% CI] | floor | margin MAE | minutes: all_rot MAE / coverage / named MAE / precision |
|---|---|---|---|---|---|---|---|
| R0 | F2 | 14.555 | -6.44 | - | 0.335 | 10.505 | 22.98 / 0 / - / - |
| R1 | F2 | 14.202 | -5.03 | -0.353 [-0.60, -0.12] | | 10.293 | 15.43 / 0.46 / 7.17 / 0.41 |
| R2 | F2 | 14.257 | -5.50 | -0.298 [-0.50, -0.11] | | 10.231 | 14.82 / 0.46 / 5.80 / 0.61 |
| R0 | F1 | 15.447 | -8.22 | - | 0.119 | 10.094 | 23.08 / 0 / - / - |
| R1 | F1 | 15.039 | -6.42 | -0.408 [-0.68, -0.13] | | 9.776 | 14.64 / 0.50 / 6.70 / 0.46 |
| R2 | F1 | 15.074 | -6.71 | -0.373 [-0.62, -0.13] | | 9.986 | 14.27 / 0.50 / 5.93 / 0.61 |

**Rule as registered.** F2: R1 delta -0.353 is below -floor (0.335), CI excludes 0, |bias| smaller than R0's: passes. R2 delta -0.298 is inside the floor: fails. F1: R1 -0.408 (floor 0.119) does not lose. Margin vetoes: none (margin MAE improves). **WINNER R1.** R2 is not adopted: it does not clear the F2 floor, though it names fewer wrong players (precision 0.61 vs 0.41) and has lower named-minutes MAE.

**Stated weakness.** The F2 floor (0.335) is one reseed draw on 218 games and is large; R1's margin over it is 0.018. Sensitivity (extra runs after the rule was applied, not part of the decision): R0 at offset 2000 = 14.559 (offset 1000 = 14.220); R1 at 1000 = 13.980; paired deltas R1 vs R0 are -0.353 (o0), -0.240 (o1000 vs o1000), -0.357 (R1 o0 vs R0 o2000), all negative. Bias still -5.0 for R1 (opening-day total bias is not this object). Both-treated cells (23 / 28 games) are UNDERPOWERED (deltas -1.2 / -1.5 with wide intervals). Eligibility is a proxy (`S-1 - start_season >= 3`), not recorded class.

**Shipped.** `make_seed_fn(arm, fallback="R1"|"R2")` and `--fallback` in `scripts/build_engine_inputs_day1prior_v1.py`, default off (bit-identical when off); the diag carries `d1p_fallback_teams`. Recommended value if the 2027 roster gap persists: `R1`. The PM decides.
