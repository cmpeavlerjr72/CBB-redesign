# Sim Sub-Models — Index

Every sim sub-model has a folder here. See `DOCUMENTATION_STANDARD.md` for the doc layout every folder must follow.

Training order and status (per `docs/FRAMEWORK_PLAN.md` §2.1 cascade order):

| # | Model | Folder | Status |
|---|---|---|---|
| L0 | Control engine (yardstick, not a candidate) | [`control_engine/`](control_engine/model.md) | BUILT AND GATED 2026-09-10 |
| L2 | Pace (possessions per game) | [`pace/`](pace/model.md) | BAKE-OFF COMPLETE 2026-09-10 -- winner `multiplicative`/`A_tempo`, target T_pbp |
| L3 | Possession event layer (pbp -> possessions and chances) | [`docs/tests/possessions_build_v2_2026-09-10.md`](../tests/possessions_build_v2_2026-09-10.md) | v2 SHIPPED 2026-09-10 -- rim-location override, per-chance attempt counts, `pbp_complete`. v1 (`data/processed/possessions/`) is frozen; `DEFAULT_POSSESSION_VERSION` is still `v1` until the PM switches it |
| L3 | Possession outcome (terminal-event mix of a chance) | [`possession_outcome/`](possession_outcome/model.md) | ROUND 2 DECIDED 2026-09-10 -- `lgbm`/`C_plus_state`/`S1` on `first`, `cascade`/`C_plus_state`/`S1` on `cont`. Round 1 adopted nothing |
| L3 | Rebound (who gets the ball after a miss) | [`rebound/`](rebound/model.md) | BAKE-OFF RUN 2026-09-10 -- see model.md for the verdict |
| L3 | Free throw (trip structure + make probability) | [`free_throw/`](free_throw/model.md) | BAKE-OFF RUN 2026-09-10 -- see model.md for the verdict |
| L3 | Field-goal make (made vs missed, three SEPARATE models by shot class) | [`fg_make/`](fg_make/model.md) | BAKE-OFF COMPLETE 2026-09-10 -- winner `lgbm`/`C_plus_state` in ALL THREE classes (F2 log loss 0.641605 / 0.642003 / 0.561085; rim and jumper clear the best passing non-tree arm by 33.7 and 32.4 noise floors). `FGA_3` was re-decided from the team-level baseline to `lgbm` under `ARCHITECTURE_DECISIONS.md` Decision 8 (decision step only, nothing retrained; experiments.md section 12). Defence is TEAM-LEVEL, not lineup-level. One clause of Decision 8 still needs its scope written down -- model.md section 4.3 |
| L4 | Rotation (who is on the floor, and for how long) | [`rotation/`](rotation/model.md) | BAKE-OFF RUN 2026-09-10, two rounds -- NO ARM ADOPTED (every arm misses a state-dependence cell); one hazard-matrix defect open, see model.md and `experiments.md` section 5 |
| L4 | Shot allocation / usage (which of the five on the floor is credited with the event) | [`usage/`](usage/model.md) | BAKE-OFF RUN 2026-09-10 -- LightGBM over the five wins all five event classes on F1; 2 replicate, 2 straddle the floor, `TOV` reverses and is UNCONFIRMED. CFB's "too narrow / too short" pair does NOT reproduce; the binding gate is top-k usage share, failing in the opposite direction. See model.md sections 4-5 |
| L4 | Player attribution (which player is credited with the secondary stat after a team-level event) | [`attribution/`](attribution/model.md) | BUILT; BAKE-OFF PARTIALLY RUN 2026-09-10 -- module, trainer, tests (27 passing) and docs complete; the F1 run was cut by the session hard stop and NO WINNER is adopted for any of the eight targets. Resume with `docs/models/attribution/RESUME.md` |
| L5 | Clock consumption (seconds per possession; pace is emergent from it, Decision 7) | [`clock/`](clock/model.md) | TWO BAKE-OFF ROUNDS RUN 2026-09-10 -- NO ARM ADOPTED (round 1: 0 of 20; round 2, finer period-end state: 0 of 6). Diagnosis model.md sections 4-5 and 10 |

Shared prerequisites built alongside a model rather than under it:

| Artifact | Built by | What it is |
|---|---|---|
| `data/processed/player_crosswalk.parquet` | `scripts/build_player_crosswalk.py` (`cbb_sim.data.player_ids`) | CBBD <-> ESPN **player** id crosswalk, the prerequisite `docs/SIM_GUARDRAILS.md` section 4 names for every lineup feature. Match rate is a reported number: see `rotation/model.md` section 4. |
| `src/cbb_sim/models/event_stream.py` | imported by `rebound.py` and `free_throw.py` | The cleaned CBBD event stream those two models train on: block rows folded into a `blocked` flag, ESPN's administrative reset rebounds dropped, free-throw trips resolved, running team-foul counts attached. Neither model can be built from the possession tables alone -- a dead-ball rebound is a no-op there and a free-throw attempt has no row at all. |
| `src/cbb_sim/models/prob_metrics.py` | imported by `rebound.py` and `free_throw.py` | The L3 round-1 scoring battery (log loss, per-class Brier, decile calibration with its level/shape split, quintile responsiveness, game-level block bootstrap) generalised from a fixed six-class vocabulary to any class list, so a three-class and a two-class target are graded by literally the same code as the six-class one. |
| `data/processed/models/usage/{events,asof}_{version}.parquet` | `scripts/train_usage_v1.py` (`cbb_sim.models.usage`) | One row per credited event with its OFFENSIVE on-floor five and its state, and one row per (season, player, game) with every pregame as-of input. Built from `cbb_sim.models.event_stream` rather than the possession tables, because the chance table has no `participant` id and the possession table records the five only once per possession -- a substitution inside a possession would be attributed to the wrong five. Coverage is a reported number: 95.0-98.4% of credited events are modelled. |
| `data/processed/models/attribution/{events_*,asof,team_asof}_{version}.parquet` | `scripts/train_attribution_v1.py` (`cbb_sim.models.attribution`) | The five attribution population tables (live offensive rebounds, live defensive rebounds, made FGA, turnovers, missed FGA) each with the CANDIDATE side's on-floor five, plus one row per (season, player, game) and one per (season, team, game) of pregame as-of inputs. Built from `attribution.build_attr_stream`, which is `event_stream` re-done with the assist columns and the blocker's / stealer's ids attached -- `event_stream` drops the block rows and carries no assist columns, so three of the four credits are not in it. |
| `data/processed/models/free_throw/bonus_era.json` | `scripts/train_free_throw_v1.py` (`cbb_sim.models.free_throw.derive_bonus_thresholds`) | The NCAA bonus thresholds RE-DERIVED from each season's own data. **The engine reads this into GameState**, per `CLAUDE.md` "rule-era flags live in GameState, not baked into sub-models". |

---

## How to start a new model

1. Create the folder under `docs/models/`.
2. Copy the three empty templates from `DOCUMENTATION_STANDARD.md`.
3. Build the training script under `scripts/train_{model}.py`.
4. Save artifacts under `data/processed/models/{model}/`.
5. Fill in all three doc files. Don't mark the model "done" until `experiments.md` shows the grid that justifies the decision.

---

## Standing result: the default TRAINING SCHEME is S1 (in-season walk-forward)

Decided by the L3 possession-outcome round-2 bake-off, 2026-09-10, on both chance populations, under a pre-registration that said the winning scheme becomes the default for every later sub-model unless that sub-model's own bake-off says otherwise (`possession_outcome/experiments.md` section 4).

**S1** means: refit at each month boundary of the test season on all prior seasons plus the test season to date, strictly before the refit date, and score each game with the most recent refit at or before its own date. Reference implementation and its leak test: `cbb_sim.models.possession_outcome.fit_predict_scheme` and `tests/test_possession_outcome.py::test_s1_never_lets_a_game_into_its_own_fit`.

Two consequences a later model must not rediscover the hard way. First, the win is a CALIBRATION win, not a log-loss win -- on L3 it moved the tree arm's worst decile gap from 2.78 pp to 0.98 pp for a log-loss gain of +0.00136, so a model graded on log loss alone would have called S1 noise. Second, the deployed artifact is a SCHEDULE: it has to be refit monthly in the live season, and a stale artifact silently degrades toward the static fit that failed the gate.
