# Sim Sub-Models — Index

Every sim sub-model has a folder here. See `DOCUMENTATION_STANDARD.md` for the doc layout every folder must follow.

Training order and status (per `docs/FRAMEWORK_PLAN.md` §2.1 cascade order):

| # | Model | Folder | Status |
|---|---|---|---|
| L0 | Control engine (yardstick, not a candidate) | [`control_engine/`](control_engine/model.md) | BUILT AND GATED 2026-09-10 |
| L2 | Pace (possessions per game) | [`pace/`](pace/model.md) | BAKE-OFF COMPLETE 2026-09-10 -- winner `multiplicative`/`A_tempo`, target T_pbp |
| L3 | Possession outcome (terminal-event mix of a chance) | [`possession_outcome/`](possession_outcome/model.md) | BAKE-OFF RUN 2026-09-10 -- see model.md for the verdict |
| L3 | Rebound (who gets the ball after a miss) | [`rebound/`](rebound/model.md) | BAKE-OFF RUN 2026-09-10 -- see model.md for the verdict |
| L3 | Free throw (trip structure + make probability) | [`free_throw/`](free_throw/model.md) | BAKE-OFF RUN 2026-09-10 -- see model.md for the verdict |
| L3 | Field-goal make (made vs missed, three SEPARATE models by shot class) | [`fg_make/`](fg_make/model.md) | BAKE-OFF COMPLETE 2026-09-10 -- winner `lgbm`/`C_plus_state` in ALL THREE classes (F2 log loss 0.641605 / 0.642003 / 0.561085; rim and jumper clear the best passing non-tree arm by 33.7 and 32.4 noise floors). `FGA_3` was re-decided from the team-level baseline to `lgbm` under `ARCHITECTURE_DECISIONS.md` Decision 8 (decision step only, nothing retrained; experiments.md section 12). Defence is TEAM-LEVEL, not lineup-level. One clause of Decision 8 still needs its scope written down -- model.md section 4.3 |
| L4 | Rotation (who is on the floor, and for how long) | [`rotation/`](rotation/model.md) | BAKE-OFF RUN 2026-09-10 -- see model.md for the verdict |
| L4 | Shot allocation / usage (which of the five on the floor is credited with the event) | [`usage/`](usage/model.md) | BAKE-OFF RUN 2026-09-10 -- LightGBM over the five wins all five event classes on F1; 2 replicate, 2 straddle the floor, `TOV` reverses and is UNCONFIRMED. CFB's "too narrow / too short" pair does NOT reproduce; the binding gate is top-k usage share, failing in the opposite direction. See model.md sections 4-5 |
| L5 | Clock consumption (seconds per possession; pace is emergent from it, Decision 7) | [`clock/`](clock/model.md) | BAKE-OFF RUN 2026-09-10 -- NO ARM ADOPTED (0 of 20 pass the emergent G1 and PIT gates); diagnosis in model.md sections 4-5 |

Shared prerequisites built alongside a model rather than under it:

| Artifact | Built by | What it is |
|---|---|---|
| `data/processed/player_crosswalk.parquet` | `scripts/build_player_crosswalk.py` (`cbb_sim.data.player_ids`) | CBBD <-> ESPN **player** id crosswalk, the prerequisite `docs/SIM_GUARDRAILS.md` section 4 names for every lineup feature. Match rate is a reported number: see `rotation/model.md` section 4. |
| `src/cbb_sim/models/event_stream.py` | imported by `rebound.py` and `free_throw.py` | The cleaned CBBD event stream those two models train on: block rows folded into a `blocked` flag, ESPN's administrative reset rebounds dropped, free-throw trips resolved, running team-foul counts attached. Neither model can be built from the possession tables alone -- a dead-ball rebound is a no-op there and a free-throw attempt has no row at all. |
| `src/cbb_sim/models/prob_metrics.py` | imported by `rebound.py` and `free_throw.py` | The L3 round-1 scoring battery (log loss, per-class Brier, decile calibration with its level/shape split, quintile responsiveness, game-level block bootstrap) generalised from a fixed six-class vocabulary to any class list, so a three-class and a two-class target are graded by literally the same code as the six-class one. |
| `data/processed/models/usage/{events,asof}_{version}.parquet` | `scripts/train_usage_v1.py` (`cbb_sim.models.usage`) | One row per credited event with its OFFENSIVE on-floor five and its state, and one row per (season, player, game) with every pregame as-of input. Built from `cbb_sim.models.event_stream` rather than the possession tables, because the chance table has no `participant` id and the possession table records the five only once per possession -- a substitution inside a possession would be attributed to the wrong five. Coverage is a reported number: 95.0-98.4% of credited events are modelled. |
| `data/processed/models/free_throw/bonus_era.json` | `scripts/train_free_throw_v1.py` (`cbb_sim.models.free_throw.derive_bonus_thresholds`) | The NCAA bonus thresholds RE-DERIVED from each season's own data. **The engine reads this into GameState**, per `CLAUDE.md` "rule-era flags live in GameState, not baked into sub-models". |

---

## How to start a new model

1. Create the folder under `docs/models/`.
2. Copy the three empty templates from `DOCUMENTATION_STANDARD.md`.
3. Build the training script under `scripts/train_{model}.py`.
4. Save artifacts under `data/processed/models/{model}/`.
5. Fill in all three doc files. Don't mark the model "done" until `experiments.md` shows the grid that justifies the decision.
