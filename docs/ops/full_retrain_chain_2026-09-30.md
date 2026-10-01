# Full retrain on the clean foundation as one command: `scripts/chain_full_retrain_v1.py` (lane D, overnight 2026-09-30)

Worker lane D, 20:41-21:50 EDT (wall-clock stamps below are `date` / `Get-Date` readings). Core cap 3, held. Local only. The full-size retrain is a box
job (`docs/ops/box_queue/laneD_1.md`). **Nothing is adopted, no consumer is switched, no served file is written.** Every
output lands in `data/processed/models/full_retrain_v1/<tag>/`, which is gitignored (section 6 has the proposed HF key).

Definition used (HANDOFF 2026-09-30, open item 2 and cross-cutting finding 3): every served sub-model that a
foundation change touches is retrained on event layer v4 (`possessions_v4`), the corrected team-foul state, and own
ratings C (`ratings_C_v1`). Engine inputs come from the v3 tag path, and the gate is graded on verified truth
(`CBB_TRUTH=verified_v1` is set explicitly in every subprocess). Team-rate features are a dimension, `--variant F_R`
(the served expanding means) or `F_T` (E3 v4 through `team_rate_adapter`). The season anchor is a switch,
`--anchor O`, default off.

## 1. Inventory of served sub-models

| served sub-model (engine flag) | served artifact | trainer of the served artifact | v4 | foul state | ratings C | E3 | sibling used by the chain (new = this lane) | status in the chain |
|---|---|---|---|---|---|---|---|---|
| possession_outcome `first` + `cont` (`ENGINE_EVENT=round2_s1`) | `engine/event_round2_s1_F2_2025/` (12 joblibs + index) | `build_engine_event_round2.py` on `round2/design.parquet` (built by `train_possession_outcome_v2.py`, `POSSESSIONS_VERSION="v2"`) | yes (chances) | yes (`in_bonus`; open-count includes the possession's own fouls) | yes (4 rating columns + 1 interaction) | yes | **new** `build_po_design_v4_v1.py` (version + ratings as arguments) -> **new** `build_foul_state_v4_v1.py` (lane A replay on the v4 machine + round-8 overlay recipe) -> **new** `train_possession_outcome_s1_par_v2.py` (lane J v1 + table AND feature tables together) | RETRAINED |
| clock (`v5b_glat_pmean` = v3c `srfloor_P3` S1 + v5b sigma) | `clock/v3c_s1/*`, `clock/v5b_bakeoff/v5b_bakeoff_report.json` | `train_clock_v2.py` (design) -> `train_clock_v3c_s1.py` -> `exp_clk5b_mean_consistent.py` | yes | no (the served cell arm has no bonus dimension) | design only (the served cell arm reads tempo, and C's tempo equals R's, so C does not move the served clock) | not in adapter scope | **new** `train_clock_chain_v1.py` (generalises lane B's `exp_clk6_r6_arms_v1.py`: root, possessions version and ratings as arguments) | RETRAINED (equals clock round 6 `L2` up to ratings) |
| rotation R2 + scheduler, S1 (`ENGINE_ROTATION=reference`, scheme `s1`) | `engine/rotation_r2_s1_F2_2025.json` -> `rotation/rotation_fit_v3_S1_<YYYYMM>.json` (6) | `train_rotation_v3b_s1.py` (reads possessions v1 through a default argument bound at definition) | yes | no | no | no | **new** `train_rotation_v3b_s1_poss_v1.py` (possessions version, own out dir, `--only-window` for parallel windows, `--fits-only`) | PARTIAL: windows 2-6 refit on v4; window 1 is the static `rotation_fit_v3.json` by construction (as served), and the static chain plus the engine-input priors stay on v1 (section 5) |
| fg_make B1 (`ENGINE_FG_MAKE=round4_B1`) | `fg_make/round4/B1/` (18) | `train_fg_make_v4_shooter_block.py` | no (raw pbp; v2 possessions only label rim/jumper, and FGA is identical in v4) | partial event-level skew, untouched (section 5) | yes | yes | lane J `train_fg_make_v4_par_v1.py` (unchanged) on a design from **new** `build_design_ratings_swap_v1.py` | RETRAINED |
| rebound S1_weekly (`ENGINE_REBOUND=s1_weekly`) | `rebound/s1_confirm/S1_weekly/F2/` (23) | `train_rebound_v2_s1.py` | no (raw pbp; v1 possessions only label miss type) | partial event-level skew, untouched | yes | yes | lane G `train_rebound_v3_par_artifacts_v1.py` arm `A0B0C0` (unchanged) on a ratings-swapped `round3/design_round3.parquet` | RETRAINED |
| free_throw (`s1_conf_aligned`) | `free_throw/s1_confirm/S1_conf_aligned/F2/` (29) | `train_free_throw_v2_s1.py` | no | serve-side skew only (the engine reads the bonus after the trip's own foul): a choice, proposed in section 5 | no rating feature | not in adapter scope | none needed | SERVED UNCHANGED |
| usage U1 (`ENGINE_USAGE=reference`) | priors baked into the inputs | `train_usage_v1/v2.py` -> `build_engine_inputs.py` | no (v2 possessions for the rim label only) | no | no | no | none | SERVED UNCHANGED |
| late_game | `late_game/round2/*` | `train_late_game_r2_v1.py` | yes | yes | -- | -- | not built | NOT SERVED (`ENGINE_LATE_GAME` defaults to off) |
| attribution | `attribution/round2_s1/` | `train_attribution_v2_s1.py` | -- | -- | -- | -- | -- | NOT READ BY THE ENGINE (props only) |
| season anchor O (`TO` arm) | none served | lane M `train_possession_outcome_s1_par_anchor_artifacts_v1.py`, `train_rebound_v3_par_anchor_artifacts_v1.py`, `build_engine_anchor_offsets_v1.py` | -- | -- | -- | -- | wired behind `--anchor O`, the in_bonus overlay is pre-applied by **new** `build_design_overlay_v1.py` | SWITCH, default off. **Dry-run only, not smoke-tested** |

Inputs: **new** `build_engine_inputs_chain_v1.py` composes three existing pieces. First, lane N's `rating_cells`, with its
parity check (0.0) run before substituting. Second, lane G's `build_engine_inputs_v3_tag_v1.py`, run unedited on the
ratings-substituted base. Third, an `overrides.json`. Serving: **new** `run_engine_overlay_v1.py` rebinds the fixed
artifact paths in the parent and in every worker (ENGINE_DIR, FG_DIR, RB_S1_MANIFEST, clock CK_DIR / V5_PARAMS,
rotation R2_S1_MANIFEST) and then runs the unedited `run_po4b_closed_loop_sample_v1.py` (sample) or `run_engine.py`
(full). This avoids Docker bind mounts. After the run it checks that every overridden family's recorded source points
inside the overrides (the section-1 trap in `engine_inputs_v3_tag_path_2026-09-30.md`), and each worker prints the
paths it serves.

## 2. Identity proofs: each sibling run on the OLD inputs against the SERVED artifact

| sibling | run | result | stated tolerance |
|---|---|---|---|
| `build_foul_state_v4_v1.py --machine v2 --identity` | replay of 2022-2025, 3 workers, 46 s | `foul_accrual_poss_v2.parquet` and `in_bonus_overlay_v2state_v1.parquet` reproduced, `DataFrame.equals` True for both (3,029,695 / 3,038,628 rows); join checks 0 / 0 / 0 | exact |
| `build_po_design_v4_v1.py --poss-version v2 --ratings-dir ratings --identity` | 11.5 s | `round2/design.parquet` reproduced, equals True | exact |
| `train_possession_outcome_s1_par_v2.py` (served design, no table) | FULL SIZE, 12 fits, 3 workers, 1,005 s | offline log loss `first` 1.515428 (the served reference value), `cont` 1.49976; 6 `cont` cascades give identical predictions (max abs diff 0.0); 6 `first` boosters give max abs prediction diff 3e-12 on 4 months and 0.0085 / 0.0020 on 2024-12 / 2025-01 (20,000 sampled 2025 rows) | not bit-exact: the served boosters were fitted multi-threaded (`n_jobs=-1`), the sibling single-threaded. LightGBM's histogram sums depend on thread count, so trees can diverge in floating point. Aggregate log loss is equal to 6 decimals. Lane J proved the same wrapper bit-identical to the serial builder at equal threads |
| `train_fg_make_v4_par_v1.py` (served design and cache) | FULL SIZE, 18 fits, 3 workers, 187 s | 18 of 18 booster texts equal once the thread-parameter line is ignored; max abs prediction diff 0.0 | exact (model) |
| `train_rebound_v3_par_artifacts_v1.py` A0B0C0 vs the served S1_weekly | first 2 weekly cuts, 3 workers | booster texts equal (thread line ignored), max abs prediction diff 0.0, n_train and max_train_date equal | exact on the cuts run. The other 21 cuts are not proven, but they are the same code on the same rows |
| `train_clock_chain_v1.py --poss-version v1 --ratings-dir ratings --identity` | whole chain, 1 core, 75 s | design equals the served `design_v2.parquet`; 6 of 6 S1 pickles byte-equal; B1_sigma 0.04702809866171037 equal | exact |
| `train_rotation_v3b_s1_poss_v1.py --poss-version v1 --max-windows 2 --identity` | window 202412, 1 core, 13 min | 202411 byte-equal. In 202412 every R2 / scheduler / tilt field is equal; only `hazard_exit` / `hazard_enter` differ (coef max 0.24), and the served manifest documents them as R5 override hazards that R2 never reads | exact on every field the engine serves. The R5 hazards differ, and the cause is not traced (the identity run used `--test-games 60 --seeds 1` instead of the trainer defaults) |
| `build_design_ratings_swap_v1.py --identity` (fg and rebound designs, served ratings) | 4.9 s / 2.5 s | equals True for both | exact |
| `build_engine_inputs_chain_v1.py` ratings step | inside every inputs build | lane N recipe vs the engine_v3 base: 0.0 / 0.0 | exact |
| `run_engine_overlay_v1.py` | smoke gate rerun, same seeds | games bit-identical across reruns; worker stamp shows the chain paths; a control run with empty overrides on the served inputs differs (e.g. mean abs home points 9.35 over 30 x 2) | -- |

## 3. Smoke (whole chain on a small slice): PASS for both variants

The smoke sizes are PO every 40th game at 15 trees, rebound 2 weekly cuts at 15 trees, rotation 2 windows, and a gate
on a 30-game verified sample (every 17th game of `stride500_verified_v1`) x 2 seeds. fg_make, the clock, the designs,
the foul state and the inputs run at full size.

| stage | F_R `smoke_FR_v1` (2 cores) | F_T `smoke_FT_v1` (3 cores, `--reuse-from smoke_FR_v1`) |
|---|---|---|
| po_design (v4 + C) | 23.9 s, 3,037,203 rows | reused |
| foul_state (v4 machine) | 122.3 s. Anti-joins 0 / 0; period, offence and start_clock mismatches 0. Per-game total fouls equal to the v2 build in 99.98% of 21,969 games, and the silent / trip split equal in 95.6%. Overlay changes 1.595% of design rows | reused |
| po_train | 28.3 s (smoke) | 23.9 s with E3 table + in_bonus overlay together (the v1 refusal is fixed) |
| clock (v4 + C) | 68.4 s, full | reused |
| fg_design / fg_train | 12.2 s / 465.9 s (full 18 fits at 1 worker) | reused / 218.1 s (E3, shooter cache rebuilt, 3 workers) |
| rb_design / rb_train | 7.2 s / 13.3 s (smoke) | reused / 20.0 s (E3) |
| rotation (v4, background) | 709 s for 2 windows | reused |
| inputs | 4.5 s: ratings C parity 0.0, mean abs change in the four rating channels 0.56-0.62 | 7.4 s: plus the 16 team-rate columns, 3 shrunk-dev slot columns and event block columns 0-7 |
| gate (30 x 2, verified truth) | 38.9 s; source check PASS (event, fg, rebound, clock, rotation all from the chain; free_throw served) | 45.7 s; source check PASS |

The smoke models are tiny on purpose, so its gate numbers are not a read and none are reported.

## 4. Measured timings (local, 1 thread per fit) and the box estimate

| sub-model | measured | core-hours per variant |
|---|---|---|
| PO | `first` 443-514 s per monthly refit (6), `cont` 1.0-1.4 s (6); 12 fits in 1,005 s wall at 3 workers | 0.80 |
| fg_make | 18 fits in 187 s at 3 workers (466 s at 1 worker); F_T adds the shooter-cache rebuild (218 s at 3 workers in total) | 0.15 (F_R) / 0.18 (F_T) |
| rebound | 140 s per weekly fit (first 2 cuts, 3 concurrent) x 23 cuts; later cuts carry more rows, so 150-180 s is an estimate | 0.9-1.15 |
| rotation | 11.5 min per window refit (202412 on v1; 709 s for the v4 smoke including loading); 5 windows | about 1.0 |
| clock, foul state, designs, inputs | 75 + 122 + 45 + 7 s | 0.07 |
| **retrain total** | | **about 3.0 core-hours (F_R); F_T about 2.0 with `--reuse-from`** |
| full gate read 5,710 x 200 | from the box tonight (section 18 of `aws_launch_chain.md`): 11-16 min at 90-96 workers | about 20 per read |

On the box with `--parallel --cores 90`, the branches run at once (PO 12 workers, fg 18, rebound 23, rotation 5
windows, clock 1). The long poles are a single PO `first` fit (about 8 min) and a single rotation window (about 13 min).
Expected wall time is about 15-20 min to inputs plus about 15 min for the gate, per variant.

Per the PM's 20:47 update, the real retrain was NOT run locally. (The measured F_R total of about 3 core-hours, at 3
cores with rotation in the background, would have fit before 02:00.)

## 5. Residuals and proposed bake-offs (not choices made here)

1. **Inputs vs training, PO style columns.** The v3 inputs build the PO team block from possessions **v2**
   (`live/features.py::po_team_block_r2`), while the chain's PO trains on **v4**. Measured on 2025
   (`scripts/diag_full_retrain_v4_style_skew_v1.py`, 10,890 team-games), mean |v4 - v2| is 0.003-0.143 on the x100
   scale, against column SDs of 2.4-7.1. The p99 is at most 0.98, and 96% of rows move. This is a small train/serve
   skew. Proposed fix: a v4 live replay of the inputs (`build_engine_inputs_v3_replay.py` with the possessions version
   as an argument; about 5 min at 64 shards on the box). It is not built tonight.
2. **Rotation static chain.** Window 1 (`rotation_fit_v3.json`) and the engine-input rotation priors (from
   `rotation_fit.json`) stay on v1 possessions. Refitting them means `train_rotation_v1.py` -> `train_rotation_v3.py`
   on v4. Rotation is PARKED.
3. **free_throw bonus-state skew** (round 8 trace: the engine reads `in_bonus` after the trip's own foul). Aligning it
   is a label or serving choice. Proposed as a free_throw round, not done here.
4. **fg_make / rebound event-level `in_bonus`** (partial skew, round 8 trace): untouched.
5. **The corrected state on `cont`.** Stage B `Tfs` was better on `first` (-1.76) and worse on `cont` (+1.21). The chain
   applies the corrected state to both populations, as the definition says. Whether `cont` keeps it is a PM decision.
6. **Anchor O path.** It is wired (`--anchor O`) but was only dry-run. Lane M's PO anchor trainer takes `--design` and
   receives the overlay-applied design. Before it is trusted it needs one smoke run (about 2 minutes).

## 6. Resume commands, sync

- Dry run: `.venv/Scripts/python.exe scripts/chain_full_retrain_v1.py --variant F_R --tag <tag> --dry-run`
- Inputs check: `... --preflight-only` lists all 25 inputs with their git / HF source.
- Run or resume: the same command. Finished stages are skipped (`<stage>/.done.json`). The PO, fg and rebound
  trainers resume from their own per-fit checkpoints. A builder stage that was cut off is cleared and rerun. On a
  failure the chain stops its own background rotation processes.
- Smoke: `--smoke`. F_T after F_R: `--variant F_T --tag <t2> --reuse-from <t1>` reuses rotation, both designs, the foul
  state and the clock.
- Identity proofs: `bash scripts/ops_full_retrain_identity_v1.sh fg rb small`, then
  `diag_full_retrain_identity_v1.py po|fg|rb --new <dir>`.
- Sync (proposal for lane J's file, not edited): add the bulk key `full_retrain` ->
  `data/processed/models/full_retrain_v1`. Today the gitignored artifacts there would ride `model_artifacts`, which
  syncs every gitignored file under `data/processed/models/`. `possessions_v4` has no key, so the chain rebuilds it from
  `raw` (about 1 min per season).

## 7. Session log and incidents

Timestamps EDT: 20:41 start; 20:44-20:45 foul-state identity; 20:46 PO design identity; 20:47-21:04 PO full-size
reproduction (3 workers); 21:04-21:08 fg; 21:08-21:11 rebound; 21:11-21:24 clock, rotation and swap identities;
21:18-21:35 F_R smoke (first attempt failed at foul_state on a stage-dir layout bug, which is fixed); 21:38-21:44 F_T
smoke; 21:45 skew diagnostic.

Incidents: after the first smoke attempt failed, its background rotation process kept running. I stopped that
process of my own (pid 36076 and its child 29040) at 21:22. Since then the chain stops its own background children
on any failure. No other lane's process was touched.

---

## 2026-09-30 22:00: second pass (PM follow-up, 21:48). The cont ruling, v4-replay inputs, the rotation trace, residual pre-registrations

### A. Ruling: the corrected foul state on `cont`. Already the case; nothing to fix

The chain applies the corrected (pre-possession) `in_bonus` to every design row, both `first` and `cont`. The overlay
covers 3,037,203 of 3,037,203 rows of the v4 design (2,649,343 `first` + 387,860 `cont`), and
`train_possession_outcome_s1_par_v2.py` applies it before any population split. The laneD_1 commands were already
correct on this point. laneD_2 (below) exists only for the inputs.

### B. The v4 replay of the engine inputs (residual 1). Built, proven and wired

- **Sibling builder:** `scripts/build_engine_inputs_v3_replay_evlayer_v1.py` (shard / assemble, `--event-layer v2|v4`).
  - The only part of the v3 inputs that reads the possession layer AND is served to the retrained PO is the round-2
    event block (16 columns), built by `live/features.py::po_team_block_r2`, which hard-codes v2. The sibling replays
    exactly that block per slate: same slate, same as-of cutoff (first tip minus 30 min), and the same
    `build_ctx` / `po_team_block_r2` / `rating_site_block` / merge / no-history fill as `build_engine_inputs_live.py`.
  - Everything else is copied from `engine_v3` unchanged.
- **Proof:** `--event-layer v2` reproduces engine_v3's event block bit for bit on all 151 dates (11,420 team-games):
  PASS.
- **Timing:** about 1.0-1.7 s per date single-core; the full 151 dates take about 195 s. Done locally (v2 proof and v4
  in parallel, 2 cores, 21:49-21:52).
- **v3 -> v4 census, by block:**

| block | cells differing | detail |
|---|---|---|
| event block cols 0-7 (PO style: off / opp_def x 3pa, rim, tov, ftr) | 10,882-10,975 of 11,420 per column (95-96%) | mean abs 0.003 (rim) to 0.142 (ftr) on the x100 scale; max 1.34 (rim) to 3.86 (ftr) |
| event block cols 8-15 (ratings, site, season, days) | 0 | |
| team_static (incl. its own PO columns, which read v1 all-chance tables through `po_team_block`), slot_static, rotation priors, roster, rebound, usage, games, names | 0 by construction (copied) | stated scope: those paths are pinned to their own possession versions and are not served to the retrained PO |

- **Wired:** new chain stage `inputs_base` (variant-independent, reusable with `--reuse-from`) shards the replay and
  assembles `engine_v3_ev4`. The `inputs` stage builds on it: ratings C, then E3 for F_T, then the overlay. A new
  `--inputs-event-layer v4|v2` (default v4) keeps the old base available.
- **Gate result names:** reads on v4 inputs carry the suffix `_ev4`.
- **`--redo <stages>`:** rebuilds the named stages of a finished tag and archives the old output as
  `<stage>.prev_<timestamp>`.
- **Smoke `smoke_FR_ev4`:** PASS. The inputs event block columns 0-7 equal the v4 replay; the overlay block equals the
  inputs block; the rating columns carry ratings C; source check PASS. The `--redo inputs,gate` path was also tested.
- **Box request `docs/ops/box_queue/laneD_2.md`:**
  - Case A (laneD_1 not started): run laneD_2 instead.
  - Case B (laneD_1 ran): rerun only `inputs_base`, `inputs` and the gate with `--redo inputs,gate`. Every retrain stage
    output of laneD_1 is reused unchanged.
- **Commit:** 7d38de5.

### C. The rotation override-hazard non-identity: traced

| comparison (window 202412, possessions v1) | fields differing |
|---|---|
| today, `--test-games 60 --seeds 1` vs the same rerun | none (the NaN-vs-NaN note only) |
| today, trainer defaults (1600 / 3) vs today 60 / 1 | none |
| today (either arguments, BLAS 1 thread) vs the served 09-11 file | only `hazard_exit` / `hazard_enter` (coef max 0.24 / 0.19) |

- **Ruled out:**
  - Non-determinism: same arguments give an identical result.
  - The test arguments: defaults and 60 / 1 give identical results.
  - Code drift: no commit to `rotation.py` or the rotation trainers after 1e6bb3e, which is the commit that wrote the
    served file.
  - Data or library drift: the possession tables, the raw pbp and the sklearn / numpy / scipy installs all predate the
    served run (09-10 08:48-12:11 EDT vs the run at 20:49 EDT).
- **Leading cause:** the BLAS thread count inside the R5 hazards' `LogisticRegression(lbfgs)`. The served run was
  unpinned; the chain pins BLAS to 1 thread.
- **Test run:** `--blas-threads 3`; result in E below (TRACED).
- **Consumers of these fields:** `R3_stint_hazard` (`models/rotation.py` line 966) only. The served R2 path and the
  engine (`grep` of `src/cbb_sim/engine`) never read them.

### D. Pre-registration text proposals (for the PM; not appended to any experiments.md)

**D1. Rotation static chain on the v4 event layer (residual 2).**
- **Candidates:**
  - `RS0`: the served static chain on v1, `rotation_fit.json` -> `rotation_fit_v3.json` (reference).
  - `RS4`: the same trainers (`train_rotation_v1.py` -> corrected hazards -> `train_rotation_v3.py`) with
    `rotation.load_team_possessions` pointed at `possessions_v4` (a sibling wrapper like `train_rotation_v3b_s1_poss_v1`).
  - `RS4p`: RS4 plus the engine-input rotation priors (`rot_share`, `rot_srank`, `rot_start`, `rot_fpm`, `rot_pavail`)
    rebuilt from the RS4 static fit by a v4 replay of `LF.rotation_priors`. That needs a possessions-version parameter
    in the replay, the same pattern as section B.
- **Features:** unchanged (a data-layer swap, not a model choice).
- **Folds:** fold 2 select, fold 1 confirm.
- **Primary:** the rotation round-10 MAE line (minutes per player-game), with Decision-8 slope and G8 cells as vetoes.
- **Noise floor:** a seed-1 refit of RS0.
- **Closed loop:** 500 x 25 paired against the laneD_2 F_R stack, Decision-12 floors.
- **Decision rule:** RS4 / RS4p ship into the retrain chain only if the primary is not worse beyond the floor. Any
  difference within the floor ships on the honesty rule (the serve-time table must match the training table).
- **Expected size:** small (v4 changes duration on 4,708 of 768,834 rows in 2025).

**D2. free_throw bonus-state alignment (residual 3).**
- **Defect:** at serve time the engine reads `st.in_bonus()` AFTER the shooting trip's own foul is added (`loop.py`
  ~503 / 703), while training (`foul_class` in {bonus_one_and_one, double_bonus}) is the trip's own class. A seventh-foul
  shooting trip reads 1 at serve and 0 in training (round-8 trace table).
- **Candidates:**
  - `FTB0`: served.
  - `FTB1`: serving fix, `in_bonus` read BEFORE the trip foul. A default-off `ENGINE_FT_BONUS_STATE=pre` switch, with
    parity proved on the off path.
  - `FTB2`: training relabel to the engine's definition (the bonus state from the team-foul counter before the trip,
    i.e. the corrected foul-state table keyed to the trip), refit on the S1_conf_aligned schedule.
- **Primary:** FT% calibration by `in_bonus` cell (bonus vs non-bonus trips), offline fold 2, plus G4 FTA/FGA and FT%
  closed loop.
- **Vetoes:** total FT%, the shooter slope (Decision 8), G5.
- **Noise floor:** a seed refit of FTB0.
- **Rule:** the arm whose serve-time definition equals its training definition is preferred when the primary ties
  within the floor; that is the leak rule, not a performance preference.
### E. Rotation trace result (22:12-22:21, `--blas-threads 3`)

| comparison (window 202412) | differing fields |
|---|---|
| BLAS 3 vs BLAS 1 (otherwise identical runs) | `hazard_enter` coef up to 0.69, `hazard_exit` up to 0.10; `notes/r7_*` / `r8_*` knob grids up to 3e-4; `notes/r5_*` up to 2e-11 |
| BLAS 3 vs served | the same fields (hazard coef up to 0.50) |
| any thread count vs served or vs each other: every R2 field (Dirichlet, scheduler, tilt tables) | 0 |

**TRACED.** The R5 override hazards (and, at the 1e-4 level, the R7 / R8 knob grids) depend on the BLAS thread count
through the lbfgs `LogisticRegression` fit. That fit is ill-conditioned (collinear deficit / surplus / time features),
so reduction-order differences move its coefficients by up to 0.7. The served 09-11 run was not thread-pinned, and its
thread count is not recorded, so its hazard values cannot be reproduced bit for bit. The chain pins 1 thread and is
deterministic run to run (C). Every field the served R2 path reads is identical at every thread count tested. The
thread-sensitive fields belong to arms R3 / R5 / R7 / R8, which are not served.

Recommendation (not done here): record the BLAS thread count in every rotation fit's notes, and standardise the
features before the lbfgs fit if any hazard arm is ever served.
---

## 2026-10-01 00:25-00:35: section 3, the E3 fg_make train/serve skew (lane A finding)

**(1) Which F_T stages go through the skewed path.**
- **fg_make: YES.** `train_fg_make_v4_par_v1.py --team-rate-table` calls `team_rate_adapter.apply`, which swaps
  `off_make_c` / `def_allow_c` but leaves `off_make_raw` / `def_allow_raw` at the served values. `fit_m` and
  `shooter_shrunk_dev_c` are then trained on the stale raw rate, while the inputs builder serves the E3-derived one.
- **Rebound: NO skew in the served arm.** The adapter does leave the round-3 rate-derived extras stale (`rawc_off`,
  `off_oreb_g*`, `off_priorc`, `*_oa*_c`, ...), but A0B0C0 reads only `off_oreb_c` and `opp_def_dreb_c` plus non-rate
  columns (feature list checked through `train_rebound_v3_round3.features_for`).
- **PO: no skew.** The adapter recomputes the interactions; lane A found `T` parity exact.

**(2) The fix.**
- **New `src/cbb_sim/team_rate_adapter_v2.py`:** v1 plus re-derivation of every raw column that depends on a swapped
  rate (fg_make raw = c + `lg_make_asof`). The rebound audit list is `RB_STALE_DERIVED`, and
  `assert_rebound_features` refuses an arm that reads one.
- **New `scripts/train_fg_make_v4_par_v2.py`:** v1 with the v2 adapter installed. The chain's fg stage now uses it.
- **New chain stage `parity`, between inputs and gate** (`scripts/diag_train_serve_parity_fg_v1.py`). It joins the
  trainer's `slot_source_v2.parquet` (the `shooter_shrunk_dev_c` it trained on) with the inputs' `slot_static` (what is
  served) on (player, date), for the 2025 test season. It FAILS the run if any class has corr < 0.999 or p99 |diff| > 1e-3,
  and it also runs the rebound feature audit.
- **F_R default path:** without a table the adapter is never called, so v2 is v1 line for line.

| run | rim corr / p99 | jump2 corr / p99 | three corr / p99 | verdict |
|---|---|---|---|---|
| F_R (`smoke_FR_ev4`, served path) | 0.999828 / 0 | 0.999815 / 0 | 0.999975 / 0 | PASS |
| F_T skewed (`smoke_FT_v1`, adapter v1) | 0.969898 / 0.0225 | 0.946771 / 0.0121 | 0.942733 / 0.0100 | FAIL (reproduces lane A's 0.94-0.97) |
| F_T fixed (`smoke_FT_v2`, adapter v2) | 0.999917 / 2.6e-8 | 0.999949 / 5.4e-9 | 0.999995 / 6.5e-9 | PASS |

The F_R residual (about 2e-4 off perfect correlation, p99 0) comes from a handful of player-dates where the v3 live
replay and the backtest design differ (the v3 replay's freshness fixes). It is the same in every variant, so it is
not a skew of the swap.

- **Smokes:**
  - `smoke_FT_v2`: whole F_T chain, 00:25-00:28, PASS.
  - `smoke_FT_v1c`: a copy of the skewed F_T tag, then `--redo fg_train,inputs,gate`, 00:28-00:31, PASS. This is
    exactly the box flow.
- **Commit:** e998b22.

**(3) Box request `docs/ops/box_queue/laneD_3.md`** (filed 00:33):
- Copy FT_box_v1 to FT_box_v2, then `--redo fg_train,inputs,gate` with `--reuse-from FR_box_v1`.
- PO and rebound F_T artifacts are reused unchanged.
- One full gate read, paired against S0 (four floor draws) and against FR_box_v1.
- About 10 min of box time.