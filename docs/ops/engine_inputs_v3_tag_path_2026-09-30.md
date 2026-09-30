# From retrained artifacts to engine inputs: tagged v3 inputs, serving overlay, runners, box job list (Lane G, 2026-09-30)

Nothing here runs on AWS. Everything was exercised locally on 2 to 4 cores. No existing file was edited. Base set: `data/processed/models/engine_v3`
(the honest fold-2 set built by replaying the live path, `docs/tests/engine_inputs_v3_replay_2026-09-30.md`, accepted by the PM).

## 0. Sync note (register only; the box operator syncs from the runbook)

- `engine_inputs_v3` is a registered `hf_sync_data.py` bulk key (added by lane J: `data/processed/models/engine_v3`, 14 MB). It is NOT synced yet.
- Tagged sets `engine_v3_<tag>` are BUILT ON THE BOX by the builder below from `engine_v3` plus artifacts, in about a minute each. They need no sync. If
  someone wants them durable, that needs a new bulk key in `hf_sync_data.py` (lane J's file; not edited here). Their sims' results travel in the `results` key.
- Committed files the box needs at the chosen SHA: `scripts/build_engine_inputs_v3_tag_v1.py`, `train_rebound_v3_par_artifacts_v1.py`, `run_po4b_closed_loop_sample_v1.py`,
  `box_run_v2.sh`, `box_v3_sims_v1.sh`, `ops_overlay_check_v1.py`, `run_engine_live.py`, `docs/ops/engine_inputs_v3_tag_path_2026-09-30.md`.
  Not committed, must be pulled from HF: `engine_v3/` (`engine_inputs_v3`), `team_rate_features_E3_v4.parquet`, `team_rate_features_E3opp_v4.parquet`,
  `team_rate_variance_O1a_v3.parquet` (`team_rate_tables`, about 43 MB), `data/processed/truth/stride500_verified_minswap_v1_F2_2025.parquet` (tracked, small).

## 1. The design, in one paragraph

The engine reads three kinds of things: the input arrays (`--input-dir`), model artifacts from FIXED paths, and (for the served event adapter) a per-row round-2 team block from
`engine/event_round2_s1_F2_2025/team_block.npz`, NOT from the inputs. So a retrained arm reaches the engine by (1) a tagged input set whose team-rate cells come from the table
through `team_rate_adapter.apply`, and (2) a serving OVERLAY tree that mirrors the three fixed artifact paths and is bind-mounted over them (`docker -v`), so no ENGINE_* flag and no
served file changes. `scripts/build_engine_inputs_v3_tag_v1.py` writes both into `data/processed/models/engine_v3_<tag>/` (refuses an existing directory and refuses `engine_v3`).
The table IS the as-of estimate on fold 2, so the replay reads it; `src/cbb_sim/live/team_rate_estimator.py` stays for future slates. TRAP, demonstrated: a sim with
`--input-dir engine_v3_<tag>` but WITHOUT the overlay mounts silently uses the SERVED v2 round-2 block on the new team_static. `scripts/ops_overlay_check_v1.py` fails
loudly if the mounted event block or manifests differ from the tag's; `box_v3_sims_v1.sh sim` runs it before every sim.

## 2. Proofs (local)

| # | claim | result |
|---|---|---|
| a | defaults (no table, served artifacts) are bit-identical to `engine_v3` | arrays, games, names, event block: sha256 of all four files equal; arrays equal; overlay event block equals the base block; the 12 served model files in the overlay are identical (sha) to the served files; overlay `index.json` equals the served one |
| b | served artifacts + E3 v4 table: only team-rate-derived cells change, equal to the adapter's values | changed `team_static` columns are exactly the 16 mapped: `off/opp_def` x `{3pa, rim, tov, ftr}` (x100), `off_oreb_c`, `opp_def_dreb_c`, `off_make_c__{rim,jump2,three}`, `def_allow_c__{rim,jump2,three}`; round-2 event block columns 0-7 (the 8 PO style columns) only; slot family `shooter_shrunk_dev_c__{rim,jump2,three}` only; all other team columns, event block columns 8-15, roster, rotation, usage, reb_rate, every other slot column bit-equal. Values equal the table read independently (value x scale, offence team's `_off`, opponent's `_def`, rebound sign -1) within 8.4e-7 (float32 rounding). The shrunk-dev formula `att (smc - c)/(m + att)` (the league rate cancels) reproduces the served v3 slot values to 2.5e-8 on all real slots. |
| c | smoke artifacts: builder runs end to end, engine runs on the result | lane J's PO trainer run tiny (60-game stride sample, 15 trees, 5 refit dates, E3 v4 with `keep_served`) gave a real event artifact dir; `train_rebound_v3_par_artifacts_v1.py` run tiny (1 cut, 15 trees) gave a rebound manifest + segment; the builder took both (plus the served fg dir as overlay) and the engine ran 6 games x 2-3 seeds through `run_engine_live.py --overlay-dir`; the run metadata shows the smoke PO dir (5 artifacts) and the smoke rebound dir (1 artifact) loaded. fg_make retrained smoke artifacts were NOT produced locally (the fg trainer needs the 2.2 M-row design and the extra-cache rebuild); the fg overlay path was exercised with a copy of the served B1 dir |
| d | lane F's draw builder takes the output directory as its base | `build_engine_inputs_trdraw_v1.py --inputs-dir engine_v3_<tag> --variance none --K 1` built (1, 5710, 2, 27) / (1, 5710, 2, 16); on an E3-substituted tag its K = 1 set equals the tag's own arrays and block exactly (max abs 0.0), i.e. the two substitution implementations agree |

## 3. Artifact and reader table (the box job graph)

| job | script | writes (under its `--out-root`/`--out-dir`, `<stem>` = table file stem, absent without a table) | builder reads |
|---|---|---|---|
| PO S1 retrain R2 / T / Topp | `train_possession_outcome_s1_par_v1.py` (lane J) | `<stem>/event_round2_s1_F2_2025/{first_<date>.joblib x6, cont_<date>.joblib x6, index.json, team_block.npz}`, `pred_{first,cont}_F2.npy`, `par_v1_report.json`, `design_overlay.parquet` (69 MB), `games_F2_2025.parquet`, `cuts/*.pkl`, `.owner.json` | `--po-artifacts <out-root>/<stem>` (12 joblibs + `index.json`; its `team_block.npz` is NOT used, the tag's own block replaces it) |
| PO anchored (TO) | `train_possession_outcome_s1_par_anchor_v1.py` (lane C) | predictions and report only; forces `--no-artifacts` | nothing (see 4) |
| fg_make B1 R2 / T | `train_fg_make_v4_par_v1.py` (lane J) | `<stem>/{m_fitted.json, slot_source_v2.parquet, leak_test.json, run_report.json, B1/manifest_FGA_{rim,jump2,three}.json, B1/FGA_*_<date>.joblib x18, cuts/}` and the `--extra-cache` parquet | `--fg-artifacts <stem>/B1` and `<stem>/m_fitted.json` (the retrained m for `shooter_shrunk_dev_c`) |
| rebound T (A0B0C0) | `train_rebound_v3_par_v1.py` (lane J) writes NO artifact: cell JSON and checkpoints without the fitted model. USE `train_rebound_v3_par_artifacts_v1.py` (this lane, same arguments) | J's `<stem>/cells/s2_F2_A0B0C0_seed0.json`, `cuts/` plus `<stem>/artifacts/s2_F2_A0B0C0_seed0/{manifest.json, seg_<refit>.joblib x23}` | `--rb-artifacts <stem>/artifacts/s2_F2_A0B0C0_seed0` |
| rebound anchored / A5 | J or lane C wrappers | offset arms | refused by the artifacts wrapper (no engine offset feed) |
| tagged inputs | `build_engine_inputs_v3_tag_v1.py` | `engine_v3_<tag>/{games,arrays,names}_F2_2025.*, event_block_F2_2025.npz, builder_report.json, docker_mounts.txt, overlay/...` | reads `engine_v3/` (4 files), the tables, the three artifact dirs, `fg_make/round4/m_fitted.json` (served m) |
| draw files S1/S2/S3 | `build_engine_inputs_trdraw_v1.py` (lane F) | `engine_v3_trdraw/{S1_K1,S2_e3_K64,S3_o1a_K64}.{npz,json}` | `--inputs-dir engine_v3_S1`, the E3 v4 table, `team_rate_variance_O1a_v3.parquet` |

## 4. Gaps and their status

- Anchored possession_outcome (and anchored rebound): NOT closable tonight. The gap is not only missing artifact files. The anchored `first` model is a LightGBM trained with an `init_score` offset
  (log of the as-of league class share minus its pooled level), and the engine has no offset feed: an anchored model served without its offset is a different (wrong) model. Writing artifacts
  plus a per-game offset table is a small wrapper, but using them needs a hook in `EventAdapter.predict` (`src/cbb_sim/engine/adapters.py`, another lane's file, not touched). Proposed diff for the PM:
  `EngineInputs.event_first_offset (G, 6)` (per game, as-of, from `cbb_sim.season_anchor`), and in the `first` branch: `raw = model.predict(X, raw_score=True) + offset[gidx]`, then softmax. About 2 to 3 h with its parity proof. `cont` (cascade) takes no offset.
- Rebound retrain had the same artifact gap (not mentioned in the brief; found by reading `_worker`): closed by `train_rebound_v3_par_artifacts_v1.py`.
- fg_make retrain artifacts: complete as written by the trainer (manifests + joblibs + `m_fitted.json`).
- Stage B `TR_MISSING`: the v4 tables cover the engine universe (0 of 5,710 x 2 missing); the training designs may still have keys the table lacks, so keep `raise` and fall back to `keep_served` only on a stop.

## 5. Runner parameters

| runner | input set | game sample | notes |
|---|---|---|---|
| `run_po4b_closed_loop_sample_v1.py` (this lane; wraps `run_po4b_closed_loop.py` unedited) | `--input-dir <engine_v3_tag>` (native) | `--sample-file <parquet with game_id>`: `stride500_verified_minswap_v1_F2_2025.parquet` (499 games common with `po4b_R_s25`) or `stride500_verified_v1_F2_2025.parquet` | every other `run_po4b_closed_loop.py` argument; stamps the sample file and the ENGINE_SHOT_BLOCK / FOUL_JOINT / TEAM_RATE_DRAW env values into `run_meta.json`. Default-off switches are taken from the environment |
| `run_aws_sweep.sh` (lane J) | `--input-dir` (native) | none (full 5,710; grading drops unplayed games under `CBB_TRUTH=verified_v1`) | needs the same overlay mounts added to its `docker run` |
| `run_engine_live.py` (this lane; local, 1 core) | `--input-dir` + `--overlay-dir <tag>/overlay` | `--max-games N` (prefix) | in-process rebinding of the artifact paths |
| `run_shot_block_closed_loop_v1.py`, foul-joint runners (other lanes) | wrap `run_po4b_closed_loop`; use the sample wrapper instead with the env switch exported | | not edited |

## 6. Box job list for the operator

Times are estimates at 192 vCPU (lane J's runbook 17.4 for retrains; mine for builds and sims: about 4 (game, seed) sims per core-second with players kept, so a 500 x 200 run is about 7 min at 64 workers).
`core` = keep if spot is reclaimed or the on-demand cap binds (core set: S0 + floors, S1, and the gate read). Everything else is optional and listed in skip order (last first).

| step | what | minutes | core |
|---|---|---:|:-:|
| S1. sync | `hf_sync_data.py pull --dirs engine_inputs model_artifacts team_rate_tables raw engine_inputs_v3` (per runbook 17.4b); confirm `engine_v3/` and the three tables exist; `git checkout` the SHA containing the files in section 0 | 8 | yes |
| S2. parity | per runbook (parity gate first, flags unset) | 5 | yes |
| S3. retrains | `TABLE_E3=data/processed/team_rate_features_E3_v4.parquet TABLE_E3OPP=..._E3opp_v4.parquet scripts/box_stageb_launch_v1.sh wave1`, BUT for the rebound T job run `scripts/box_run.sh scripts/train_rebound_v3_par_artifacts_v1.py <the same arguments as job rb_T>` instead of J's `rb_T` (J's leaves no artifact). Arms: PO R2 (seed 1, served features, the floor), T (E3), Topp (E3opp, optional); fg R2, T; rebound T. `R2` arms are offline floors, not served | 40 (wave 1) | PO T, fg T, rebound T yes; R2 floors yes for the Stage B read; Topp optional |
| S4. tagged inputs | `scripts/box_v3_sims_v1.sh builds` (S0 served artifacts on v3; S1 = E3 + T artifacts for PO, fg, rebound; S1opp if Topp ran; S1_K1 / S2_e3_K64 / S3_o1a_K64 draw files). A sub-model whose T lost: rerun the S1 build with `--no-table-for po|fg|rb` and without that artifact flag (mixed stack) | 6 | S0, S1 yes; S1opp, draws optional |
| S5. sims, core | `box_v3_sims_v1.sh sim S0`, `sim S0f1`, `sim S0f2`, `sim S0f3`, `sim S0f4` (seed offsets 0, 1000, 2000, 3000, 4000: the four floor draws), `sim S1`; three concurrent at 64 workers | 7 each; 25 wall for all six | yes |
| S6. grade | `box_v3_sims_v1.sh grade` (`CBB_TRUTH=verified_v1`), pair S0 vs S1 with S0f1..S0f4 as floors (`diag_pair_gate_reports.py`), plus `grade_po4b_closed_loop.py` | 10 | yes |
| S7. push | `hf_sync_data.py push --dirs results model_artifacts`; status shows 0 outstanding | 10 | yes |
| S8. sims, optional | `sim S2`, `sim S3` (draws), `sim S1opp`, then the pending fixes on v3 with SERVED artifacts: `sim K2O` (`ENGINE_SHOT_BLOCK=K2_Ocell`), `sim R8b`, `sim R8bS` (`ENGINE_FOUL_JOINT`) | 7 each; 25 wall for three at a time | no (skip order: R8bS, R8b, K2O, S1opp, S3, S2) |
| S9. optional retrains | rebound stage-2 drift cells (wave 2), PO Topp if not started | 40 | no |
| S10. terminate | runbook 17.4d | 5 | yes |

Preflight additions: `scripts/ops_overlay_check_v1.py --input-dir data/processed/models/engine_v3_S1 --root /app` must print PASS inside the container before any S5/S8 sim (the launcher does it);
run `bash -n scripts/box_v3_sims_v1.sh`; Docker was not available on this machine, so `box_run_v2.sh` and the nested bind mounts are UNVERIFIED until the box (a nested `-v` under `-v $PWD:/app` is standard Docker).
Memory: each tagged set is 14 MB of arrays plus 45 to 190 MB of artifacts per sim worker set, negligible next to the draw files (126 MB each for S2 and S3, loaded per worker).

## 7. Open risks

1. Nested bind mounts and `box_run_v2.sh` untested on Docker. Fallback without Docker mounts: copy the overlay tree onto a scratch checkout (`cp -r engine_v3_<tag>/overlay/data .`) before the sim; the check script catches a wrong state either way.
2. fg_make retrained artifacts were never produced locally end to end; their `B1/` directory layout is read from `assemble()` in the trainer, and the overlay builder consumes exactly that (manifests with relative `path` entries).
3. The rebound artifact wrapper was tested on one cut with 15 trees; the 23-cut run on the box is unmeasured (same fits as J's job plus a model pickle per cut, about 8 MB each).
4. The S2/S3 draw engine path reads the event block from the draw file; with the draw ON the overlay's event block is not consulted for the event adapter, but the fixed-path artifacts still are.
