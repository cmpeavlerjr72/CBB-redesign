# Data triage and box runbook, 2026-10-14 (Sonnet ops worker)

Brief: PROJECT_STATUS "SESSION 2026-10-14" Ops queue. Rule applied: CLAUDE.md data rules (model-artifact directories over 20 MB are gitignored and synced to `mvpeav/cbb-sim-data`; small, non-derivable, chain-consumed files are committed; derivable scratch is gitignored).

## Key naming decision
No new bulk key was created. The existing convention (`.gitignore` comments for r6, r8_K2, r9_env, fold1_v1, lane A dirs) is the generic `model_artifacts` key, pushed for named directories by `scripts/ops_hf_push_paths_v1.py` (it uploads only the named dirs, not other lanes' in-progress files that `hf_sync_data.py push --dirs model_artifacts` would sweep up). A `clock_rounds` key would have duplicated every file, since `model_artifacts` already mirrors all gitignored files under `data/processed/models/`. Pull path: `hf_sync_data.py pull --dirs model_artifacts --only 'clock/r8_*/**'`.

## (a) Clock r7 / r8 dumps
Ignored by `data/processed/models/clock/r7_*/` and `clock/r8_*/`. Files already tracked in those dirs (r7_A2/F2, r8_K1/F2, r8_K2M/F2, r8_M2D/F2, bake-off reports; listed by `git ls-files -ci`) stay tracked, because ignore rules do not apply to tracked paths.

| dir | untracked files | bytes (approx) | HF prefix (under model_artifacts/clock/) |
|---|---|---|---|
| r7_A1 | 14 | 6.0 MB | r7_A1 |
| r7_A2/F1 | 7 | 3.0 MB | r7_A2 |
| r7_A3 | 16 | 7.9 MB | r7_A3 |
| r7_C0 | 14 | 18.8 MB | r7_C0 |
| r8_G2D | 14 | 6.0 MB | r8_G2D |
| r8_K1/F1 | 7 | 9.4 MB | r8_K1 |
| r8_K2M/F1 | 7 | 16.7 MB | r8_K2M |
| r8_M1, r8_M2 | 14 each | 6.0 MB each | r8_M1, r8_M2 |
| r8_M2D/F1, F2_seed1 | 14 | 6.0 MB | r8_M2D |
| r8_K2 (already ignored; F1_seed1 and F2_seed1 were missing on remote) | 14 | 101 MB | r8_K2 |

Pushed: 149 files that were missing on remote, about 178 MB (whole dirs were uploaded; files already present were skipped by hash). A missing-set check over `model_artifacts` and `team_rate_tables` afterwards returned no clock, engine or free_throw entries.

## (b) Per-directory triage

Writer and reader columns come from `grep` over `scripts/`, `src/`, `tests/`. The served daily chain (`chain_daily_v3.py`, `run_daily_sim_v1.py`, `run_daily_publish_v1.py`) reads only the tracked `engine_live/*_D1P_A3_*` files and `build_diag_LIVE_F2_*_{slate_date}.json` for the slate date; none of the ignored directories below is read by it. Tests read `engine_v3/arrays_F2_2025.npz` and `team_rate_features_E3_v{1,2,3}.parquet` only.

| path | size / files | writer | read by | derivable | decision |
|---|---|---|---|---|---|
| `models/engine_live/*_LIVE_F2_2025_*` (151 dates) | 17 MB / 605 | `build_engine_inputs_v3_replay.py` (fold-2 replay staging) | replay chain only; the served chain glob matches the slate date; the 5 tracked `D1P_A3_F2_2027_2026-11-02` files stay | yes, rerun the replay | gitignored; mirrored under `model_artifacts` (that key syncs every ignored file under `models/`, so "no sync" is not possible there; 17 MB, harmless) |
| `models/engine_v3/` | 13 MB / 4 | `build_engine_inputs_v3_replay.py` | tests `test_shot_block_v3in`, `test_trajectory`; box sims; HF key `engine_inputs_v3` | yes but slow; test input | COMMITTED (under 20 MB) |
| `models/engine_v3_*` 53 sibling dirs (arm / lane inputs) | 4.3 GB / 2,246 files; 13 to 274 MB each; 36 of 53 over 20 MB | `build_engine_inputs_{v3_combine,v3_A1,d1p_live_early,day1prior,anon_window,seeded,trdraw}_v1.py`, `build_ft_prior_serving_v1.py` | box sim scripts and `diag_*`/`exp_aggregation_*` pass them by tag; not the served chain | yes, per-arm builders | gitignored (`engine_v3_*/`, the glob does not match `engine_v3/`); pushed to `model_artifacts/engine_v3_*` |
| `models/engine_seed_stage/` | 16 MB / 453 | `build_engine_inputs_seeded_v1.py` (stage dir) | same script | yes | gitignored; mirrored |
| `models/free_throw/laneI_N1/` | 80 MB / 31 | lane I taps (`chain_laneI_taps_v1.sh`, `box_op1001d_*`) | box jobs | yes | gitignored; mirrored |
| `team_rate_features_E3{,opp}_v{1..4}.parquet` | 25 to 29 MB each, 8 files | `exp_team_rate_estimator_v*.py` | `cbb_sim.team_rate_adapter`, `test_team_rate_adapter` (v1-v3), `box_*` jobs | yes | gitignored (`data/processed/team_rate_features_*.parquet`); already on HF under `team_rate_tables` (8 of 8 present) |
| `team_rate_variance_O1a_v{1,2,3}.parquet` | 13.5 MB each | `exp_team_rate_estimator_v4.py` | `build_engine_inputs_trdraw_v1.py`, `diag_team_rate_draw_proofs_v1.py`, box jobs | yes | gitignored (`team_rate_variance_*.parquet`); `hf_sync_data.py` `team_rate_tables` glob extended (`TEAM_RATE_GLOBS`) and the 3 files pushed. Not committed: 40 MB of git history for rebuildable tables |
| `data/processed/injuries/injuries_2026-10-14.parquet` | 5.7 KB | daily injury feed | served chain | no (point-in-time snapshot) | COMMITTED |
| clock r7/r8 | see (a) | `exp_clk*`, `train_clock_r8_*` | bake-off reads | yes | gitignored; pushed |

Nothing was deleted. After the edits `git status --short data/processed` shows only the two committed items.

HF mirror: every family above was `missing_on_remote` before the pushes and absent from the missing set after. Uploaded: 4,322 MB engine_v3_*, 114 MB (engine_live, engine_seed_stage, laneI_N1), 178 MB clock, 3 variance tables. The engine pushes took 3 min 26 s because most arm dirs share byte-identical arrays. The 648 files still reported by `status` are under `raw/` and `results/` and are not part of this triage.

## (c) Box runbook
`docs/ops/aws_launch_chain.md` section 20 added. `scripts/ops_launch_spot_v1.sh`: the root volume was a hard-coded 100; it is now `VOLGB` with default 200, plus the AZ order note (`us-east-2c us-east-2b us-east-2a`, 2a last; the 2a spot was reclaimed after 8.3 min on 2026-10-14 and the 2b spot ran to completion). No other `box_*` script sets a volume size (`grep VolumeSize scripts/*.sh`).

## Commands run
- `git status --short data/processed`, `du -sk` per dir, `find ... -printf '%s'` for byte totals
- `grep -rl <path pattern> scripts src tests` per directory, `git ls-files`, `git ls-files -ci --exclude-standard`
- `.venv/Scripts/python.exe scripts/hf_sync_data.py status`, and a missing-set script (same method as `status`) before and after
- `.venv/Scripts/python.exe scripts/ops_hf_push_paths_v1.py clock/r7_A1 ... clock/r8_M2D`, then the same for `engine_v3_* engine_seed_stage engine_live free_throw/laneI_N1`
- `.venv/Scripts/python.exe scripts/hf_sync_data.py push --dirs team_rate_tables`
- `.venv/Scripts/python.exe -m pytest -q`: 847 passed, 1 skipped (288 s), run after the `.gitignore` and `hf_sync_data.py` edits
